import time as time_module
from collections.abc import Callable
from datetime import datetime
from typing import Any

import glfw
import imageio
import moderngl
import numpy as np
from loguru import logger
from PIL import Image

standard_vertex_shader = """
#version 460 core
in vec2 in_position;
out vec2 vs_uv;
void main() {
    vs_uv = (in_position + 1.0) * 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

shadertoy_vertex_shader = """
#version 330 core
in vec2 in_position;
out vec2 vs_uv;
void main() {
    vs_uv = (in_position + 1.0) * 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

# Default to standard vertex shader
vertex_shader_source = standard_vertex_shader
logger.opt(colors=True).info(
    f"<blue>Vertex Shader GLSL:\n{vertex_shader_source}</blue>"
)


def _prepare_shader_code(
    shader_input: Callable[..., Any] | str,
    backend_type: Any = None,
) -> str:
    """Prepare shader code from input.

    Args:
        shader_input: Shader function or GLSL string
        backend_type: Backend type to use for transpilation

    Returns:
        Generated GLSL code
    """
    if callable(shader_input):
        from py2glsl.transpiler import transpile

        if backend_type is not None:
            glsl_code, _ = transpile(shader_input, backend_type=backend_type)
        else:
            glsl_code, _ = transpile(shader_input)
    else:
        glsl_code = shader_input
    return glsl_code


def _init_context(
    size: tuple[int, int],
    windowed: bool,
    window_title: str = "GLSL Shader",
    use_gles: bool = False,
) -> tuple[moderngl.Context, glfw._GLFWwindow | None]:
    """Initialize ModernGL context and optional GLFW window.

    Args:
        size: Window size as (width, height)
        windowed: Whether to create a window or use offscreen rendering
        window_title: Title of the window
        use_gles: Whether to use OpenGL ES (for Shadertoy compatibility)

    Returns:
        Tuple of (ModernGL context, GLFW window or None)
    """
    if windowed:
        if not glfw.init():
            logger.error("Failed to initialize GLFW")
            raise RuntimeError("Failed to initialize GLFW")

        # Set OpenGL version based on whether we're using GLES
        if use_gles:
            # Use OpenGL compatibility profile for Shadertoy
            # Note: We don't use ES API when windowed as it's not well supported
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        else:
            # Use OpenGL 4.6 Core profile
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        window = glfw.create_window(size[0], size[1], window_title, None, None)
        if not window:
            logger.error("Failed to create GLFW window")
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        glfw.make_context_current(window)
        ctx = moderngl.create_context()
    else:
        window = None
        if use_gles:
            # Use OpenGL 3.3 Core for Shadertoy (more compatible than ES)
            ctx = moderngl.create_context(standalone=True, require=330)
        else:
            # Use OpenGL 4.6 for standalone
            ctx = moderngl.create_context(standalone=True, require=460)
    return ctx, window


def _compile_program(
    ctx: moderngl.Context,
    glsl_code: str,
    backend_type: Any = None,
) -> moderngl.Program:
    """Compile shader program.

    Args:
        ctx: ModernGL context
        glsl_code: Fragment shader code
        backend_type: Backend type used to generate the shader

    Returns:
        Compiled shader program
    """
    # Determine which vertex shader to use based on backend type
    from py2glsl.transpiler.backends.models import BackendType

    vertex_shader = standard_vertex_shader
    if backend_type == BackendType.SHADERTOY:
        vertex_shader = shadertoy_vertex_shader

    try:
        program = ctx.program(vertex_shader=vertex_shader, fragment_shader=glsl_code)
        # Store the backend type with the program for later reference
        if backend_type:
            # Add custom attribute to store backend type
            # This is safe because we control both Program creation and attribute usage
            # We have to use a bit of a hack with moderngl.Program
            # The moderngl.Program object doesn't have this attribute in type
            # definitions, but it does allow for custom attributes at runtime
            setattr(program, "_backend_type", backend_type)
        logger.info("Shader program compiled successfully")
        logger.info(f"Available uniforms: {list(program)}")
        return program
    except Exception as e:
        logger.error(f"Shader compilation error: {e}")
        raise


def _setup_primitives(
    ctx: moderngl.Context, program: moderngl.Program
) -> tuple[moderngl.Buffer, moderngl.VertexArray]:
    """Create vertex buffer and vertex array."""
    vertices = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype="f4")
    vbo = ctx.buffer(vertices)
    vao = ctx.simple_vertex_array(program, vbo, "in_position")
    return vbo, vao


def _setup_mouse_tracking(
    window: glfw._GLFWwindow, size: tuple[int, int]
) -> tuple[list[float], list[float]]:
    """Set up mouse tracking for windowed mode."""
    mouse_pos = [size[0] / 2, size[1] / 2]
    mouse_uv = [0.5, 0.5]

    def mouse_callback(_window: glfw._GLFWwindow, xpos: float, ypos: float) -> None:
        mouse_pos[0] = xpos
        mouse_pos[1] = ypos
        mouse_uv[0] = xpos / size[0]
        mouse_uv[1] = 1.0 - ypos / size[1]

    glfw.set_cursor_pos_callback(window, mouse_callback)
    return mouse_pos, mouse_uv


def _render_frame(
    ctx: moderngl.Context,
    program: moderngl.Program,
    vao: moderngl.VertexArray,
    target: moderngl.Framebuffer | moderngl.Context,
    size: tuple[int, int],
    time: float,
    uniforms: dict[str, float | tuple[float, ...]] | None,
    mouse_pos: list[float] | None = None,
    mouse_uv: list[float] | None = None,
) -> Any | None:  # Using Any to avoid numpy typing issues
    """Render a single frame."""
    if isinstance(target, moderngl.Framebuffer):
        target.use()
    # If it's a context, no need to call use()
    ctx.clear(0.0, 0.0, 0.0, 1.0)

    from py2glsl.transpiler.backends.models import BackendType

    uniforms = uniforms or {}
    default_uniforms = {}

    # Set uniforms based on backend type
    # Check if we're using the Shadertoy backend
    has_backend = hasattr(program, "_backend_type")
    # We have to use a bit of a hack with moderngl.Program
    is_shadertoy = False
    if has_backend:
        # The Program object allows custom attributes at runtime
        backend_type = getattr(program, "_backend_type")
        is_shadertoy = backend_type == BackendType.SHADERTOY
    if is_shadertoy:
        # Get current date for date uniform
        current_date = datetime.now()

        # Create all Shadertoy compatible uniforms
        default_uniforms = {
            "iResolution": (size[0], size[1], 1.0),  # x, y, pixel_ratio
            "iTime": time,  # shader playback time
            "iTimeDelta": 1.0 / 60.0,  # approx render time (default 60fps)
            "iFrame": int(time * 60),  # approximate frame number at 60fps
            # Current date (year, month, day, seconds)
            "iDate": (
                current_date.year,  # year
                current_date.month,  # month
                current_date.day,  # day
                (
                    current_date.hour * 3600
                    + current_date.minute * 60
                    + current_date.second
                ),  # seconds of the current day
            ),
            "iSampleRate": 44100.0,  # standard audio sample rate
            # Channel resolutions (assuming standard texture sizes)
            "iChannelResolution[0]": (256.0, 256.0, 0.0),
            "iChannelResolution[1]": (256.0, 256.0, 0.0),
            "iChannelResolution[2]": (256.0, 256.0, 0.0),
            "iChannelResolution[3]": (256.0, 256.0, 0.0),
            # Channel times (same as main time by default)
            "iChannelTime[0]": time,
            "iChannelTime[1]": time,
            "iChannelTime[2]": time,
            "iChannelTime[3]": time,
        }

        if mouse_pos and mouse_uv:
            # Shadertoy uses iMouse(x, y, click_x, click_y)
            # We don't track clicks, so use zeros for click coords
            default_uniforms["iMouse"] = (mouse_pos[0], mouse_pos[1], 0.0, 0.0)
    else:
        # Standard uniforms
        default_uniforms = {
            "u_resolution": size,
            "u_time": time,
            "u_aspect": size[0] / size[1],
        }
        if mouse_pos and mouse_uv:
            default_uniforms["u_mouse_pos"] = tuple(mouse_pos)
            default_uniforms["u_mouse_uv"] = tuple(mouse_uv)

    # Add any user-provided uniforms
    default_uniforms.update(uniforms)

    for name, value in default_uniforms.items():
        if name in program:
            # We need to handle moderngl uniform types which may be different
            uniform = program[name]
            try:
                # Try the most common approach first - for moderngl.Uniform
                if hasattr(uniform, "value"):
                    uniform.value = value
                # Some uniform types use write method
                elif hasattr(uniform, "write"):
                    uniform.write(value)
                else:
                    # For types we don't know how to handle, log a warning
                    logger.warning(f"Unknown uniform type for {name}, can't set value")
            except (AttributeError, TypeError) as e:
                # Log but don't fail
                logger.warning(f"Could not set uniform {name} to {value}: {e}")

    vao.render(moderngl.TRIANGLE_STRIP)
    if isinstance(target, moderngl.Framebuffer):
        data = target.read(components=4, dtype="f1")
        # Reshape the data to image dimensions
        img = np.frombuffer(data, dtype=np.uint8).reshape(size[1], size[0], 4)

        # CRITICAL FIX: Flip the image vertically
        # OpenGL has Y=0 at the bottom, but image formats have Y=0 at the top
        # Ensures consistent orientation across all output formats
        flipped_img = np.flipud(img)

        return flipped_img
    return None


def _cleanup(
    ctx: moderngl.Context,
    program: moderngl.Program,
    vbo: moderngl.Buffer,
    vao: moderngl.VertexArray,
    fbo: moderngl.Framebuffer | None = None,
    window: glfw._GLFWwindow | None = None,
) -> None:
    """Release rendering resources."""
    if fbo:
        fbo.release()
    vao.release()
    vbo.release()
    program.release()
    ctx.release()
    if window:
        glfw.terminate()


def animate(
    shader_input: Callable[..., Any] | str,
    size: tuple[int, int] = (1200, 800),
    window_title: str = "GLSL Shader",
    uniforms: dict[str, float | tuple[float, ...]] | None = None,
    backend_type: Any = None,
) -> None:
    """Run a real-time shader animation in a window.

    Args:
        shader_input: Shader function or GLSL string
        size: Window size as (width, height)
        window_title: Title of the window
        uniforms: Additional uniform values to pass to the shader
        backend_type: Backend type to use for transpilation (e.g., STANDARD, SHADERTOY)
    """
    from py2glsl.transpiler.backends.models import BackendType

    # Determine if we should use OpenGL ES for Shadertoy
    use_gles = backend_type == BackendType.SHADERTOY

    glsl_code = _prepare_shader_code(shader_input, backend_type)
    ctx, window = _init_context(
        size, windowed=True, window_title=window_title, use_gles=use_gles
    )
    program = _compile_program(ctx, glsl_code, backend_type)
    vbo, vao = _setup_primitives(ctx, program)
    mouse_pos, mouse_uv = _setup_mouse_tracking(window, size)

    try:
        frame_count = 0
        last_time = time_module.time()
        while not glfw.window_should_close(window):
            current_time = time_module.time()
            frame_count += 1
            if current_time - last_time >= 1.0:
                fps = frame_count / (current_time - last_time)
                logger.info(f"FPS: {fps:.2f}")
                frame_count = 0
                last_time = current_time

            _render_frame(
                ctx,
                program,
                vao,
                ctx.screen,
                size,
                glfw.get_time(),
                uniforms,
                mouse_pos,
                mouse_uv,
            )
            glfw.swap_buffers(window)
            glfw.poll_events()
    finally:
        _cleanup(ctx, program, vbo, vao, window=window)


def render_array(
    shader_input: Callable[..., Any] | str,
    size: tuple[int, int] = (1200, 800),
    time: float = 0.0,
    uniforms: dict[str, float | tuple[float, ...]] | None = None,
    backend_type: Any = None,
) -> Any:  # Use Any type to bypass mypy issues with numpy typings
    """Render shader to a numpy array.

    Args:
        shader_input: Shader function or GLSL string
        size: Image size as (width, height)
        time: Shader time value
        uniforms: Additional uniform values to pass to the shader
        backend_type: Backend type to use for transpilation (e.g., STANDARD, SHADERTOY)

    Returns:
        Numpy array containing the rendered image
    """
    from py2glsl.transpiler.backends.models import BackendType

    # Determine if we should use OpenGL ES for Shadertoy
    use_gles = backend_type == BackendType.SHADERTOY

    logger.info("Rendering to array")
    glsl_code = _prepare_shader_code(shader_input, backend_type)
    ctx, _ = _init_context(size, windowed=False, use_gles=use_gles)
    program = _compile_program(ctx, glsl_code, backend_type)
    vbo, vao = _setup_primitives(ctx, program)
    fbo = ctx.simple_framebuffer(size)

    try:
        result = _render_frame(ctx, program, vao, fbo, size, time, uniforms)
        if result is None:
            raise RuntimeError("Rendering failed to produce an image")
        return result
    finally:
        _cleanup(ctx, program, vbo, vao, fbo)


def render_image(
    shader_input: Callable[..., Any] | str,
    size: tuple[int, int] = (1200, 800),
    time: float = 0.0,
    uniforms: dict[str, float | tuple[float, ...]] | None = None,
    output_path: str | None = None,
    image_format: str = "PNG",
    backend_type: Any = None,
) -> Image.Image:
    """Render shader to a PIL Image.

    Args:
        shader_input: Shader function or GLSL string
        size: Image size as (width, height)
        time: Shader time value
        uniforms: Additional uniform values to pass to the shader
        output_path: Path to save the image, if desired
        image_format: Format to save the image in (e.g., "PNG", "JPEG")
        backend_type: Backend type to use for transpilation (e.g., STANDARD, SHADERTOY)

    Returns:
        PIL Image containing the rendered image
    """
    from py2glsl.transpiler.backends.models import BackendType

    # Determine if we should use OpenGL ES for Shadertoy
    use_gles = backend_type == BackendType.SHADERTOY

    logger.info("Rendering to image")
    glsl_code = _prepare_shader_code(shader_input, backend_type)
    ctx, _ = _init_context(size, windowed=False, use_gles=use_gles)
    program = _compile_program(ctx, glsl_code, backend_type)
    vbo, vao = _setup_primitives(ctx, program)
    fbo = ctx.simple_framebuffer(size)

    try:
        array = _render_frame(ctx, program, vao, fbo, size, time, uniforms)
        assert array is not None
        image = Image.fromarray(array, mode="RGBA")
        if output_path:
            image.save(output_path, format=image_format)
            logger.info(f"Image saved to {output_path}")
        return image
    finally:
        _cleanup(ctx, program, vbo, vao, fbo)


def render_gif(
    shader_input: Callable[..., Any] | str,
    size: tuple[int, int] = (1200, 800),
    duration: float = 5.0,
    fps: int = 30,
    uniforms: dict[str, float | tuple[float, ...]] | None = None,
    output_path: str | None = None,
    backend_type: Any = None,
    time_offset: float = 0.0,
) -> tuple[Image.Image, list[Any]]:
    """Render shader to an animated GIF, returning first frame and raw frames.

    Args:
        shader_input: Shader function or GLSL string
        size: Image size as (width, height)
        duration: Animation duration in seconds
        fps: Frames per second
        uniforms: Additional uniform values to pass to the shader
        output_path: Path to save the GIF, if desired
        backend_type: Backend type to use for transpilation (e.g., STANDARD, SHADERTOY)
        time_offset: Starting time for the animation (seconds)

    Returns:
        Tuple of (first frame as PIL Image, list of raw frames as numpy arrays)
    """
    from py2glsl.transpiler.backends.models import BackendType

    # Determine if we should use OpenGL ES for Shadertoy
    use_gles = backend_type == BackendType.SHADERTOY

    logger.info("Rendering to GIF")
    glsl_code = _prepare_shader_code(shader_input, backend_type)
    ctx, _ = _init_context(size, windowed=False, use_gles=use_gles)
    program = _compile_program(ctx, glsl_code, backend_type)
    vbo, vao = _setup_primitives(ctx, program)
    fbo = ctx.simple_framebuffer(size)

    try:
        num_frames = int(duration * fps)
        raw_frames: list[Any] = []
        pil_frames: list[Image.Image] = []
        for i in range(num_frames):
            # Add offset to make animations consistent with interactive mode
            frame_time = time_offset + (i / fps)
            array = _render_frame(ctx, program, vao, fbo, size, frame_time, uniforms)
            assert array is not None
            raw_frames.append(array)
            pil_frames.append(Image.fromarray(array, mode="RGBA"))

        if output_path:
            pil_frames[0].save(
                output_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=int(1000 / fps),
                loop=0,
            )
            logger.info(f"GIF saved to {output_path}")
        return pil_frames[0], raw_frames
    finally:
        _cleanup(ctx, program, vbo, vao, fbo)


def render_video(
    shader_input: Callable[..., Any] | str,
    size: tuple[int, int] = (1200, 800),
    duration: float = 5.0,
    fps: int = 30,
    output_path: str = "shader_output.mp4",
    codec: str = "h264",
    quality: int = 8,
    pixel_format: str = "yuv420p",
    uniforms: dict[str, float | tuple[float, ...]] | None = None,
    backend_type: Any = None,
    time_offset: float = 0.0,
) -> tuple[str, list[Any]]:
    """Render shader to a video file, returning path and raw frames.

    Args:
        shader_input: Shader function or GLSL string
        size: Image size as (width, height)
        duration: Video duration in seconds
        fps: Frames per second
        output_path: Path to save the video
        codec: Video codec (e.g., "h264", "vp9")
        quality: Video quality (0-10, higher is better)
        pixel_format: Pixel format (e.g., "yuv420p")
        uniforms: Additional uniform values to pass to the shader
        backend_type: Backend type to use for transpilation (e.g., STANDARD, SHADERTOY)
        time_offset: Starting time for the animation (seconds)

    Returns:
        Tuple of (output path, list of raw frames as numpy arrays)
    """
    from py2glsl.transpiler.backends.models import BackendType

    # Determine if we should use OpenGL ES for Shadertoy
    use_gles = backend_type == BackendType.SHADERTOY

    logger.info(f"Rendering to video file {output_path} with {codec} codec")
    glsl_code = _prepare_shader_code(shader_input, backend_type)
    ctx, _ = _init_context(size, windowed=False, use_gles=use_gles)
    program = _compile_program(ctx, glsl_code, backend_type)
    vbo, vao = _setup_primitives(ctx, program)
    fbo = ctx.simple_framebuffer(size)

    try:
        writer = imageio.get_writer(
            output_path,
            fps=fps,
            codec=codec,
            quality=quality,
            pixelformat=pixel_format,
        )
        num_frames = int(duration * fps)
        raw_frames: list[Any] = []
        for i in range(num_frames):
            # Add offset to make animations consistent with interactive mode
            frame_time = time_offset + (i / fps)
            array = _render_frame(ctx, program, vao, fbo, size, frame_time, uniforms)
            assert array is not None
            raw_frames.append(array)
            writer.append_data(array)
        writer.close()
        logger.info(f"Video saved to {output_path}")
        return output_path, raw_frames
    finally:
        _cleanup(ctx, program, vbo, vao, fbo)
