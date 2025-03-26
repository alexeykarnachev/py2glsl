import time as time_module
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import glfw
import imageio
import moderngl
import numpy as np
from loguru import logger
from PIL import Image

from py2glsl.transpiler.core.interfaces import RenderInterface
from py2glsl.transpiler.render.opengl import (
    ShadertoyOpenGLRenderer,
    StandardOpenGLRenderer,
)

# Define a custom error type for ModernGL errors
ModernGLError = Exception

# Type aliases for better abstraction
WindowHandle = Any  # Avoids direct dependency on glfw._GLFWwindow

@dataclass
class RenderContext:
    """Holds all resources for rendering.

    This class manages both OpenGL resources and rendering configuration,
    providing a unified interface for the rendering functions.
    """

    ctx: moderngl.Context
    program: moderngl.Program
    vbo: moderngl.Buffer
    vao: moderngl.VertexArray
    renderer: RenderInterface
    fbo: moderngl.Framebuffer | None = None
    window: WindowHandle | None = None

    def __del__(self) -> None:
        """Clean up resources when the context is garbage collected."""
        try:
            if self.fbo:
                self.fbo.release()
            if self.vao:
                self.vao.release()
            if self.vbo:
                self.vbo.release()
            if self.program:
                self.program.release()
            # Note: ctx and window are handled by _setup_rendering_context
        except Exception:
            # Ignore errors during cleanup
            pass


# Type for uniform values (could be single value, vector, or matrix)
UniformValue = float | tuple[float, ...] | tuple[tuple[float, ...], ...]

@dataclass
class FrameParams:
    """Parameters for rendering a single frame.

    This class contains all the information needed to render a single frame,
    including OpenGL resources, uniforms, and rendering configuration.
    """

    ctx: moderngl.Context
    program: moderngl.Program
    vao: moderngl.VertexArray
    target: moderngl.Framebuffer | moderngl.Context | None  # None for error handling
    size: tuple[int, int]
    time: float
    uniforms: dict[str, UniformValue] | None = None
    mouse_pos: list[float] | None = None
    mouse_uv: list[float] | None = None
    resolution: tuple[int, int] | None = None
    frame_num: int = 0
    renderer: RenderInterface | None = None


def _init_glfw(
    size: tuple[int, int],
    windowed: bool,
    window_title: str = "GLSL Shader",
    gl_version: tuple[int, int] = (4, 6),
    gl_profile: str = "core",
) -> tuple[moderngl.Context, glfw._GLFWwindow | None]:
    """Initialize ModernGL context and optional GLFW window.

    Args:
        size: Window size as (width, height)
        windowed: Whether to create a window or use offscreen rendering
        window_title: Title of the window
        gl_version: OpenGL version as (major, minor)
        gl_profile: OpenGL profile ("core" or "compatibility")

    Returns:
        Tuple of (ModernGL context, GLFW window or None)
    """
    major_version, minor_version = gl_version
    profile = gl_profile
    require_version = major_version * 100 + minor_version * 10  # e.g. 4.6 -> 460

    if windowed:
        try:
            if not glfw.init():
                raise RuntimeError("Failed to initialize GLFW")

            # Set OpenGL version from backend
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, major_version)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, minor_version)

            # Set profile from backend
            if profile == "core":
                glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            elif profile == "compatibility":
                glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)
            else:
                log_warning(f"Unknown OpenGL profile: {profile}, using core profile")
                glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)

            window = glfw.create_window(size[0], size[1], window_title, None, None)
            if not window:
                raise RuntimeError("Failed to create GLFW window")
        except Exception as e:
            log_error(f"GLFW initialization error: {e}")
            if 'glfw' in locals() and glfw._glfw is not None:
                glfw.terminate()
            raise
        glfw.make_context_current(window)

        # Setup key callback to close window on ESC key
        def key_callback(
            win: glfw._GLFWwindow, key: int, _scancode: int, action: int, _mods: int
        ) -> None:
            if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
                glfw.set_window_should_close(win, True)

        glfw.set_key_callback(window, key_callback)

        ctx = moderngl.create_context()
    else:
        window = None
        # Create standalone context with required OpenGL version
        ctx = moderngl.create_context(standalone=True, require=require_version)

    return ctx, window


def log_info(message: str, detached: bool = False) -> None:
    """Log info message if not in detached mode.

    Args:
        message: Message to log
        detached: Whether running in detached mode
    """
    if not detached:
        logger.info(message)


def log_warning(message: str, detached: bool = False) -> None:
    """Log warning message if not in detached mode.

    Args:
        message: Message to log
        detached: Whether running in detached mode
    """
    if not detached:
        logger.warning(message)


def log_error(message: str, detached: bool = False) -> None:
    """Log error message if not in detached mode.

    Args:
        message: Message to log
        detached: Whether running in detached mode
    """
    if not detached:
        logger.error(message)


def _compile_program(
    ctx: moderngl.Context,
    glsl_code: str,
    renderer: RenderInterface,
    detached: bool = False,
) -> moderngl.Program:
    """Compile shader program.

    Args:
        ctx: ModernGL context
        glsl_code: Fragment shader code
        renderer: The render interface to use
        detached: Whether to run in detached mode (no output/logging)

    Returns:
        Compiled shader program
    """
    # Get vertex shader from renderer
    vertex_shader = renderer.get_vertex_code()

    try:
        program = ctx.program(vertex_shader=vertex_shader, fragment_shader=glsl_code)
        # Store the renderer with the program for later reference
        program._renderer = renderer  # Add attribute directly

        log_info("Shader program compiled successfully", detached)
        log_info(f"Available uniforms: {list(program)}", detached)
        return program
    except Exception:
        log_error("Shader compilation error", detached)
        # Re-raise without arguments to preserve the original traceback
        raise


def _setup_primitives(
    ctx: moderngl.Context, program: moderngl.Program
) -> tuple[moderngl.Buffer, moderngl.VertexArray]:
    """Create vertex buffer and vertex array."""
    vertices = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype="f4")
    vbo = ctx.buffer(vertices)
    vao = ctx.simple_vertex_array(program, vbo, "in_position")
    return vbo, vao


@contextmanager
def _setup_rendering_context(
    shader_input: Callable[..., Any] | str,
    size: tuple[int, int],
    windowed: bool = False,
    window_title: str = "GLSL Shader",
    backend_type: Any = None,
    detached: bool = False,
) -> Generator[RenderContext, None, None]:
    """Sets up all rendering resources and cleans them up when done.

    Args:
        shader_input: Shader function or GLSL string
        size: Window/framebuffer size as (width, height)
        windowed: Whether to create a window or use offscreen rendering
        window_title: Title of the window (if windowed is True)
        backend_type: Backend type enum
        detached: Whether to run in detached mode (no output/logging)

    Yields:
        RenderContext object containing all rendering resources
    """
    from py2glsl.transpiler.backends.models import BackendType
    from py2glsl.transpiler.core.interfaces import TargetLanguageType

    # Create the appropriate renderer based on backend type
    renderer: RenderInterface
    if backend_type == BackendType.SHADERTOY:
        renderer = ShadertoyOpenGLRenderer()
        target_type = TargetLanguageType.SHADERTOY
    else:
        renderer = StandardOpenGLRenderer()
        target_type = TargetLanguageType.GLSL

    # Prepare shader code
    if callable(shader_input):
        from py2glsl.transpiler import transpile

        glsl_code, _ = transpile(shader_input, target_type=target_type)
    else:
        glsl_code = shader_input

    # Get OpenGL requirements from the renderer
    reqs = renderer.get_render_requirements()
    gl_version = (reqs["version_major"], reqs["version_minor"])
    gl_profile = reqs["profile"]

    # Initialize context and window
    ctx, window = _init_glfw(size, windowed, window_title, gl_version, gl_profile)

    # Compile shader program
    program = _compile_program(ctx, glsl_code, renderer, detached)

    # Setup rendering primitives
    vbo, vao = _setup_primitives(ctx, program)

    # Create framebuffer if needed (for offscreen rendering)
    fbo = None
    if not windowed:
        fbo = ctx.simple_framebuffer(size)

    # Create context object with the renderer
    render_ctx = RenderContext(
        ctx=ctx,
        program=program,
        vbo=vbo,
        vao=vao,
        renderer=renderer,
        fbo=fbo,
        window=window,
    )

    try:
        yield render_ctx
    finally:
        # Release all OpenGL resources
        if render_ctx.fbo:
            render_ctx.fbo.release()
        render_ctx.vao.release()
        render_ctx.vbo.release()
        render_ctx.program.release()
        render_ctx.ctx.release()
        if render_ctx.window:
            glfw.terminate()


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


def _render_frame(params: FrameParams) -> np.ndarray | None:
    """Render a single frame.

    Args:
        params: Frame rendering parameters

    Returns:
        Numpy array with rendered image if rendering to a framebuffer, None otherwise
    """
    if params.target is None:
        log_error("No target specified for rendering")
        return None

    # Set the appropriate rendering target
    if isinstance(params.target, moderngl.Framebuffer):
        params.target.use()
    elif not isinstance(params.target, moderngl.Context):
        log_error(f"Unsupported target type: {type(params.target)}")
        return None

    # Clear the screen/framebuffer
    params.ctx.clear(0.0, 0.0, 0.0, 1.0)

    # Prepare standard uniforms dict - start with user-provided uniforms
    standard_uniforms: dict[str, UniformValue] = {}
    if params.uniforms:
        # Copy from user-provided uniforms
        for key, value in params.uniforms.items():
            standard_uniforms[key] = value

    # Always include these standard uniforms
    base_uniforms: dict[str, UniformValue] = {
        "u_resolution": params.size,
        "u_time": params.time,
        "u_aspect": params.size[0] / params.size[1],
    }

    # Add mouse uniforms if available
    if params.mouse_pos and params.mouse_uv:
        base_uniforms["u_mouse_pos"] = tuple(params.mouse_pos)
        base_uniforms["u_mouse_uv"] = tuple(params.mouse_uv)

    # Combine base uniforms with any user-provided uniforms
    for key, value in base_uniforms.items():
        standard_uniforms[key] = value

    # Get renderer from FrameParams or from program object
    renderer = params.renderer
    if renderer is None and hasattr(params.program, "_renderer"):
        # This is a custom attribute added to the ModernGL program object
        renderer = getattr(params.program, "_renderer", None)

    # Apply renderer-specific uniform transformations
    if renderer:
        # Transform standard uniforms to renderer-specific uniforms
        uniforms_to_set = renderer.setup_uniforms(standard_uniforms)
    else:
        # Fallback to just using standard uniforms
        uniforms_to_set = standard_uniforms

    # Set all uniforms on the program
    for name, value in uniforms_to_set.items():
        if name in params.program:
            uniform = params.program[name]
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

    # Render the quad
    params.vao.render(moderngl.TRIANGLE_STRIP)

    # Return pixel data if rendering to framebuffer
    if params.target is not None and isinstance(params.target, moderngl.Framebuffer):
        data = params.target.read(components=4, dtype="f1")
        # Reshape the data to image dimensions
        img = np.frombuffer(data, dtype=np.uint8).reshape(
            params.size[1], params.size[0], 4
        )

        # CRITICAL FIX: Flip the image vertically
        # OpenGL has Y=0 at the bottom, but image formats have Y=0 at the top
        # Ensures consistent orientation across all output formats
        flipped_img = np.flipud(img)

        return flipped_img

    # If rendering to screen or no target, no pixel data to return
    return None


def _configure_frame_rate(fps: int, detached: bool = False) -> tuple[float, float]:
    """Configure frame rate settings for animation.

    Args:
        fps: Target frame rate (0 = unlimited)
        detached: Whether to run in detached mode (no output/logging)

    Returns:
        Tuple of (frame_interval, previous_time)
    """
    if fps > 0:
        # Try to use vertical sync if available
        try:
            glfw.swap_interval(1)
            log_info("Using vsync for frame timing", detached)
        except Exception:
            log_info("Could not set vsync, using manual timing", detached)

        frame_interval = 1.0 / fps
        log_info(f"Target FPS: {fps}", detached)
    else:
        # Run at maximum speed
        glfw.swap_interval(0)  # Disable vsync
        frame_interval = 0
        log_info("Target FPS: unlimited", detached)

    # Initial time tracking
    previous_time = time_module.time()
    return frame_interval, previous_time


def _reload_shader(
    render_ctx: RenderContext,
    reload_function: Callable[[], tuple[str, Any]],
    detached: bool = False,
) -> None:
    """Reload shader during animation.

    Args:
        render_ctx: Current rendering context
        reload_function: Function that returns new shader code and backend type
        detached: Whether to run in detached mode (no output/logging)
    """
    try:
        # Get new shader code and backend type
        new_shader_code, new_backend_type = reload_function()

        # Release old program
        render_ctx.program.release()

        # Create new renderer based on backend type
        from py2glsl.transpiler.backends.models import BackendType

        # Always use a specific backend type
        if new_backend_type == BackendType.SHADERTOY:
            render_ctx.renderer = ShadertoyOpenGLRenderer()
        else:
            render_ctx.renderer = StandardOpenGLRenderer()

        # Recompile program
        render_ctx.program = _compile_program(
            render_ctx.ctx, new_shader_code, render_ctx.renderer, detached
        )

        # Recreate VAO
        render_ctx.vao.release()
        _, render_ctx.vao = _setup_primitives(render_ctx.ctx, render_ctx.program)

        log_info("Shader reloaded successfully", detached)
    except ModernGLError as e:
        log_error(f"GL error when reloading shader: {e}", detached)
    except ValueError as e:
        log_error(f"Shader code error: {e}", detached)
    except Exception as e:
        log_error(f"Error reloading shader: {e}", detached)


def _should_render_frame(
    fps: int, lag: float, frame_interval: float
) -> tuple[bool, float]:
    """Determine if a frame should be rendered based on FPS settings.

    Args:
        fps: Target frame rate
        lag: Current lag accumulator
        frame_interval: Time interval between frames

    Returns:
        Tuple of (should_render, updated_lag)
    """
    if fps <= 0:
        return True, lag

    if lag < frame_interval:
        return False, lag

    # Consume a frame worth of time
    lag -= frame_interval

    # Prevent spiral of death (if rendering is too slow)
    if lag > frame_interval * 5:
        lag = 0.0

    return True, lag


def animate(
    shader_input: Callable[..., Any] | str,
    size: tuple[int, int] = (1200, 800),
    window_title: str = "GLSL Shader",
    uniforms: dict[str, UniformValue] | None = None,
    backend_type: Any = None,
    fps: int = 0,  # 0 means unlimited
    reload_callback: Callable[[], bool] | None = None,
    reload_function: Callable[[], tuple[str, Any]] | None = None,
    detached: bool = False,
    max_runtime: float | None = None,
) -> None:
    """Run a real-time shader animation in a window.

    Args:
        shader_input: Shader function or GLSL string
        size: Window size as (width, height)
        window_title: Title of the window
        uniforms: Additional uniform values to pass to the shader
        backend_type: Backend type (e.g., BackendType.SHADERTOY)
        fps: Target frame rate (0 = unlimited)
        reload_callback: Function that returns True when shader should be reloaded
        reload_function: Function that returns new shader code and backend type
        detached: Whether to run in detached mode (no output/logging)
        max_runtime: Maximum runtime in seconds (stops animation after this time)
    """
    with _setup_rendering_context(
        shader_input,
        size,
        windowed=True,
        window_title=window_title,
        backend_type=backend_type,
        detached=detached,
    ) as render_ctx:
        # Setup mouse tracking and FPS tracking
        mouse_pos, mouse_uv = _setup_mouse_tracking(render_ctx.window, size)
        frame_count, fps_timer = 0, time_module.time()
        start_time = time_module.time()

        # Configure frame rate
        frame_interval, previous_time = _configure_frame_rate(fps, detached)
        lag = 0.0

        # Print startup message
        if max_runtime:
            msg = f"Running animation at {fps}fps for {max_runtime:.2f}s"
            msg += " (press ESC to exit)"
            log_info(msg, detached)
        else:
            log_info(f"Running animation at {fps}fps (press ESC to exit)", detached)

        # Main animation loop
        while not glfw.window_should_close(render_ctx.window):
            # Check if we've exceeded max_runtime
            elapsed_total = time_module.time() - start_time
            if max_runtime is not None and elapsed_total >= max_runtime:
                log_info(
                    f"Maximum runtime of {max_runtime:.2f}s reached, stopping",
                    detached
                )
                break
            # Calculate time delta with fixed time step approach
            current_time = time_module.time()
            elapsed = current_time - previous_time
            previous_time = current_time
            lag += elapsed

            # Process input
            glfw.poll_events()

            # Reload shader if needed
            if reload_callback and reload_function and reload_callback():
                _reload_shader(render_ctx, reload_function, detached)

            # Check if we should render this frame
            should_render, lag = _should_render_frame(fps, lag, frame_interval)
            if not should_render:
                continue

            # FPS measurement and logging
            frame_count += 1
            if current_time - fps_timer >= 1.0:
                measured_fps = frame_count / (current_time - fps_timer)
                log_info(f"FPS: {measured_fps:.2f}", detached)
                frame_count, fps_timer = 0, current_time

            # Create frame parameters and render
            frame_params = FrameParams(
                ctx=render_ctx.ctx,
                program=render_ctx.program,
                vao=render_ctx.vao,
                target=render_ctx.ctx.screen,
                size=size,
                time=glfw.get_time(),
                uniforms=uniforms,
                mouse_pos=mouse_pos,
                mouse_uv=mouse_uv,
                renderer=render_ctx.renderer,
            )
            _render_frame(frame_params)
            glfw.swap_buffers(render_ctx.window)

            # Small delay to prevent 100% CPU usage when not vsync limited
            if fps <= 0:
                time_module.sleep(0.001)


def render_array(
    shader_input: Callable[..., Any] | str,
    size: tuple[int, int] = (1200, 800),
    time: float = 0.0,
    uniforms: dict[str, UniformValue] | None = None,
    backend_type: Any = None,
    detached: bool = False,
) -> np.ndarray:
    """Render shader to a numpy array.

    Args:
        shader_input: Shader function or GLSL string
        size: Image size as (width, height)
        time: Shader time value
        uniforms: Additional uniform values to pass to the shader
        backend_type: Backend type (e.g., BackendType.SHADERTOY)
        detached: Whether to run in detached mode (no output/logging)

    Returns:
        Numpy array containing the rendered image
    """
    log_info("Rendering to array", detached)

    with _setup_rendering_context(
        shader_input,
        size,
        windowed=False,
        backend_type=backend_type,
        detached=detached,
    ) as render_ctx:
        # Create frame parameters
        frame_params = FrameParams(
            ctx=render_ctx.ctx,
            program=render_ctx.program,
            vao=render_ctx.vao,
            target=render_ctx.fbo,
            size=size,
            time=time,
            uniforms=uniforms,
        )

        # Render frame
        result = _render_frame(frame_params)

        if result is None:
            raise RuntimeError("Rendering failed to produce an image")
        return result


def render_image(
    shader_input: Callable[..., Any] | str,
    size: tuple[int, int] = (1200, 800),
    time: float = 0.0,
    uniforms: dict[str, UniformValue] | None = None,
    output_path: str | None = None,
    image_format: str = "PNG",
    backend_type: Any = None,
    detached: bool = False,
) -> Image.Image:
    """Render shader to a PIL Image.

    Args:
        shader_input: Shader function or GLSL string
        size: Image size as (width, height)
        time: Shader time value
        uniforms: Additional uniform values to pass to the shader
        output_path: Path to save the image, if desired
        image_format: Format to save the image in (e.g., "PNG", "JPEG")
        backend_type: Backend type (e.g., BackendType.SHADERTOY)
        detached: Whether to run in detached mode (no output/logging)

    Returns:
        PIL Image containing the rendered image
    """
    log_info("Rendering to image", detached)

    with _setup_rendering_context(
        shader_input, size, windowed=False, backend_type=backend_type, detached=detached
    ) as render_ctx:
        # Create frame parameters
        frame_params = FrameParams(
            ctx=render_ctx.ctx,
            program=render_ctx.program,
            vao=render_ctx.vao,
            target=render_ctx.fbo,
            size=size,
            time=time,
            uniforms=uniforms,
        )

        # Render frame
        array = _render_frame(frame_params)

        if array is None:
            raise RuntimeError("Rendering failed to produce an image")
        image = Image.fromarray(array, mode="RGBA")
        if output_path:
            image.save(output_path, format=image_format)
            logger.info(f"Image saved to {output_path}")
        return image


def render_gif(
    shader_input: Callable[..., Any] | str,
    size: tuple[int, int] = (1200, 800),
    duration: float = 5.0,
    fps: int = 30,
    uniforms: dict[str, UniformValue] | None = None,
    output_path: str | None = None,
    time_offset: float = 0.0,
    backend_type: Any = None,
    detached: bool = False,
) -> tuple[Image.Image, list[np.ndarray]]:
    """Render shader to an animated GIF, returning first frame and raw frames.

    Args:
        shader_input: Shader function or GLSL string
        size: Image size as (width, height)
        duration: Animation duration in seconds
        fps: Frames per second
        uniforms: Additional uniform values to pass to the shader
        output_path: Path to save the GIF, if desired
        time_offset: Starting time for the animation (seconds)
        backend_type: Backend type (e.g., BackendType.SHADERTOY)
        detached: Whether to run in detached mode (no output/logging)

    Returns:
        Tuple of (first frame as PIL Image, list of raw frames as numpy arrays)
    """
    log_info("Rendering to GIF", detached)

    with _setup_rendering_context(
        shader_input, size, windowed=False, backend_type=backend_type, detached=detached
    ) as render_ctx:
        num_frames = int(duration * fps)
        raw_frames: list[Any] = []
        pil_frames: list[Image.Image] = []

        for i in range(num_frames):
            # Add offset to make animations consistent with interactive mode
            frame_time = time_offset + (i / fps)

            # Create frame parameters
            frame_params = FrameParams(
                ctx=render_ctx.ctx,
                program=render_ctx.program,
                vao=render_ctx.vao,
                target=render_ctx.fbo,
                size=size,
                time=frame_time,
                uniforms=uniforms,
            )

            # Render frame
            array = _render_frame(frame_params)

            if array is None:
                raise RuntimeError(f"Rendering failed on frame {i+1}/{num_frames}")
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
            log_info(f"GIF saved to {output_path}", detached)
        return pil_frames[0], raw_frames


def render_video(
    shader_input: Callable[..., Any] | str,
    size: tuple[int, int] = (1200, 800),
    duration: float = 5.0,
    fps: int = 30,
    output_path: str = "shader_output.mp4",
    codec: str = "h264",
    quality: int = 8,
    pixel_format: str = "yuv420p",
    uniforms: dict[str, UniformValue] | None = None,
    time_offset: float = 0.0,
    backend_type: Any = None,
    detached: bool = False,
) -> tuple[str, list[np.ndarray]]:
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
        time_offset: Starting time for the animation (seconds)
        backend_type: Backend type (e.g., BackendType.SHADERTOY)
        detached: Whether to run in detached mode (no output/logging)

    Returns:
        Tuple of (output path, list of raw frames as numpy arrays)
    """
    log_info(f"Rendering to video file {output_path} with {codec} codec", detached)

    with _setup_rendering_context(
        shader_input, size, windowed=False, backend_type=backend_type, detached=detached
    ) as render_ctx:
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

            # Create frame parameters
            frame_params = FrameParams(
                ctx=render_ctx.ctx,
                program=render_ctx.program,
                vao=render_ctx.vao,
                target=render_ctx.fbo,
                size=size,
                time=frame_time,
                uniforms=uniforms,
            )

            # Render frame
            array = _render_frame(frame_params)

            if array is None:
                raise RuntimeError(f"Rendering failed on frame {i+1}/{num_frames}")
            raw_frames.append(array)
            writer.append_data(array)

        writer.close()
        log_info(f"Video saved to {output_path}", detached)
        return output_path, raw_frames
