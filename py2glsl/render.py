import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, TypeVar, cast

import glfw
import moderngl
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from PIL import Image

# Type for ModernGL window
GLFWwindow = TypeVar("GLFWwindow", bound=glfw._GLFWwindow)


# Define more specific protocol for ModernGL uniform handling
class UniformProtocol(Protocol):
    value: float | int | bool | tuple[float, float] | Sequence[float]
    array_length: int
    dimension: int
    dtype: str


# Vertex shader source
vertex_shader_source = """
#version 460 core
in vec2 in_position;
out vec2 vs_uv;
void main() {
    vs_uv = (in_position + 1.0) * 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""
logger.opt(colors=True).info(
    f"<blue>Vertex Shader GLSL:\n{vertex_shader_source}</blue>"
)


def _prepare_shader_code(
    shader_input: Callable[..., None] | str,
) -> tuple[str, set[str]]:
    """Prepare shader code from input.

    Transpiles a function to GLSL or uses direct GLSL code.

    Args:
        shader_input: Either a shader function or GLSL code string

    Returns:
        Tuple of (GLSL code, set of used uniforms)
    """
    if callable(shader_input):
        # If a function is provided, transpile it to GLSL
        from py2glsl.transpiler import transpile

        glsl_code, used_uniforms = transpile(shader_input)
    else:
        # If GLSL code is provided directly, use it as-is
        glsl_code = shader_input
        used_uniforms = set()
        logger.warning("No uniform information provided with GLSL code")

    return glsl_code, used_uniforms


def _init_glfw_window(
    size: tuple[int, int], window_title: str
) -> tuple[glfw._GLFWwindow, moderngl.Context]:
    """Initialize GLFW window and ModernGL context.

    Args:
        size: Window size as (width, height) tuple
        window_title: Title for the display window

    Returns:
        Tuple of (GLFW window, ModernGL context)

    Raises:
        RuntimeError: If GLFW initialization or window creation fails
    """
    if not glfw.init():
        logger.error("Failed to initialize GLFW")
        raise RuntimeError("Failed to initialize GLFW")

    # Set OpenGL version and profile
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    # Create window
    window = glfw.create_window(size[0], size[1], window_title, None, None)
    if not window:
        logger.error("Failed to create GLFW window")
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    # Create ModernGL context
    glfw.make_context_current(window)
    ctx = moderngl.create_context()
    logger.info(f"OpenGL version: {ctx.version_code}")

    return window, ctx


def _compile_shader_program(ctx: moderngl.Context, glsl_code: str) -> moderngl.Program:
    """Compile shader program from GLSL code.

    Args:
        ctx: ModernGL context
        glsl_code: GLSL fragment shader code

    Returns:
        Compiled shader program

    Raises:
        Exception: If shader compilation fails
    """
    try:
        program = ctx.program(
            vertex_shader=vertex_shader_source, fragment_shader=glsl_code
        )
        logger.info("Shader program compiled successfully")

        # Log available uniforms for debugging
        available_uniforms = list(program)
        logger.info(f"Available uniforms: {available_uniforms}")
        return program
    except Exception as e:
        logger.error(f"Shader compilation error: {e}")
        glfw.terminate()
        raise


def _create_rendering_primitives(
    ctx: moderngl.Context, program: moderngl.Program
) -> tuple[moderngl.Buffer, moderngl.VertexArray]:
    """Create vertex buffer and vertex array for rendering.

    Args:
        ctx: ModernGL context
        program: Shader program

    Returns:
        Tuple of (vertex buffer, vertex array)
    """
    # Full-screen quad vertices
    vertices = np.array(
        [
            -1.0,
            -1.0,  # Bottom-left
            1.0,
            -1.0,  # Bottom-right
            -1.0,
            1.0,  # Top-left
            1.0,
            1.0,  # Top-right
        ],
        dtype="f4",
    )
    vbo = ctx.buffer(vertices)
    vao = ctx.simple_vertex_array(program, vbo, "in_position")
    return vbo, vao


def _setup_mouse_tracking(
    window: glfw._GLFWwindow, size: tuple[int, int]
) -> tuple[list[float], list[float]]:
    """Set up mouse position tracking.

    Args:
        window: GLFW window
        size: Window size as (width, height) tuple

    Returns:
        Tuple of (mouse_pos, mouse_uv) lists for tracking
    """
    # Initialize mouse position and UV coordinates
    mouse_pos: list[float] = [0.0, 0.0]
    mouse_uv: list[float] = [0.5, 0.5]  # Default centered

    # Define mouse callback function
    def mouse_callback(window: GLFWwindow, xpos: float, ypos: float) -> None:
        mouse_pos[0] = xpos
        mouse_pos[1] = ypos
        mouse_uv[0] = xpos / size[0]
        mouse_uv[1] = 1.0 - (ypos / size[1])  # Flip Y coordinate

    # Set the callback
    glfw.set_cursor_pos_callback(window, mouse_callback)
    return mouse_pos, mouse_uv


def _update_common_uniforms(
    program: moderngl.Program,
    size: tuple[int, int],
    mouse_pos: list[float],
    mouse_uv: list[float],
) -> None:
    """Update common uniform values in the shader program.

    Args:
        program: Shader program
        size: Window size as (width, height) tuple
        mouse_pos: Mouse position in window coordinates
        mouse_uv: Mouse position in UV coordinates
    """
    # Resolution uniform
    if "u_resolution" in program:
        uniform = cast(UniformProtocol, program["u_resolution"])
        uniform.value = (size[0], size[1])

    # Time uniform
    if "u_time" in program:
        uniform = cast(UniformProtocol, program["u_time"])
        uniform.value = glfw.get_time()

    # Aspect ratio uniform
    if "u_aspect" in program:
        uniform = cast(UniformProtocol, program["u_aspect"])
        uniform.value = size[0] / size[1]

    # Mouse position uniforms
    if "u_mouse_pos" in program:
        uniform = cast(UniformProtocol, program["u_mouse_pos"])
        uniform.value = mouse_pos

    if "u_mouse_uv" in program:
        uniform = cast(UniformProtocol, program["u_mouse_uv"])
        uniform.value = mouse_uv


def _update_custom_uniforms(program: moderngl.Program) -> None:
    """Update any custom uniforms with default values.

    Args:
        program: Shader program
    """
    common_uniforms = {
        "u_resolution",
        "u_time",
        "u_aspect",
        "u_mouse_pos",
        "u_mouse_uv",
    }

    # Set defaults for any uniforms not handled above
    for uniform_name in program:
        if uniform_name not in common_uniforms:
            try:
                uniform = cast(UniformProtocol, program[uniform_name])
                if uniform.array_length == 1 and uniform.dimension == 1:
                    if uniform.dtype.startswith("f"):
                        uniform.value = 1.0  # Default float value
                    elif uniform.dtype.startswith("i"):
                        uniform.value = 1  # Default int value
                    elif uniform.dtype.startswith("b"):
                        uniform.value = True  # Default bool value
            except Exception as e:
                logger.warning(f"Could not set uniform {uniform_name}: {e}")


@dataclass
class RenderContext:
    """Rendering context for the shader animation loop.

    Attributes:
        window: GLFW window
        ctx: ModernGL context
        program: Shader program
        vao: Vertex array
        size: Window size as (width, height) tuple
        mouse_pos: Mouse position in window coordinates
        mouse_uv: Mouse position in UV coordinates
    """

    window: glfw._GLFWwindow
    ctx: moderngl.Context
    program: moderngl.Program
    vao: moderngl.VertexArray
    size: tuple[int, int]
    mouse_pos: list[float]
    mouse_uv: list[float]


def _run_render_loop(render_ctx: RenderContext) -> None:
    """Run the main rendering loop.

    Args:
        render_ctx: Rendering context containing all necessary objects and state
    """
    frame_count = 0
    last_time = time.time()

    while not glfw.window_should_close(render_ctx.window):
        current_time = time.time()
        frame_count += 1

        # Calculate and display FPS once per second
        if current_time - last_time >= 1.0:
            fps = frame_count / (current_time - last_time)
            logger.info(f"FPS: {fps:.2f}")
            frame_count = 0
            last_time = current_time

        # Update uniform values
        _update_common_uniforms(
            render_ctx.program,
            render_ctx.size,
            render_ctx.mouse_pos,
            render_ctx.mouse_uv,
        )
        _update_custom_uniforms(render_ctx.program)

        # Render the frame
        render_ctx.ctx.clear(0.0, 0.0, 0.0, 1.0)
        render_ctx.vao.render(moderngl.TRIANGLE_STRIP)
        glfw.swap_buffers(render_ctx.window)
        glfw.poll_events()


def animate(
    shader_input: Callable[..., None] | str,
    used_uniforms: set[str] | None = None,
    size: tuple[int, int] = (1200, 800),
    window_title: str = "GLSL Shader",
) -> None:
    """Run the shader with an animation loop and FPS calculation.

    Args:
        shader_input: Either a shader function or GLSL code string
        used_uniforms: Set of uniform names used by the shader
            (only needed if shader_input is GLSL code)
        size: Window size as (width, height) tuple
        window_title: Title for the display window

    Note:
        This function opens a window and runs the shader in real-time.
        It will block until the window is closed.
    """
    try:
        # Prepare shader code
        glsl_code, _ = _prepare_shader_code(shader_input)
        # Note: used_uniforms parameter is for compatibility with the previous API
        # but we don't actually use it in the rendering process

        # Initialize GLFW and ModernGL
        window, ctx = _init_glfw_window(size, window_title)

        # Compile shader program
        program = _compile_shader_program(ctx, glsl_code)

        # Create rendering primitives
        vbo, vao = _create_rendering_primitives(ctx, program)

        # Set up mouse tracking
        mouse_pos, mouse_uv = _setup_mouse_tracking(window, size)

        # Create render context
        render_ctx = RenderContext(
            window=window,
            ctx=ctx,
            program=program,
            vao=vao,
            size=size,
            mouse_pos=mouse_pos,
            mouse_uv=mouse_uv,
        )

        # Run the render loop
        _run_render_loop(render_ctx)

    finally:
        # Clean up GLFW resources
        glfw.terminate()


def render_array(
    glsl_code: str, size: tuple[int, int] = (1200, 800)
) -> NDArray[np.uint8]:
    """Render the shader to a numpy array.

    Args:
        glsl_code: GLSL code for the fragment shader
        size: Width and height of the output image

    Returns:
        A numpy RGBA array of shape (height, width, 4) with values in range 0-255
    """
    logger.info("Rendering to array")
    # Placeholder implementation - would actually render the shader
    return np.zeros((*size[::-1], 4), dtype=np.uint8)  # Shape: (height, width, 4)


def render_gif(
    glsl_code: str,
    size: tuple[int, int] = (1200, 800),
    duration: float = 5.0,
    fps: int = 30,
    output_path: str | None = None,
) -> Image.Image:
    """Render the shader to an animated GIF.

    Args:
        glsl_code: GLSL code for the fragment shader
        size: Width and height of the output image
        duration: Duration of the GIF in seconds
        fps: Frames per second
        output_path: Optional path to save the GIF

    Returns:
        A PIL Image object containing the animated GIF
    """
    logger.info("Rendering to GIF")

    # Calculate number of frames
    num_frames = int(duration * fps)

    # Placeholder - would actually render the shader at different timestamps
    frames = []
    for _i in range(num_frames):
        # For each frame, you would actually set time uniform and render shader
        frame_array = np.zeros((*size[::-1], 3), dtype=np.uint8)
        frame = Image.fromarray(frame_array, mode="RGB")
        frames.append(frame)

    # Create a simple animation
    result = frames[0].copy()
    if output_path:
        result.save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / fps),  # milliseconds per frame
            loop=0,  # loop forever
        )

    return result


@dataclass
class VideoConfig:
    """Configuration for video rendering.

    Attributes:
        size: Width and height of the output image
        duration: Duration of the video in seconds
        fps: Frames per second
        output_path: Path to save the video file
        codec: Video codec to use (e.g., 'h264', 'vp9')
        quality: Video quality (1-10, with 10 being highest quality)
        pixel_format: Pixel format for the video
    """

    size: tuple[int, int] = field(default_factory=lambda: (1200, 800))
    duration: float = 5.0
    fps: int = 30
    output_path: str = "shader_output.mp4"
    codec: str = "h264"
    quality: int = 8
    pixel_format: str = "yuv420p"


def render_video(
    glsl_code: str,
    config: VideoConfig | None = None,
) -> str:
    """Render the shader to a video file.

    Args:
        glsl_code: GLSL code for the fragment shader
        config: Video configuration parameters

    Returns:
        Path to the generated video file
    """
    # Use default config if none provided
    if config is None:
        config = VideoConfig()

    logger.info(
        f"Rendering to video file {config.output_path} with {config.codec} codec"
    )

    try:
        import imageio

        writer = imageio.get_writer(
            config.output_path,
            fps=config.fps,
            codec=config.codec,
            quality=config.quality,
            pixelformat=config.pixel_format,
        )

        # Calculate number of frames
        num_frames = int(config.duration * config.fps)

        # For each frame, render the shader at the appropriate time
        for _i in range(num_frames):
            # Placeholder - would actually render the shader using time_value
            # Create a blank frame that would be filled by actual shader code
            frame = np.zeros((*config.size[::-1], 3), dtype=np.uint8)
            writer.append_data(frame)

        writer.close()
        logger.info(f"Video saved to {config.output_path}")
        return config.output_path

    except ImportError:
        logger.warning(
            "imageio module not available. Video rendering requires imageio."
        )
        # Create an empty file to indicate the operation was attempted
        with open(config.output_path, "w") as f:
            f.write("# Video rendering requires imageio module")
        return config.output_path


def render_image(
    glsl_code: str,
    size: tuple[int, int] = (1200, 800),
    output_path: str | None = None,
    image_format: str = "PNG",
) -> Image.Image:
    """Render the shader to a single image.

    Args:
        glsl_code: GLSL code for the fragment shader
        size: Width and height of the output image (width, height)
        output_path: Optional path to save the image
        image_format: Output format when saving (PNG, JPEG, etc.)

    Returns:
        A PIL Image object containing the rendered image
    """
    logger.info("Rendering shader to image")

    # Placeholder implementation - would actually render the shader
    # Create a blank image that would be filled by actual shader code
    width, height = size
    array = np.zeros((height, width, 3), dtype=np.uint8)

    # Convert to PIL Image
    image = Image.fromarray(array, mode="RGB")

    # Save if output path is provided
    if output_path:
        image.save(output_path, format=image_format)
        logger.info(f"Image saved to {output_path}")

    return image
