"""Rendering module for py2glsl shaders.

This module provides functions for rendering shaders to various outputs:
- animate(): Real-time interactive preview in a window
- render_array(): Render to a numpy array
- render_image(): Render to a PIL Image
- render_gif(): Render to an animated GIF
- render_video(): Render to a video file
"""

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

from py2glsl.transpiler import TargetType, transpile_to_result
from py2glsl.transpiler.target import Target


@dataclass
class RenderContext:
    """Holds all resources for rendering."""

    ctx: moderngl.Context
    program: moderngl.Program
    vbo: moderngl.Buffer
    vao: moderngl.VertexArray
    target: Target
    fbo: moderngl.Framebuffer | None = None
    window: glfw._GLFWwindow | None = None


@dataclass
class FrameParams:
    """Parameters for rendering a single frame."""

    ctx: moderngl.Context
    program: moderngl.Program
    vao: moderngl.VertexArray
    target_fbo: moderngl.Framebuffer | moderngl.Context | None
    size: tuple[int, int]
    time: float
    target: Target
    uniforms: dict[str, float | tuple[float, ...]] | None = None
    mouse_pos: list[float] | None = None
    mouse_uv: list[float] | None = None
    resolution: tuple[int, int] | None = None
    frame_num: int = 0

    @classmethod
    def from_render_context(
        cls,
        render_ctx: "RenderContext",
        size: tuple[int, int],
        time: float,
        uniforms: dict[str, float | tuple[float, ...]] | None = None,
        mouse_pos: list[float] | None = None,
        mouse_uv: list[float] | None = None,
        use_screen: bool = False,
    ) -> "FrameParams":
        """Create FrameParams from a RenderContext.

        Args:
            render_ctx: The rendering context
            size: Frame size as (width, height)
            time: Shader time value
            uniforms: Additional uniform values
            mouse_pos: Mouse position in pixels
            mouse_uv: Mouse position in UV coordinates
            use_screen: Use screen instead of FBO (for windowed mode)

        Returns:
            FrameParams configured for rendering
        """
        return cls(
            ctx=render_ctx.ctx,
            program=render_ctx.program,
            vao=render_ctx.vao,
            target_fbo=render_ctx.ctx.screen if use_screen else render_ctx.fbo,
            size=size,
            time=time,
            target=render_ctx.target,
            uniforms=uniforms,
            mouse_pos=mouse_pos,
            mouse_uv=mouse_uv,
        )


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
        gl_profile: OpenGL profile ("core" or "es")

    Returns:
        Tuple of (ModernGL context, GLFW window or None)
    """
    major_version, minor_version = gl_version
    require_version = major_version * 100 + minor_version * 10  # e.g. 4.6 -> 460

    if windowed:
        if not glfw.init():
            logger.error("Failed to initialize GLFW")
            raise RuntimeError("Failed to initialize GLFW")

        # Set OpenGL version
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, major_version)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, minor_version)

        # Set profile
        if gl_profile == "core":
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        elif gl_profile == "compatibility":
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)

        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)

        window = glfw.create_window(size[0], size[1], window_title, None, None)
        if not window:
            logger.error("Failed to create GLFW window")
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        glfw.make_context_current(window)
        ctx = moderngl.create_context()
    else:
        window = None
        ctx = moderngl.create_context(standalone=True, require=require_version)

    return ctx, window


def _compile_program(
    ctx: moderngl.Context,
    glsl_code: str,
    target: Target,
) -> moderngl.Program:
    """Compile shader program.

    Args:
        ctx: ModernGL context
        glsl_code: Fragment shader code
        target: The compilation target

    Returns:
        Compiled shader program
    """
    vertex_shader = target.get_vertex_shader()

    try:
        program = ctx.program(vertex_shader=vertex_shader, fragment_shader=glsl_code)
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


@contextmanager
def _setup_rendering_context(
    shader_input: Callable[..., Any] | str,
    size: tuple[int, int],
    windowed: bool = False,
    window_title: str = "GLSL Shader",
    target: TargetType = TargetType.OPENGL46,
) -> Generator[RenderContext, None, None]:
    """Sets up all rendering resources and cleans them up when done.

    Args:
        shader_input: Shader function or GLSL string
        size: Window/framebuffer size as (width, height)
        windowed: Whether to create a window or use offscreen rendering
        window_title: Title of the window (if windowed is True)
        target: Target platform

    Yields:
        RenderContext object containing all rendering resources
    """
    target_instance = target.create()

    # Check if target supports rendering
    if target_instance.is_export_only():
        raise ValueError(
            f"{target.name} target is for code export only and cannot be rendered. "
            f"Use 'py2glsl export --target {target.name.lower()}' to export code. "
            f"For development and preview, use OpenGL46 or OpenGL33 target."
        )

    # Prepare shader code
    if callable(shader_input):
        result = transpile_to_result(shader_input, target=target)
        glsl_code = result.code
    else:
        glsl_code = shader_input

    # Get OpenGL requirements from target
    gl_version = target_instance.get_gl_version()
    gl_profile = target_instance.get_gl_profile()

    # Initialize context and window
    ctx, window = _init_glfw(size, windowed, window_title, gl_version, gl_profile)

    # Compile shader program
    program = _compile_program(ctx, glsl_code, target_instance)

    # Setup rendering primitives
    vbo, vao = _setup_primitives(ctx, program)

    # Create framebuffer if needed (for offscreen rendering)
    fbo = None
    if not windowed:
        fbo = ctx.simple_framebuffer(size)

    render_ctx = RenderContext(
        ctx=ctx,
        program=program,
        vbo=vbo,
        vao=vao,
        target=target_instance,
        fbo=fbo,
        window=window,
    )

    try:
        yield render_ctx
    finally:
        _cleanup(
            render_ctx.ctx,
            render_ctx.program,
            render_ctx.vbo,
            render_ctx.vao,
            render_ctx.fbo,
            render_ctx.window,
        )


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


def _render_frame(params: FrameParams) -> Any | None:
    """Render a single frame.

    Args:
        params: Frame rendering parameters

    Returns:
        Numpy array with rendered image if rendering to a framebuffer, None otherwise
    """
    if params.target_fbo is None:
        logger.error("No target specified for rendering")
        return None

    if isinstance(params.target_fbo, moderngl.Framebuffer):
        params.target_fbo.use()
    params.ctx.clear(0.0, 0.0, 0.0, 1.0)

    # Build canonical uniform values
    canonical_uniforms: dict[str, Any] = {
        "u_resolution": params.size,
        "u_time": params.time,
        "u_aspect": params.size[0] / params.size[1],
    }

    # Add mouse uniforms if available
    if params.mouse_pos and params.mouse_uv:
        canonical_uniforms["u_mouse_pos"] = tuple(params.mouse_pos)
        canonical_uniforms["u_mouse_uv"] = tuple(params.mouse_uv)

    # Add user-provided uniforms
    if params.uniforms:
        for key, value in params.uniforms.items():
            canonical_uniforms[key] = value

    # Transform uniforms using target's mapping
    uniforms_to_set = params.target.transform_uniform_values(canonical_uniforms)

    # Set all uniforms on the program
    for name, value in uniforms_to_set.items():
        if name in params.program:
            uniform: Any = params.program[name]
            try:
                if hasattr(uniform, "value"):
                    uniform.value = value
                elif hasattr(uniform, "write"):
                    uniform.write(value)
                else:
                    logger.warning(f"Unknown uniform type for {name}, can't set value")
            except (AttributeError, TypeError) as e:
                logger.warning(f"Could not set uniform {name} to {value}: {e}")

    # Render the quad
    params.vao.render(moderngl.TRIANGLE_STRIP)

    # Return pixel data if rendering to framebuffer
    if params.target_fbo is not None and isinstance(
        params.target_fbo, moderngl.Framebuffer
    ):
        data = params.target_fbo.read(components=4, dtype="f1")
        img = np.frombuffer(data, dtype=np.uint8).reshape(
            params.size[1], params.size[0], 4
        )
        # Flip vertically (OpenGL has Y=0 at bottom)
        return np.flipud(img)

    return None


def _render_frames(
    render_ctx: RenderContext,
    size: tuple[int, int],
    duration: float,
    fps: int,
    time_offset: float,
    uniforms: dict[str, float | tuple[float, ...]] | None,
) -> Generator[Any, None, None]:
    """Generate rendered frames for an animation.

    Args:
        render_ctx: The rendering context
        size: Frame size as (width, height)
        duration: Animation duration in seconds
        fps: Frames per second
        time_offset: Starting time for the animation
        uniforms: Additional uniform values

    Yields:
        Numpy arrays containing rendered frame data
    """
    num_frames = int(duration * fps)
    for i in range(num_frames):
        frame_time = time_offset + (i / fps)
        frame_params = FrameParams.from_render_context(
            render_ctx, size=size, time=frame_time, uniforms=uniforms
        )
        array = _render_frame(frame_params)
        if array is None:
            raise RuntimeError("Failed to render frame")
        yield array


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
    target: TargetType = TargetType.OPENGL46,
    fps: int = 0,
) -> None:
    """Run a real-time shader animation in a window.

    Args:
        shader_input: Shader function or GLSL string
        size: Window size as (width, height)
        window_title: Title of the window
        uniforms: Additional uniform values to pass to the shader
        target: Target platform
        fps: Target frame rate (0 = unlimited)
    """
    with _setup_rendering_context(
        shader_input,
        size,
        windowed=True,
        window_title=window_title,
        target=target,
    ) as render_ctx:
        if render_ctx.window is None:
            raise RuntimeError("Window is required for animate mode")
        mouse_pos, mouse_uv = _setup_mouse_tracking(render_ctx.window, size)

        frame_count = 0
        fps_timer = time_module.time()

        if fps > 0:
            try:
                glfw.swap_interval(1)
                logger.info("Using vsync for frame timing")
            except Exception:
                logger.info("Could not set vsync, using manual timing")
            frame_interval = 1.0 / fps
            logger.info(f"Target FPS: {fps}")
        else:
            glfw.swap_interval(0)
            frame_interval = 0
            logger.info("Target FPS: unlimited")

        previous_time = time_module.time()
        lag = 0.0

        while not glfw.window_should_close(render_ctx.window):
            current_time = time_module.time()
            elapsed = current_time - previous_time
            previous_time = current_time
            lag += elapsed

            glfw.poll_events()

            should_render = True
            if fps > 0:
                if lag < frame_interval:
                    should_render = False
                else:
                    lag -= frame_interval
                    if lag > frame_interval * 5:
                        lag = 0.0

            if not should_render:
                continue

            frame_count += 1
            if current_time - fps_timer >= 1.0:
                measured_fps = frame_count / (current_time - fps_timer)
                logger.info(f"FPS: {measured_fps:.2f}")
                frame_count = 0
                fps_timer = current_time

            frame_params = FrameParams.from_render_context(
                render_ctx,
                size=size,
                time=glfw.get_time(),
                uniforms=uniforms,
                mouse_pos=mouse_pos,
                mouse_uv=mouse_uv,
                use_screen=True,
            )

            _render_frame(frame_params)
            glfw.swap_buffers(render_ctx.window)

            if fps <= 0:
                time_module.sleep(0.001)


def render_array(
    shader_input: Callable[..., Any] | str,
    size: tuple[int, int] = (1200, 800),
    time: float = 0.0,
    uniforms: dict[str, float | tuple[float, ...]] | None = None,
    target: TargetType = TargetType.OPENGL46,
) -> Any:
    """Render shader to a numpy array.

    Args:
        shader_input: Shader function or GLSL string
        size: Image size as (width, height)
        time: Shader time value
        uniforms: Additional uniform values to pass to the shader
        target: Target platform

    Returns:
        Numpy array containing the rendered image
    """
    logger.info("Rendering to array")

    with _setup_rendering_context(
        shader_input,
        size,
        windowed=False,
        target=target,
    ) as render_ctx:
        frame_params = FrameParams.from_render_context(
            render_ctx, size=size, time=time, uniforms=uniforms
        )
        result = _render_frame(frame_params)

        if result is None:
            raise RuntimeError("Rendering failed to produce an image")
        return result


def render_image(
    shader_input: Callable[..., Any] | str,
    size: tuple[int, int] = (1200, 800),
    time: float = 0.0,
    uniforms: dict[str, float | tuple[float, ...]] | None = None,
    output_path: str | None = None,
    image_format: str = "PNG",
    target: TargetType = TargetType.OPENGL46,
) -> Image.Image:
    """Render shader to a PIL Image.

    Args:
        shader_input: Shader function or GLSL string
        size: Image size as (width, height)
        time: Shader time value
        uniforms: Additional uniform values to pass to the shader
        output_path: Path to save the image, if desired
        image_format: Format to save the image in (e.g., "PNG", "JPEG")
        target: Target platform

    Returns:
        PIL Image containing the rendered image
    """
    logger.info("Rendering to image")

    with _setup_rendering_context(
        shader_input, size, windowed=False, target=target
    ) as render_ctx:
        frame_params = FrameParams.from_render_context(
            render_ctx, size=size, time=time, uniforms=uniforms
        )
        array = _render_frame(frame_params)
        if array is None:
            raise RuntimeError("Failed to render frame")
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
    uniforms: dict[str, float | tuple[float, ...]] | None = None,
    output_path: str | None = None,
    time_offset: float = 0.0,
    target: TargetType = TargetType.OPENGL46,
) -> tuple[Image.Image, list[Any]]:
    """Render shader to an animated GIF, returning first frame and raw frames.

    Args:
        shader_input: Shader function or GLSL string
        size: Image size as (width, height)
        duration: Animation duration in seconds
        fps: Frames per second
        uniforms: Additional uniform values to pass to the shader
        output_path: Path to save the GIF, if desired
        time_offset: Starting time for the animation (seconds)
        target: Target platform

    Returns:
        Tuple of (first frame as PIL Image, list of raw frames as numpy arrays)
    """
    logger.info("Rendering to GIF")

    with _setup_rendering_context(
        shader_input, size, windowed=False, target=target
    ) as render_ctx:
        raw_frames: list[Any] = []
        pil_frames: list[Image.Image] = []

        for array in _render_frames(
            render_ctx, size, duration, fps, time_offset, uniforms
        ):
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
    time_offset: float = 0.0,
    target: TargetType = TargetType.OPENGL46,
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
        time_offset: Starting time for the animation (seconds)
        target: Target platform

    Returns:
        Tuple of (output path, list of raw frames as numpy arrays)
    """
    logger.info(f"Rendering to video file {output_path} with {codec} codec")

    with _setup_rendering_context(
        shader_input, size, windowed=False, target=target
    ) as render_ctx:
        writer = imageio.get_writer(
            output_path,
            fps=fps,
            codec=codec,
            quality=quality,
            pixelformat=pixel_format,
        )

        raw_frames: list[Any] = []

        for array in _render_frames(
            render_ctx, size, duration, fps, time_offset, uniforms
        ):
            raw_frames.append(array)
            writer.append_data(array)

        writer.close()
        logger.info(f"Video saved to {output_path}")
        return output_path, raw_frames
