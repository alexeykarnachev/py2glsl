import time
from collections.abc import Callable

import glfw
import imageio
import moderngl
import numpy as np
from loguru import logger
from numpy.typing import NDArray
from PIL import Image

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


def _prepare_shader_code(shader_input: Callable[..., None] | str) -> str:
    """Prepare shader code from input."""
    if callable(shader_input):
        from py2glsl.transpiler import transpile

        glsl_code, _ = transpile(shader_input)
    else:
        glsl_code = shader_input
    return glsl_code


def _init_context(
    size: tuple[int, int], windowed: bool, window_title: str = "GLSL Shader"
) -> tuple[moderngl.Context, glfw._GLFWwindow | None]:
    """Initialize ModernGL context and optional GLFW window."""
    if windowed:
        if not glfw.init():
            logger.error("Failed to initialize GLFW")
            raise RuntimeError("Failed to initialize GLFW")
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
        ctx = moderngl.create_context(standalone=True, require=460)
    return ctx, window


def _compile_program(ctx: moderngl.Context, glsl_code: str) -> moderngl.Program:
    """Compile shader program."""
    try:
        program = ctx.program(
            vertex_shader=vertex_shader_source, fragment_shader=glsl_code
        )
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
) -> NDArray[np.uint8] | None:
    """Render a single frame."""
    target.use()
    ctx.clear(0.0, 0.0, 0.0, 1.0)

    uniforms = uniforms or {}
    default_uniforms = {
        "u_resolution": size,
        "u_time": time,
        "u_aspect": size[0] / size[1],
    }
    if mouse_pos and mouse_uv:
        default_uniforms["u_mouse_pos"] = tuple(mouse_pos)
        default_uniforms["u_mouse_uv"] = tuple(mouse_uv)
    default_uniforms.update(uniforms)

    for name, value in default_uniforms.items():
        if name in program:
            program[name].value = value

    vao.render(moderngl.TRIANGLE_STRIP)
    if isinstance(target, moderngl.Framebuffer):
        data = target.read(components=4, dtype="f1")
        return np.frombuffer(data, dtype=np.uint8).reshape(size[1], size[0], 4)
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
    shader_input: Callable[..., None] | str,
    size: tuple[int, int] = (1200, 800),
    window_title: str = "GLSL Shader",
    uniforms: dict[str, float | tuple[float, ...]] | None = None,
) -> None:
    """Run a real-time shader animation in a window."""
    glsl_code = _prepare_shader_code(shader_input)
    ctx, window = _init_context(size, windowed=True, window_title=window_title)
    program = _compile_program(ctx, glsl_code)
    vbo, vao = _setup_primitives(ctx, program)
    mouse_pos, mouse_uv = _setup_mouse_tracking(window, size)

    try:
        frame_count = 0
        last_time = time.time()
        while not glfw.window_should_close(window):
            current_time = time.time()
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
    shader_input: Callable[..., None] | str,
    size: tuple[int, int] = (1200, 800),
    time: float = 0.0,
    uniforms: dict[str, float | tuple[float, ...]] | None = None,
) -> NDArray[np.uint8]:
    """Render shader to a numpy array."""
    logger.info("Rendering to array")
    glsl_code = _prepare_shader_code(shader_input)
    ctx, _ = _init_context(size, windowed=False)
    program = _compile_program(ctx, glsl_code)
    vbo, vao = _setup_primitives(ctx, program)
    fbo = ctx.simple_framebuffer(size)

    try:
        return _render_frame(ctx, program, vao, fbo, size, time, uniforms)
    finally:
        _cleanup(ctx, program, vbo, vao, fbo)


def render_image(
    shader_input: Callable[..., None] | str,
    size: tuple[int, int] = (1200, 800),
    time: float = 0.0,
    uniforms: dict[str, float | tuple[float, ...]] | None = None,
    output_path: str | None = None,
    image_format: str = "PNG",
) -> Image.Image:
    """Render shader to a PIL Image."""
    logger.info("Rendering to image")
    glsl_code = _prepare_shader_code(shader_input)
    ctx, _ = _init_context(size, windowed=False)
    program = _compile_program(ctx, glsl_code)
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
    shader_input: Callable[..., None] | str,
    size: tuple[int, int] = (1200, 800),
    duration: float = 5.0,
    fps: int = 30,
    uniforms: dict[str, float | tuple[float, ...]] | None = None,
    output_path: str | None = None,
) -> tuple[Image.Image, list[NDArray[np.uint8]]]:
    """Render shader to an animated GIF, returning first frame and raw frames."""
    logger.info("Rendering to GIF")
    glsl_code = _prepare_shader_code(shader_input)
    ctx, _ = _init_context(size, windowed=False)
    program = _compile_program(ctx, glsl_code)
    vbo, vao = _setup_primitives(ctx, program)
    fbo = ctx.simple_framebuffer(size)

    try:
        num_frames = int(duration * fps)
        raw_frames = []
        pil_frames = []
        for i in range(num_frames):
            time = i / fps
            array = _render_frame(ctx, program, vao, fbo, size, time, uniforms)
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
    shader_input: Callable[..., None] | str,
    size: tuple[int, int] = (1200, 800),
    duration: float = 5.0,
    fps: int = 30,
    output_path: str = "shader_output.mp4",
    codec: str = "h264",
    quality: int = 8,
    pixel_format: str = "yuv420p",
    uniforms: dict[str, float | tuple[float, ...]] | None = None,
) -> tuple[str, list[NDArray[np.uint8]]]:
    """Render shader to a video file, returning path and raw frames."""
    logger.info(f"Rendering to video file {output_path} with {codec} codec")
    glsl_code = _prepare_shader_code(shader_input)
    ctx, _ = _init_context(size, windowed=False)
    program = _compile_program(ctx, glsl_code)
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
        raw_frames = []
        for i in range(num_frames):
            time = i / fps
            array = _render_frame(ctx, program, vao, fbo, size, time, uniforms)
            assert array is not None
            raw_frames.append(array)
            writer.append_data(array)
        writer.close()
        logger.info(f"Video saved to {output_path}")
        return output_path, raw_frames
    finally:
        _cleanup(ctx, program, vbo, vao, fbo)
