import asyncio
import time
from contextlib import contextmanager
from typing import Callable, Iterator, Optional, Tuple, Union

import glfw
import imageio
import moderngl
import numpy as np
from PIL import Image

from py2glsl.transpiler import VERTEX_SHADER, py2glsl
from py2glsl.types import vec4

Size = Tuple[int, int]


@contextmanager
def gl_context(size: Size) -> Iterator[moderngl.Context]:
    """Create standalone OpenGL context"""
    ctx = moderngl.create_standalone_context()
    try:
        yield ctx
    finally:
        ctx.release()


def render_array(shader_func, size: Size = (512, 512), **uniforms) -> np.ndarray:
    """Render shader to numpy array"""
    with gl_context(size) as ctx:
        # Create fullscreen quad with proper UV coordinates
        quad_vertices = np.array(
            [
                -1.0,
                -1.0,
                0.0,
                0.0,
                1.0,
                -1.0,
                1.0,
                0.0,
                -1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            dtype="f4",
        )
        quad = ctx.buffer(quad_vertices.tobytes())

        # Convert shader to GLSL
        shader_result = py2glsl(shader_func)

        # Create shader program
        program = ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=shader_result.fragment_source,
        )

        # Set built-in uniforms
        if "u_resolution" in shader_result.uniforms:
            program["u_resolution"].value = tuple(map(int, size))

        # Set custom uniforms
        for name, value in uniforms.items():
            if name in shader_result.uniforms:
                try:
                    uniform_type = shader_result.uniforms[name]
                    if uniform_type == "int":
                        program[name].value = int(value)  # Keep integers as integers
                    elif uniform_type == "float":
                        program[name].value = float(value)
                    elif isinstance(value, (tuple, list, np.ndarray)):
                        # Convert sequence elements maintaining their types
                        program[name].value = tuple(
                            int(x) if isinstance(x, int) else float(x) for x in value
                        )
                except KeyError:
                    continue

        vao = ctx.vertex_array(
            program,
            [(quad, "2f 2f", "in_pos", "in_uv")],
            skip_errors=True,
        )

        # Create framebuffer
        fbo = ctx.framebuffer(color_attachments=[ctx.texture(size, 4)])
        fbo.use()

        # Clear and render
        ctx.clear()
        vao.render(mode=moderngl.TRIANGLE_STRIP)

        # Read pixels without normalization for integer uniforms
        data = np.frombuffer(fbo.read(components=4, alignment=1), dtype=np.uint8)
        data = data.reshape(*size, 4)

        return data.astype(np.float32) / 255.0


def render_image(shader_func, size: Size = (512, 512), **uniforms) -> Image.Image:
    """Render shader to PIL Image"""
    data = render_array(shader_func, size, **uniforms)
    return Image.fromarray((data * 255).astype(np.uint8))


def render_gif(
    shader_func,
    filename: str,
    duration: float,
    fps: float = 30.0,
    size: Size = (512, 512),
    **uniforms,
) -> None:
    """Render shader animation to GIF"""
    frames = []
    frame_count = int(duration * fps)

    for frame in range(frame_count):
        time = frame / fps
        frame_uniforms = {
            **uniforms,
            "u_time": time,
            "u_frame": frame,
        }
        frame_data = render_array(shader_func, size=size, **frame_uniforms)
        frames.append((frame_data * 255).astype(np.uint8))

    imageio.mimsave(filename, frames, fps=fps, format="GIF")


def render_video(
    shader_func,
    filename: str,
    duration: float,
    fps: float = 30.0,
    size: Size = (512, 512),
    codec: str = "h264",
    **uniforms,
) -> None:
    """Render shader animation to video file"""
    writer = imageio.get_writer(
        filename, fps=fps, codec=codec, quality=8, pixelformat="yuv420p"
    )

    frame_count = int(duration * fps)
    try:
        for frame in range(frame_count):
            time = frame / fps
            frame_uniforms = {
                **uniforms,
                "u_time": time,
                "u_frame": frame,
            }
            frame_data = render_array(shader_func, size=size, **frame_uniforms)
            writer.append_data((frame_data * 255).astype(np.uint8))
    finally:
        writer.close()


def animate(
    shader_func: Callable[..., vec4],
    size: Size = (512, 512),
    fps: float = 60.0,
    title: str = "Shader Preview",
    **uniforms,
) -> None:
    """Animate shader in real-time window."""
    if not glfw.init():
        raise RuntimeError("Could not initialize GLFW")

    try:
        # Configure OpenGL context
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.SAMPLES, 4)

        # Create window
        window = glfw.create_window(size[0], size[1], title, None, None)
        if not window:
            raise RuntimeError("Could not create GLFW window")

        glfw.make_context_current(window)
        glfw.swap_interval(1)  # Enable vsync

        # Setup ModernGL
        ctx = moderngl.create_context()
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # Create fullscreen quad
        quad_vertices = np.array(
            [
                # x,    y,     u,    v
                -1.0,
                -1.0,
                0.0,
                0.0,
                1.0,
                -1.0,
                1.0,
                0.0,
                -1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            dtype="f4",
        )
        quad = ctx.buffer(quad_vertices.tobytes())

        # Convert shader to GLSL
        shader_result = py2glsl(shader_func)

        print("Vertex shader:", VERTEX_SHADER)
        print("Fragment shader:", shader_result.fragment_source)

        # Create shader program
        program = ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=shader_result.fragment_source,
        )

        print("Attribute locations:", program._attribute_locations)
        print("Attribute types:", program._attribute_types)

        vao = ctx.vertex_array(
            program,
            [(quad, "2f 2f", "in_pos", "in_uv")],
            skip_errors=True,
        )

        # Track mouse position
        mouse_pos = [0.0, 0.0]

        def mouse_callback(window, xpos: float, ypos: float) -> None:
            width, height = glfw.get_window_size(window)
            mouse_pos[0] = xpos / width
            mouse_pos[1] = 1.0 - (ypos / height)

        glfw.set_cursor_pos_callback(window, mouse_callback)

        # Setup timing
        start_time = time.perf_counter()
        last_frame = start_time
        frame = 0
        frame_time = 1.0 / fps

        # Main loop
        while not glfw.window_should_close(window):
            current_time = time.perf_counter()

            if current_time - last_frame >= frame_time:
                last_frame = current_time
                elapsed = current_time - start_time

                width, height = glfw.get_framebuffer_size(window)
                ctx.viewport = (0, 0, width, height)

                # Update uniforms
                frame_uniforms = {
                    **uniforms,
                    "u_time": elapsed,
                    "u_frame": frame,
                    "u_resolution": (width, height),
                    "u_mouse": tuple(mouse_pos),
                }

                # Set uniforms
                for name, value in frame_uniforms.items():
                    if name in shader_result.uniforms:
                        if isinstance(value, (float, int)):
                            program[name].value = float(value)
                        elif isinstance(value, (tuple, list, np.ndarray)):
                            program[name].value = tuple(map(float, value))

                # Render
                ctx.clear(0.0, 0.0, 0.0, 0.0)
                vao.render(mode=moderngl.TRIANGLE_STRIP)

                glfw.swap_buffers(window)
                frame += 1

            glfw.poll_events()

            if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
                glfw.set_window_should_close(window, True)

            time.sleep(max(0.0, frame_time - (time.perf_counter() - current_time)))

    finally:
        glfw.terminate()


def get_pixel_centered_quad(size: Size) -> np.ndarray:
    """Create quad vertices with UV coordinates centered on pixels."""
    w, h = size
    # Map UV coordinates to pixel centers
    u_min = 0.5 / w
    u_max = (w - 0.5) / w
    v_min = 0.5 / h
    v_max = (h - 0.5) / h

    return np.array(
        [
            -1.0,
            -1.0,
            u_min,
            v_min,  # bottom left
            1.0,
            -1.0,
            u_max,
            v_min,  # bottom right
            -1.0,
            1.0,
            u_min,
            v_max,  # top left
            1.0,
            1.0,
            u_max,
            v_max,  # top right
        ],
        dtype="f4",
    )
