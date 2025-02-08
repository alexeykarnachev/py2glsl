import time
from contextlib import contextmanager
from typing import Callable, Iterator, Optional, Tuple, Union

import glfw
import imageio
import moderngl
import numpy as np
from PIL import Image

from py2glsl.types import vec4

Size = Tuple[int, int]


@contextmanager
def gl_context(size: Size) -> Iterator[moderngl.Context]:
    """Create standalone OpenGL context"""
    ctx = moderngl.create_standalone_context()

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

    # Basic vertex shader that just passes UV coords
    vertex_shader = """
        #version 460
        in vec2 in_position;
        in vec2 in_uv;
        out vec2 vs_uv;
        void main() {
            gl_Position = vec4(in_position, 0.0, 1.0);
            vs_uv = in_uv;
        }
    """

    try:
        yield ctx
    finally:
        ctx.release()


def render_array(shader_func, size: Size = (512, 512), **uniforms) -> np.ndarray:
    """Render shader to numpy array"""
    with gl_context(size) as ctx:
        # Convert shader to GLSL
        shader_result = py2glsl(shader_func)

        # Create shader program
        program = ctx.program(
            vertex_shader=vertex_shader, fragment_shader=shader_result.fragment_source
        )

        # Set uniforms
        for name, value in uniforms.items():
            if isinstance(value, (float, int)):
                program[name].value = float(value)
            elif isinstance(value, (tuple, list, np.ndarray)):
                program[name].value = tuple(map(float, value))

        # Create framebuffer
        fbo = ctx.framebuffer(color_attachments=[ctx.texture(size, 4)])
        fbo.use()

        # Render
        vao = ctx.vertex_array(program, [(quad, "2f 2f", "in_position", "in_uv")])
        vao.render(mode=moderngl.TRIANGLE_STRIP)

        # Read pixels
        data = np.frombuffer(
            fbo.read(components=4, alignment=1), dtype=np.uint8
        ).reshape(*size, 4)

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
        frame_data = render_array(
            shader_func, size=size, u_time=time, u_frame=frame, **uniforms
        )
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
            frame_data = render_array(
                shader_func, size=size, u_time=time, u_frame=frame, **uniforms
            )
            writer.append_data((frame_data * 255).astype(np.uint8))
    finally:
        writer.close()


def animate(
    shader_func: Callable[..., vec4],
    size: Tuple[int, int] = (512, 512),
    fps: float = 60.0,
    title: str = "Shader Preview",
    **uniforms,
) -> None:
    """Animate shader in real-time window.

    Args:
        shader_func: Shader function that takes vs_uv and returns vec4 color
        size: Window size (width, height)
        fps: Target frames per second
        title: Window title
        **uniforms: Additional uniform values to pass to shader

    The following uniforms are automatically provided:
        u_time (float): Time since start in seconds
        u_frame (int): Frame number
        u_resolution (vec2): Window size in pixels
        u_mouse (vec2): Mouse position in UV coordinates (0-1)
    """
    if not glfw.init():
        raise RuntimeError("Could not initialize GLFW")

    try:
        # Configure OpenGL context
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.SAMPLES, 4)  # Enable MSAA

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

        # Convert shader to GLSL
        shader_result = py2glsl(shader_func)

        # Create shader program
        program = ctx.program(
            vertex_shader="""
                #version 460
                in vec2 in_position;
                in vec2 in_uv;
                out vec2 vs_uv;
                void main() {
                    gl_Position = vec4(in_position, 0.0, 1.0);
                    vs_uv = in_uv;
                }
            """,
            fragment_shader=shader_result.fragment_source,
        )

        # Create fullscreen quad
        vertices = np.array(
            [
                # x,    y,     u,    v
                -1.0,
                -1.0,
                0.0,
                0.0,  # bottom left
                1.0,
                -1.0,
                1.0,
                0.0,  # bottom right
                -1.0,
                1.0,
                0.0,
                1.0,  # top left
                1.0,
                1.0,
                1.0,
                1.0,  # top right
            ],
            dtype="f4",
        )

        quad = ctx.buffer(vertices.tobytes())
        vao = ctx.vertex_array(program, [(quad, "2f 2f", "in_position", "in_uv")])

        # Track mouse position
        mouse_pos = [0.0, 0.0]

        def mouse_callback(window, xpos: float, ypos: float) -> None:
            """Update mouse position in UV coordinates"""
            width, height = glfw.get_window_size(window)
            mouse_pos[0] = xpos / width
            mouse_pos[1] = 1.0 - (ypos / height)  # Flip Y

        glfw.set_cursor_pos_callback(window, mouse_callback)

        # Setup timing
        start_time = time.perf_counter()
        last_frame = start_time
        frame = 0
        frame_time = 1.0 / fps

        # Main loop
        while not glfw.window_should_close(window):
            current_time = time.perf_counter()

            # Control frame rate
            if current_time - last_frame >= frame_time:
                # Update timing
                last_frame = current_time
                elapsed = current_time - start_time

                # Get current window size
                width, height = glfw.get_framebuffer_size(window)
                ctx.viewport = (0, 0, width, height)

                # Set built-in uniforms
                program["u_time"].value = elapsed
                program["u_frame"].value = frame
                program["u_resolution"].value = (width, height)
                program["u_mouse"].value = tuple(mouse_pos)

                # Set custom uniforms
                for name, value in uniforms.items():
                    if isinstance(value, (float, int)):
                        program[name].value = float(value)
                    elif isinstance(value, (tuple, list, np.ndarray)):
                        program[name].value = tuple(map(float, value))
                    # Could add support for other types here

                # Render
                ctx.clear(0.0, 0.0, 0.0, 0.0)
                vao.render(mode=moderngl.TRIANGLE_STRIP)

                # Swap buffers and poll events
                glfw.swap_buffers(window)
                frame += 1

            # Poll events even when not rendering
            glfw.poll_events()

            # Handle keyboard input
            if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
                glfw.set_window_should_close(window, True)

            # Optional sleep to prevent busy-waiting
            time.sleep(max(0.0, frame_time - (time.perf_counter() - current_time)))

    finally:
        glfw.terminate()
