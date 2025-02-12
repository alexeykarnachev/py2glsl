"""Real-time shader animation window."""

import time

import glfw
import moderngl
import numpy as np

from ..transpiler import py2glsl
from ..transpiler.constants import VERTEX_SHADER
from ..types import vec4
from .buffers import create_quad_buffer, create_vertex_array
from .context import setup_context


def animate(
    shader_func,
    size=(512, 512),
    fps=60.0,
    title="Shader Preview",
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
        setup_context(ctx)

        # Create quad and program
        quad = create_quad_buffer(ctx, size)
        shader_result = py2glsl(shader_func)
        program = ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=shader_result.fragment_source,
        )
        vao = create_vertex_array(ctx, program, quad)

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
                        uniform_type = shader_result.uniforms[name]
                        if uniform_type == "int":
                            program[name].value = int(value)
                        elif isinstance(value, (tuple, list, np.ndarray)):
                            program[name].value = tuple(map(float, value))
                        else:
                            program[name].value = float(value)

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
