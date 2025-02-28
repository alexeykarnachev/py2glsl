import time
from typing import Any, Callable, Dict, Protocol, Union, cast

import glfw
import moderngl
import numpy as np
from loguru import logger

# Define protocol for ModernGL uniform handling
class UniformProtocol(Protocol):
    value: Any
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


def animate(shader_input: Callable[..., Any] | str, used_uniforms: set[str] | None = None, size: tuple[int, int] = (1200, 800)) -> None:
    """Run the shader with an animation loop and FPS calculation.

    Args:
        shader_input: Either a shader function or GLSL code string
        used_uniforms: Set of uniform names used by the shader (only needed if shader_input is GLSL code)
        size: Window size as (width, height) tuple
    """
    if callable(shader_input):
        # If a function is provided, transpile it to GLSL
        from py2glsl.transpiler import transpile

        glsl_code, used_uniforms = transpile(shader_input)
    else:
        # If GLSL code is provided directly, use it as-is
        glsl_code = shader_input
        if used_uniforms is None:
            logger.warning("No uniform information provided with GLSL code")
            used_uniforms = set()

    if not glfw.init():
        logger.error("Failed to initialize GLFW")
        raise RuntimeError("Failed to initialize GLFW")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(size[0], size[1], "Shader Animation", None, None)
    if not window:
        logger.error("Failed to create GLFW window")
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    glfw.make_context_current(window)
    ctx = moderngl.create_context()
    logger.info(f"OpenGL version: {ctx.version_code}")

    try:
        program = ctx.program(
            vertex_shader=vertex_shader_source, fragment_shader=glsl_code
        )
        logger.info("Shader program compiled successfully")
        # Log available uniforms for debugging
        available_uniforms = [name for name in program]
        logger.info(f"Available uniforms: {available_uniforms}")
    except Exception as e:
        logger.error(f"Shader compilation error: {e}")
        glfw.terminate()
        raise

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

    # Mouse position tracking
    mouse_pos: list[float] = [0.0, 0.0]
    mouse_uv: list[float] = [0.5, 0.5]  # Default centered

    def mouse_callback(window: glfw._GLFWwindow, xpos: float, ypos: float) -> None:
        mouse_pos[0] = xpos
        mouse_pos[1] = ypos
        mouse_uv[0] = xpos / size[0]
        mouse_uv[1] = 1.0 - (ypos / size[1])  # Flip Y coordinate

    glfw.set_cursor_pos_callback(window, mouse_callback)

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

        # Always set common uniforms if they exist in the program
        if "u_resolution" in program:
            uniform = cast(UniformProtocol, program["u_resolution"])
            uniform.value = (size[0], size[1])

        if "u_time" in program:
            uniform = cast(UniformProtocol, program["u_time"])
            uniform.value = glfw.get_time()

        if "u_aspect" in program:
            uniform = cast(UniformProtocol, program["u_aspect"])
            uniform.value = size[0] / size[1]

        if "u_mouse_pos" in program:
            uniform = cast(UniformProtocol, program["u_mouse_pos"])
            uniform.value = mouse_pos

        if "u_mouse_uv" in program:
            uniform = cast(UniformProtocol, program["u_mouse_uv"])
            uniform.value = mouse_uv

        # Set any other custom uniforms that might be in the shader
        # This loop is a fallback for uniforms not covered above
        for uniform_name in program:
            if uniform_name not in [
                "u_resolution",
                "u_time",
                "u_aspect",
                "u_mouse_pos",
                "u_mouse_uv",
            ]:
                try:
                    uniform = cast(UniformProtocol, program[uniform_name])
                    if uniform.array_length == 1:  # Not an array uniform
                        if uniform.dimension == 1:
                            if uniform.dtype.startswith("f"):
                                uniform.value = 1.0  # Default float value
                            elif uniform.dtype.startswith("i"):
                                uniform.value = 1  # Default int value
                            elif uniform.dtype.startswith("b"):
                                uniform.value = True  # Default bool value
                except Exception as e:
                    logger.warning(f"Could not set uniform {uniform_name}: {e}")

        ctx.clear(0.0, 0.0, 0.0, 1.0)
        vao.render(moderngl.TRIANGLE_STRIP)
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


def render_array(glsl_code: str, size: tuple[int, int] = (1200, 800)) -> None:
    """Render the shader to an array or buffer."""
    logger.info("Rendering to array")
    # Placeholder


def render_gif(glsl_code: str, size: tuple[int, int] = (1200, 800), duration: float = 5.0, fps: int = 30) -> None:
    """Render the shader to a GIF file."""
    logger.info("Rendering to GIF")
    # Placeholder


def render_video(glsl_code: str, size: tuple[int, int] = (1200, 800), duration: float = 5.0, fps: int = 30) -> None:
    """Render the shader to a video file."""
    logger.info("Rendering to video")
    # Placeholder


def render_image(glsl_code: str, size: tuple[int, int] = (1200, 800)) -> None:
    """Render the shader to a single image."""
    logger.info("Rendering to image")
    # Placeholder
