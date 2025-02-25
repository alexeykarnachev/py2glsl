import time

import glfw
import moderngl
import numpy as np
from loguru import logger

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


def animate(shader_input, used_uniforms=None, size=(1200, 800)):
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

        # Set uniforms only if they exist in the program
        if "u_time" in used_uniforms and "u_time" in program:
            program["u_time"].value = glfw.get_time()
        if "u_aspect" in used_uniforms and "u_aspect" in program:
            program["u_aspect"].value = size[0] / size[1]
        if "u_resolution" in used_uniforms and "u_resolution" in program:
            program["u_resolution"].value = (size[0], size[1])
        # Add other uniforms as needed

        ctx.clear(0.0, 0.0, 0.0, 1.0)
        vao.render(moderngl.TRIANGLE_STRIP)
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


def render_array(glsl_code, size=(1200, 800)):
    """Render the shader to an array or buffer."""
    logger.info("Rendering to array")
    # Placeholder


def render_gif(glsl_code, size=(1200, 800), duration=5.0, fps=30):
    """Render the shader to a GIF file."""
    logger.info("Rendering to GIF")
    # Placeholder


def render_video(glsl_code, size=(1200, 800), duration=5.0, fps=30):
    """Render the shader to a video file."""
    logger.info("Rendering to video")
    # Placeholder


def render_image(glsl_code, size=(1200, 800)):
    """Render the shader to a single image."""
    logger.info("Rendering to image")
    # Placeholder
