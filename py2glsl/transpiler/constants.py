"""GLSL shader constants."""

# Default vertex shader for fragment shader rendering
VERTEX_SHADER = """#version 460
layout(location = 0) in vec2 in_pos;
layout(location = 1) in vec2 in_uv;
out vec2 vs_uv;

void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    vs_uv = in_uv;
}
"""

# Common GLSL version
GLSL_VERSION = 460

# Common uniform names
BUILTIN_UNIFORMS = {
    "u_time": "float",  # Time in seconds
    "u_frame": "int",  # Frame number
    "u_resolution": "vec2",  # Window/framebuffer resolution
    "u_mouse": "vec2",  # Mouse position in normalized coordinates
}

__all__ = [
    "VERTEX_SHADER",
    "GLSL_VERSION",
    "BUILTIN_UNIFORMS",
]
