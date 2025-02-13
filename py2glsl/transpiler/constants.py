"""GLSL shader constants."""

from ..types import FLOAT, INT, VEC2

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

# Common uniform names with their GLSL types
BUILTIN_UNIFORMS = {
    "u_time": FLOAT,  # Time in seconds
    "u_frame": INT,  # Frame number
    "u_aspect": FLOAT,  # Width/height ratio
    "u_mouse": VEC2,  # Mouse position in normalized coordinates
}

__all__ = [
    "VERTEX_SHADER",
    "GLSL_VERSION",
    "BUILTIN_UNIFORMS",
]
