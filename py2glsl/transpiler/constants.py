"""Constants used in GLSL code generation."""

# Basic vertex shader that just passes UV coords
VERTEX_SHADER = """#version 460
layout(location = 0) in vec2 in_pos;
layout(location = 1) in vec2 in_uv;
out vec2 vs_uv;

void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    vs_uv = in_uv;
}
"""

# GLSL version header
GLSL_VERSION = "#version 460"

# Common GLSL type names
FLOAT_TYPE = "float"
INT_TYPE = "int"
BOOL_TYPE = "bool"
VEC2_TYPE = "vec2"
VEC3_TYPE = "vec3"
VEC4_TYPE = "vec4"

# Special shader function names
MAIN_FUNC = "main"
SHADER_FUNC = "shader"

# Default indentation
INDENT = "    "
