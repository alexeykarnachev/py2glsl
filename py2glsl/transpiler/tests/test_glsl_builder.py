import pytest

from py2glsl.transpiler.glsl_builder import GLSLBuilder, GLSLCodeError


def test_vertex_shader_generation():
    builder = GLSLBuilder()
    builder.add_vertex_attribute(0, "vec2", "a_pos")
    builder.add_interface_block("VertexOutput", "out", {"vs_uv": "vec2"})

    expected = """
#version 460 core
layout(location = 0) in vec2 a_pos;
out VertexOutput {
    vec2 vs_uv;
};
void main() {
    vs_uv = a_pos * 0.5 + 0.5;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
""".strip()
    assert builder.build_vertex_shader().strip() == expected


def test_fragment_shader_generation():
    builder = GLSLBuilder()
    builder.add_uniform("u_time", "float")
    builder.add_interface_block("VertexInput", "in", {"vs_uv": "vec2"})
    builder.add_output("fs_color", "vec4")
    builder.add_function(
        return_type="vec4",
        name="main_shader",
        parameters=["vec2 uv", "float time"],
        body=["return vec4(uv, time, 1.0);"],
    )
    builder.main_body = ["fs_color = main_shader(vs_uv, u_time);"]

    expected = """
#version 460 core
uniform float u_time;
in VertexInput {
    vec2 vs_uv;
};
out vec4 fs_color;
vec4 main_shader(vec2 uv, float time) {
    return vec4(uv, time, 1.0);
}
void main() {
    fs_color = main_shader(vs_uv, u_time);
}
""".strip()
    assert builder.build_fragment_shader(entry_point="main_shader").strip() == expected


def test_naming_conventions():
    builder = GLSLBuilder()

    with pytest.raises(GLSLCodeError):
        builder.add_uniform("time", "float")

    with pytest.raises(GLSLCodeError):
        builder.add_vertex_attribute(0, "vec2", "position")

    with pytest.raises(GLSLCodeError):
        builder.add_output("color", "vec4")


def test_interface_blocks():
    builder = GLSLBuilder()
    builder.add_interface_block(
        "LightData", "in", {"direction": "vec3", "color": "vec4"}
    )

    expected = """
in LightData {
    vec3 direction;
    vec4 color;
};""".strip()
    assert expected in builder.build_fragment_shader(entry_point="main")


def test_struct_generation():
    builder = GLSLBuilder()
    builder.add_struct("Material", {"albedo": "vec3", "roughness": "float"})

    expected = """
struct Material {
    vec3 albedo;
    float roughness;
};""".strip()
    assert expected in builder.build_fragment_shader(entry_point="main")
