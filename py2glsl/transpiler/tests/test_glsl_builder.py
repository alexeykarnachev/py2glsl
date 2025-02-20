from typing import Callable

import pytest

from py2glsl.transpiler.glsl_builder import GLSLBuilder, GLSLCodeError


def test_vertex_shader_generation_basic():
    builder = GLSLBuilder()
    builder.add_vertex_attribute(0, "vec2", "vs_uv")
    # Add required interface blocks
    builder.add_interface_block("VertexData", "out", {"vs_uv": "vec2"})
    builder.add_interface_block("VertexData", "in", {"vs_uv": "vec2"})

    builder.vertex_main_body = [
        "VertexData.vs_uv = vs_uv;",
        "gl_Position = vec4(vs_uv * 2.0 - 1.0, 0.0, 1.0);",
    ]

    result = builder.build_vertex_shader()

    assert "layout(location = 0) in vec2 vs_uv" in result
    assert "out VertexData" in result
    assert "gl_Position = vec4(vs_uv * 2.0 - 1.0, 0.0, 1.0);" in result


def test_fragment_shader_with_uniforms():
    builder = GLSLBuilder()
    builder.add_uniform("u_time", "float")
    builder.add_uniform("u_resolution", "vec2")
    builder.fragment_main_body = [
        "float aspect = u_resolution.x / u_resolution.y;",
        "fs_color = vec4(sin(u_time), aspect, 0.5, 1.0);",
    ]

    result = builder.build_fragment_shader()

    assert "uniform float u_time" in result
    assert "uniform vec2 u_resolution" in result
    assert "aspect = u_resolution.x / u_resolution.y" in result
    assert "fs_color = vec4(sin(u_time), aspect, 0.5, 1.0)" in result


@pytest.mark.parametrize(
    "type_name", ["vec2", "vec3", "vec4", "mat3", "mat4", "float", "int", "bool"]
)
def test_uniform_type_declarations(type_name):
    builder = GLSLBuilder()
    builder.add_uniform(f"u_{type_name}", type_name)
    shader_src = builder.build_fragment_shader()
    assert f"uniform {type_name} u_{type_name}" in shader_src


def test_interface_block_matching():
    builder = GLSLBuilder()
    builder.add_interface_block("VertexData", "out", {"vs_uv": "vec2", "color": "vec3"})
    builder.add_interface_block("VertexData", "in", {"vs_uv": "vec2", "color": "vec3"})

    vertex_src = builder.build_vertex_shader()
    fragment_src = builder.build_fragment_shader()

    # Verify matching interface blocks
    v_out = next(line for line in vertex_src.splitlines() if "out VertexData" in line)
    f_in = next(line for line in fragment_src.splitlines() if "in VertexData" in line)
    assert v_out == f_in.replace("in", "out")


def test_struct_declaration():
    builder = GLSLBuilder()
    builder.add_struct(
        "Material", {"albedo": "vec3", "roughness": "float", "uv_scale": "vec2"}
    )

    shader_src = builder.build_fragment_shader()
    assert "struct Material {" in shader_src
    assert "vec3 albedo;" in shader_src
    assert "float roughness;" in shader_src
    assert "vec2 uv_scale;" in shader_src


def test_function_declaration_with_parameters():
    builder = GLSLBuilder()
    builder.add_function(
        return_type="vec3",
        name="calculate_normal",
        parameters=[("vec3", "pos"), ("vec2", "uv"), ("float", "intensity")],
        body=[
            "vec3 dx = dFdx(pos);",
            "vec3 dy = dFdy(pos);",
            "return normalize(cross(dx, dy)) * intensity;",
        ],
    )

    shader_src = builder.build_fragment_shader()
    assert "vec3 calculate_normal(vec3 pos, vec2 uv, float intensity)" in shader_src
    assert "vec3 dx = dFdx(pos)" in shader_src
    assert "return normalize(cross(dx, dy)) * intensity" in shader_src


def test_swizzle_validation():
    builder = GLSLBuilder()
    with pytest.raises(GLSLCodeError) as exc:
        builder.add_function(
            return_type="vec4",
            name="invalid_swizzle",
            parameters=[("vec3", "v")],
            body=["return v.xyzw;"],
        )
    assert "Invalid swizzle 'xyzw' for vec3" in str(exc.value)


def test_duplicate_uniform_declaration():
    builder = GLSLBuilder()
    builder.add_uniform("u_time", "float")
    with pytest.raises(GLSLCodeError) as exc:
        builder.add_uniform("u_time", "float")
    assert "Duplicate declaration: u_time" in str(exc.value)


def test_reserved_keyword_validation():
    builder = GLSLBuilder()
    with pytest.raises(GLSLCodeError) as exc:
        builder.add_uniform("float", "vec2")
    assert "Reserved GLSL keyword: float" in str(exc.value)


def test_shader_output_order():
    builder = GLSLBuilder()
    builder.add_uniform("u_time", "float")
    builder.add_vertex_attribute(0, "vec2", "vs_uv")
    builder.add_interface_block("VertexData", "out", {"vs_uv": "vec2"})
    builder.add_interface_block("VertexData", "in", {"vs_uv": "vec2"})
    builder.add_struct("Data", {"value": "float"})
    builder.add_function(
        return_type="float",
        name="noise",
        parameters=[("vec2", "uv")],
        body=["return fract(sin(dot(uv, vec2(12.9898,78.233))) * 43758.5453);"],
    )

    vertex_src = builder.build_vertex_shader()
    sections = vertex_src.split("\n")

    def find_line(predicate: Callable) -> int:
        """Returns index of first line matching the predicate"""
        return next(i for i, line in enumerate(sections) if predicate(line))

    # Find key line indexes using the helper
    version_idx = find_line(lambda l: l == "#version 460 core")
    attr_idx = find_line(lambda l: "layout(location = 0)" in l)
    interface_idx = find_line(lambda l: "out VertexData" in l)
    struct_idx = find_line(lambda l: "struct Data" in l)
    func_idx = find_line(lambda l: "float noise(vec2 uv)" in l)
    main_idx = find_line(lambda l: l == "void main() {")

    # Verify ordering
    assert version_idx < attr_idx < interface_idx < struct_idx < func_idx < main_idx

    # Verify interface block structure
    assert sections[interface_idx + 1].strip() == "vec2 vs_uv;"

    # Verify struct contents
    assert sections[struct_idx + 1].strip() == "float value;"
