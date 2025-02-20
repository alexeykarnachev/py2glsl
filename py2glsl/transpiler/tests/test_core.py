import pytest

from py2glsl.glsl.types import vec2, vec4
from py2glsl.transpiler.core import GLSLTypeError, TranspilerResult, transpile
from py2glsl.transpiler.glsl_builder import GLSLCodeError


def test_basic_transpilation():
    def shader(vs_uv: vec2, /, u_time: float) -> vec4:
        return vec4(vs_uv, 0.0, 1.0)

    result = transpile(shader)

    # Vertex shader checks
    assert "layout(location = 0) in vec2 vs_uv" in result.vertex_src
    assert "gl_Position = vec4(VertexData.vs_uv, 0.0, 1.0)" in result.vertex_src
    assert "out VertexData" in result.vertex_src

    # Fragment shader checks
    assert "in VertexData" in result.fragment_src
    assert "uniform float u_time" in result.fragment_src
    assert "fs_color = shader(VertexData.vs_uv, u_time)" in result.fragment_src


def test_invalid_signatures():
    # Case 1: Bad return type
    with pytest.raises(GLSLTypeError) as exc:

        def shader1(vs_uv: vec2, /) -> int:
            return vec4(0)

        transpile(shader1)
    assert "Shader must return vec4" in str(exc.value)

    # Case 2: Missing positional parameter
    with pytest.raises(GLSLTypeError) as exc:

        def shader2(*, uv: vec2) -> vec4:
            return vec4(uv, 0.0, 1.0)

        transpile(shader2)
    assert "Must have one positional-only vec2 parameter" in str(exc.value)

    # Case 3: Wrong positional parameter type
    with pytest.raises(GLSLTypeError) as exc:

        def shader3(vs_uv: float, /) -> vec4:
            return vec4(0.0)

        transpile(shader3)
    assert "Must have one positional-only vec2 parameter" in str(exc.value)


def test_parameter_separation():
    # Multiple uniforms
    def multi_uniform(vs_uv: vec2, /, time: float, res: vec2) -> vec4:
        return vec4(vs_uv, 0.0, 1.0)

    result = transpile(multi_uniform)
    assert "uniform float time" in result.fragment_src
    assert "uniform vec2 res" in result.fragment_src


def test_type_validation():
    # Valid mixed types
    def valid_types(vs_uv: vec2, /, flag: bool, count: int) -> vec4:
        return vec4(0.0)

    result = transpile(valid_types)
    assert "uniform bool flag" in result.fragment_src
    assert "uniform int count" in result.fragment_src


def test_body_translation():
    def shader(vs_uv: vec2, /) -> vec4:
        color = vs_uv * 2.0
        return vec4(color, 0.0, 1.0)

    result = transpile(shader)
    assert "color = vs_uv * 2.0" in result.fragment_src
    assert "return vec4(color, 0.0, 1.0)" in result.fragment_src


def test_interface_blocks():
    def interface_shader(vs_uv: vec2, /, res: vec2) -> vec4:
        uv = vs_uv * res
        return vec4(uv, 0.0, 1.0)

    result = transpile(interface_shader)
    assert "out VertexData" in result.vertex_src
    assert "in VertexData" in result.fragment_src
    assert "uniform vec2 res" in result.fragment_src
    assert "fs_color = interface_shader(VertexData.vs_uv, res)" in result.fragment_src


def test_edge_cases():
    def shader(vs_uv: vec2, /) -> vec4:
        return vec4(vs_uv, 0.0, 1.0)

    result = transpile(shader)
    assert "uniform" not in result.fragment_src
    assert "fs_color = shader(VertexData.vs_uv)" in result.fragment_src


def test_complex_shader():
    def complex_shader(vs_uv: vec2, /, u_time: float, u_res: vec2) -> vec4:
        coord = vs_uv * u_res
        wave = (coord.x + coord.y) / u_time
        return vec4(wave, wave * 0.5, 1.0 - wave, 1.0)

    result = transpile(complex_shader)
    assert "uniform float u_time" in result.fragment_src
    assert "uniform vec2 u_res" in result.fragment_src
    assert (
        "vec4 complex_shader(vec2 vs_uv, float u_time, vec2 u_res)"
        in result.fragment_src
    )
    assert (
        "fs_color = complex_shader(VertexData.vs_uv, u_time, u_res)"
        in result.fragment_src
    )
