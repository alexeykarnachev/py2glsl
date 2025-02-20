import pytest

from py2glsl.glsl.types import mat4, vec2, vec4
from py2glsl.transpiler.core import GLSLTypeError, transpile
from py2glsl.transpiler.glsl_builder import GLSLCodeError


# Valid shader functions
def shader_basic(vs_uv: vec2, /, u_time: float) -> vec4:
    return vec4(vs_uv, 0.0, 1.0)


def shader_multiple_uniforms(vs_uv: vec2, /, time: float, res: vec2) -> vec4:
    return vec4(vs_uv, 0.0, 1.0)


def shader_vector_ops(vs_uv: vec2, /) -> vec4:
    color = vs_uv * 2.0
    return vec4(color, 0.0, 1.0)


def shader_matrix_uniform(vs_uv: vec2, /, mvp: mat4) -> vec4:
    pos = mvp * vec4(vs_uv, 0.0, 1.0)
    return pos


def shader_complex_expressions(vs_uv: vec2, /, u_time: float) -> vec4:
    wave = (vs_uv.x + vs_uv.y) * u_time
    return vec4(wave, wave * 0.5, 1.0 - wave, 1.0)


def shader_no_uniforms(vs_uv: vec2, /) -> vec4:
    return vec4(vs_uv, 0.0, 1.0)


valid_shader_cases = [
    (shader_basic, ["vs_uv"], ["u_time"], ["vec4(vs_uv, 0.0, 1.0)"], "Basic shader"),
    (
        shader_multiple_uniforms,
        ["vs_uv"],
        ["time", "res"],
        ["uniform float time", "uniform vec2 res"],
        "Multiple uniforms",
    ),
    (shader_vector_ops, ["vs_uv"], [], ["color = vs_uv * 2.0"], "Vector operations"),
    (
        shader_matrix_uniform,
        ["vs_uv"],
        ["mvp"],
        ["uniform mat4 mvp", "mvp * vec4(vs_uv, 0.0, 1.0)"],
        "Matrix uniform",
    ),
    (
        shader_complex_expressions,
        ["vs_uv"],
        ["u_time"],
        ["wave = (vs_uv.x + vs_uv.y) * u_time"],
        "Complex expressions",
    ),
    (
        shader_no_uniforms,
        ["vs_uv"],
        [],
        ["fs_color = shader_no_uniforms(VertexData.vs_uv)"],
        "No uniforms",
    ),
]


@pytest.mark.parametrize(
    "shader_func,exp_attrs,exp_uniforms,exp_frag,test_id",
    valid_shader_cases,
    ids=[case[4] for case in valid_shader_cases],
)
def test_valid_shaders(shader_func, exp_attrs, exp_uniforms, exp_frag, test_id):
    result = transpile(shader_func)

    # Verify vertex attributes
    for attr in exp_attrs:
        assert (
            f"layout(location = {exp_attrs.index(attr)}) in vec2 {attr}"
            in result.vertex_src
        )

    # Verify uniforms
    for uniform in exp_uniforms:
        assert uniform in result.fragment_src

    # Verify fragment body
    for frag_code in exp_frag:
        assert frag_code in result.fragment_src


# Error case functions
def shader_bad_return_type(vs_uv: vec2, /) -> int:
    return vec4(0)


def shader_missing_positional(*, uv: vec2) -> vec4:
    return vec4(uv, 0.0, 1.0)


def shader_wrong_positional_type(vs_uv: float, /) -> vec4:
    return vec4(0.0)


def shader_invalid_uniform_type(vs_uv: vec2, /, invalid: list) -> vec4:
    return vec4(0)


def shader_multiple_positional(pos: vec2, uv: vec2, /) -> vec4:
    return vec4(0)


def shader_reserved_keyword(vs_uv: vec2, /, float: float) -> vec4:
    return vec4(0)


error_cases = [
    (shader_bad_return_type, "Shader must return vec4", "Bad return type"),
    (
        shader_missing_positional,
        "Must have one positional-only vec2 parameter",
        "Missing positional param",
    ),
    (
        shader_wrong_positional_type,
        "Must have one positional-only vec2 parameter",
        "Wrong positional type",
    ),
    (shader_invalid_uniform_type, "Unsupported type: list", "Invalid uniform type"),
    (
        shader_multiple_positional,
        "Must have one positional-only vec2 parameter",
        "Multiple positional params",
    ),
    (shader_reserved_keyword, "Reserved GLSL keyword: float", "Reserved keyword param"),
]


@pytest.mark.parametrize(
    "shader_func,error_msg,test_id",
    error_cases,
    ids=[case[2] for case in error_cases],
)
def test_error_cases(shader_func, error_msg, test_id):
    with pytest.raises((GLSLTypeError, GLSLCodeError)) as exc:
        transpile(shader_func)

    assert error_msg in str(exc.value)


def test_complex_shader_structure():
    def complex_shader(vs_uv: vec2, /, u_time: float, u_res: vec2) -> vec4:
        coord = vs_uv * u_res
        wave = (coord.x + coord.y) / u_time
        return vec4(wave, wave * 0.5, 1.0 - wave, 1.0)

    result = transpile(complex_shader)

    # Verify interface
    assert "out VertexData" in result.vertex_src
    assert "in VertexData" in result.fragment_src

    # Verify uniforms
    assert "uniform float u_time" in result.fragment_src
    assert "uniform vec2 u_res" in result.fragment_src

    # Verify calculations
    assert "coord = vs_uv * u_res" in result.fragment_src
    assert "wave = (coord.x + coord.y) / u_time" in result.fragment_src
    assert "vec4(wave, wave * 0.5, 1.0 - wave, 1.0)" in result.fragment_src
