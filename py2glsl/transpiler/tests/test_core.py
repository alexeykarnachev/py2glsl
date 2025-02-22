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
    (shader_basic, [], ["u_time"], ["vec4(vs_uv, 0.0, 1.0)"], "Basic shader"),
    (
        shader_multiple_uniforms,
        [],
        ["time", "res"],
        ["uniform float time", "uniform vec2 res"],
        "Multiple uniforms",
    ),
    (shader_vector_ops, [], [], ["color = vs_uv * 2.0"], "Vector operations"),
    (
        shader_matrix_uniform,
        [],
        ["mvp"],
        [
            "uniform mat4 mvp",
            "mvp * vec4(vs_uv, 0.0, 1.0)",
            "return pos",
        ],
        "Matrix uniform",
    ),
    (
        shader_complex_expressions,
        [],
        ["u_time"],
        ["wave = (vs_uv.x + vs_uv.y) * u_time"],
        "Complex expressions",
    ),
    (
        shader_no_uniforms,
        [],
        [],
        ["fs_color = shader_no_uniforms(vs_uv)"],
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
    assert "layout(location = 0) in vec2 a_pos" in result.vertex_src

    # Verify interface variables
    for name in exp_attrs:
        assert f"out vec2 {name}" in result.vertex_src
        assert f"in vec2 {name}" in result.fragment_src

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
