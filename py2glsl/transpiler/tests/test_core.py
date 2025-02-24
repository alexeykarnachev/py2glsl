import pytest

from py2glsl.glsl.builtins import dot, fract, length, mix, normalize, sin, smoothstep
from py2glsl.glsl.types import mat4, vec2, vec3, vec4
from py2glsl.transpiler.core import GLSLTypeError, transpile
from py2glsl.transpiler.glsl_builder import GLSLCodeError


# Basic test cases remain the same
def shader_basic(vs_uv: vec2, /, u_time: float) -> vec4:
    return vec4(vs_uv, 0.0, 1.0)


def shader_multiple_uniforms(vs_uv: vec2, /, time: float, res: vec2) -> vec4:
    return vec4(vs_uv, 0.0, 1.0)


def shader_matrix_uniform(vs_uv: vec2, /, mvp: mat4) -> vec4:
    pos = mvp * vec4(vs_uv, 0.0, 1.0)
    return pos


def shader_complex_expressions(vs_uv: vec2, /, u_time: float) -> vec4:
    wave = (vs_uv.x + vs_uv.y) * u_time
    return vec4(wave, wave * 0.5, 1.0 - wave, 1.0)


def shader_no_uniforms(vs_uv: vec2, /) -> vec4:
    return vec4(vs_uv, 0.0, 1.0)


def nested_helper_1(uv: vec2) -> float:
    return (uv.x + uv.y) * 0.5


def nested_helper_2(color: vec3) -> vec4:
    return vec4(color, 1.0)


def shader_with_nested_calls(vs_uv: vec2, /, u_time: float) -> vec4:
    intensity = nested_helper_1(vs_uv) * sin(u_time)
    return nested_helper_2(vec3(intensity, intensity * 0.5, 1.0 - intensity))


def recursive_shader(vs_uv: vec2, /, depth: int) -> vec4:
    if depth > 0:
        return recursive_shader(vs_uv * 0.5, depth=depth - 1)
    return vec4(vs_uv, 0.0, 1.0)


def shader_with_builtins(vs_uv: vec2, /, u_time: float) -> vec4:
    dist = length(vs_uv - vec2(0.5))
    wave = smoothstep(0.2, 0.8, sin(dist * 10.0 - u_time))
    return vec4(wave, wave * 0.5, 1.0 - wave, 1.0)


def deep_nested_shader(vs_uv: vec2, /) -> vec4:
    def level1(uv: vec2) -> vec3:
        def level2(x: float) -> float:
            return fract(x * 10.0)

        return vec3(level2(uv.x), level2(uv.y), 0.5)

    return vec4(level1(vs_uv), 1.0)


def shader_matrix_ops(vs_uv: vec2, /, mvp: mat4) -> vec4:
    pos = mvp @ vec4(vs_uv, 0.0, 1.0)
    normal = normalize(vec3(pos.xy, 1.0))
    return vec4(dot(normal, vec3(0.0, 0.0, 1.0)))


def shader_vector_ops(vs_uv: vec2, /, u_time: float) -> vec4:
    v1 = vec3(vs_uv, fract(u_time))
    v2 = normalize(v1) * 0.5
    return vec4(mix(v1, v2, 0.5), 1.0)


def shader_type_mismatch_1(vs_uv: vec2, /, scale: int) -> vec4:
    return vec4(vs_uv * scale, 0.0, 1.0)


def shader_type_mismatch_2(vs_uv: vec2, /, colors: vec3) -> vec4:
    return colors


def shader_type_inference(vs_uv: vec2, /, t: float) -> vec4:
    a = vs_uv.x * 2.0
    b = vec2(a, t)
    c = b.yxx * 0.5
    return vec4(c, a)


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
    (recursive_shader, "Recursive function calls are not supported", "Recursive call"),
    (
        deep_nested_shader,
        "Nested function definitions are not supported",
        "Deep nesting",
    ),
    (shader_type_mismatch_1, "Type mismatch: vec2 vs float", "Vector/scalar mismatch"),
    (
        shader_type_mismatch_2,
        "Return type mismatch: vec3 vs vec4",
        "Invalid return type",
    ),
]

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
    (
        shader_with_nested_calls,
        [],
        ["u_time"],
        [
            "nested_helper_1(vs_uv)",
            "sin(u_time)",
            "nested_helper_2(vec3(intensity, intensity * 0.5, 1.0 - intensity))",
        ],
        "Nested function calls",
    ),
    (
        shader_with_builtins,
        [],
        ["u_time"],
        [
            "length(vs_uv - vec2(0.5))",
            "smoothstep(0.2, 0.8, sin(dist * 10.0 - u_time))",
        ],
        "Builtin functions",
    ),
    (
        shader_matrix_ops,
        [],
        ["mvp"],
        [
            "mvp * vec4(vs_uv, 0.0, 1.0)",
            "normalize(vec3(pos.xy, 1.0))",
            "dot(normal, vec3(0.0, 0.0, 1.0))",
        ],
        "Matrix operations",
    ),
    (
        shader_vector_ops,
        [],
        ["u_time"],
        ["vec3(vs_uv, fract(u_time))", "normalize(v1) * 0.5", "mix(v1, v2, 0.5)"],
        "Vector operations",
    ),
    (
        shader_type_inference,
        [],
        ["t"],
        ["vec2(a, t)", "b.yxx * 0.5", "vec4(c, a)"],
        "Type inference",
    ),
]


# ------------------------------------------------------------------------
# Tests
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


@pytest.mark.parametrize(
    "shader_func,error_msg,test_id",
    error_cases,
    ids=[case[2] for case in error_cases],
)
def test_error_cases(shader_func, error_msg, test_id):
    with pytest.raises((GLSLTypeError, GLSLCodeError)) as exc:
        transpile(shader_func)
    assert error_msg in str(exc.value)


def test_nested_functions():
    result = transpile(shader_with_nested_calls)
    assert "float nested_helper_1(vec2 uv)" in result.fragment_src
    assert "vec4 nested_helper_2(vec3 color)" in result.fragment_src
    assert "nested_helper_1(vs_uv)" in result.fragment_src
    assert (
        "nested_helper_2(vec3(intensity, intensity * 0.5, 1.0 - intensity))"
        in result.fragment_src
    )


def test_deep_nesting():
    with pytest.raises(GLSLCodeError) as exc:
        transpile(deep_nested_shader)
    assert "Nested function definitions are not supported" in str(exc.value)


def test_recursive_calls():
    with pytest.raises(GLSLCodeError) as exc:
        transpile(recursive_shader)
    assert "Recursive function calls are not supported" in str(exc.value)


def test_builtin_functions():
    result = transpile(shader_with_builtins)
    assert "length(" in result.fragment_src
    assert "smoothstep(" in result.fragment_src
    assert "sin(" in result.fragment_src


def test_matrix_operations():
    result = transpile(shader_matrix_ops)
    assert "mat4 mvp" in result.fragment_src
    assert "mvp * vec4(vs_uv, 0.0, 1.0)" in result.fragment_src
    assert "dot(normal, vec3(0.0, 0.0, 1.0))" in result.fragment_src


def test_vector_operations():
    result = transpile(shader_vector_ops)
    assert "vec3(vs_uv, fract(u_time))" in result.fragment_src
    assert "normalize(v1) * 0.5" in result.fragment_src
    assert "mix(v1, v2, 0.5)" in result.fragment_src


def test_type_inference():
    result = transpile(shader_type_inference)
    assert "float a = vs_uv.x * 2.0" in result.fragment_src
    assert "vec2 b = vec2(a, t)" in result.fragment_src
    assert "vec3 c = b.yxx * 0.5" in result.fragment_src
    assert "vec4(c, a)" in result.fragment_src


def test_type_mismatch_errors():
    with pytest.raises(GLSLTypeError):
        transpile(shader_type_mismatch_1)
    with pytest.raises(GLSLTypeError):
        transpile(shader_type_mismatch_2)
