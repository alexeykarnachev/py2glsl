import pytest

from py2glsl.transpiler.glsl_builder import GLSLBuilder, GLSLCodeError


@pytest.fixture
def builder():
    return GLSLBuilder()


test_cases = [
    # Basic uniforms
    (
        {"u_time": "float"},
        {"vs_uv": "vec2"},
        ["fs_color = vec4(u_time);"],
        ["uniform float u_time;"],
        ["fs_color = vec4(u_time);"],
        [
            "gl_Position = vec4(a_pos, 0.0, 1.0);",
            "vs_uv = a_pos * 0.5 + 0.5;",
        ],
    ),
    # Vector operations
    (
        {"u_res": "vec2"},
        {"vs_uv": "vec2"},
        ["vec2 scaled = u_res * 2.0;", "fs_color = vec4(scaled, 0.0, 1.0);"],
        ["uniform vec2 u_res;"],
        ["fs_color = vec4(scaled, 0.0, 1.0);"],
        ["vs_uv = a_pos * 0.5 + 0.5;"],
    ),
    # Matrix uniforms
    (
        {"mvp": "mat4"},
        {"position": "vec3"},
        ["vec4 pos = mvp * vec4(position, 1.0);", "fs_color = pos;"],
        ["uniform mat4 mvp;"],
        ["fs_color = pos;"],
        ["gl_Position = vec4(a_pos, 0.0, 1.0);"],
    ),
    # Multiple uniforms/attributes
    (
        {"u_time": "float", "u_res": "vec2"},
        {"vs_uv": "vec2", "normal": "vec3"},
        ["fs_color = vec4(u_time / u_res.x, normal, 1.0);"],
        ["uniform float u_time;\nuniform vec2 u_res;"],
        ["fs_color = vec4(u_time / u_res.x, normal, 1.0);"],
        [],
    ),
    # Struct definitions
    (
        {"light": "Light"},
        {"vs_uv": "vec2"},
        [
            "struct Light { vec3 color; float intensity; };",
            "fs_color = vec4(light.color, light.intensity);",
        ],
        ["uniform Light light;"],
        ["struct Light", "fs_color = vec4(light.color, light.intensity);"],
        [],
    ),
    # Custom functions
    (
        {},
        {"vs_uv": "vec2"},
        ["float custom(float a) { return a * 2.0; }", "fs_color = vec4(custom(0.5));"],
        [],
        ["float custom(float a)", "return a * 2.0;"],
        [],
    ),
    # Swizzle operations
    (
        {},
        {"vs_uv": "vec2"},
        ["vec3 color = vec3(vs_uv.xy, 0.5);", "fs_color = vec4(color.rgb, 1.0);"],
        [],
        ["vec3 color = vec3(vs_uv.xy, 0.5);"],
        [],
    ),
    # Conditionals
    (
        {"u_threshold": "float"},
        {"vs_uv": "vec2"},
        [
            "if (u_threshold > 0.5) {",
            "    fs_color = vec4(1.0, 0.0, 0.0, 1.0);",
            "} else {",
            "    fs_color = vec4(0.0, 1.0, 0.0, 1.0);",
            "}",
        ],
        ["uniform float u_threshold;"],
        ["if (u_threshold > 0.5)", "fs_color = vec4(1.0, 0.0, 0.0, 1.0);"],
        [],
    ),
    # Loops
    (
        {},
        {"vs_uv": "vec2"},
        [
            "float sum = 0.0;",
            "for (int i = 0; i < 10; i++) {",
            "    sum += 0.1;",
            "}",
            "fs_color = vec4(sum);",
        ],
        [],
        ["for (int i = 0; i < 10; i++)", "sum += 0.1;"],
        [],
    ),
    # Vector constructors
    (
        {},
        {"vs_uv": "vec2"},
        ["vec4 color = vec4(vec3(vs_uv, 0.5), 1.0);"],
        [],
        ["vec4 color = vec4(vec3(vs_uv, 0.5), 1.0);"],
        [],
    ),
]


@pytest.mark.parametrize(
    "uniforms,attributes,body_lines,exp_uniforms,exp_frag,exp_vert", test_cases
)
def test_glsl_builder(
    uniforms, attributes, body_lines, exp_uniforms, exp_frag, exp_vert, builder
):
    # Configure builder
    builder.configure_shader_transpiler(
        uniforms=uniforms,
        attributes=attributes,
        func_name="main_shader",
        shader_body=body_lines,
        called_functions={},
    )

    # Verify fragment shader
    frag_src = builder.build_fragment_shader()
    for expected in exp_uniforms + exp_frag:
        assert expected in frag_src

    # Verify vertex shader
    vert_src = builder.build_vertex_shader()
    for expected in exp_vert:
        assert expected in vert_src

    # Verify vertex attributes
    assert "layout(location = 0) in vec2 a_pos" in vert_src
    for name in attributes:
        assert f"out {attributes[name]} {name}" in vert_src

    # Verify fragment inputs
    for name in attributes:
        assert f"in {attributes[name]} {name}" in frag_src


# Error case tests
error_cases = [
    (
        {"float": "float"},
        {"vs_uv": "vec2"},
        ["fs_color = vec4(0.0);"],
        "Reserved GLSL keyword: float",
        "invalid uniform name",
    ),
    (
        {"u_time": "vec2"},
        {"vs_uv": "vec2"},
        ["vec4 color = vec4(vs_uv.xyz, 1.0);"],
        "Invalid swizzle 'xyz' for vec2",
        "invalid swizzle operation",
    ),
    (
        {"u_res": "vec2"},
        {"gl_position": "vec3"},
        ["fs_color = vec4(0.0);"],
        "Reserved GLSL prefix 'gl_'",
        "reserved prefix check",
    ),
]


@pytest.mark.parametrize(
    "uniforms,attributes,body_lines,error_msg,test_id", error_cases
)
def test_error_cases(uniforms, attributes, body_lines, error_msg, test_id, builder):
    with pytest.raises(GLSLCodeError) as exc:
        builder.configure_shader_transpiler(
            uniforms=uniforms,
            attributes=attributes,
            func_name="main_shader",
            shader_body=body_lines,
            called_functions={},
        )

    assert error_msg in str(exc.value)


def test_complex_shader_structure(builder):
    """Test full shader structure with multiple components"""
    builder.configure_shader_transpiler(
        uniforms={"u_time": "float", "u_res": "vec2"},
        attributes={"vs_uv": "vec2", "normal": "vec3"},
        func_name="main_shader",
        shader_body=[
            "vec3 color = vec3(vs_uv, u_time);",
            "fs_color = vec4(color * u_res.x, 1.0);",
        ],
        called_functions={},
    )

    frag_src = builder.build_fragment_shader()
    vert_src = builder.build_vertex_shader()

    # Verify fragment components
    assert "uniform float u_time" in frag_src
    assert "uniform vec2 u_res" in frag_src
    assert "vec3 color = vec3(vs_uv, u_time)" in frag_src
    assert "fs_color = vec4(color * u_res.x, 1.0);" in frag_src

    # Verify vertex components
    assert "layout(location = 0) in vec2 a_pos" in vert_src
    assert "out vec2 vs_uv" in vert_src
    assert "out vec3 normal" in vert_src
