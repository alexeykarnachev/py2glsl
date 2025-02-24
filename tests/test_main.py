import ast

import pytest

from py2glsl.main import GLSLGenerator, builtins, transpile


@pytest.fixture
def generator():
    """Fixture providing a fresh GLSLGenerator instance with default uniforms."""
    default_uniforms = {
        "u_time": "float",
        "u_aspect": "float",
        "u_resolution": "vec2",
        "u_mouse_pos": "vec2",
        "u_mouse_uv": "vec2",
    }
    return GLSLGenerator({}, {}, {}, builtins, default_uniforms)


@pytest.fixture
def symbols():
    """Fixture providing a sample symbol table."""
    return {
        "hex_coord": "vec2",
        "color": "vec4",
        "uv": "vec2",
        "time": "float",
        "i": "int",
        "test": "Test",  # For struct test
    }


def test_generate_vec2_attribute(generator, symbols):
    """Test generating and typing a vec2 attribute access."""
    node = ast.parse("hex_coord.x", mode="eval").body
    code = generator.generate_expr(node, symbols)
    expr_type = generator.get_expr_type(node, symbols)
    assert code == "hex_coord.x"
    assert expr_type == "float"


def test_generate_vec4_rgb(generator, symbols):
    """Test generating and typing a vec4 rgb swizzle."""
    node = ast.parse("color.rgb", mode="eval").body
    code = generator.generate_expr(node, symbols)
    expr_type = generator.get_expr_type(node, symbols)
    assert code == "color.rgb"
    assert expr_type == "vec3"


def test_binop_vec2_float(generator, symbols):
    """Test generating and typing a vec2 * float operation."""
    node = ast.parse("uv * 2.0", mode="eval").body
    code = generator.generate_expr(node, symbols)
    expr_type = generator.get_expr_type(node, symbols)
    assert code == "(uv * 2.0)"
    assert expr_type == "vec2"


def test_invalid_attribute_raises_error(generator, symbols):
    """Test that an invalid attribute access raises a ValueError."""
    node = ast.parse("hex_coord.z", mode="eval").body
    with pytest.raises(
        ValueError, match="Cannot infer type for attribute 'z' of 'vec2'"
    ):
        generator.get_expr_type(node, symbols)


def test_version_directive_first_line():
    """Test that #version 460 core is the first line in the generated GLSL."""
    shader_code = """
def main_shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    return vec4(1.0, 0.0, 0.0, 1.0)
"""
    glsl_code, _ = transpile(shader_code)
    lines = glsl_code.splitlines()
    assert (
        lines[0] == "#version 460 core"
    ), "The #version directive must be the first line"


def test_shader_compilation():
    """Test that the generated GLSL code has the expected structure."""
    shader_code = """
def main_shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    return vec4(sin(u_time), 0.0, 0.0, 1.0)
"""
    glsl_code, _ = transpile(shader_code)
    assert "main_shader" in glsl_code
    assert "void main()" in glsl_code
    assert "fragColor" in glsl_code


def test_vec4_rgba_swizzle(generator, symbols):
    """Test generating and typing a vec4 rgba swizzle."""
    node = ast.parse("color.rgba", mode="eval").body
    code = generator.generate_expr(node, symbols)
    expr_type = generator.get_expr_type(node, symbols)
    assert code == "color.rgba"
    assert expr_type == "vec4"


def test_vec4_xy_swizzle(generator, symbols):
    """Test generating and typing a vec4 xy swizzle."""
    node = ast.parse("color.xy", mode="eval").body
    code = generator.generate_expr(node, symbols)
    expr_type = generator.get_expr_type(node, symbols)
    assert code == "color.xy"
    assert expr_type == "vec2"


def test_binop_vec4_vec4_addition(generator, symbols):
    """Test generating and typing a vec4 + vec4 operation."""
    node = ast.parse("color + color", mode="eval").body
    code = generator.generate_expr(node, symbols)
    expr_type = generator.get_expr_type(node, symbols)
    assert code == "(color + color)"
    assert expr_type == "vec4"


def test_function_call_with_args(generator, symbols):
    """Test generating and typing a function call with arguments."""
    generator.functions = {"wave": ("float", ["vec2", "float"])}
    node = ast.parse("wave(uv, time)", mode="eval").body
    code = generator.generate_expr(node, symbols)
    expr_type = generator.get_expr_type(node, symbols)
    assert code == "wave(uv, time)"
    assert expr_type == "float"


def test_nested_function_call(generator, symbols):
    """Test generating and typing a nested function call."""
    generator.functions = {"sin": ("float", ["float"]), "length": ("float", ["vec2"])}
    node = ast.parse("sin(length(uv))", mode="eval").body
    code = generator.generate_expr(node, symbols)
    expr_type = generator.get_expr_type(node, symbols)
    assert code == "sin(length(uv))"
    assert expr_type == "float"


def test_missing_main_shader_raises_error():
    """Test that a missing main_shader raises a ValueError."""
    shader_code = """
def helper(uv: 'vec2') -> 'float':
    return sin(uv.x)
"""
    with pytest.raises(
        ValueError, match="Main shader function 'main_shader' not found"
    ):
        transpile(shader_code)


def test_multiple_helper_functions():
    """Test transpilation with multiple helper functions."""
    shader_code = """
def helper1(uv: 'vec2') -> 'float':
    return sin(uv.x)

def helper2(uv: 'vec2', time: 'float') -> 'float':
    return cos(uv.y + time)

def main_shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    return vec4(helper1(vs_uv), helper2(vs_uv, u_time), 0.0, 1.0)
"""
    glsl_code, _ = transpile(shader_code)
    assert "helper1" in glsl_code
    assert "helper2" in glsl_code
    assert "main_shader" in glsl_code


def test_uniform_declarations():
    """Test that uniforms are correctly declared in GLSL."""
    shader_code = """
def main_shader(vs_uv: 'vec2', u_time: 'float', u_scale: 'float') -> 'vec4':
    return vec4(vs_uv * u_scale, 0.0, 1.0)
"""
    glsl_code, used_uniforms = transpile(shader_code)
    assert "uniform float u_time;" in glsl_code
    assert "uniform float u_scale;" in glsl_code
    assert used_uniforms == {"u_time", "u_scale"}


def test_no_return_type_raises_error():
    """Test that a missing return type raises an error."""
    shader_code = """
def main_shader(vs_uv: 'vec2', u_time: 'float'):
    uv = vs_uv
"""
    with pytest.raises(
        ValueError, match="Function 'main_shader' must have a return type annotation"
    ):
        transpile(shader_code)


def test_complex_expression_in_shader():
    """Test a shader with a complex expression."""
    shader_code = """
def main_shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    uv = vs_uv * 2.0
    color = vec4(sin(uv.x + u_time), cos(uv.y - u_time), 0.5, 1.0)
    return color
"""
    glsl_code, _ = transpile(shader_code)
    assert "uv = (vs_uv * 2.0)" in glsl_code
    assert "sin((uv.x + u_time))" in glsl_code


def test_unsupported_binop_raises_error(generator, symbols):
    """Test that an unsupported binary operation raises an error."""
    node = ast.parse("uv % 2.0", mode="eval").body
    with pytest.raises(ValueError, match="Unsupported expression type"):
        generator.generate_expr(node, symbols)


def test_empty_shader_raises_error():
    """Test that an empty shader raises an error."""
    shader_code = ""
    with pytest.raises(
        ValueError, match="Main shader function 'main_shader' not found"
    ):
        transpile(shader_code)


def test_invalid_function_call_raises_error(generator, symbols):
    """Test that an unknown function call raises an error."""
    node = ast.parse("unknown(uv)", mode="eval").body
    with pytest.raises(ValueError, match="Unknown function: unknown"):
        generator.get_expr_type(node, symbols)


def test_shader_with_no_body_raises_error():
    """Test that a main_shader with no body raises an error."""
    shader_code = """
def main_shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    pass
"""
    with pytest.raises(ValueError, match="Unsupported statement type"):
        transpile(shader_code)


def test_augmented_assignment():
    """Test that augmented assignment is correctly transpiled."""
    shader_code = """
def main_shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    uv = vs_uv
    uv *= 2.0
    return vec4(uv, 0.0, 1.0)
"""
    glsl_code, _ = transpile(shader_code)
    assert "uv = (uv * 2.0)" in glsl_code


def test_struct_definition():
    """Test that a struct is correctly defined and used."""
    shader_code = """
class TestStruct:
    x: 'float'
    y: 'vec2'

def main_shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    test: 'Test' = TestStruct(1.0, vs_uv)
    return vec4(test.y, 0.0, 1.0)
"""
    glsl_code, _ = transpile(shader_code)
    assert "struct Test {" in glsl_code
    assert "float x;" in glsl_code
    assert "vec2 y;" in glsl_code
    assert "Test test = Test(1.0, vs_uv)" in glsl_code


def test_while_loop():
    """Test that a while loop is correctly transpiled."""
    shader_code = """
def main_shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    i = 0
    while i < 10:
        i += 1
    return vec4(float(i) * 0.1, 0.0, 0.0, 1.0)
"""
    glsl_code, _ = transpile(shader_code)
    assert "while (i < 10)" in glsl_code
    assert "i = (i + 1)" in glsl_code


def test_attribute_assignment():
    """Test that attribute assignment is correctly transpiled."""
    shader_code = """
class TestStruct:
    x: 'float'
    y: 'vec2'

def main_shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    test: 'Test' = TestStruct(1.0, vs_uv)
    test.y = vs_uv * 2.0
    return vec4(test.y, 0.0, 1.0)
"""
    glsl_code, _ = transpile(shader_code)
    assert "test.y = (vs_uv * 2.0)" in glsl_code


def test_if_statement():
    """Test that an if statement is correctly transpiled."""
    shader_code = """
def main_shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    color = vec3(0.0)
    if u_time > 1.0:
        color = vec3(1.0, 0.0, 0.0)
    else:
        color = vec3(0.0, 1.0, 0.0)
    return vec4(color, 1.0)
"""
    glsl_code, _ = transpile(shader_code)
    assert "if (u_time > 1.0)" in glsl_code
    assert "else" in glsl_code
    assert "color = vec3(1.0, 0.0, 0.0)" in glsl_code
    assert "color = vec3(0.0, 1.0, 0.0)" in glsl_code


def test_break_in_loop():
    """Test that a break statement in a loop is correctly transpiled."""
    shader_code = """
def main_shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    i = 0
    while i < 10:
        if i > 5:
            break
        i += 1
    return vec4(float(i) * 0.1, 0.0, 0.0, 1.0)
"""
    glsl_code, _ = transpile(shader_code)
    assert "while (i < 10)" in glsl_code
    assert "if (i > 5)" in glsl_code
    assert "break;" in glsl_code


def test_for_loop():
    """Test that a for loop is correctly transpiled."""
    shader_code = """
def main_shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    i = 0
    for i in range(10):
        i += 1
    return vec4(float(i) * 0.1, 0.0, 0.0, 1.0)
"""
    glsl_code, _ = transpile(shader_code)
    assert "for (int i = 0; i < 10; i += 1)" in glsl_code


def test_boolean_operation():
    """Test that a boolean operation is correctly transpiled."""
    shader_code = """
def main_shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    i = 0
    while i < 10 or u_time > 1.0:
        i += 1
    return vec4(float(i) * 0.1, 0.0, 0.0, 1.0)
"""
    glsl_code, _ = transpile(shader_code)
    assert "while ((i < 10) || (u_time > 1.0))" in glsl_code


def test_global_variables():
    """Test that global variables are correctly transpiled."""
    shader_code = """
PI: 'float' = 3.141592
MAX_STEPS: 'int' = 10

def main_shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    return vec4(sin(PI * u_time), 0.0, 0.0, 1.0)
"""
    glsl_code, _ = transpile(shader_code)
    assert "float PI = 3.141592;" in glsl_code
    assert "int MAX_STEPS = 10;" in glsl_code
    assert "sin((PI * u_time))" in glsl_code


def test_default_uniforms_included():
    """Test that default uniforms are included only if used."""
    shader_code = """
def main_shader(vs_uv: 'vec2', u_time: 'float', u_aspect: 'float') -> 'vec4':
    return vec4(sin(u_time), cos(u_aspect), 0.0, 1.0)
"""
    glsl_code, used_uniforms = transpile(shader_code)
    assert "uniform float u_time;" in glsl_code
    assert "uniform float u_aspect;" in glsl_code
    assert used_uniforms == {"u_time", "u_aspect"}


def test_unused_default_uniforms_not_included():
    """Test that unused default uniforms are not included in GLSL."""
    shader_code = """
def main_shader(vs_uv: 'vec2') -> 'vec4':
    return vec4(vs_uv, 0.0, 1.0)
"""
    glsl_code, used_uniforms = transpile(shader_code)
    assert "uniform float u_time;" not in glsl_code
    assert "uniform float u_aspect;" not in glsl_code
    assert used_uniforms == set()


def test_custom_uniforms_included():
    """Test that custom uniforms are included if used."""
    shader_code = """
def main_shader(vs_uv: 'vec2', u_custom: 'vec3') -> 'vec4':
    return vec4(u_custom, 1.0)
"""
    glsl_code, used_uniforms = transpile(shader_code)
    assert "uniform vec3 u_custom;" in glsl_code
    assert used_uniforms == {"u_custom"}


def test_mixed_uniforms():
    """Test that both default and custom uniforms are handled correctly."""
    shader_code = """
def main_shader(vs_uv: 'vec2', u_time: 'float', u_custom: 'float') -> 'vec4':
    return vec4(sin(u_time * u_custom), 0.0, 0.0, 1.0)
"""
    glsl_code, used_uniforms = transpile(shader_code)
    assert "uniform float u_time;" in glsl_code
    assert "uniform float u_custom;" in glsl_code
    assert used_uniforms == {"u_time", "u_custom"}
