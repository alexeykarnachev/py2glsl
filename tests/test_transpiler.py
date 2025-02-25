import ast
from dataclasses import dataclass

import pytest

from py2glsl.builtins import cos, normalize, sin, vec2, vec3, vec4
from py2glsl.transpiler import (
    FunctionCollector,
    GLSLGenerator,
    TranspilerError,
    transpile,
)


@pytest.fixture
def generator():
    """Fixture providing a fresh GLSLGenerator instance for testing."""
    collector = FunctionCollector()
    return GLSLGenerator(collector)


@pytest.fixture
def symbols():
    """Fixture providing a sample symbol table."""
    return {
        "hex_coord": "vec2",
        "color": "vec4",
        "uv": "vec2",
        "time": "float",
        "i": "int",
        "test": "Test",
    }


def test_generate_vec2_attribute(generator, symbols):
    node = ast.parse("hex_coord.x", mode="eval").body
    code = generator._generate_expr(node, symbols)
    expr_type = generator._get_expr_type(node, symbols)
    assert code == "hex_coord.x"
    assert expr_type == "float"


def test_main_shader_no_return_type():
    shader_code = """
def shader(vs_uv: 'vec2', u_time: 'float'):
    return vec4(1.0, 0.0, 0.0, 1.0)
"""
    glsl_code, uniforms = transpile(shader_code)
    assert "uniform float u_time;" in glsl_code
    assert "vec4 shader(vec2 vs_uv, float u_time) {" in glsl_code
    assert "return vec4(1.0, 0.0, 0.0, 1.0);" in glsl_code
    assert "void main() {" in glsl_code
    assert "fragColor = shader(u_time, vs_uv);" in glsl_code
    assert uniforms == {"u_time"}


def test_helper_function_requires_return_type():
    shader_code = """
def helper(x: 'float'):
    return x * 2.0
def shader(vs_uv: 'vec2'):
    return vec4(helper(1.0), 0.0, 0.0, 1.0)
"""
    with pytest.raises(
        TranspilerError, match="Helper function 'helper' lacks return type annotation"
    ):
        transpile(shader_code)


def test_struct_and_uniforms():
    shader_code = """
from dataclasses import dataclass
@dataclass
class Material:
    color: 'vec3'
    shininess: 'float'
def shader(vs_uv: 'vec2', u_mat: 'Material'):
    return vec4(u_mat.color, 1.0)
"""
    glsl_code, uniforms = transpile(shader_code)
    assert "struct Material {" in glsl_code
    assert "    vec3 color;" in glsl_code
    assert "    float shininess;" in glsl_code
    assert "uniform Material u_mat;" in glsl_code
    assert "vec4 shader(vec2 vs_uv, Material u_mat) {" in glsl_code
    assert "return vec4(u_mat.color, 1.0);" in glsl_code
    assert uniforms == {"u_mat"}


def test_multi_uniform_struct():
    @dataclass
    class UniStruct:
        offset: "vec3"
        active: "bool"

    def shader(vs_uv: "vec2", u_time: "float", u_offset: "vec3"):
        s = UniStruct(offset=u_offset, active=(sin(u_time) > 0.0))
        return vec4(s.offset, 1.0) if s.active else vec4(0.0, 0.0, 0.0, 1.0)

    glsl_code, used_uniforms = transpile(shader)
    assert "struct UniStruct {" in glsl_code
    assert "uniform float u_time;" in glsl_code
    assert "uniform vec3 u_offset;" in glsl_code
    assert "UniStruct s = UniStruct(u_offset, sin(u_time) > 0.0);" in glsl_code
    assert (
        "return (s.active ? vec4(s.offset, 1.0) : vec4(0.0, 0.0, 0.0, 1.0));"
        in glsl_code
    )
    assert used_uniforms == {"u_time", "u_offset"}


def test_generate_vec4_rgb(generator, symbols):
    node = ast.parse("color.rgb", mode="eval").body
    code = generator._generate_expr(node, symbols)
    expr_type = generator._get_expr_type(node, symbols)
    assert code == "color.rgb"
    assert expr_type == "vec3"


def test_binop_vec2_float(generator, symbols):
    node = ast.parse("uv * 2.0", mode="eval").body
    code = generator._generate_expr(node, symbols)
    expr_type = generator._get_expr_type(node, symbols)
    assert code == "uv * 2.0"
    assert expr_type == "vec2"


def test_invalid_attribute_raises_error(generator, symbols):
    node = ast.parse("hex_coord.z", mode="eval").body
    with pytest.raises(TranspilerError, match="Invalid swizzle 'z' for vec2"):
        generator._get_expr_type(node, symbols)


def test_version_directive_first_line():
    shader_code = """
def shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    return vec4(1.0, 0.0, 0.0, 1.0)
"""
    glsl_code, _ = transpile(shader_code)
    lines = glsl_code.splitlines()
    assert (
        lines[0] == "#version 460 core"
    ), "The #version directive must be the first line"


def test_shader_compilation():
    shader_code = """
def shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    return vec4(sin(u_time), 0.0, 0.0, 1.0)
"""
    glsl_code, _ = transpile(shader_code)
    assert "shader" in glsl_code
    assert "void main()" in glsl_code
    assert "fragColor" in glsl_code


def test_vec4_rgba_swizzle(generator, symbols):
    node = ast.parse("color.rgba", mode="eval").body
    code = generator._generate_expr(node, symbols)
    expr_type = generator._get_expr_type(node, symbols)
    assert code == "color.rgba"
    assert expr_type == "vec4"


def test_vec4_xy_swizzle(generator, symbols):
    node = ast.parse("color.xy", mode="eval").body
    code = generator._generate_expr(node, symbols)
    expr_type = generator._get_expr_type(node, symbols)
    assert code == "color.xy"
    assert expr_type == "vec2"


def test_binop_vec4_vec4_addition(generator, symbols):
    node = ast.parse("color + color", mode="eval").body
    code = generator._generate_expr(node, symbols)
    expr_type = generator._get_expr_type(node, symbols)
    assert code == "color + color"
    assert expr_type == "vec4"


def test_function_call_with_args(generator, symbols):
    dummy_node = ast.FunctionDef(
        name="wave", args=ast.arguments(args=[]), body=[ast.Pass()]
    )
    generator.collector.functions["wave"] = ("float", ["vec2", "float"], dummy_node)
    node = ast.parse("wave(uv, time)", mode="eval").body
    code = generator._generate_expr(node, symbols)
    assert code == "wave(uv, time)"
    assert generator._get_expr_type(node, symbols) == "float"


def test_nested_function_call(generator, symbols):
    generator.collector.functions["sin"] = (
        "float",
        ["float"],
        ast.FunctionDef(name="sin", args=ast.arguments(args=[]), body=[]),
    )
    generator.collector.functions["length"] = (
        "float",
        ["vec2"],
        ast.FunctionDef(name="length", args=ast.arguments(args=[]), body=[]),
    )
    node = ast.parse("sin(length(uv))", mode="eval").body
    code = generator._generate_expr(node, symbols)
    expr_type = generator._get_expr_type(node, symbols)
    assert code == "sin(length(uv))"
    assert expr_type == "float"


def test_missing_main_shader_raises_error():
    shader_code = """
def helper(uv: 'vec2') -> 'float':
    return sin(uv.x)
"""
    with pytest.raises(TranspilerError, match="Main function 'shader' not found"):
        transpile(shader_code)


def test_multiple_helper_functions():
    shader_code = """
def helper1(uv: 'vec2') -> 'float':
    return sin(uv.x)
def helper2(uv: 'vec2', time: 'float') -> 'float':
    return cos(uv.y + time)
def shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    return vec4(helper1(vs_uv), helper2(vs_uv, u_time), 0.0, 1.0)
"""
    glsl_code, _ = transpile(shader_code)
    assert "helper1" in glsl_code
    assert "helper2" in glsl_code
    assert "shader" in glsl_code


def test_uniform_declarations():
    shader_code = """
def shader(vs_uv: 'vec2', u_time: 'float', u_scale: 'float') -> 'vec4':
    return vec4(vs_uv * u_scale, 0.0, 1.0)
"""
    glsl_code, used_uniforms = transpile(shader_code)
    assert "uniform float u_time;" in glsl_code
    assert "uniform float u_scale;" in glsl_code
    assert used_uniforms == {"u_time", "u_scale"}


def test_no_return_type_raises_error():
    shader_code = """
def helper(vs_uv: 'vec2', u_time: 'float'):
    uv = vs_uv
def shader(vs_uv: 'vec2', u_time: 'float'):
    return vec4(1.0, 0.0, 0.0, 1.0)
"""
    with pytest.raises(
        TranspilerError, match="Helper function 'helper' lacks return type annotation"
    ):
        transpile(shader_code)


def test_complex_expression_in_shader():
    shader_code = """
def shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    uv = vs_uv * 2.0
    color = vec4(sin(uv.x + u_time), cos(uv.y - u_time), 0.5, 1.0)
    return color
"""
    glsl_code, _ = transpile(shader_code)
    assert "uv = vs_uv * 2.0" in glsl_code
    assert "sin(uv.x + u_time)" in glsl_code


def test_unsupported_binop_raises_error(generator, symbols):
    node = ast.parse("uv % 2.0", mode="eval").body
    with pytest.raises(TranspilerError, match="Unsupported binary op: Mod"):
        generator._generate_expr(node, symbols)


def test_empty_shader_raises_error():
    shader_code = ""
    with pytest.raises(TranspilerError, match="Empty shader code provided"):
        transpile(shader_code)


def test_invalid_function_call_raises_error(generator, symbols):
    node = ast.parse("unknown(uv)", mode="eval").body
    with pytest.raises(TranspilerError, match="Unknown function call: unknown"):
        generator._generate_expr(node, symbols)


def test_shader_with_no_body_raises_error():
    shader_code = """
def shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    pass
"""
    with pytest.raises(
        TranspilerError, match="Pass statements are not supported in GLSL"
    ):
        transpile(shader_code)


def test_augmented_assignment():
    shader_code = """
def shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    uv = vs_uv
    uv *= 2.0
    return vec4(uv, 0.0, 1.0)
"""
    glsl_code, _ = transpile(shader_code)
    assert "uv = uv * 2.0" in glsl_code


def test_struct_definition():
    shader_code = """
from dataclasses import dataclass
@dataclass
class Test:
    x: 'float'
    y: 'vec2'
def shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    test: 'Test' = Test(1.0, vs_uv)
    return vec4(test.y, 0.0, 1.0)
"""
    glsl_code, _ = transpile(shader_code)
    assert "struct Test {" in glsl_code
    assert "float x;" in glsl_code
    assert "vec2 y;" in glsl_code
    assert "Test test = Test(1.0, vs_uv)" in glsl_code


def test_while_loop():
    shader_code = """
def shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    i = 0
    while i < 10:
        i += 1
    return vec4(float(i) * 0.1, 0.0, 0.0, 1.0)
"""
    glsl_code, _ = transpile(shader_code)
    assert "while (i < 10)" in glsl_code
    assert "i = i + 1;" in glsl_code


def test_attribute_assignment():
    shader_code = """
from dataclasses import dataclass
@dataclass
class Test:
    x: 'float'
    y: 'vec2'
def shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    test: 'Test' = Test(1.0, vs_uv)
    test.y = vs_uv * 2.0
    return vec4(test.y, 0.0, 1.0)
"""
    glsl_code, _ = transpile(shader_code)
    assert "test.y = vs_uv * 2.0" in glsl_code


def test_if_statement():
    shader_code = """
def shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    color = vec3(0.0)
    if u_time > 1.0:
        color = vec3(1.0, 0.0, 0.0)
    else:
        color = vec3(0.0, 1.0, 0.0)
    return vec4(color, 1.0)
"""
    glsl_code, _ = transpile(shader_code)
    assert "if (u_time > 1.0)" in glsl_code
    assert "color = vec3(1.0, 0.0, 0.0);" in glsl_code
    assert "else" in glsl_code
    assert "color = vec3(0.0, 1.0, 0.0);" in glsl_code


def test_break_in_loop():
    shader_code = """
def shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
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
    shader_code = """
def shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    i = 0
    for i in range(10):
        i += 1
    return vec4(float(i) * 0.1, 0.0, 0.0, 1.0)
"""
    glsl_code, _ = transpile(shader_code)
    assert "for (int i = 0; i < 10; i += 1)" in glsl_code


def test_boolean_operation():
    shader_code = """
def shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    i = 0
    while i < 10 or u_time > 1.0:
        i += 1
    return vec4(float(i) * 0.1, 0.0, 0.0, 1.0)
"""
    glsl_code, _ = transpile(shader_code)
    assert "while (i < 10 || u_time > 1.0)" in glsl_code


def test_global_variables():
    shader_code = """
PI: 'float' = 3.141592
MAX_STEPS: 'int' = 10
def shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    return vec4(sin(PI * u_time), 0.0, 0.0, 1.0)
"""
    glsl_code, _ = transpile(shader_code)
    assert "const float PI = 3.141592;" in glsl_code
    assert "const int MAX_STEPS = 10;" in glsl_code
    assert "sin(PI * u_time)" in glsl_code


def test_default_uniforms_included():
    shader_code = """
def shader(vs_uv: 'vec2', u_time: 'float', u_aspect: 'float') -> 'vec4':
    return vec4(sin(u_time), cos(u_aspect), 0.0, 1.0)
"""
    glsl_code, used_uniforms = transpile(shader_code)
    assert "uniform float u_time;" in glsl_code
    assert "uniform float u_aspect;" in glsl_code
    assert used_uniforms == {"u_time", "u_aspect"}


def test_unused_default_uniforms_not_included():
    shader_code = """
def shader(vs_uv: 'vec2') -> 'vec4':
    return vec4(vs_uv, 0.0, 1.0)
"""
    glsl_code, used_uniforms = transpile(shader_code)
    assert "uniform float u_time;" not in glsl_code
    assert "uniform float u_aspect;" not in glsl_code
    assert used_uniforms == set()


def test_custom_uniforms_included():
    shader_code = """
def shader(vs_uv: 'vec2', u_custom: 'vec3') -> 'vec4':
    return vec4(u_custom, 1.0)
"""
    glsl_code, used_uniforms = transpile(shader_code)
    assert "uniform vec3 u_custom;" in glsl_code
    assert used_uniforms == {"u_custom"}


def test_mixed_uniforms():
    shader_code = """
def shader(vs_uv: 'vec2', u_time: 'float', u_custom: 'float') -> 'vec4':
    return vec4(sin(u_time * u_custom), 0.0, 0.0, 1.0)
"""
    glsl_code, used_uniforms = transpile(shader_code)
    assert "uniform float u_time;" in glsl_code
    assert "uniform float u_custom;" in glsl_code
    assert used_uniforms == {"u_time", "u_custom"}


def test_struct_initialization_keywords():
    @dataclass
    class SimpleStruct:
        x: "float"
        y: "vec3"
        z: "int"

    def shader(vs_uv: "vec2") -> "vec4":
        s = SimpleStruct(x=1.0, y=vec3(2.0, 3.0, 4.0), z=5)
        return vec4(s.y, s.x)

    glsl_code, _ = transpile(shader)
    assert "SimpleStruct s = SimpleStruct(1.0, vec3(2.0, 3.0, 4.0), 5);" in glsl_code
    assert "return vec4(s.y, s.x);" in glsl_code


def test_struct_partial_init_with_defaults():
    @dataclass
    class DefaultStruct:
        a: "float" = 0.0
        b: "vec3" = vec3(1.0, 1.0, 1.0)
        c: "int" = 42

    def shader(vs_uv: "vec2") -> "vec4":
        s = DefaultStruct(b=vec3(2.0, 3.0, 4.0))
        return vec4(s.b, s.a)

    glsl_code, _ = transpile(shader)
    assert "DefaultStruct s = DefaultStruct(0.0, vec3(2.0, 3.0, 4.0), 42);" in glsl_code


def test_uniform_declaration_and_usage():
    def shader(vs_uv: "vec2", u_time: "float", u_scale: "float") -> "vec4":
        pos = vec3(sin(u_time) * u_scale, cos(u_time), 0.0)
        return vec4(pos, 1.0)

    glsl_code, used_uniforms = transpile(shader)
    assert "uniform float u_time;" in glsl_code
    assert "uniform float u_scale;" in glsl_code
    assert "u_time" in used_uniforms and "u_scale" in used_uniforms


def test_unused_uniforms():
    def shader(vs_uv: "vec2", u_unused: "float") -> "vec4":
        return vec4(vs_uv.x, vs_uv.y, 0.0, 1.0)

    glsl_code, used_uniforms = transpile(shader)
    assert "uniform float u_unused;" in glsl_code
    assert "u_unused" in used_uniforms


def test_arithmetic_and_builtins():
    def shader(vs_uv: "vec2") -> "vec4":
        v = vec3(vs_uv.x + 1.0, vs_uv.y * 2.0, sin(vs_uv.x))
        return vec4(normalize(v), 1.0)

    glsl_code, _ = transpile(shader)
    assert "vec3 v = vec3(vs_uv.x + 1.0, vs_uv.y * 2.0, sin(vs_uv.x));" in glsl_code
    assert "return vec4(normalize(v), 1.0);" in glsl_code


def test_loop_with_struct():
    @dataclass
    class LoopStruct:
        count: "int"
        value: "float"

    def shader(vs_uv: "vec2") -> "vec4":
        s = LoopStruct(count=0, value=vs_uv.x)
        for i in range(5):
            s.count = i
            s.value = s.value + 1.0
        return vec4(s.value, 0.0, 0.0, 1.0)

    glsl_code, _ = transpile(shader)
    assert "LoopStruct s = LoopStruct(0, vs_uv.x);" in glsl_code
    assert "for (int i = 0; i < 5; i += 1) {" in glsl_code
    assert "s.count = i;" in glsl_code
    assert "s.value = s.value + 1.0;" in glsl_code


def test_conditional_struct():
    @dataclass
    class CondStruct:
        flag: "bool"
        color: "vec3"

    def shader(vs_uv: "vec2") -> "vec4":
        s = CondStruct(flag=vs_uv.x > 0.5, color=vec3(1.0, 0.0, 0.0))
        if s.flag:
            s.color = vec3(0.0, 1.0, 0.0)
        return vec4(s.color, 1.0)

    glsl_code, _ = transpile(shader)
    assert "CondStruct s = CondStruct(vs_uv.x > 0.5, vec3(1.0, 0.0, 0.0));" in glsl_code
    assert "if (s.flag) {" in glsl_code
    assert "s.color = vec3(0.0, 1.0, 0.0);" in glsl_code


def test_missing_required_fields():
    @dataclass
    class RequiredStruct:
        x: "float"
        y: "vec3"

    def shader(vs_uv: "vec2") -> "vec4":
        s = RequiredStruct(x=1.0)  # Missing y
        return vec4(s.y, 1.0)

    with pytest.raises(
        TranspilerError, match="Wrong number of arguments for struct RequiredStruct"
    ):
        transpile(shader)


def test_invalid_field_name():
    @dataclass
    class ValidStruct:
        a: "int"
        b: "vec3"

    def shader(vs_uv: "vec2") -> "vec4":
        s = ValidStruct(a=1, z=vec3(1.0, 2.0, 3.0))  # 'z' is invalid
        return vec4(s.b, 1.0)

    with pytest.raises(
        TranspilerError, match="Unknown field 'z' in struct 'ValidStruct'"
    ):
        transpile(shader)


def test_nested_structs():
    @dataclass
    class InnerStruct:
        v: "vec2"

    @dataclass
    class OuterStruct:
        inner: "InnerStruct"
        scale: "float"

    def shader(vs_uv: "vec2") -> "vec4":
        inner = InnerStruct(v=vs_uv)
        outer = OuterStruct(inner=inner, scale=2.0)
        return vec4(outer.inner.v.x * outer.scale, outer.inner.v.y, 0.0, 1.0)

    glsl_code, _ = transpile(shader)
    assert "InnerStruct inner = InnerStruct(vs_uv);" in glsl_code
    assert "OuterStruct outer = OuterStruct(inner, 2.0);" in glsl_code
    assert (
        "return vec4(outer.inner.v.x * outer.scale, outer.inner.v.y, 0.0, 1.0);"
        in glsl_code
    )


def test_nan_prevention():
    @dataclass
    class SafeStruct:
        pos: "vec3"
        speed: "float"

    def shader(vs_uv: "vec2") -> "vec4":
        s = SafeStruct(pos=vec3(vs_uv.x, vs_uv.y, 0.0), speed=1.0)
        s.pos = s.pos + vec3(s.speed, 0.0, 0.0)
        return vec4(s.pos, 1.0)

    glsl_code, _ = transpile(shader)
    assert "SafeStruct s = SafeStruct(vec3(vs_uv.x, vs_uv.y, 0.0), 1.0);" in glsl_code
    assert "s.pos = s.pos + vec3(s.speed, 0.0, 0.0);" in glsl_code


def test_loops_and_conditionals():
    shader_code = """
def shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    total: 'float' = 0.0
    for i in range(3):
        if u_time > float(i):
            total = total + 1.0
    return vec4(total / 3.0, 0.0, 0.0, 1.0)
"""
    glsl_code, _ = transpile(shader_code)
    assert "for (int i = 0; i < 3; i += 1) {" in glsl_code
    assert "if (u_time > float(i)) {" in glsl_code
    assert "total = total + 1.0;" in glsl_code
    assert "return vec4(total / 3.0, 0.0, 0.0, 1.0);" in glsl_code


def test_arithmetic_and_builtins():
    shader_code = """
def shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    x: 'float' = sin(u_time) + 2.0 * vs_uv.x
    return vec4(x, 0.0, 0.0, 1.0)
"""
    glsl_code, _ = transpile(shader_code)
    assert "float x = sin(u_time) + 2.0 * vs_uv.x;" in glsl_code
    assert "return vec4(x, 0.0, 0.0, 1.0);" in glsl_code
