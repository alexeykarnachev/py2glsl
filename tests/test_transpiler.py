"""Tests for the GLSL shader transpiler."""

import ast
from dataclasses import dataclass
from typing import Dict

import pytest

from py2glsl.builtins import cos, normalize, sin, vec2, vec3, vec4
from py2glsl.transpiler import (
    BUILTIN_FUNCTIONS,
    OPERATOR_PRECEDENCE,
    CollectedInfo,
    FunctionInfo,
    StructDefinition,
    StructField,
    TranspilerError,
    collect_info,
    generate_annotated_assignment,
    generate_assignment,
    generate_attribute_expr,
    generate_augmented_assignment,
    generate_binary_op_expr,
    generate_body,
    generate_bool_op_expr,
    generate_call_expr,
    generate_compare_expr,
    generate_constant_expr,
    generate_expr,
    generate_for_loop,
    generate_glsl,
    generate_if_expr,
    generate_if_statement,
    generate_name_expr,
    generate_return_statement,
    generate_simple_expr,
    generate_struct_constructor,
    generate_while_loop,
    get_annotation_type,
    get_expr_type,
    parse_shader_code,
    transpile,
)


@pytest.fixture
def collected_info() -> CollectedInfo:
    """Fixture providing basic collected info for testing."""
    return CollectedInfo()


@pytest.fixture
def symbols() -> Dict[str, str]:
    """Fixture providing a sample symbol table."""
    return {
        "hex_coord": "vec2",
        "color": "vec4",
        "uv": "vec2",
        "time": "float",
        "i": "int",
        "test": "Test",
    }


def test_builtin_functions():
    """Test that built-in functions are defined with correct signatures."""
    assert "sin" in BUILTIN_FUNCTIONS
    assert BUILTIN_FUNCTIONS["sin"] == ("float", ["float"])
    assert "vec3" in BUILTIN_FUNCTIONS
    assert BUILTIN_FUNCTIONS["vec3"] == ("vec3", ["float", "float", "float"])


def test_operator_precedence():
    """Test that operator precedence is defined correctly."""
    assert "+" in OPERATOR_PRECEDENCE
    assert "-" in OPERATOR_PRECEDENCE
    assert "*" in OPERATOR_PRECEDENCE
    assert "/" in OPERATOR_PRECEDENCE
    assert "&&" in OPERATOR_PRECEDENCE
    assert "||" in OPERATOR_PRECEDENCE

    # Check relative precedence
    assert (
        OPERATOR_PRECEDENCE["*"] > OPERATOR_PRECEDENCE["+"]
    )  # Multiplication has higher precedence
    assert (
        OPERATOR_PRECEDENCE["&&"] > OPERATOR_PRECEDENCE["||"]
    )  # AND has higher precedence than OR


def test_get_annotation_type():
    """Test getting type from annotations."""
    # Test with Name annotation
    name_annotation = ast.Name(id="vec2", ctx=ast.Load())
    assert get_annotation_type(name_annotation) == "vec2"

    # Test with string annotation
    str_annotation = ast.Constant(value="float", kind=None)
    assert get_annotation_type(str_annotation) == "float"

    # Test with None
    assert get_annotation_type(None) is None

    # Test with unsupported annotation
    list_annotation = ast.List(elts=[], ctx=ast.Load())
    assert get_annotation_type(list_annotation) is None


def test_generate_simple_expr():
    """Test generating simple expressions for globals and defaults."""
    # Test with numeric constant
    num_node = ast.Constant(value=3.14, kind=None)
    assert generate_simple_expr(num_node) == "3.14"

    # Test with bool constant
    bool_node = ast.Constant(value=True, kind=None)
    assert generate_simple_expr(bool_node) == "true"

    # Test with string constant
    str_node = ast.Constant(value="test", kind=None)
    assert generate_simple_expr(str_node) == "test"

    # Test with vector call
    vec_node = ast.Call(
        func=ast.Name(id="vec3", ctx=ast.Load()),
        args=[
            ast.Constant(value=1.0, kind=None),
            ast.Constant(value=2.0, kind=None),
            ast.Constant(value=3.0, kind=None),
        ],
        keywords=[],
    )
    assert generate_simple_expr(vec_node) == "vec3(1.0, 2.0, 3.0)"

    # Test with unsupported node
    with pytest.raises(TranspilerError):
        generate_simple_expr(
            ast.BinOp(
                left=ast.Constant(value=1, kind=None),
                op=ast.Add(),
                right=ast.Constant(value=2, kind=None),
            )
        )


def test_collect_info():
    """Test the collect_info function."""
    code = """
from dataclasses import dataclass

@dataclass
class TestStruct:
    x: 'float'
    y: 'vec2'

PI: 'float' = 3.14159

def helper(pos: 'vec2') -> 'float':
    return pos.x + pos.y

def shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    return vec4(helper(vs_uv), 0.0, 0.0, 1.0)
"""
    tree = ast.parse(code)
    collected = collect_info(tree)

    # Check structs
    assert "TestStruct" in collected.structs
    assert len(collected.structs["TestStruct"].fields) == 2
    assert collected.structs["TestStruct"].fields[0].name == "x"
    assert collected.structs["TestStruct"].fields[0].type_name == "float"

    # Check globals
    assert "PI" in collected.globals
    assert collected.globals["PI"] == ("float", "3.14159")

    # Check functions
    assert "helper" in collected.functions
    assert "shader" in collected.functions
    assert collected.functions["helper"].return_type == "float"
    assert collected.functions["shader"].param_types == ["vec2", "float"]


def test_get_expr_type(symbols: Dict[str, str], collected_info: CollectedInfo):
    """Test determining expression types."""
    # Name expression
    name_node = ast.Name(id="color", ctx=ast.Load())
    assert get_expr_type(name_node, symbols, collected_info) == "vec4"

    # Undefined variable
    with pytest.raises(TranspilerError, match="Undefined variable"):
        get_expr_type(ast.Name(id="undefined", ctx=ast.Load()), symbols, collected_info)

    # Constants
    assert (
        get_expr_type(ast.Constant(value=1.0, kind=None), symbols, collected_info)
        == "float"
    )
    assert (
        get_expr_type(ast.Constant(value=True, kind=None), symbols, collected_info)
        == "bool"
    )

    # Binary operation with same types
    bin_op_node = ast.BinOp(
        left=ast.Name(id="color", ctx=ast.Load()),
        op=ast.Add(),
        right=ast.Name(id="color", ctx=ast.Load()),
    )
    assert get_expr_type(bin_op_node, symbols, collected_info) == "vec4"

    # Binary operation with mixed types (vec * float)
    mixed_op_node = ast.BinOp(
        left=ast.Name(id="uv", ctx=ast.Load()),
        op=ast.Mult(),
        right=ast.Constant(value=2.0, kind=None),
    )
    assert get_expr_type(mixed_op_node, symbols, collected_info) == "vec2"

    # Binary operation returning float
    float_op_node = ast.BinOp(
        left=ast.Constant(value=1.0, kind=None),
        op=ast.Add(),
        right=ast.Constant(value=2, kind=None),
    )
    assert get_expr_type(float_op_node, symbols, collected_info) == "float"

    # Built-in function call
    collected_info.functions["sin"] = FunctionInfo(
        name="sin",
        return_type="float",
        param_types=["float"],
        node=ast.FunctionDef(name="sin", args=ast.arguments(args=[]), body=[]),
    )
    call_node = ast.Call(
        func=ast.Name(id="sin", ctx=ast.Load()),
        args=[ast.Name(id="time", ctx=ast.Load())],
        keywords=[],
    )
    assert get_expr_type(call_node, symbols, collected_info) == "float"

    # Comparison expression
    comp_node = ast.Compare(
        left=ast.Name(id="time", ctx=ast.Load()),
        ops=[ast.Gt()],
        comparators=[ast.Constant(value=1.0, kind=None)],
    )
    assert get_expr_type(comp_node, symbols, collected_info) == "bool"

    # Boolean operation
    bool_op_node = ast.BoolOp(
        op=ast.And(),
        values=[ast.Name(id="i", ctx=ast.Load()), ast.Constant(value=True, kind=None)],
    )
    assert get_expr_type(bool_op_node, symbols, collected_info) == "bool"

    # Conditional expression
    if_expr_node = ast.IfExp(
        test=ast.Constant(value=True, kind=None),
        body=ast.Name(id="color", ctx=ast.Load()),
        orelse=ast.Name(id="color", ctx=ast.Load()),
    )
    assert get_expr_type(if_expr_node, symbols, collected_info) == "vec4"

    # Conditional expression with type mismatch
    type_mismatch_node = ast.IfExp(
        test=ast.Constant(value=True, kind=None),
        body=ast.Name(id="color", ctx=ast.Load()),
        orelse=ast.Name(id="uv", ctx=ast.Load()),
    )
    with pytest.raises(TranspilerError, match="Ternary expression types mismatch"):
        get_expr_type(type_mismatch_node, symbols, collected_info)


def test_generate_name_expr(symbols: Dict[str, str]):
    """Test generating code for a name expression."""
    node = ast.Name(id="uv", ctx=ast.Load())
    result = generate_name_expr(node, symbols)
    assert result == "uv"


def test_generate_constant_expr():
    """Test generating code for constant expressions."""
    # Integer
    node_int = ast.Constant(value=42, kind=None)
    assert generate_constant_expr(node_int) == "42"

    # Float
    node_float = ast.Constant(value=3.14, kind=None)
    assert generate_constant_expr(node_float) == "3.14"

    # Boolean
    node_true = ast.Constant(value=True, kind=None)
    node_false = ast.Constant(value=False, kind=None)
    assert generate_constant_expr(node_true) == "true"
    assert generate_constant_expr(node_false) == "false"

    # String should raise error (not supported in GLSL)
    node_str = ast.Constant(value="test", kind=None)
    with pytest.raises(TranspilerError):
        generate_constant_expr(node_str)


def test_generate_binary_op_expr(
    symbols: Dict[str, str], collected_info: CollectedInfo
):
    """Test generating code for binary operations."""
    # Test addition
    add_node = ast.BinOp(
        left=ast.Name(id="time", ctx=ast.Load()),
        op=ast.Add(),
        right=ast.Constant(value=1.0, kind=None),
    )
    assert generate_binary_op_expr(add_node, symbols, 0, collected_info) == "time + 1.0"

    # Test multiplication
    mul_node = ast.BinOp(
        left=ast.Name(id="uv", ctx=ast.Load()),
        op=ast.Mult(),
        right=ast.Constant(value=2.0, kind=None),
    )
    assert generate_binary_op_expr(mul_node, symbols, 0, collected_info) == "uv * 2.0"

    # Test precedence (multiplication inside addition)
    nested_node = ast.BinOp(
        left=ast.Name(id="time", ctx=ast.Load()),
        op=ast.Add(),
        right=ast.BinOp(
            left=ast.Name(id="i", ctx=ast.Load()),
            op=ast.Mult(),
            right=ast.Constant(value=2.0, kind=None),
        ),
    )
    # Multiplication should not need parentheses when nested in addition
    assert (
        generate_binary_op_expr(nested_node, symbols, 0, collected_info)
        == "time + i * 2.0"
    )

    # Test unsupported operation
    mod_node = ast.BinOp(
        left=ast.Name(id="i", ctx=ast.Load()),
        op=ast.Mod(),
        right=ast.Constant(value=10, kind=None),
    )
    with pytest.raises(TranspilerError, match="Unsupported binary op: Mod"):
        generate_binary_op_expr(mod_node, symbols, 0, collected_info)


def test_generate_compare_expr(symbols: Dict[str, str], collected_info: CollectedInfo):
    """Test generating code for comparison expressions."""
    # Simple comparison
    gt_node = ast.Compare(
        left=ast.Name(id="time", ctx=ast.Load()),
        ops=[ast.Gt()],
        comparators=[ast.Constant(value=1.0, kind=None)],
    )
    assert generate_compare_expr(gt_node, symbols, 0, collected_info) == "time > 1.0"

    # Test with precedence
    eq_node = ast.Compare(
        left=ast.Name(id="i", ctx=ast.Load()),
        ops=[ast.Eq()],
        comparators=[ast.Constant(value=0, kind=None)],
    )
    assert generate_compare_expr(eq_node, symbols, 5, collected_info) == "(i == 0)"

    # Unsupported comparison
    with pytest.raises(TranspilerError, match="Multiple comparisons not supported"):
        generate_compare_expr(
            ast.Compare(
                left=ast.Name(id="i", ctx=ast.Load()),
                ops=[ast.Lt(), ast.Lt()],
                comparators=[
                    ast.Constant(value=5, kind=None),
                    ast.Constant(value=10, kind=None),
                ],
            ),
            symbols,
            0,
            collected_info,
        )


def test_generate_bool_op_expr(symbols: Dict[str, str], collected_info: CollectedInfo):
    """Test generating code for boolean operations."""
    # AND operation
    and_node = ast.BoolOp(
        op=ast.And(),
        values=[ast.Name(id="i", ctx=ast.Load()), ast.Constant(value=True, kind=None)],
    )
    assert generate_bool_op_expr(and_node, symbols, 0, collected_info) == "i && true"

    # OR operation
    or_node = ast.BoolOp(
        op=ast.Or(),
        values=[
            ast.Compare(
                left=ast.Name(id="time", ctx=ast.Load()),
                ops=[ast.Gt()],
                comparators=[ast.Constant(value=1.0, kind=None)],
            ),
            ast.Compare(
                left=ast.Name(id="i", ctx=ast.Load()),
                ops=[ast.Lt()],
                comparators=[ast.Constant(value=0, kind=None)],
            ),
        ],
    )
    assert (
        generate_bool_op_expr(or_node, symbols, 0, collected_info)
        == "time > 1.0 || i < 0"
    )

    # Test with precedence
    assert generate_bool_op_expr(and_node, symbols, 1, collected_info) == "(i && true)"

    # Unsupported operator
    with pytest.raises(TranspilerError, match="Unsupported boolean op"):
        generate_bool_op_expr(
            ast.BoolOp(
                op=ast.operator(),  # This is not a real operator
                values=[
                    ast.Constant(value=True, kind=None),
                    ast.Constant(value=False, kind=None),
                ],
            ),
            symbols,
            0,
            collected_info,
        )


def test_generate_attribute_expr(
    symbols: Dict[str, str], collected_info: CollectedInfo
):
    """Test generating code for attribute access."""
    # Simple attribute
    node = ast.Attribute(
        value=ast.Name(id="color", ctx=ast.Load()), attr="r", ctx=ast.Load()
    )
    assert generate_attribute_expr(node, symbols, 0, collected_info) == "color.r"

    # Nested attribute (struct field)
    collected_info.structs["Test"] = StructDefinition(
        name="Test", fields=[StructField(name="position", type_name="vec3")]
    )
    nested_node = ast.Attribute(
        value=ast.Attribute(
            value=ast.Name(id="test", ctx=ast.Load()), attr="position", ctx=ast.Load()
        ),
        attr="y",
        ctx=ast.Load(),
    )
    assert (
        generate_attribute_expr(nested_node, symbols, 0, collected_info)
        == "test.position.y"
    )


def test_generate_if_expr(symbols: Dict[str, str], collected_info: CollectedInfo):
    """Test generating code for conditional expressions."""
    # Simple conditional
    if_node = ast.IfExp(
        test=ast.Compare(
            left=ast.Name(id="time", ctx=ast.Load()),
            ops=[ast.Gt()],
            comparators=[ast.Constant(value=1.0, kind=None)],
        ),
        body=ast.Name(id="color", ctx=ast.Load()),
        orelse=ast.Call(
            func=ast.Name(id="vec4", ctx=ast.Load()),
            args=[
                ast.Constant(value=0.0, kind=None),
                ast.Constant(value=0.0, kind=None),
                ast.Constant(value=0.0, kind=None),
                ast.Constant(value=1.0, kind=None),
            ],
            keywords=[],
        ),
    )
    assert (
        generate_if_expr(if_node, symbols, 0, collected_info)
        == "time > 1.0 ? color : vec4(0.0, 0.0, 0.0, 1.0)"
    )

    # With precedence
    assert (
        generate_if_expr(if_node, symbols, 5, collected_info)
        == "(time > 1.0 ? color : vec4(0.0, 0.0, 0.0, 1.0))"
    )


def test_generate_call_expr(symbols: Dict[str, str], collected_info: CollectedInfo):
    """Test generating code for function calls."""
    # Simple function call
    func_node = ast.Call(
        func=ast.Name(id="vec3", ctx=ast.Load()),
        args=[
            ast.Constant(value=1.0, kind=None),
            ast.Constant(value=2.0, kind=None),
            ast.Constant(value=3.0, kind=None),
        ],
        keywords=[],
    )
    assert (
        generate_call_expr(func_node, symbols, collected_info) == "vec3(1.0, 2.0, 3.0)"
    )

    # Function from collected info
    collected_info.functions["helper"] = FunctionInfo(
        name="helper",
        return_type="float",
        param_types=["vec2"],
        node=ast.FunctionDef(name="helper", args=ast.arguments(args=[]), body=[]),
    )
    helper_call = ast.Call(
        func=ast.Name(id="helper", ctx=ast.Load()),
        args=[ast.Name(id="uv", ctx=ast.Load())],
        keywords=[],
    )
    assert generate_call_expr(helper_call, symbols, collected_info) == "helper(uv)"

    # Unknown function
    unknown_call = ast.Call(
        func=ast.Name(id="unknown", ctx=ast.Load()), args=[], keywords=[]
    )
    with pytest.raises(TranspilerError, match="Unknown function call: unknown"):
        generate_call_expr(unknown_call, symbols, collected_info)


def test_generate_struct_constructor(
    symbols: Dict[str, str], collected_info: CollectedInfo
):
    """Test generating code for struct constructors."""
    # Define a struct
    collected_info.structs["Material"] = StructDefinition(
        name="Material",
        fields=[
            StructField(name="color", type_name="vec3"),
            StructField(name="shininess", type_name="float"),
        ],
    )

    # Positional arguments
    pos_args_call = ast.Call(
        func=ast.Name(id="Material", ctx=ast.Load()),
        args=[
            ast.Call(
                func=ast.Name(id="vec3", ctx=ast.Load()),
                args=[
                    ast.Constant(value=1.0, kind=None),
                    ast.Constant(value=0.0, kind=None),
                    ast.Constant(value=0.0, kind=None),
                ],
                keywords=[],
            ),
            ast.Constant(value=32.0, kind=None),
        ],
        keywords=[],
    )
    assert (
        generate_struct_constructor("Material", pos_args_call, symbols, collected_info)
        == "Material(vec3(1.0, 0.0, 0.0), 32.0)"
    )

    # Keyword arguments
    kw_args_call = ast.Call(
        func=ast.Name(id="Material", ctx=ast.Load()),
        args=[],
        keywords=[
            ast.keyword(
                arg="color",
                value=ast.Call(
                    func=ast.Name(id="vec3", ctx=ast.Load()),
                    args=[
                        ast.Constant(value=0.0, kind=None),
                        ast.Constant(value=1.0, kind=None),
                        ast.Constant(value=0.0, kind=None),
                    ],
                    keywords=[],
                ),
            ),
            ast.keyword(arg="shininess", value=ast.Constant(value=64.0, kind=None)),
        ],
    )
    assert (
        generate_struct_constructor("Material", kw_args_call, symbols, collected_info)
        == "Material(vec3(0.0, 1.0, 0.0), 64.0)"
    )

    # Wrong number of arguments
    with pytest.raises(TranspilerError, match="Wrong number of arguments for struct"):
        generate_struct_constructor(
            "Material",
            ast.Call(
                func=ast.Name(id="Material", ctx=ast.Load()),
                args=[ast.Constant(value=1.0, kind=None)],
                keywords=[],
            ),
            symbols,
            collected_info,
        )

    # Unknown field
    with pytest.raises(TranspilerError, match="Unknown field"):
        generate_struct_constructor(
            "Material",
            ast.Call(
                func=ast.Name(id="Material", ctx=ast.Load()),
                args=[],
                keywords=[
                    ast.keyword(arg="unknown", value=ast.Constant(value=1.0, kind=None))
                ],
            ),
            symbols,
            collected_info,
        )

    # No arguments
    with pytest.raises(TranspilerError, match="initialization requires arguments"):
        generate_struct_constructor(
            "Material",
            ast.Call(
                func=ast.Name(id="Material", ctx=ast.Load()), args=[], keywords=[]
            ),
            symbols,
            collected_info,
        )


def test_generate_expr(symbols: Dict[str, str], collected_info: CollectedInfo):
    """Test the combined expression generator."""
    # Simple name
    name_node = ast.Name(id="color", ctx=ast.Load())
    assert generate_expr(name_node, symbols, 0, collected_info) == "color"

    # Constant
    constant_node = ast.Constant(value=1.0, kind=None)
    assert generate_expr(constant_node, symbols, 0, collected_info) == "1.0"

    # Binary operation
    bin_op_node = ast.BinOp(
        left=ast.Name(id="time", ctx=ast.Load()),
        op=ast.Add(),
        right=ast.Constant(value=1.0, kind=None),
    )
    assert generate_expr(bin_op_node, symbols, 0, collected_info) == "time + 1.0"

    # Unsupported expression
    with pytest.raises(TranspilerError, match="Unsupported expression"):
        generate_expr(ast.List(elts=[], ctx=ast.Load()), symbols, 0, collected_info)


def test_generate_assignment(symbols: Dict[str, str], collected_info: CollectedInfo):
    """Test generating code for assignments."""
    # Simple assignment with new variable
    assign_node = ast.Assign(
        targets=[ast.Name(id="x", ctx=ast.Store())],
        value=ast.Constant(value=1.0, kind=None),
    )
    assert (
        generate_assignment(assign_node, symbols, "    ", collected_info)
        == "    float x = 1.0;"
    )
    assert "x" in symbols
    assert symbols["x"] == "float"

    # Assignment to existing variable
    assign_existing = ast.Assign(
        targets=[ast.Name(id="x", ctx=ast.Store())],
        value=ast.Constant(value=2.0, kind=None),
    )
    assert (
        generate_assignment(assign_existing, symbols, "    ", collected_info)
        == "    x = 2.0;"
    )

    # Assignment to attribute
    assign_attr = ast.Assign(
        targets=[
            ast.Attribute(
                value=ast.Name(id="color", ctx=ast.Load()), attr="r", ctx=ast.Store()
            )
        ],
        value=ast.Constant(value=1.0, kind=None),
    )
    assert (
        generate_assignment(assign_attr, symbols, "    ", collected_info)
        == "    color.r = 1.0;"
    )

    # Unsupported target
    with pytest.raises(TranspilerError, match="Unsupported assignment target"):
        generate_assignment(
            ast.Assign(
                targets=[ast.List(elts=[], ctx=ast.Store())],
                value=ast.Constant(value=1.0, kind=None),
            ),
            symbols,
            "    ",
            collected_info,
        )


def test_generate_annotated_assignment(symbols: Dict[str, str]):
    """Test generating code for annotated assignments."""
    # Annotated assignment with initialization
    ann_assign = ast.AnnAssign(
        target=ast.Name(id="pos", ctx=ast.Store()),
        annotation=ast.Constant(value="vec2", kind=None),
        value=ast.Name(id="uv", ctx=ast.Load()),
        simple=1,
    )
    assert (
        generate_annotated_assignment(ann_assign, symbols, "    ")
        == "    vec2 pos = uv;"
    )

    # Annotated assignment without initialization
    ann_assign_no_val = ast.AnnAssign(
        target=ast.Name(id="result", ctx=ast.Store()),
        annotation=ast.Constant(value="float", kind=None),
        value=None,
        simple=1,
    )
    assert (
        generate_annotated_assignment(ann_assign_no_val, symbols, "    ")
        == "    float result;"
    )

    # Unsupported target
    with pytest.raises(
        TranspilerError, match="Unsupported annotated assignment target"
    ):
        generate_annotated_assignment(
            ast.AnnAssign(
                target=ast.Attribute(
                    value=ast.Name(id="x", ctx=ast.Load()), attr="y", ctx=ast.Store()
                ),
                annotation=ast.Constant(value="float", kind=None),
                value=None,
                simple=0,
            ),
            symbols,
            "    ",
        )


def test_generate_augmented_assignment(
    symbols: Dict[str, str], collected_info: CollectedInfo
):
    """Test generating code for augmented assignments."""
    # Addition assignment
    add_assign = ast.AugAssign(
        target=ast.Name(id="i", ctx=ast.Store()),
        op=ast.Add(),
        value=ast.Constant(value=1, kind=None),
    )
    assert (
        generate_augmented_assignment(add_assign, symbols, "    ", collected_info)
        == "    i = i + 1;"
    )

    # Multiplication assignment
    mult_assign = ast.AugAssign(
        target=ast.Name(id="uv", ctx=ast.Store()),
        op=ast.Mult(),
        value=ast.Constant(value=2.0, kind=None),
    )
    assert (
        generate_augmented_assignment(mult_assign, symbols, "    ", collected_info)
        == "    uv = uv * 2.0;"
    )

    # Unsupported operator
    with pytest.raises(TranspilerError, match="Unsupported augmented operator"):
        generate_augmented_assignment(
            ast.AugAssign(
                target=ast.Name(id="i", ctx=ast.Store()),
                op=ast.Mod(),
                value=ast.Constant(value=10, kind=None),
            ),
            symbols,
            "    ",
            collected_info,
        )


def test_generate_for_loop(symbols: Dict[str, str], collected_info: CollectedInfo):
    """Test generating code for for loops."""
    # Simple range loop
    for_node = ast.For(
        target=ast.Name(id="i", ctx=ast.Store()),
        iter=ast.Call(
            func=ast.Name(id="range", ctx=ast.Load()),
            args=[ast.Constant(value=10, kind=None)],
            keywords=[],
        ),
        body=[
            ast.Assign(
                targets=[ast.Name(id="x", ctx=ast.Store())],
                value=ast.BinOp(
                    left=ast.Name(id="x", ctx=ast.Load()),
                    op=ast.Add(),
                    right=ast.Constant(value=1.0, kind=None),
                ),
            )
        ],
        orelse=[],
    )
    symbols["x"] = "float"  # Define x for the test
    result = generate_for_loop(for_node, symbols, "    ", collected_info)
    assert len(result) == 3
    assert result[0] == "    for (int i = 0; i < 10; i += 1) {"
    assert result[1] == "        x = x + 1.0;"
    assert result[2] == "    }"

    # Range with start and end
    range_start_end = ast.For(
        target=ast.Name(id="j", ctx=ast.Store()),
        iter=ast.Call(
            func=ast.Name(id="range", ctx=ast.Load()),
            args=[ast.Constant(value=5, kind=None), ast.Constant(value=15, kind=None)],
            keywords=[],
        ),
        body=[ast.Expr(value=ast.Constant(value=0, kind=None))],
        orelse=[],
    )
    result = generate_for_loop(range_start_end, symbols, "    ", collected_info)
    assert result[0] == "    for (int j = 5; j < 15; j += 1) {"

    # Range with start, end, and step
    range_full = ast.For(
        target=ast.Name(id="k", ctx=ast.Store()),
        iter=ast.Call(
            func=ast.Name(id="range", ctx=ast.Load()),
            args=[
                ast.Constant(value=0, kind=None),
                ast.Constant(value=20, kind=None),
                ast.Constant(value=2, kind=None),
            ],
            keywords=[],
        ),
        body=[ast.Pass()],
        orelse=[],
    )
    result = generate_for_loop(range_full, symbols, "    ", collected_info)
    assert result[0] == "    for (int k = 0; k < 20; k += 2) {"

    # Non-range loop (unsupported)
    with pytest.raises(
        TranspilerError, match="Only range-based for loops are supported"
    ):
        generate_for_loop(
            ast.For(
                target=ast.Name(id="x", ctx=ast.Store()),
                iter=ast.Name(id="items", ctx=ast.Load()),
                body=[ast.Pass()],
                orelse=[],
            ),
            symbols,
            "    ",
            collected_info,
        )


def test_generate_while_loop(symbols: Dict[str, str], collected_info: CollectedInfo):
    """Test generating code for while loops."""
    # Simple while loop
    while_node = ast.While(
        test=ast.Compare(
            left=ast.Name(id="i", ctx=ast.Load()),
            ops=[ast.Lt()],
            comparators=[ast.Constant(value=10, kind=None)],
        ),
        body=[
            ast.AugAssign(
                target=ast.Name(id="i", ctx=ast.Store()),
                op=ast.Add(),
                value=ast.Constant(value=1, kind=None),
            )
        ],
        orelse=[],
    )
    result = generate_while_loop(while_node, symbols, "    ", collected_info)
    assert len(result) == 3
    assert result[0] == "    while (i < 10) {"
    assert result[1] == "        i = i + 1;"
    assert result[2] == "    }"


def test_generate_if_statement(symbols: Dict[str, str], collected_info: CollectedInfo):
    """Test generating code for if statements."""
    # If without else
    if_node = ast.If(
        test=ast.Compare(
            left=ast.Name(id="time", ctx=ast.Load()),
            ops=[ast.Gt()],
            comparators=[ast.Constant(value=1.0, kind=None)],
        ),
        body=[
            ast.Assign(
                targets=[ast.Name(id="color", ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id="vec4", ctx=ast.Load()),
                    args=[
                        ast.Constant(value=1.0, kind=None),
                        ast.Constant(value=0.0, kind=None),
                        ast.Constant(value=0.0, kind=None),
                        ast.Constant(value=1.0, kind=None),
                    ],
                    keywords=[],
                ),
            )
        ],
        orelse=[],
    )
    result = generate_if_statement(if_node, symbols, "    ", collected_info)
    assert len(result) == 3
    assert result[0] == "    if (time > 1.0) {"
    assert result[1] == "        color = vec4(1.0, 0.0, 0.0, 1.0);"
    assert result[2] == "    }"

    # If with else
    if_else_node = ast.If(
        test=ast.Compare(
            left=ast.Name(id="time", ctx=ast.Load()),
            ops=[ast.Gt()],
            comparators=[ast.Constant(value=1.0, kind=None)],
        ),
        body=[
            ast.Assign(
                targets=[ast.Name(id="color", ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id="vec4", ctx=ast.Load()),
                    args=[
                        ast.Constant(value=1.0, kind=None),
                        ast.Constant(value=0.0, kind=None),
                        ast.Constant(value=0.0, kind=None),
                        ast.Constant(value=1.0, kind=None),
                    ],
                    keywords=[],
                ),
            )
        ],
        orelse=[
            ast.Assign(
                targets=[ast.Name(id="color", ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id="vec4", ctx=ast.Load()),
                    args=[
                        ast.Constant(value=0.0, kind=None),
                        ast.Constant(value=1.0, kind=None),
                        ast.Constant(value=0.0, kind=None),
                        ast.Constant(value=1.0, kind=None),
                    ],
                    keywords=[],
                ),
            )
        ],
    )
    result = generate_if_statement(if_else_node, symbols, "    ", collected_info)
    assert len(result) == 5
    assert result[0] == "    if (time > 1.0) {"
    assert result[1] == "        color = vec4(1.0, 0.0, 0.0, 1.0);"
    assert result[2] == "    } else {"
    assert result[3] == "        color = vec4(0.0, 1.0, 0.0, 1.0);"
    assert result[4] == "    }"


def test_generate_return_statement(
    symbols: Dict[str, str], collected_info: CollectedInfo
):
    """Test generating code for return statements."""
    return_node = ast.Return(
        value=ast.Call(
            func=ast.Name(id="vec4", ctx=ast.Load()),
            args=[
                ast.Name(id="uv", ctx=ast.Load()),
                ast.Constant(value=0.0, kind=None),
                ast.Constant(value=1.0, kind=None),
            ],
            keywords=[],
        )
    )
    result = generate_return_statement(return_node, symbols, "    ", collected_info)
    assert result == "    return vec4(uv, 0.0, 1.0);"


def test_generate_body(symbols: Dict[str, str], collected_info: CollectedInfo):
    """Test generating code for function bodies."""
    body = [
        # Variable declaration
        ast.AnnAssign(
            target=ast.Name(id="result", ctx=ast.Store()),
            annotation=ast.Constant(value="float", kind=None),
            value=ast.Constant(value=0.0, kind=None),
            simple=1,
        ),
        # For loop
        ast.For(
            target=ast.Name(id="i", ctx=ast.Store()),
            iter=ast.Call(
                func=ast.Name(id="range", ctx=ast.Load()),
                args=[ast.Constant(value=5, kind=None)],
                keywords=[],
            ),
            body=[
                ast.AugAssign(
                    target=ast.Name(id="result", ctx=ast.Store()),
                    op=ast.Add(),
                    value=ast.Constant(value=1.0, kind=None),
                )
            ],
            orelse=[],
        ),
        # Return statement
        ast.Return(value=ast.Name(id="result", ctx=ast.Load())),
    ]

    result = generate_body(body, symbols.copy(), collected_info)
    assert "float result = 0.0;" in result
    assert "for (int i = 0; i < 5; i += 1) {" in result
    assert "result = result + 1.0;" in result
    assert "return result;" in result


def test_generate_glsl(collected_info: CollectedInfo):
    """Test GLSL code generation from collected info."""
    # Create a simple shader function
    shader_node = ast.FunctionDef(
        name="simple_shader",
        args=ast.arguments(
            posonlyargs=[],
            args=[
                ast.arg(arg="vs_uv", annotation=ast.Constant(value="vec2", kind=None)),
                ast.arg(
                    arg="u_time", annotation=ast.Constant(value="float", kind=None)
                ),
            ],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[
            ast.Return(
                value=ast.Call(
                    func=ast.Name(id="vec4", ctx=ast.Load()),
                    args=[
                        ast.Constant(value=1.0, kind=None),
                        ast.Constant(value=0.0, kind=None),
                        ast.Constant(value=0.0, kind=None),
                        ast.Constant(value=1.0, kind=None),
                    ],
                    keywords=[],
                )
            )
        ],
        decorator_list=[],
        returns=ast.Constant(value="vec4", kind=None),
    )

    # Add function to collected info
    collected_info.functions["simple_shader"] = FunctionInfo(
        name="simple_shader",
        return_type="vec4",
        param_types=["vec2", "float"],
        node=shader_node,
    )

    # Add a global constant
    collected_info.globals["PI"] = ("float", "3.14159")

    # Generate GLSL code
    glsl_code, uniforms = generate_glsl(collected_info, "simple_shader")

    # Check output
    assert "#version 460 core" in glsl_code
    assert "uniform float u_time;" in glsl_code
    assert "const float PI = 3.14159;" in glsl_code
    assert "vec4 simple_shader(vec2 vs_uv, float u_time)" in glsl_code
    assert "return vec4(1.0, 0.0, 0.0, 1.0);" in glsl_code
    assert "void main() {" in glsl_code
    assert "fragColor = simple_shader(vs_uv, u_time);" in glsl_code
    assert uniforms == {"u_time"}


def test_parse_shader_code():
    """Test parsing shader code."""
    # Parse string
    shader_code = """
def shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    return vec4(1.0, 0.0, 0.0, 1.0)
"""
    tree, main_func = parse_shader_code(shader_code)
    assert isinstance(tree, ast.Module)
    assert main_func == "shader"

    # Parse with custom main function
    tree, main_func = parse_shader_code(shader_code, main_func="custom_main")
    assert main_func == "custom_main"

    # Empty code
    with pytest.raises(TranspilerError, match="Empty shader code provided"):
        parse_shader_code("")


def test_struct_definition():
    """Test struct definition representation."""
    struct_def = StructDefinition(
        name="Material",
        fields=[
            StructField(name="color", type_name="vec3"),
            StructField(name="shininess", type_name="float", default_value="1.0"),
        ],
    )

    assert struct_def.name == "Material"
    assert len(struct_def.fields) == 2
    assert struct_def.fields[0].name == "color"
    assert struct_def.fields[0].type_name == "vec3"
    assert struct_def.fields[0].default_value is None
    assert struct_def.fields[1].name == "shininess"
    assert struct_def.fields[1].type_name == "float"
    assert struct_def.fields[1].default_value == "1.0"


def test_generate_vec2_attribute(
    symbols: Dict[str, str], collected_info: CollectedInfo
) -> None:
    """Test generating code for accessing vec2 attributes."""
    node = ast.parse("hex_coord.x", mode="eval").body
    code = generate_expr(node, symbols, 0, collected_info)
    expr_type = get_expr_type(node, symbols, collected_info)
    assert code == "hex_coord.x"
    assert expr_type == "float"


def test_main_shader_no_return_type() -> None:
    """Test transpiling a shader function without explicit return type."""
    shader_code = """
def shader(vs_uv: 'vec2', u_time: 'float'):
    return vec4(1.0, 0.0, 0.0, 1.0)
"""
    glsl_code, uniforms = transpile(shader_code)
    assert "uniform float u_time;" in glsl_code
    assert "vec4 shader(vec2 vs_uv, float u_time) {" in glsl_code
    assert "return vec4(1.0, 0.0, 0.0, 1.0);" in glsl_code
    assert "void main() {" in glsl_code
    assert "fragColor = shader(vs_uv, u_time);" in glsl_code  # Matches definition order
    assert uniforms == {"u_time"}


def test_helper_function_requires_return_type() -> None:
    """Test that helper functions require explicit return type annotations."""
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


def test_struct_and_uniforms() -> None:
    """Test transpiling code with structs and uniform variables."""
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


def test_multi_uniform_struct() -> None:
    """Test handling multiple uniform structs in a shader."""

    @dataclass
    class UniStruct:
        offset: "vec3"
        active: "bool"

    def shader(vs_uv: "vec2", u_time: "float", u_offset: "vec3") -> "vec4":
        s = UniStruct(offset=u_offset, active=(sin(u_time) > 0.0))
        return vec4(s.offset, 1.0) if s.active else vec4(0.0, 0.0, 0.0, 1.0)

    glsl_code, used_uniforms = transpile(UniStruct, shader)
    assert "struct UniStruct {" in glsl_code
    assert "uniform float u_time;" in glsl_code
    assert "uniform vec3 u_offset;" in glsl_code
    assert "UniStruct s = UniStruct(u_offset, sin(u_time) > 0.0);" in glsl_code
    assert (
        "return (s.active ? vec4(s.offset, 1.0) : vec4(0.0, 0.0, 0.0, 1.0));"
        in glsl_code
    )
    assert used_uniforms == {"u_time", "u_offset"}


def test_generate_vec4_rgb(
    symbols: Dict[str, str], collected_info: CollectedInfo
) -> None:
    """Test generating code for accessing vec4 RGB components."""
    node = ast.parse("color.rgb", mode="eval").body
    code = generate_expr(node, symbols, 0, collected_info)
    expr_type = get_expr_type(node, symbols, collected_info)
    assert code == "color.rgb"
    assert expr_type == "vec3"


def test_binop_vec2_float(
    symbols: Dict[str, str], collected_info: CollectedInfo
) -> None:
    """Test binary operations between vec2 and float."""
    node = ast.parse("uv * 2.0", mode="eval").body
    code = generate_expr(node, symbols, 0, collected_info)
    expr_type = get_expr_type(node, symbols, collected_info)
    assert code == "uv * 2.0"
    assert expr_type == "vec2"


def test_invalid_attribute_raises_error(
    symbols: Dict[str, str], collected_info: CollectedInfo
) -> None:
    """Test that accessing invalid attributes raises errors."""
    node = ast.parse("hex_coord.z", mode="eval").body
    with pytest.raises(TranspilerError, match="Invalid swizzle 'z' for vec2"):
        get_expr_type(node, symbols, collected_info)


def test_version_directive_first_line() -> None:
    """Test that the GLSL version directive appears on the first line."""
    shader_code = """
def shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    return vec4(1.0, 0.0, 0.0, 1.0)
"""
    glsl_code, _ = transpile(shader_code)
    lines = glsl_code.splitlines()
    assert (
        lines[0] == "#version 460 core"
    ), "The #version directive must be the first line"


def test_shader_compilation() -> None:
    """Test basic shader compilation."""
    shader_code = """
def shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    return vec4(sin(u_time), 0.0, 0.0, 1.0)
"""
    glsl_code, _ = transpile(shader_code)
    assert "shader" in glsl_code
    assert "void main()" in glsl_code
    assert "fragColor" in glsl_code


def test_vec4_rgba_swizzle(
    symbols: Dict[str, str], collected_info: CollectedInfo
) -> None:
    """Test vec4 RGBA swizzling."""
    node = ast.parse("color.rgba", mode="eval").body
    code = generate_expr(node, symbols, 0, collected_info)
    expr_type = get_expr_type(node, symbols, collected_info)
    assert code == "color.rgba"
    assert expr_type == "vec4"


def test_vec4_xy_swizzle(
    symbols: Dict[str, str], collected_info: CollectedInfo
) -> None:
    """Test vec4 XY swizzling."""
    node = ast.parse("color.xy", mode="eval").body
    code = generate_expr(node, symbols, 0, collected_info)
    expr_type = get_expr_type(node, symbols, collected_info)
    assert code == "color.xy"
    assert expr_type == "vec2"


def test_binop_vec4_vec4_addition(
    symbols: Dict[str, str], collected_info: CollectedInfo
) -> None:
    """Test addition operation between vec4 values."""
    node = ast.parse("color + color", mode="eval").body
    code = generate_expr(node, symbols, 0, collected_info)
    expr_type = get_expr_type(node, symbols, collected_info)
    assert code == "color + color"
    assert expr_type == "vec4"


def test_function_call_with_args(
    symbols: Dict[str, str], collected_info: CollectedInfo
) -> None:
    """Test function calls with arguments."""
    dummy_node = ast.FunctionDef(
        name="wave", args=ast.arguments(args=[]), body=[ast.Pass()]
    )
    collected_info.functions["wave"] = FunctionInfo(
        name="wave", return_type="float", param_types=["vec2", "float"], node=dummy_node
    )
    node = ast.parse("wave(uv, time)", mode="eval").body
    code = generate_expr(node, symbols, 0, collected_info)
    assert code == "wave(uv, time)"
    assert get_expr_type(node, symbols, collected_info) == "float"


def test_nested_function_call(
    symbols: Dict[str, str], collected_info: CollectedInfo
) -> None:
    """Test nested function calls."""
    dummy_sin_node = ast.FunctionDef(name="sin", args=ast.arguments(args=[]), body=[])
    dummy_length_node = ast.FunctionDef(
        name="length", args=ast.arguments(args=[]), body=[]
    )

    collected_info.functions["sin"] = FunctionInfo(
        name="sin", return_type="float", param_types=["float"], node=dummy_sin_node
    )
    collected_info.functions["length"] = FunctionInfo(
        name="length", return_type="float", param_types=["vec2"], node=dummy_length_node
    )

    node = ast.parse("sin(length(uv))", mode="eval").body
    code = generate_expr(node, symbols, 0, collected_info)
    expr_type = get_expr_type(node, symbols, collected_info)
    assert code == "sin(length(uv))"
    assert expr_type == "float"


def test_missing_main_shader_raises_error() -> None:
    """Test that missing main function raises an error."""
    shader_code = """
def helper(uv: 'vec2') -> 'float':
    return sin(uv.x)
"""
    with pytest.raises(TranspilerError, match="Main function 'shader' not found"):
        transpile(shader_code)


def test_multiple_helper_functions() -> None:
    """Test using multiple helper functions in a shader."""
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


def test_uniform_declarations() -> None:
    """Test uniform variable declarations."""
    shader_code = """
def shader(vs_uv: 'vec2', u_time: 'float', u_scale: 'float') -> 'vec4':
    return vec4(vs_uv * u_scale, 0.0, 1.0)
"""
    glsl_code, used_uniforms = transpile(shader_code)
    assert "uniform float u_time;" in glsl_code
    assert "uniform float u_scale;" in glsl_code
    assert used_uniforms == {"u_time", "u_scale"}


def test_no_return_type_raises_error() -> None:
    """Test that helper functions without return types raise errors."""
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


def test_complex_expression_in_shader() -> None:
    """Test handling complex expressions in shaders."""
    shader_code = """
def shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    uv = vs_uv * 2.0
    color = vec4(sin(uv.x + u_time), cos(uv.y - u_time), 0.5, 1.0)
    return color
"""
    glsl_code, _ = transpile(shader_code)
    assert "uv = vs_uv * 2.0" in glsl_code
    assert "sin(uv.x + u_time)" in glsl_code


def test_unsupported_binop_raises_error(
    symbols: Dict[str, str], collected_info: CollectedInfo
) -> None:
    """Test that unsupported binary operations raise errors."""
    node = ast.parse("uv % 2.0", mode="eval").body
    with pytest.raises(TranspilerError, match="Unsupported binary op: Mod"):
        generate_expr(node, symbols, 0, collected_info)


def test_empty_shader_raises_error() -> None:
    """Test that empty shader code raises an error."""
    shader_code = ""
    with pytest.raises(TranspilerError, match="Empty shader code provided"):
        transpile(shader_code)


def test_invalid_function_call_raises_error(
    symbols: Dict[str, str], collected_info: CollectedInfo
) -> None:
    """Test that invalid function calls raise errors."""
    node = ast.parse("unknown(uv)", mode="eval").body
    with pytest.raises(TranspilerError, match="Unknown function call: unknown"):
        generate_expr(node, symbols, 0, collected_info)


def test_shader_with_no_body_raises_error() -> None:
    """Test that shaders with no body (only pass) raise errors."""
    shader_code = """
def shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    pass
"""
    with pytest.raises(
        TranspilerError, match="Pass statements are not supported in GLSL"
    ):
        transpile(shader_code)


def test_augmented_assignment() -> None:
    """Test handling augmented assignment operations."""
    shader_code = """
def shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    uv = vs_uv
    uv *= 2.0
    return vec4(uv, 0.0, 1.0)
"""
    glsl_code, _ = transpile(shader_code)
    assert "uv = uv * 2.0" in glsl_code


def test_struct_definition() -> None:
    """Test struct definition generation."""
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


def test_while_loop() -> None:
    """Test while loop generation."""
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


def test_attribute_assignment() -> None:
    """Test struct attribute assignment."""
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


def test_if_statement() -> None:
    """Test if-else statement generation."""
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


def test_break_in_loop() -> None:
    """Test break statement in loops."""
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


def test_for_loop() -> None:
    """Test for loop generation."""
    shader_code = """
def shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    i = 0
    for i in range(10):
        i += 1
    return vec4(float(i) * 0.1, 0.0, 0.0, 1.0)
"""
    glsl_code, _ = transpile(shader_code)
    assert "for (int i = 0; i < 10; i += 1)" in glsl_code


def test_boolean_operation() -> None:
    """Test boolean operations generation."""
    shader_code = """
def shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    i = 0
    while i < 10 or u_time > 1.0:
        i += 1
    return vec4(float(i) * 0.1, 0.0, 0.0, 1.0)
"""
    glsl_code, _ = transpile(shader_code)
    assert "while (i < 10 || u_time > 1.0)" in glsl_code


def test_global_variables() -> None:
    """Test global variable declarations."""
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


def test_default_uniforms_included() -> None:
    """Test that default uniforms are included correctly."""
    shader_code = """
def shader(vs_uv: 'vec2', u_time: 'float', u_aspect: 'float') -> 'vec4':
    return vec4(sin(u_time), cos(u_aspect), 0.0, 1.0)
"""
    glsl_code, used_uniforms = transpile(shader_code)
    assert "uniform float u_time;" in glsl_code
    assert "uniform float u_aspect;" in glsl_code
    assert used_uniforms == {"u_time", "u_aspect"}


def test_unused_default_uniforms_not_included() -> None:
    """Test that unused default uniforms are not included."""
    shader_code = """
def shader(vs_uv: 'vec2') -> 'vec4':
    return vec4(vs_uv, 0.0, 1.0)
"""
    glsl_code, used_uniforms = transpile(shader_code)
    assert "uniform float u_time;" not in glsl_code
    assert "uniform float u_aspect;" not in glsl_code
    assert used_uniforms == set()


def test_custom_uniforms_included() -> None:
    """Test that custom uniforms are included correctly."""
    shader_code = """
def shader(vs_uv: 'vec2', u_custom: 'vec3') -> 'vec4':
    return vec4(u_custom, 1.0)
"""
    glsl_code, used_uniforms = transpile(shader_code)
    assert "uniform vec3 u_custom;" in glsl_code
    assert used_uniforms == {"u_custom"}


def test_mixed_uniforms() -> None:
    """Test handling of mixed uniform types."""
    shader_code = """
def shader(vs_uv: 'vec2', u_time: 'float', u_custom: 'float') -> 'vec4':
    return vec4(sin(u_time * u_custom), 0.0, 0.0, 1.0)
"""
    glsl_code, used_uniforms = transpile(shader_code)
    assert "uniform float u_time;" in glsl_code
    assert "uniform float u_custom;" in glsl_code
    assert used_uniforms == {"u_time", "u_custom"}


def test_struct_initialization_keywords() -> None:
    """Test struct initialization with keyword arguments."""

    @dataclass
    class SimpleStruct:
        x: "float"
        y: "vec3"
        z: "int"

    def shader(vs_uv: "vec2") -> "vec4":
        s = SimpleStruct(x=1.0, y=vec3(2.0, 3.0, 4.0), z=5)
        return vec4(s.y, s.x)

    glsl_code, _ = transpile(SimpleStruct, shader)
    assert "SimpleStruct s = SimpleStruct(1.0, vec3(2.0, 3.0, 4.0), 5);" in glsl_code
    assert "return vec4(s.y, s.x);" in glsl_code


def test_struct_partial_init_with_defaults() -> None:
    """Test struct initialization with default values."""

    @dataclass
    class DefaultStruct:
        a: "float" = 0.0
        b: "vec3" = "vec3(1.0, 1.0, 1.0)"
        c: "int" = 42

    def shader(vs_uv: "vec2") -> "vec4":
        s = DefaultStruct(b=vec3(2.0, 3.0, 4.0))
        return vec4(s.b, s.a)

    glsl_code, _ = transpile(DefaultStruct, shader)
    assert "DefaultStruct s = DefaultStruct(0.0, vec3(2.0, 3.0, 4.0), 42);" in glsl_code


def test_uniform_declaration_and_usage() -> None:
    """Test declaration and usage of uniform variables."""

    def shader(vs_uv: "vec2", u_time: "float", u_scale: "float") -> "vec4":
        pos = vec3(sin(u_time) * u_scale, cos(u_time), 0.0)
        return vec4(pos, 1.0)

    glsl_code, used_uniforms = transpile(shader)
    assert "uniform float u_time;" in glsl_code
    assert "uniform float u_scale;" in glsl_code
    assert "u_time" in used_uniforms and "u_scale" in used_uniforms


def test_unused_uniforms() -> None:
    """Test handling of unused uniform variables."""

    def shader(vs_uv: "vec2", u_unused: "float") -> "vec4":
        return vec4(vs_uv.x, vs_uv.y, 0.0, 1.0)

    glsl_code, used_uniforms = transpile(shader)
    assert "uniform float u_unused;" in glsl_code
    assert "u_unused" in used_uniforms


def test_arithmetic_and_builtins() -> None:
    """Test arithmetic operations and built-in function usage."""

    def shader(vs_uv: "vec2") -> "vec4":
        v = vec3(vs_uv.x + 1.0, vs_uv.y * 2.0, sin(vs_uv.x))
        return vec4(normalize(v), 1.0)

    glsl_code, _ = transpile(shader)
    assert "vec3 v = vec3(vs_uv.x + 1.0, vs_uv.y * 2.0, sin(vs_uv.x));" in glsl_code
    assert "return vec4(normalize(v), 1.0);" in glsl_code


def test_loop_with_struct() -> None:
    """Test loops with struct variables."""

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

    glsl_code, _ = transpile(LoopStruct, shader)
    assert "LoopStruct s = LoopStruct(0, vs_uv.x);" in glsl_code
    assert "for (int i = 0; i < 5; i += 1) {" in glsl_code
    assert "s.count = i;" in glsl_code
    assert "s.value = s.value + 1.0;" in glsl_code


def test_conditional_struct() -> None:
    """Test structs with conditional logic."""

    @dataclass
    class CondStruct:
        flag: "bool"
        color: "vec3"

    def shader(vs_uv: "vec2") -> "vec4":
        s = CondStruct(flag=vs_uv.x > 0.5, color=vec3(1.0, 0.0, 0.0))
        if s.flag:
            s.color = vec3(0.0, 1.0, 0.0)
        return vec4(s.color, 1.0)

    glsl_code, _ = transpile(CondStruct, shader)
    assert "CondStruct s = CondStruct(vs_uv.x > 0.5, vec3(1.0, 0.0, 0.0));" in glsl_code
    assert "if (s.flag) {" in glsl_code
    assert "s.color = vec3(0.0, 1.0, 0.0);" in glsl_code


def test_missing_required_fields() -> None:
    """Test that missing required struct fields raise errors."""

    @dataclass
    class RequiredStruct:
        x: "float"
        y: "vec3"

    def shader(vs_uv: "vec2") -> "vec4":
        s = RequiredStruct(x=1.0)  # Missing y
        return vec4(s.y, 1.0)

    with pytest.raises(
        TranspilerError, match="Missing required fields in struct RequiredStruct"
    ):
        transpile(RequiredStruct, shader)


def test_invalid_field_name() -> None:
    """Test that invalid struct field names raise errors."""

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
        transpile(ValidStruct, shader)


def test_nested_structs() -> None:
    """Test handling of nested struct definitions."""

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

    glsl_code, _ = transpile(InnerStruct, OuterStruct, shader)
    assert "InnerStruct inner = InnerStruct(vs_uv);" in glsl_code
    assert "OuterStruct outer = OuterStruct(inner, 2.0);" in glsl_code
    assert (
        "return vec4(outer.inner.v.x * outer.scale, outer.inner.v.y, 0.0, 1.0);"
        in glsl_code
    )


def test_nan_prevention() -> None:
    """Test NaN prevention in GLSL code generation."""

    @dataclass
    class SafeStruct:
        pos: "vec3"
        speed: "float"

    def shader(vs_uv: "vec2") -> "vec4":
        s = SafeStruct(pos=vec3(vs_uv.x, vs_uv.y, 0.0), speed=1.0)
        s.pos = s.pos + vec3(s.speed, 0.0, 0.0)
        return vec4(s.pos, 1.0)

    glsl_code, _ = transpile(SafeStruct, shader)
    assert "SafeStruct s = SafeStruct(vec3(vs_uv.x, vs_uv.y, 0.0), 1.0);" in glsl_code
    assert "s.pos = s.pos + vec3(s.speed, 0.0, 0.0);" in glsl_code


def test_loops_and_conditionals() -> None:
    """Test complex loops and conditionals in shaders."""
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


def test_arithmetic_operations() -> None:
    """Test arithmetic operations in shaders."""
    shader_code = """
def shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    x: 'float' = sin(u_time) + 2.0 * vs_uv.x
    return vec4(x, 0.0, 0.0, 1.0)
"""
    glsl_code, _ = transpile(shader_code)
    assert "float x = sin(u_time) + 2.0 * vs_uv.x;" in glsl_code
    assert "return vec4(x, 0.0, 0.0, 1.0);" in glsl_code


def test_helper_function_with_struct() -> None:
    """Test helper functions that return structs."""
    shader_code = """
from dataclasses import dataclass

@dataclass
class Result:
    value: 'float'
    flag: 'bool'

def helper(pos: 'vec2') -> 'Result':
    return Result(length(pos), pos.x > 0.0)

def shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    res = helper(vs_uv)
    return vec4(res.value, 0.0, 0.0, 1.0) if res.flag else vec4(0.0, 0.0, 0.0, 1.0)
"""
    glsl_code, used_uniforms = transpile(shader_code)
    assert "struct Result {" in glsl_code
    assert "float value;" in glsl_code
    assert "bool flag;" in glsl_code
    assert "Result helper(vec2 pos) {" in glsl_code
    assert "return Result(length(pos), pos.x > 0.0);" in glsl_code
    assert "Result res = helper(vs_uv);" in glsl_code
    assert (
        "return res.flag ? vec4(res.value, 0.0, 0.0, 1.0) : vec4(0.0, 0.0, 0.0, 1.0);"
        in glsl_code
    )
    assert used_uniforms == {"u_time"}


def test_ray_march_style_shader() -> None:
    """Test ray marching-style shader with complex structs."""
    shader_code = """
from dataclasses import dataclass

@dataclass
class MarchResult:
    hit: 'bool'
    distance: 'float'

def march_step(start: 'vec2', step: 'float') -> 'MarchResult':
    pos = start
    dist = 0.0
    for i in range(5):
        dist = dist + step
        pos = pos + vec2(step, 0.0)
    return MarchResult(dist < 1.0, dist)

def shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
    result = march_step(vs_uv, u_time)
    return vec4(result.distance, 0.0, 0.0, 1.0) if result.hit else vec4(0.0, 0.0, 0.0, 1.0)
"""
    glsl_code, used_uniforms = transpile(shader_code)
    assert "struct MarchResult {" in glsl_code
    assert "bool hit;" in glsl_code
    assert "float distance;" in glsl_code
    assert "MarchResult march_step(vec2 start, float step) {" in glsl_code
    assert "for (int i = 0; i < 5; i += 1) {" in glsl_code
    assert "return MarchResult(dist < 1.0, dist);" in glsl_code
    assert "MarchResult result = march_step(vs_uv, u_time);" in glsl_code
    assert (
        "return (result.hit ? vec4(result.distance, 0.0, 0.0, 1.0) : vec4(0.0, 0.0, 0.0, 1.0));"
        in glsl_code
    )
    assert used_uniforms == {"u_time"}


def test_shader_with_test_prefix() -> None:
    """Test that shaders with test_ prefix are excluded."""

    def test_shader(vs_uv: "vec2") -> "vec4":
        return vec4(1.0, 0.0, 0.0, 1.0)

    with pytest.raises(
        TranspilerError,
        match="Main function 'test_shader' excluded due to 'test_' prefix",
    ):
        transpile(test_shader)
