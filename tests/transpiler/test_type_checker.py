"""Tests for the transpiler type_checker module."""

import ast

import pytest

from py2glsl.transpiler.errors import TranspilerError
from py2glsl.transpiler.models import (
    CollectedInfo,
    FunctionInfo,
    StructDefinition,
    StructField,
)
from py2glsl.transpiler.type_checker import get_expr_type


@pytest.fixture
def symbols():
    """Fixture providing a sample symbol table."""
    return {
        "uv": "vec2",
        "color": "vec4",
        "time": "float",
        "count": "int",
        "flag": "bool",
        "test_struct": "TestStruct",
    }


@pytest.fixture
def collected_info():
    """Fixture providing a sample collected info structure."""
    info = CollectedInfo()

    # Add a test struct
    info.structs["TestStruct"] = StructDefinition(
        name="TestStruct",
        fields=[
            StructField(name="position", type_name="vec3"),
            StructField(name="value", type_name="float"),
        ],
    )

    # Add a test function
    dummy_node = ast.FunctionDef(
        name="test_func",
        args=ast.arguments(
            args=[], posonlyargs=[], kwonlyargs=[], kw_defaults=[], defaults=[]
        ),
        body=[],
        decorator_list=[],
    )
    info.functions["test_func"] = FunctionInfo(
        name="test_func",
        return_type="vec3",
        param_types=["vec2", "float"],
        node=dummy_node,
    )

    return info


class TestGetExprType:
    """Tests for the get_expr_type function."""

    def test_name_expr_type(self, symbols, collected_info):
        """Test determining type of name expressions."""
        # Test with a vec2 variable
        node_vec2 = ast.parse("uv", mode="eval").body
        assert get_expr_type(node_vec2, symbols, collected_info) == "vec2"

        # Test with a float variable
        node_float = ast.parse("time", mode="eval").body
        assert get_expr_type(node_float, symbols, collected_info) == "float"

        # Test with an int variable
        node_int = ast.parse("count", mode="eval").body
        assert get_expr_type(node_int, symbols, collected_info) == "int"

        # Test with a bool variable
        node_bool = ast.parse("flag", mode="eval").body
        assert get_expr_type(node_bool, symbols, collected_info) == "bool"

    def test_undefined_variable(self, symbols, collected_info):
        """Test that accessing an undefined variable raises an error."""
        node = ast.parse("undefined_var", mode="eval").body
        with pytest.raises(TranspilerError, match="Undefined variable: undefined_var"):
            get_expr_type(node, symbols, collected_info)

    def test_constant_expr_type(self, symbols, collected_info):
        """Test determining type of constant expressions."""
        # Test with an integer constant
        node_int = ast.parse("42", mode="eval").body
        assert get_expr_type(node_int, symbols, collected_info) == "int"

        # Test with a float constant
        node_float = ast.parse("3.14", mode="eval").body
        assert get_expr_type(node_float, symbols, collected_info) == "float"

        # Test with boolean constants
        node_true = ast.parse("True", mode="eval").body
        node_false = ast.parse("False", mode="eval").body
        assert get_expr_type(node_true, symbols, collected_info) == "bool"
        assert get_expr_type(node_false, symbols, collected_info) == "bool"

    def test_binary_op_expr_type(self, symbols, collected_info):
        """Test determining type of binary operations."""
        # Test with vec2 + vec2
        node_vec_add = ast.parse("uv + uv", mode="eval").body
        assert get_expr_type(node_vec_add, symbols, collected_info) == "vec2"

        # Test with vec2 * float
        node_vec_mul = ast.parse("uv * time", mode="eval").body
        assert get_expr_type(node_vec_mul, symbols, collected_info) == "vec2"

        # Test with float + int
        node_float_add = ast.parse("time + count", mode="eval").body
        assert get_expr_type(node_float_add, symbols, collected_info) == "float"

        # Test with int + int
        node_int_add = ast.parse("count + count", mode="eval").body
        assert get_expr_type(node_int_add, symbols, collected_info) == "int"

    def test_call_expr_type(self, symbols, collected_info):
        """Test determining type of function call expressions."""
        # Test with built-in function
        node_sin = ast.parse("sin(time)", mode="eval").body
        assert get_expr_type(node_sin, symbols, collected_info) == "float"

        # Test with user-defined function
        node_user_func = ast.parse("test_func(uv, time)", mode="eval").body
        assert get_expr_type(node_user_func, symbols, collected_info) == "vec3"

        # Test with struct constructor
        node_struct = ast.parse("TestStruct(position, value)", mode="eval").body
        assert get_expr_type(node_struct, symbols, collected_info) == "TestStruct"

        # Test with unknown function
        node_unknown = ast.parse("unknown_func()", mode="eval").body
        with pytest.raises(TranspilerError, match="Unknown function: unknown_func"):
            get_expr_type(node_unknown, symbols, collected_info)

    def test_attribute_expr_type(self, symbols, collected_info):
        """Test determining type of attribute access expressions."""
        # Test with struct field access
        node_struct_field = ast.parse("test_struct.position", mode="eval").body
        assert get_expr_type(node_struct_field, symbols, collected_info) == "vec3"

        # Test with vector swizzle (single component)
        node_vec_x = ast.parse("uv.x", mode="eval").body
        assert get_expr_type(node_vec_x, symbols, collected_info) == "float"

        # Test with vector swizzle (multiple components)
        node_vec_xy = ast.parse("uv.xy", mode="eval").body
        assert get_expr_type(node_vec_xy, symbols, collected_info) == "vec2"

        # Test with vec4 swizzle
        node_vec4_rgb = ast.parse("color.rgb", mode="eval").body
        assert get_expr_type(node_vec4_rgb, symbols, collected_info) == "vec3"

        # Test with invalid swizzle
        node_invalid_swizzle = ast.parse("uv.z", mode="eval").body
        with pytest.raises(TranspilerError, match="Invalid swizzle 'z' for vec2"):
            get_expr_type(node_invalid_swizzle, symbols, collected_info)

        # Test with unknown struct field
        node_unknown_field = ast.parse("test_struct.unknown", mode="eval").body
        with pytest.raises(
            TranspilerError, match="Unknown field 'unknown' in struct 'TestStruct'"
        ):
            get_expr_type(node_unknown_field, symbols, collected_info)

    def test_if_expr_type(self, symbols, collected_info):
        """Test determining type of conditional expressions."""
        # Test with matching types
        node_if_match = ast.parse("uv if flag else uv", mode="eval").body
        assert get_expr_type(node_if_match, symbols, collected_info) == "vec2"

        # Test with mismatched types
        node_if_mismatch = ast.parse("uv if flag else time", mode="eval").body
        with pytest.raises(TranspilerError, match="Ternary expression types mismatch"):
            get_expr_type(node_if_mismatch, symbols, collected_info)

    def test_compare_expr_type(self, symbols, collected_info):
        """Test determining type of comparison expressions."""
        # Test with a comparison
        node_compare = ast.parse("time > 1.0", mode="eval").body
        assert get_expr_type(node_compare, symbols, collected_info) == "bool"

    def test_bool_op_expr_type(self, symbols, collected_info):
        """Test determining type of boolean operation expressions."""
        # Test with a boolean operation
        node_bool_op = ast.parse("flag and count > 0", mode="eval").body
        assert get_expr_type(node_bool_op, symbols, collected_info) == "bool"

    def test_unary_op_expr_type(
        self, symbols: dict[str, str | None], collected_info: CollectedInfo
    ) -> None:
        """Test determining type of unary operation expressions."""
        # Test with unary minus on a float
        node_neg_float = ast.parse("-time", mode="eval").body
        assert get_expr_type(node_neg_float, symbols, collected_info) == "float"

        # Test with unary minus on an int
        node_neg_int = ast.parse("-count", mode="eval").body
        assert get_expr_type(node_neg_int, symbols, collected_info) == "int"

        # Test with unary minus on a vector
        node_neg_vec = ast.parse("-uv", mode="eval").body
        assert get_expr_type(node_neg_vec, symbols, collected_info) == "vec2"

        # Test with logical not
        node_not = ast.parse("not flag", mode="eval").body
        assert get_expr_type(node_not, symbols, collected_info) == "bool"

        # Test with unary plus
        node_pos = ast.parse("+time", mode="eval").body
        assert get_expr_type(node_pos, symbols, collected_info) == "float"

    def test_unsupported_expr_type(self, symbols, collected_info):
        """Test that unsupported expressions raise an error."""

        # Create a node type that's not handled by get_expr_type
        class UnsupportedNode(ast.AST):
            pass

        node = UnsupportedNode()
        with pytest.raises(
            TranspilerError, match="Cannot determine type for: UnsupportedNode"
        ):
            get_expr_type(node, symbols, collected_info)
