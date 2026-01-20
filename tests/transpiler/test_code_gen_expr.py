"""Tests for the transpiler code_gen_expr module."""

import ast

import pytest

from py2glsl.transpiler.code_gen_expr import (
    generate_attribute_expr,
    generate_binary_op_expr,
    generate_bool_op_expr,
    generate_call_expr,
    generate_compare_expr,
    generate_constant_expr,
    generate_expr,
    generate_if_expr,
    generate_name_expr,
    generate_struct_constructor,
    generate_subscript_expr,
    generate_unary_op_expr,
)
from py2glsl.transpiler.models import (
    CollectedInfo,
    FunctionInfo,
    StructDefinition,
    StructField,
    TranspilerError,
)


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


class TestGenerateNameExpr:
    """Tests for the generate_name_expr function."""

    def test_generate_name_expr(self):
        """Test generating code for a name expression."""
        # Arrange
        node = ast.Name(id="uv", ctx=ast.Load())

        # Act
        result = generate_name_expr(node)

        # Assert
        assert result == "uv"


class TestGenerateConstantExpr:
    """Tests for the generate_constant_expr function."""

    def test_generate_constant_expr_int(self):
        """Test generating code for an integer constant."""
        # Arrange
        node = ast.Constant(value=42, kind=None)

        # Act
        result = generate_constant_expr(node)

        # Assert
        assert result == "42"

    def test_generate_constant_expr_float(self):
        """Test generating code for a float constant."""
        # Arrange
        node = ast.Constant(value=3.14, kind=None)

        # Act
        result = generate_constant_expr(node)

        # Assert
        assert result == "3.14"

    def test_generate_constant_expr_bool(self):
        """Test generating code for boolean constants."""
        # Arrange
        node_true = ast.Constant(value=True, kind=None)
        node_false = ast.Constant(value=False, kind=None)

        # Act
        result_true = generate_constant_expr(node_true)
        result_false = generate_constant_expr(node_false)

        # Assert
        assert result_true == "true"
        assert result_false == "false"

    def test_generate_constant_expr_unsupported(self):
        """Test that unsupported constants raise TranspilerError."""
        # Arrange
        node = ast.Constant(value="string", kind=None)

        # Act & Assert
        with pytest.raises(TranspilerError, match="Unsupported constant type"):
            generate_constant_expr(node)


class TestGenerateBinaryOpExpr:
    """Tests for the generate_binary_op_expr function."""

    def test_generate_binary_op_expr_add(self, symbols, collected_info):
        """Test generating code for addition operation."""
        # Arrange
        node = ast.parse("uv + uv", mode="eval").body

        # Act
        result = generate_binary_op_expr(node, symbols, 0, collected_info)

        # Assert
        assert result == "uv + uv"

    def test_generate_binary_op_expr_mult(self, symbols, collected_info):
        """Test generating code for multiplication operation."""
        # Arrange
        node = ast.parse("uv * 2.0", mode="eval").body

        # Act
        result = generate_binary_op_expr(node, symbols, 0, collected_info)

        # Assert
        assert result == "uv * 2.0"

    def test_generate_binary_op_expr_with_precedence(self, symbols, collected_info):
        """Test generating code for binary operation with precedence considerations."""
        # Arrange
        node = ast.parse("uv * 2.0", mode="eval").body

        # Act - with higher parent precedence
        result = generate_binary_op_expr(node, symbols, 10, collected_info)

        # Assert - should be wrapped in parentheses
        assert result == "(uv * 2.0)"

        # Act - with lower parent precedence
        result = generate_binary_op_expr(node, symbols, 5, collected_info)

        # Assert - should not be wrapped in parentheses
        assert result == "uv * 2.0"

    def test_generate_binary_op_expr_power(self, symbols, collected_info):
        """Test generating code for power operation (converts to pow())."""
        # Arrange
        node = ast.parse("time ** 2.0", mode="eval").body

        # Act
        result = generate_binary_op_expr(node, symbols, 0, collected_info)

        # Assert
        assert result == "pow(time, 2.0)"

    def test_generate_binary_op_expr_power_complex(self, symbols, collected_info):
        """Test generating code for complex power expression."""
        # Arrange
        node = ast.parse("(time + 1.0) ** (2.0 * count)", mode="eval").body

        # Act
        result = generate_binary_op_expr(node, symbols, 0, collected_info)

        # Assert - parentheses are preserved based on precedence rules
        assert result == "pow(time + 1.0, 2.0 * count)"

    def test_generate_binary_op_expr_unsupported(self, symbols, collected_info):
        """Test that unsupported binary operations raise TranspilerError."""
        # Arrange - modulo operator isn't supported in our implementation
        node = ast.parse("count % 10", mode="eval").body

        # Act & Assert
        with pytest.raises(TranspilerError, match="Unsupported binary op"):
            generate_binary_op_expr(node, symbols, 0, collected_info)


class TestGenerateCompareExpr:
    """Tests for the generate_compare_expr function."""

    def test_generate_compare_expr_equality(self, symbols, collected_info):
        """Test generating code for equality comparison."""
        # Arrange
        node = ast.parse("time == 1.0", mode="eval").body

        # Act
        result = generate_compare_expr(node, symbols, 0, collected_info)

        # Assert
        assert result == "time == 1.0"

    def test_generate_compare_expr_inequality(self, symbols, collected_info):
        """Test generating code for inequality comparison."""
        # Arrange
        node = ast.parse("time != 1.0", mode="eval").body

        # Act
        result = generate_compare_expr(node, symbols, 0, collected_info)

        # Assert
        assert result == "time != 1.0"

    def test_generate_compare_expr_relational(self, symbols, collected_info):
        """Test generating code for relational comparisons."""
        # Arrange
        node_lt = ast.parse("time < 1.0", mode="eval").body
        node_gt = ast.parse("time > 1.0", mode="eval").body
        node_lte = ast.parse("time <= 1.0", mode="eval").body
        node_gte = ast.parse("time >= 1.0", mode="eval").body

        # Act
        result_lt = generate_compare_expr(node_lt, symbols, 0, collected_info)
        result_gt = generate_compare_expr(node_gt, symbols, 0, collected_info)
        result_lte = generate_compare_expr(node_lte, symbols, 0, collected_info)
        result_gte = generate_compare_expr(node_gte, symbols, 0, collected_info)

        # Assert
        assert result_lt == "time < 1.0"
        assert result_gt == "time > 1.0"
        assert result_lte == "time <= 1.0"
        assert result_gte == "time >= 1.0"

    def test_generate_compare_expr_with_precedence(self, symbols, collected_info):
        """Test generating code for comparison with precedence considerations."""
        # Arrange
        node = ast.parse("time > 1.0", mode="eval").body

        # Act - with higher parent precedence
        result = generate_compare_expr(node, symbols, 10, collected_info)

        # Assert - should be wrapped in parentheses
        assert result == "(time > 1.0)"

    def test_generate_compare_expr_multiple_comparisons(self, symbols, collected_info):
        """Test that multiple comparisons raise TranspilerError."""
        # Arrange - chained comparison which isn't directly supported in GLSL
        node = ast.parse("0 < time < 1.0", mode="eval").body

        # Act & Assert
        with pytest.raises(TranspilerError, match="Multiple comparisons not supported"):
            generate_compare_expr(node, symbols, 0, collected_info)


class TestGenerateBoolOpExpr:
    """Tests for the generate_bool_op_expr function."""

    def test_generate_bool_op_expr_and(self, symbols, collected_info):
        """Test generating code for logical AND operation."""
        # Arrange
        node = ast.parse("flag and time > 1.0", mode="eval").body

        # Act
        result = generate_bool_op_expr(node, symbols, 0, collected_info)

        # Assert
        assert result == "flag && time > 1.0"

    def test_generate_bool_op_expr_or(self, symbols, collected_info):
        """Test generating code for logical OR operation."""
        # Arrange
        node = ast.parse("flag or time > 1.0", mode="eval").body

        # Act
        result = generate_bool_op_expr(node, symbols, 0, collected_info)

        # Assert
        assert result == "flag || time > 1.0"

    def test_generate_bool_op_expr_with_precedence(self, symbols, collected_info):
        """Test generating code for boolean operation with precedence considerations."""
        # Arrange
        node = ast.parse("flag and time > 1.0", mode="eval").body

        # Act - with higher parent precedence
        result = generate_bool_op_expr(node, symbols, 10, collected_info)

        # Assert - should be wrapped in parentheses
        assert result == "(flag && time > 1.0)"

    def test_generate_bool_op_expr_multiple_values(self, symbols, collected_info):
        """Test generating code for boolean operation with multiple values."""
        # Arrange
        node = ast.parse("flag and time > 1.0 and count > 0", mode="eval").body

        # Act
        result = generate_bool_op_expr(node, symbols, 0, collected_info)

        # Assert
        assert result == "flag && time > 1.0 && count > 0"


class TestGenerateAttributeExpr:
    """Tests for the generate_attribute_expr function."""

    def test_generate_attribute_expr_struct(self, symbols, collected_info):
        """Test generating code for struct field access."""
        # Arrange
        node = ast.parse("test_struct.position", mode="eval").body

        # Act
        result = generate_attribute_expr(node, symbols, collected_info)

        # Assert
        assert result == "test_struct.position"

    def test_generate_attribute_expr_vec_component(self, symbols, collected_info):
        """Test generating code for vector component access."""
        # Arrange
        node = ast.parse("uv.x", mode="eval").body

        # Act
        result = generate_attribute_expr(node, symbols, collected_info)

        # Assert
        assert result == "uv.x"

    def test_generate_attribute_expr_vec_swizzle(self, symbols, collected_info):
        """Test generating code for vector swizzling."""
        # Arrange
        node = ast.parse("color.rgb", mode="eval").body

        # Act
        result = generate_attribute_expr(node, symbols, collected_info)

        # Assert
        assert result == "color.rgb"

    def test_generate_attribute_expr_nested(self, symbols, collected_info):
        """Test generating code for nested attribute access."""
        # Arrange - create a nested attribute manually since it's not valid Python
        node = ast.Attribute(
            value=ast.Attribute(
                value=ast.Name(id="test_struct", ctx=ast.Load()),
                attr="position",
                ctx=ast.Load(),
            ),
            attr="y",
            ctx=ast.Load(),
        )

        # Act
        result = generate_attribute_expr(node, symbols, collected_info)

        # Assert
        assert result == "test_struct.position.y"


class TestGenerateIfExpr:
    """Tests for the generate_if_expr function."""

    def test_generate_if_expr(self, symbols, collected_info):
        """Test generating code for conditional expression."""
        # Arrange
        node = ast.parse("uv if flag else uv * 2.0", mode="eval").body

        # Act
        result = generate_if_expr(node, symbols, 0, collected_info)

        # Assert
        assert result == "flag ? uv : uv * 2.0"

    def test_generate_if_expr_with_precedence(self, symbols, collected_info):
        """Test generating code for conditional expression with precedence handling.

        Verifies proper parentheses are added when needed based on operator precedence.
        """
        # Arrange
        node = ast.parse("uv if flag else uv * 2.0", mode="eval").body

        # Act - with higher parent precedence
        result = generate_if_expr(node, symbols, 15, collected_info)

        # Assert - should be wrapped in parentheses
        assert result == "(flag ? uv : uv * 2.0)"


class TestGenerateCallExpr:
    """Tests for the generate_call_expr function."""

    def test_generate_call_expr_builtin(self, symbols, collected_info):
        """Test generating code for built-in function call."""
        # Arrange
        node = ast.parse("sin(time)", mode="eval").body

        # Act
        result = generate_call_expr(node, symbols, collected_info)

        # Assert
        assert result == "sin(time)"

    def test_generate_call_expr_user_function(self, symbols, collected_info):
        """Test generating code for user-defined function call."""
        # Arrange
        node = ast.parse("test_func(uv, time)", mode="eval").body

        # Act
        result = generate_call_expr(node, symbols, collected_info)

        # Assert
        assert result == "test_func(uv, time)"

    def test_generate_call_expr_unknown_function(self, symbols, collected_info):
        """Test that unknown function calls raise TranspilerError."""
        # Arrange
        node = ast.parse("unknown_func()", mode="eval").body

        # Act & Assert
        with pytest.raises(
            TranspilerError, match="Unknown function call: unknown_func"
        ):
            generate_call_expr(node, symbols, collected_info)


class TestGenerateStructConstructor:
    """Tests for the generate_struct_constructor function."""

    def test_generate_struct_constructor_positional(self, symbols, collected_info):
        """Test generating code for struct constructor with positional arguments."""
        # Arrange
        node = ast.parse("TestStruct(vec3(1.0, 0.0, 0.0), 5.0)", mode="eval").body

        # Act
        result = generate_struct_constructor(
            "TestStruct", node, symbols, collected_info
        )

        # Assert
        assert result == "TestStruct(vec3(1.0, 0.0, 0.0), 5.0)"

    def test_generate_struct_constructor_keyword(self, symbols, collected_info):
        """Test generating code for struct constructor with keyword arguments."""
        # Arrange
        node = ast.parse(
            "TestStruct(position=vec3(0.0, 1.0, 0.0), value=10.0)", mode="eval"
        ).body

        # Act
        result = generate_struct_constructor(
            "TestStruct", node, symbols, collected_info
        )

        # Assert
        assert result == "TestStruct(vec3(0.0, 1.0, 0.0), 10.0)"

    def test_generate_struct_constructor_wrong_arg_count(self, symbols, collected_info):
        """Test that wrong argument count raises TranspilerError."""
        # Arrange
        node = ast.parse("TestStruct(vec3(0.0, 1.0, 0.0))", mode="eval").body

        # Act & Assert
        with pytest.raises(TranspilerError, match="Wrong number of arguments"):
            generate_struct_constructor("TestStruct", node, symbols, collected_info)

    def test_generate_struct_constructor_unknown_field(self, symbols, collected_info):
        """Test that unknown field raises TranspilerError."""
        # Arrange
        node = ast.parse(
            "TestStruct(pos=vec3(0.0, 1.0, 0.0), value=10.0)", mode="eval"
        ).body

        # Act & Assert
        with pytest.raises(TranspilerError, match="Unknown field 'pos'"):
            generate_struct_constructor("TestStruct", node, symbols, collected_info)

    def test_generate_struct_constructor_missing_field(self, symbols, collected_info):
        """Test that missing required field raises TranspilerError."""
        # Arrange
        node = ast.parse("TestStruct(value=10.0)", mode="eval").body

        # Act & Assert
        with pytest.raises(TranspilerError, match="Missing required fields"):
            generate_struct_constructor("TestStruct", node, symbols, collected_info)

    def test_generate_struct_constructor_no_args(self, symbols, collected_info):
        """Test that no arguments raises TranspilerError."""
        # Arrange
        node = ast.parse("TestStruct()", mode="eval").body

        # Act & Assert
        with pytest.raises(TranspilerError, match="initialization requires arguments"):
            generate_struct_constructor("TestStruct", node, symbols, collected_info)


class TestGenerateUnaryOpExpr:
    """Tests for the generate_unary_op_expr function."""

    def test_generate_unary_op_expr_minus(self, symbols, collected_info):
        """Test generating code for unary minus operation."""
        # Arrange
        node = ast.parse("-time", mode="eval").body

        # Act
        result = generate_unary_op_expr(node, symbols, 0, collected_info)

        # Assert
        assert result == "-time"

    def test_generate_unary_op_expr_not(self, symbols, collected_info):
        """Test generating code for logical not operation."""
        # Arrange
        node = ast.parse("not flag", mode="eval").body

        # Act
        result = generate_unary_op_expr(node, symbols, 0, collected_info)

        # Assert
        assert result == "!flag"

    def test_generate_unary_op_expr_with_precedence(self, symbols, collected_info):
        """Test generating code for unary operation with precedence considerations."""
        # Arrange
        node = ast.parse("-time", mode="eval").body

        # Act - with higher parent precedence (e.g., function call precedence is 9)
        result_high = generate_unary_op_expr(node, symbols, 9, collected_info)
        # Act - with lower parent precedence (e.g., addition precedence is 6)
        result_low = generate_unary_op_expr(node, symbols, 6, collected_info)

        # Assert
        assert result_high == "(-time)"  # Wrapped due to higher precedence
        assert result_low == "-time"  # Not wrapped due to lower precedence

    def test_generate_unary_op_expr_unsupported(self, symbols, collected_info):
        """Test that unsupported unary operations raise TranspilerError."""
        # Arrange - ast.UAdd (unary plus) is not supported
        node = ast.UnaryOp(
            op=ast.UAdd(),
            operand=ast.Name(id="time", ctx=ast.Load()),
        )

        # Act & Assert
        with pytest.raises(TranspilerError, match="Unsupported unary op"):
            generate_unary_op_expr(node, symbols, 0, collected_info)


class TestGenerateSubscriptExpr:
    """Tests for the generate_subscript_expr function."""

    def test_generate_subscript_expr_constant_index(self, symbols, collected_info):
        """Test generating code for subscript with constant index."""
        # Arrange - add an array variable to symbols
        symbols["arr"] = "float[3]"
        node = ast.parse("arr[0]", mode="eval").body

        # Act
        result = generate_subscript_expr(node, symbols, collected_info)

        # Assert
        assert result == "arr[0]"

    def test_generate_subscript_expr_variable_index(self, symbols, collected_info):
        """Test generating code for subscript with variable index."""
        # Arrange - add matrix to symbols
        symbols["matrix"] = "mat3"
        node = ast.parse("matrix[count]", mode="eval").body

        # Act
        result = generate_subscript_expr(node, symbols, collected_info)

        # Assert
        assert result == "matrix[count]"

    def test_generate_subscript_expr_expression_index(self, symbols, collected_info):
        """Test generating code for subscript with expression index."""
        # Arrange
        symbols["matrix"] = "mat4"
        node = ast.parse("matrix[count + 1]", mode="eval").body

        # Act
        result = generate_subscript_expr(node, symbols, collected_info)

        # Assert
        assert result == "matrix[count + 1]"

    def test_generate_subscript_expr_nested(self, symbols, collected_info):
        """Test generating code for nested subscript (2D array access)."""
        # Arrange
        symbols["matrix"] = "mat2"
        node = ast.parse("matrix[0][1]", mode="eval").body

        # Act
        result = generate_subscript_expr(node, symbols, collected_info)

        # Assert
        assert result == "matrix[0][1]"


class TestGenerateExpr:
    """Tests for the generate_expr function."""

    def test_generate_expr_name(self, symbols, collected_info):
        """Test generating code for name expression."""
        # Arrange
        node = ast.parse("uv", mode="eval").body

        # Act
        result = generate_expr(node, symbols, 0, collected_info)

        # Assert
        assert result == "uv"

    def test_generate_expr_constant(self, symbols, collected_info):
        """Test generating code for constant expression."""
        # Arrange
        node = ast.parse("42", mode="eval").body

        # Act
        result = generate_expr(node, symbols, 0, collected_info)

        # Assert
        assert result == "42"

    def test_generate_expr_binary_op(self, symbols, collected_info):
        """Test generating code for binary operation expression."""
        # Arrange
        node = ast.parse("uv * 2.0", mode="eval").body

        # Act
        result = generate_expr(node, symbols, 0, collected_info)

        # Assert
        assert result == "uv * 2.0"

    def test_generate_expr_compare(self, symbols, collected_info):
        """Test generating code for comparison expression."""
        # Arrange
        node = ast.parse("time > 1.0", mode="eval").body

        # Act
        result = generate_expr(node, symbols, 0, collected_info)

        # Assert
        assert result == "time > 1.0"

    def test_generate_expr_bool_op(self, symbols, collected_info):
        """Test generating code for boolean operation expression."""
        # Arrange
        node = ast.parse("flag and time > 1.0", mode="eval").body

        # Act
        result = generate_expr(node, symbols, 0, collected_info)

        # Assert
        assert result == "flag && time > 1.0"

    def test_generate_expr_call(self, symbols, collected_info):
        """Test generating code for function call expression."""
        # Arrange
        node = ast.parse("sin(time)", mode="eval").body

        # Act
        result = generate_expr(node, symbols, 0, collected_info)

        # Assert
        assert result == "sin(time)"

    def test_generate_expr_attribute(self, symbols, collected_info):
        """Test generating code for attribute expression."""
        # Arrange
        node = ast.parse("uv.x", mode="eval").body

        # Act
        result = generate_expr(node, symbols, 0, collected_info)

        # Assert
        assert result == "uv.x"

    def test_generate_expr_if_exp(self, symbols, collected_info):
        """Test generating code for conditional expression."""
        # Arrange
        node = ast.parse("uv if flag else uv * 2.0", mode="eval").body

        # Act
        result = generate_expr(node, symbols, 0, collected_info)

        # Assert
        assert result == "flag ? uv : uv * 2.0"

    def test_generate_expr_unary_op(self, symbols, collected_info):
        """Test generating code for unary operation expression."""
        # Arrange
        node_minus = ast.parse("-time", mode="eval").body
        node_not = ast.parse("not flag", mode="eval").body

        # Act
        result_minus = generate_expr(node_minus, symbols, 0, collected_info)
        result_not = generate_expr(node_not, symbols, 0, collected_info)

        # Assert
        assert result_minus == "-time"
        assert result_not == "!flag"

    def test_generate_expr_unsupported(self, symbols, collected_info):
        """Test that unsupported expressions raise TranspilerError."""
        # Arrange - list comprehension isn't supported
        node = ast.ListComp(
            elt=ast.Name(id="x", ctx=ast.Load()),
            generators=[
                ast.comprehension(
                    target=ast.Name(id="x", ctx=ast.Store()),
                    iter=ast.Name(id="range", ctx=ast.Load()),
                    ifs=[],
                    is_async=0,
                )
            ],
        )

        # Act & Assert
        with pytest.raises(TranspilerError, match="Unsupported expression"):
            generate_expr(node, symbols, 0, collected_info)
