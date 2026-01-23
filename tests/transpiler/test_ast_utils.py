"""Tests for ast_utils module."""

import ast

from py2glsl.transpiler.ast_utils import (
    eval_const_expr,
    eval_constant,
    infer_literal_glsl_type,
    substitute_name,
)


class TestSubstituteName:
    """Tests for substitute_name function."""

    def test_substitute_simple_name(self) -> None:
        """Test substituting a simple variable name."""
        node = ast.Name(id="i", ctx=ast.Load())
        result = substitute_name(node, "i", 5)
        assert isinstance(result, ast.Constant)
        assert result.value == 5

    def test_substitute_different_name(self) -> None:
        """Test that different names are not substituted."""
        node = ast.Name(id="j", ctx=ast.Load())
        result = substitute_name(node, "i", 5)
        assert isinstance(result, ast.Name)
        assert result.id == "j"

    def test_substitute_in_binop(self) -> None:
        """Test substitution in binary operations: i + 1."""
        node = ast.BinOp(
            left=ast.Name(id="i", ctx=ast.Load()),
            op=ast.Add(),
            right=ast.Constant(value=1),
        )
        result = substitute_name(node, "i", 3)
        assert isinstance(result, ast.BinOp)
        assert isinstance(result.left, ast.Constant)
        assert result.left.value == 3

    def test_substitute_in_call(self) -> None:
        """Test substitution in function calls: vec3(i, 0.0, 0.0)."""
        node = ast.Call(
            func=ast.Name(id="vec3", ctx=ast.Load()),
            args=[
                ast.Name(id="i", ctx=ast.Load()),
                ast.Constant(value=0.0),
                ast.Constant(value=0.0),
            ],
            keywords=[],
        )
        result = substitute_name(node, "i", 2)
        assert isinstance(result, ast.Call)
        assert isinstance(result.args[0], ast.Constant)
        assert result.args[0].value == 2

    def test_substitute_in_call_with_keywords(self) -> None:
        """Test substitution in function calls with keyword arguments: func(x=i)."""
        node = ast.Call(
            func=ast.Name(id="func", ctx=ast.Load()),
            args=[],
            keywords=[
                ast.keyword(arg="x", value=ast.Name(id="i", ctx=ast.Load())),
            ],
        )
        result = substitute_name(node, "i", 7)
        assert isinstance(result, ast.Call)
        assert isinstance(result.keywords[0].value, ast.Constant)
        assert result.keywords[0].value.value == 7

    def test_substitute_in_subscript(self) -> None:
        """Test substitution in subscript expressions: arr[i]."""
        node = ast.Subscript(
            value=ast.Name(id="arr", ctx=ast.Load()),
            slice=ast.Name(id="i", ctx=ast.Load()),
            ctx=ast.Load(),
        )
        result = substitute_name(node, "i", 0)
        assert isinstance(result, ast.Subscript)
        assert isinstance(result.slice, ast.Constant)
        assert result.slice.value == 0

    def test_substitute_preserves_constant(self) -> None:
        """Test that constants are preserved unchanged."""
        node = ast.Constant(value=42)
        result = substitute_name(node, "i", 5)
        assert isinstance(result, ast.Constant)
        assert result.value == 42


class TestEvalConstant:
    """Tests for eval_constant function."""

    def test_eval_integer_literal(self) -> None:
        """Test evaluating integer literals."""
        node = ast.Constant(value=42)
        result = eval_constant(node, {})
        assert result == 42

    def test_eval_float_returns_none(self) -> None:
        """Test that float literals return None."""
        node = ast.Constant(value=3.14)
        result = eval_constant(node, {})
        assert result is None

    def test_eval_global_int_constant(self) -> None:
        """Test evaluating global integer constants."""
        node = ast.Name(id="MAX_SIZE", ctx=ast.Load())
        globals_dict = {"MAX_SIZE": ("int", "10")}
        result = eval_constant(node, globals_dict)
        assert result == 10

    def test_eval_global_float_constant_returns_none(self) -> None:
        """Test that global float constants return None."""
        node = ast.Name(id="PI", ctx=ast.Load())
        globals_dict = {"PI": ("float", "3.14159")}
        result = eval_constant(node, globals_dict)
        assert result is None

    def test_eval_unary_minus(self) -> None:
        """Test evaluating unary minus."""
        node = ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=5))
        result = eval_constant(node, {})
        assert result == -5

    def test_eval_addition(self) -> None:
        """Test evaluating addition."""
        node = ast.BinOp(
            left=ast.Constant(value=3),
            op=ast.Add(),
            right=ast.Constant(value=4),
        )
        result = eval_constant(node, {})
        assert result == 7

    def test_eval_subtraction(self) -> None:
        """Test evaluating subtraction."""
        node = ast.BinOp(
            left=ast.Constant(value=10),
            op=ast.Sub(),
            right=ast.Constant(value=3),
        )
        result = eval_constant(node, {})
        assert result == 7

    def test_eval_multiplication(self) -> None:
        """Test evaluating multiplication."""
        node = ast.BinOp(
            left=ast.Constant(value=6),
            op=ast.Mult(),
            right=ast.Constant(value=7),
        )
        result = eval_constant(node, {})
        assert result == 42

    def test_eval_floor_division(self) -> None:
        """Test evaluating floor division."""
        node = ast.BinOp(
            left=ast.Constant(value=17),
            op=ast.FloorDiv(),
            right=ast.Constant(value=5),
        )
        result = eval_constant(node, {})
        assert result == 3

    def test_eval_unknown_variable_returns_none(self) -> None:
        """Test that unknown variables return None."""
        node = ast.Name(id="unknown", ctx=ast.Load())
        result = eval_constant(node, {})
        assert result is None

    def test_eval_non_constant_binop_returns_none(self) -> None:
        """Test that binops with non-constant operands return None."""
        node = ast.BinOp(
            left=ast.Name(id="x", ctx=ast.Load()),
            op=ast.Add(),
            right=ast.Constant(value=1),
        )
        result = eval_constant(node, {})
        assert result is None


class TestEvalConstExpr:
    """Tests for eval_const_expr function (general constant evaluation)."""

    def test_eval_bool_literal(self) -> None:
        """Test evaluating boolean literals."""
        node = ast.Constant(value=True)
        result = eval_const_expr(node, {})
        assert result == ("bool", True)

        node = ast.Constant(value=False)
        result = eval_const_expr(node, {})
        assert result == ("bool", False)

    def test_eval_float_literal(self) -> None:
        """Test evaluating float literals."""
        node = ast.Constant(value=3.14)
        result = eval_const_expr(node, {})
        assert result == ("float", 3.14)

    def test_eval_int_literal(self) -> None:
        """Test evaluating integer literals."""
        node = ast.Constant(value=42)
        result = eval_const_expr(node, {})
        assert result == ("int", 42)

    def test_eval_string_literal_returns_none(self) -> None:
        """Test that string literals return None."""
        node = ast.Constant(value="hello")
        result = eval_const_expr(node, {})
        assert result is None

    def test_eval_global_float_constant(self) -> None:
        """Test evaluating reference to global float constant."""
        node = ast.Name(id="PI", ctx=ast.Load())
        globals_dict = {"PI": ("float", "3.14159")}
        result = eval_const_expr(node, globals_dict)
        assert result == ("float", 3.14159)

    def test_eval_global_int_constant(self) -> None:
        """Test evaluating reference to global int constant."""
        node = ast.Name(id="SIZE", ctx=ast.Load())
        globals_dict = {"SIZE": ("int", "10")}
        result = eval_const_expr(node, globals_dict)
        assert result == ("int", 10)

    def test_eval_global_bool_constant(self) -> None:
        """Test evaluating reference to global bool constant."""
        node = ast.Name(id="DEBUG", ctx=ast.Load())
        globals_dict = {"DEBUG": ("bool", "true")}
        result = eval_const_expr(node, globals_dict)
        assert result == ("bool", True)

    def test_eval_float_addition(self) -> None:
        """Test evaluating float addition."""
        node = ast.BinOp(
            left=ast.Constant(value=1.5),
            op=ast.Add(),
            right=ast.Constant(value=2.5),
        )
        result = eval_const_expr(node, {})
        assert result == ("float", 4.0)

    def test_eval_mixed_int_float_becomes_float(self) -> None:
        """Test that int + float produces float result."""
        node = ast.BinOp(
            left=ast.Constant(value=1),
            op=ast.Add(),
            right=ast.Constant(value=2.5),
        )
        result = eval_const_expr(node, {})
        assert result is not None
        assert result[0] == "float"
        assert result[1] == 3.5

    def test_eval_division(self) -> None:
        """Test evaluating division."""
        node = ast.BinOp(
            left=ast.Constant(value=10.0),
            op=ast.Div(),
            right=ast.Constant(value=4.0),
        )
        result = eval_const_expr(node, {})
        assert result == ("float", 2.5)

    def test_eval_modulo(self) -> None:
        """Test evaluating modulo."""
        node = ast.BinOp(
            left=ast.Constant(value=17),
            op=ast.Mod(),
            right=ast.Constant(value=5),
        )
        result = eval_const_expr(node, {})
        assert result == ("int", 2)

    def test_eval_power(self) -> None:
        """Test evaluating power."""
        node = ast.BinOp(
            left=ast.Constant(value=2.0),
            op=ast.Pow(),
            right=ast.Constant(value=3.0),
        )
        result = eval_const_expr(node, {})
        assert result == ("float", 8.0)

    def test_eval_unary_minus_float(self) -> None:
        """Test evaluating unary minus on float."""
        node = ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=3.14))
        result = eval_const_expr(node, {})
        assert result == ("float", -3.14)

    def test_eval_unary_plus(self) -> None:
        """Test evaluating unary plus."""
        node = ast.UnaryOp(op=ast.UAdd(), operand=ast.Constant(value=5))
        result = eval_const_expr(node, {})
        assert result == ("int", 5)

    def test_eval_chained_reference(self) -> None:
        """Test evaluating expression referencing another constant: TAU = PI * 2."""
        node = ast.BinOp(
            left=ast.Name(id="PI", ctx=ast.Load()),
            op=ast.Mult(),
            right=ast.Constant(value=2.0),
        )
        globals_dict = {"PI": ("float", "3.14159")}
        result = eval_const_expr(node, globals_dict)
        assert result is not None
        assert result[0] == "float"
        assert abs(result[1] - 6.28318) < 0.0001

    def test_eval_unknown_name_returns_none(self) -> None:
        """Test that unknown names return None."""
        node = ast.Name(id="unknown", ctx=ast.Load())
        result = eval_const_expr(node, {})
        assert result is None

    def test_eval_unsupported_binop_returns_none(self) -> None:
        """Test that unsupported binary ops return None."""
        node = ast.BinOp(
            left=ast.Constant(value=1),
            op=ast.BitAnd(),
            right=ast.Constant(value=2),
        )
        result = eval_const_expr(node, {})
        assert result is None


class TestInferLiteralGlslType:
    """Tests for infer_literal_glsl_type function."""

    def test_bool_true(self) -> None:
        """Test inferring type from True literal."""
        node = ast.Constant(value=True)
        result = infer_literal_glsl_type(node)
        assert result == ("bool", "true")

    def test_bool_false(self) -> None:
        """Test inferring type from False literal."""
        node = ast.Constant(value=False)
        result = infer_literal_glsl_type(node)
        assert result == ("bool", "false")

    def test_float_literal(self) -> None:
        """Test inferring type from float literal."""
        node = ast.Constant(value=3.14)
        result = infer_literal_glsl_type(node)
        assert result == ("float", "3.14")

    def test_int_literal(self) -> None:
        """Test inferring type from int literal."""
        node = ast.Constant(value=42)
        result = infer_literal_glsl_type(node)
        assert result == ("int", "42")

    def test_string_literal_returns_none(self) -> None:
        """Test that string literals return None."""
        node = ast.Constant(value="hello")
        result = infer_literal_glsl_type(node)
        assert result is None

    def test_none_literal_returns_none(self) -> None:
        """Test that None literals return None."""
        node = ast.Constant(value=None)
        result = infer_literal_glsl_type(node)
        assert result is None
