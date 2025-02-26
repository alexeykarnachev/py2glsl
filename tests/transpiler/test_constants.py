"""Tests for the transpiler constants module."""

from py2glsl.transpiler.constants import BUILTIN_FUNCTIONS, OPERATOR_PRECEDENCE


class TestBuiltinFunctions:
    """Tests for the BUILTIN_FUNCTIONS dictionary."""

    def test_builtin_functions_structure(self):
        """Test that the built-in functions dictionary has the expected structure."""
        # All entries should be tuples of (return_type, param_types)
        for func_name, func_info in BUILTIN_FUNCTIONS.items():
            assert isinstance(func_name, str)
            assert isinstance(func_info, tuple)
            assert len(func_info) == 2

            return_type, param_types = func_info
            assert isinstance(return_type, str)
            assert isinstance(param_types, list)
            assert all(isinstance(param, str) for param in param_types)

    def test_common_functions_exist(self):
        """Test that common GLSL functions exist in the dictionary."""
        expected_functions = [
            "sin",
            "cos",
            "tan",
            "abs",
            "min",
            "max",
            "length",
            "normalize",
            "cross",
            "vec2",
            "vec3",
            "vec4",
        ]

        for func in expected_functions:
            assert func in BUILTIN_FUNCTIONS, f"Function {func} should be defined"

    def test_vec_constructors(self):
        """Test that vector constructors have correct signatures."""
        # Check vec2
        assert BUILTIN_FUNCTIONS["vec2"][0] == "vec2"
        assert len(BUILTIN_FUNCTIONS["vec2"][1]) == 2
        assert all(param == "float" for param in BUILTIN_FUNCTIONS["vec2"][1])

        # Check one of the vec3 constructors
        assert BUILTIN_FUNCTIONS["vec3"][0] == "vec3"
        # We can't check specific parameter types because there are multiple overloads

        # Check one of the vec4 constructors
        assert BUILTIN_FUNCTIONS["vec4"][0] == "vec4"
        # Again, can't check specific parameters due to overloads


class TestOperatorPrecedence:
    """Tests for the OPERATOR_PRECEDENCE dictionary."""

    def test_operator_precedence_structure(self):
        """Test that the operator precedence dictionary has the expected structure."""
        for op, precedence in OPERATOR_PRECEDENCE.items():
            assert isinstance(op, str)
            assert isinstance(precedence, int)
            assert precedence > 0  # Precedence should be positive

    def test_multiplication_higher_than_addition(self):
        """Test that multiplication has higher precedence than addition."""
        assert OPERATOR_PRECEDENCE["*"] > OPERATOR_PRECEDENCE["+"]
        assert OPERATOR_PRECEDENCE["/"] > OPERATOR_PRECEDENCE["+"]

    def test_and_higher_than_or(self):
        """Test that logical AND has higher precedence than logical OR."""
        assert OPERATOR_PRECEDENCE["&&"] > OPERATOR_PRECEDENCE["||"]

    def test_equality_higher_than_logical(self):
        """Test that equality operators have higher precedence than logical operators."""
        assert OPERATOR_PRECEDENCE["=="] > OPERATOR_PRECEDENCE["||"]
        assert OPERATOR_PRECEDENCE["!="] > OPERATOR_PRECEDENCE["||"]

    def test_relational_higher_than_equality(self):
        """Test that relational operators have higher precedence than equality."""
        assert OPERATOR_PRECEDENCE["<"] > OPERATOR_PRECEDENCE["=="]
        assert OPERATOR_PRECEDENCE[">"] > OPERATOR_PRECEDENCE["=="]

    def test_assignment_lowest_precedence(self):
        """Test that assignment has the lowest precedence."""
        assignment_precedence = OPERATOR_PRECEDENCE["="]
        for op, precedence in OPERATOR_PRECEDENCE.items():
            if op != "=":
                assert precedence > assignment_precedence
