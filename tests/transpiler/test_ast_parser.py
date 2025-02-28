"""Tests for the transpiler ast_parser module."""

import ast

import pytest

from py2glsl.transpiler.ast_parser import (
    generate_simple_expr,
    get_annotation_type,
    parse_shader_code,
)
from py2glsl.transpiler.errors import TranspilerError


class TestGetAnnotationType:
    """Tests for the get_annotation_type function."""

    def test_get_annotation_type_name(self):
        """Test extracting type from Name annotation."""
        # Arrange
        annotation = ast.Name(id="vec2", ctx=ast.Load())

        # Act
        result = get_annotation_type(annotation)

        # Assert
        assert result == "vec2"

    def test_get_annotation_type_string(self):
        """Test extracting type from string constant annotation."""
        # Arrange
        annotation = ast.Constant(value="float", kind=None)

        # Act
        result = get_annotation_type(annotation)

        # Assert
        assert result == "float"

    def test_get_annotation_type_none(self):
        """Test handling None annotation."""
        # Act
        result = get_annotation_type(None)

        # Assert
        assert result is None

    def test_get_annotation_type_unsupported(self):
        """Test handling unsupported annotation."""
        # Arrange
        annotation = ast.List(elts=[], ctx=ast.Load())

        # Act
        result = get_annotation_type(annotation)

        # Assert
        assert result is None


class TestGenerateSimpleExpr:
    """Tests for the generate_simple_expr function."""

    def test_generate_simple_expr_int(self):
        """Test generating code for an integer constant."""
        # Arrange
        node = ast.Constant(value=42, kind=None)

        # Act
        result = generate_simple_expr(node)

        # Assert
        assert result == "42"

    def test_generate_simple_expr_float(self):
        """Test generating code for a float constant."""
        # Arrange
        node = ast.Constant(value=3.14, kind=None)

        # Act
        result = generate_simple_expr(node)

        # Assert
        assert result == "3.14"

    def test_generate_simple_expr_bool(self):
        """Test generating code for boolean constants."""
        # Arrange
        node_true = ast.Constant(value=True, kind=None)
        node_false = ast.Constant(value=False, kind=None)

        # Act
        result_true = generate_simple_expr(node_true)
        result_false = generate_simple_expr(node_false)

        # Assert
        assert result_true == "true"
        assert result_false == "false"

    def test_generate_simple_expr_string(self):
        """Test generating code for a string constant."""
        # Arrange
        node = ast.Constant(value="test", kind=None)

        # Act
        result = generate_simple_expr(node)

        # Assert
        assert result == "test"

    def test_generate_simple_expr_vec(self):
        """Test generating code for a vector constructor."""
        # Arrange
        node = ast.Call(
            func=ast.Name(id="vec3", ctx=ast.Load()),
            args=[
                ast.Constant(value=1.0, kind=None),
                ast.Constant(value=2.0, kind=None),
                ast.Constant(value=3.0, kind=None),
            ],
            keywords=[],
        )

        # Act
        result = generate_simple_expr(node)

        # Assert
        assert result == "vec3(1.0, 2.0, 3.0)"

    def test_generate_simple_expr_unsupported(self):
        """Test that unsupported expressions raise TranspilerError."""
        # Arrange
        node = ast.BinOp(
            left=ast.Constant(value=1, kind=None),
            op=ast.Add(),
            right=ast.Constant(value=2, kind=None),
        )

        # Act & Assert
        with pytest.raises(TranspilerError, match="Unsupported expression"):
            generate_simple_expr(node)


class TestParseShaderCode:
    """Tests for the parse_shader_code function."""

    def test_parse_shader_code_string(self):
        """Test parsing a string of shader code."""
        # Arrange
        shader_code = """
        def shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
            return vec4(1.0, 0.0, 0.0, 1.0)
        """

        # Act
        tree, main_func = parse_shader_code(shader_code)

        # Assert
        assert isinstance(tree, ast.Module)
        assert main_func == "shader"

    def test_parse_shader_code_with_custom_main(self):
        """Test parsing shader code with a custom main function name."""
        # Arrange
        shader_code = """
        def my_shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
            return vec4(1.0, 0.0, 0.0, 1.0)
        """

        # Act
        tree, main_func = parse_shader_code(shader_code, main_func="my_shader")

        # Assert
        assert isinstance(tree, ast.Module)
        assert main_func == "my_shader"

    def test_parse_shader_code_dict(self):
        """Test parsing a dictionary of callables."""

        # Arrange
        def test_func(vs_uv: "vec2") -> "vec4":  # type: ignore # noqa: F821
            return vs_uv.x, vs_uv.y, 0.0, 1.0

        shader_input = {"test_func": test_func}

        # Act
        tree, main_func = parse_shader_code(shader_input)

        # Assert
        assert isinstance(tree, ast.Module)
        assert main_func == "test_func"

    def test_parse_shader_code_empty(self):
        """Test that empty shader code raises TranspilerError."""
        # Act & Assert
        with pytest.raises(TranspilerError, match="Empty shader code provided"):
            parse_shader_code("")

    def test_parse_shader_code_invalid_type(self):
        """Test that invalid input type raises TranspilerError."""
        # Act & Assert
        with pytest.raises(
            TranspilerError, match="Shader input must be a string or context dictionary"
        ):
            parse_shader_code(123)  # type: ignore
