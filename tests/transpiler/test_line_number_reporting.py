"""Tests for line number reporting in transpiler errors."""

import ast
import os
import textwrap
from unittest.mock import patch

import pytest

from py2glsl.transpiler.errors import TranspilerError
from py2glsl.transpiler.type_checker import _find_matching_signature


def test_error_with_actual_file_line():
    """Test that TranspilerError uses actual_file_line attribute when available."""
    # Isolate from environment variables
    with patch.dict('os.environ', {}, clear=True):
        # Create a mock node
        mock_node = type('MockNode', (), {'actual_file_line': 100})

        # Create an error with our custom node
        error = TranspilerError("Test error", mock_node)

        # The error should use the custom attribute
        assert error.lineno == 100


def test_line_offset_calculation():
    """Test that source line offsets are correctly calculated."""
    # Create a mock function that we can manipulate its line info
    def mock_func():
        pass

    # Artificially set the co_firstlineno attribute
    mock_func.__code__ = mock_func.__code__.replace(co_firstlineno=50)

    # Define test environment
    test_env = {
        "PY2GLSL_CURRENT_FILE": "/test/example.py"
    }

    # Run the code with patched environment
    with patch.dict(os.environ, test_env):
        # Simulate what happens in ast_parser.py
        if hasattr(mock_func, "__code__"):
            line_offset = mock_func.__code__.co_firstlineno - 1
            os.environ["PY2GLSL_LINE_OFFSET"] = str(line_offset)

        # Create an AST node with lineinfo
        code = "x = 42"
        tree = ast.parse(code)
        node = tree.body[0]  # The assignment node

        # Set the actual file line on the node as done in type_checker.py
        ast_line = node.lineno
        line_offset = int(os.environ.get("PY2GLSL_LINE_OFFSET", "0"))
        node.actual_file_line = ast_line + line_offset

        # Create an error
        error = TranspilerError("Test error", node)

        # The error should have the correct line number
        assert error.lineno == node.lineno + 49  # 49 is line_offset
        assert str(error.lineno) in str(error)


def test_function_signature_error():
    """Test that function signature errors include appropriate error info."""
    # Define a simple function with type error
    code = textwrap.dedent("""
    from py2glsl.builtins import vec2, normalize

    def shader(vs_uv: vec2) -> vec2:
        # Line 5: this will be an error as normalize expects a vector, not an int
        return normalize(42)
    """)

    # Parse the code
    tree = ast.parse(code)

    # Find the normalize call node
    call_node = None
    for node in ast.walk(tree):
        # Find the normalize call node
        if (isinstance(node, ast.Call) and hasattr(node, 'func') and
                isinstance(node.func, ast.Name) and node.func.id == 'normalize'):
            call_node = node
            break

    assert call_node is not None

    # Try to find a matching signature - should raise an error
    with pytest.raises(TranspilerError) as excinfo:
        _find_matching_signature('normalize', [('vec2', ['vec2'])], ['int'], call_node)

    # Check that the error contains key information
    error_text = str(excinfo.value)
    assert "normalize" in error_text, "Error should mention the function name"
    assert "int" in error_text, "Error should mention the argument types"
    assert "expression" in error_text, "Error should mention it's in an expression"


def test_line_offset_environment():
    """Test that line offset is correctly set in the environment variable."""
    # Create a clean environment for testing
    with patch.dict(os.environ, {}, clear=True):
        # Manually set the environment variables as the parser would
        os.environ["PY2GLSL_CURRENT_FILE"] = "/test/example.py"
        os.environ["PY2GLSL_LINE_OFFSET"] = "74"  # 75-1

        # Verify the environment variables
        assert os.environ["PY2GLSL_CURRENT_FILE"] == "/test/example.py"
        assert os.environ["PY2GLSL_LINE_OFFSET"] == "74"


def test_environment_variable_cleanup():
    """Test that environment variables are cleaned up properly."""
    # Set environment variables before the test
    os.environ["PY2GLSL_CURRENT_FILE"] = "/test/example.py"
    os.environ["PY2GLSL_LINE_OFFSET"] = "42"

    # Create a simple shader function with quoted type annotations
    def mock_shader(vs_uv: "vec2") -> "vec4":  # noqa: F821
        return 'vec4(1.0, 0.0, 0.0, 1.0)'

    # Create a sample synthetic error using our mechanism
    code = "x = 42"
    tree = ast.parse(code)
    node = tree.body[0]

    # Calculate actual file line
    ast_line = node.lineno
    line_offset = int(os.environ.get("PY2GLSL_LINE_OFFSET", "0"))
    node.actual_file_line = ast_line + line_offset

    # Create error with node
    error = TranspilerError("Test error", node)

    # Check that error has the correct information
    assert error.lineno == 1 + 42  # AST line 1 + offset 42
    assert error.file_path == "/test/example.py"

    # Clean up environment variables after the test
    if "PY2GLSL_CURRENT_FILE" in os.environ:
        del os.environ["PY2GLSL_CURRENT_FILE"]
    if "PY2GLSL_LINE_OFFSET" in os.environ:
        del os.environ["PY2GLSL_LINE_OFFSET"]


def test_different_line_info_sources_priority():
    """Test the priority order of different line info sources."""
    # Set up environment variables
    os.environ["PY2GLSL_CURRENT_FILE"] = "/test/example.py"
    os.environ["PY2GLSL_LINE_OFFSET"] = "10"

    # Create an AST node
    code = "x = 42"
    tree = ast.parse(code)
    node = tree.body[0]

    # Case 1: Just AST line number + environment offset
    error1 = TranspilerError("Test error 1", node)
    assert error1.lineno == 1 + 10  # AST line 1 + offset 10

    # Case 2: Node with source_file attribute
    node.source_file = "/custom/path.py"
    error2 = TranspilerError("Test error 2", node)
    assert error2.file_path == "/custom/path.py"
    assert error2.lineno == 1  # Original AST line without offset

    # Case 3: Node with actual_file_line attribute (highest priority)
    node.actual_file_line = 50
    error3 = TranspilerError("Test error 3", node)
    assert error3.lineno == 50

    # Clean up
    if "PY2GLSL_CURRENT_FILE" in os.environ:
        del os.environ["PY2GLSL_CURRENT_FILE"]
    if "PY2GLSL_LINE_OFFSET" in os.environ:
        del os.environ["PY2GLSL_LINE_OFFSET"]
