"""Tests for the transpiler errors module."""

import pytest

from py2glsl.transpiler.errors import TranspilerError


def test_transpiler_error():
    """Test that TranspilerError can be raised with a message."""
    # Arrange
    error_message = "Test error message"

    # Act & Assert
    with pytest.raises(TranspilerError) as excinfo:
        raise TranspilerError(error_message)

    assert str(excinfo.value) == error_message


def test_transpiler_error_inheritance():
    """Test that TranspilerError inherits from Exception."""
    # Act
    error = TranspilerError("Test")

    # Assert
    assert isinstance(error, Exception)
    assert issubclass(TranspilerError, Exception)
