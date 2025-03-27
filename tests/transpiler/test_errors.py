"""Tests for the transpiler errors module."""

import os
import pytest
from unittest.mock import patch

from py2glsl.transpiler.errors import TranspilerError


def test_transpiler_error():
    """Test that TranspilerError includes both message and location info."""
    # Arrange
    error_message = "Test error message"

    # Act & Assert
    with pytest.raises(TranspilerError) as excinfo:
        raise TranspilerError(error_message)

    # The error should include the original message
    assert error_message in str(excinfo.value)
    
    # The error should also include location information
    # It might include "in filename" or other file reference
    error_str = str(excinfo.value)
    assert len(error_str) > len(error_message), "Error should contain additional context info"
    assert "in " in error_str or "at " in error_str, "Error should contain location markers"


def test_transpiler_error_inheritance():
    """Test that TranspilerError inherits from Exception."""
    # Act
    error = TranspilerError("Test")

    # Assert
    assert isinstance(error, Exception)
    assert issubclass(TranspilerError, Exception)
