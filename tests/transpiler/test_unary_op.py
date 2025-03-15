"""Tests for unary operations in the type checker."""

import ast

import pytest

from py2glsl.transpiler.models import CollectedInfo
from py2glsl.transpiler.type_checker import get_expr_type


@pytest.fixture  # type: ignore
def symbols() -> dict[str, str | None]:
    """Fixture providing a sample symbol table."""
    return {
        "uv": "vec2",
        "time": "float",
        "count": "int",
        "flag": "bool",
    }


@pytest.fixture  # type: ignore
def collected_info() -> CollectedInfo:
    """Fixture providing a sample collected info structure."""
    return CollectedInfo()


def test_unary_minus_float(
    symbols: dict[str, str | None], collected_info: CollectedInfo
) -> None:
    """Test unary minus with float type."""
    node = ast.parse("-time", mode="eval").body
    assert get_expr_type(node, symbols, collected_info) == "float"


def test_unary_minus_int(
    symbols: dict[str, str | None], collected_info: CollectedInfo
) -> None:
    """Test unary minus with int type."""
    node = ast.parse("-count", mode="eval").body
    assert get_expr_type(node, symbols, collected_info) == "int"


def test_unary_minus_vector(
    symbols: dict[str, str | None], collected_info: CollectedInfo
) -> None:
    """Test unary minus with vector type."""
    node = ast.parse("-uv", mode="eval").body
    assert get_expr_type(node, symbols, collected_info) == "vec2"


def test_unary_not(
    symbols: dict[str, str | None], collected_info: CollectedInfo
) -> None:
    """Test logical not operation."""
    node = ast.parse("not flag", mode="eval").body
    assert get_expr_type(node, symbols, collected_info) == "bool"


def test_unary_plus(
    symbols: dict[str, str | None], collected_info: CollectedInfo
) -> None:
    """Test unary plus operation."""
    node = ast.parse("+time", mode="eval").body
    assert get_expr_type(node, symbols, collected_info) == "float"
