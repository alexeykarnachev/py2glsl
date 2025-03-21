"""
AST parsing utilities for the GLSL shader transpiler.

This module provides functions for parsing Python code into AST nodes
and extracting basic information like type annotations.
"""

import ast
import inspect
import textwrap
from typing import Any, Protocol, TypeVar

from loguru import logger

from py2glsl.transpiler.errors import TranspilerError

# Type variable for shader functions
T = TypeVar("T")


class ShaderFunction(Protocol):
    """Protocol for shader functions that can be converted to GLSL."""

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Callable shader function protocol."""
        ...


def get_annotation_type(annotation: ast.AST | None) -> str | None:
    """Extract the type name from an AST annotation node.

    Args:
        annotation: AST node representing a type annotation

    Returns:
        String representation of the type or None if no valid annotation
    """
    if annotation is None:
        return None
    if isinstance(annotation, ast.Name):
        return annotation.id
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        return annotation.value
    return None


def generate_simple_expr(node: ast.AST) -> str:
    """Generate GLSL code for simple expressions used in global constants or defaults.

    Args:
        node: AST node for a simple expression

    Returns:
        String representation of the expression in GLSL

    Raises:
        TranspilerError: If the expression is not supported for globals/defaults
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return "true" if node.value else "false"  # GLSL uses lowercase
        elif isinstance(node.value, int | float):
            return str(node.value)
        elif isinstance(node.value, str):
            return node.value
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"vec2", "vec3", "vec4"}
    ):
        args = [generate_simple_expr(arg) for arg in node.args]
        return f"{node.func.id}({', '.join(args)})"
    raise TranspilerError("Unsupported expression in global or default value")


def parse_shader_code(
    shader_input: str | dict[str, Any], main_func: str | None = None
) -> tuple[ast.AST, str | None]:
    """Parse the input Python code into an AST.

    Args:
        shader_input: The input Python code (string or dict of callables)
        main_func: Name of the main function to use as shader entry point

    Returns:
        Tuple of (AST of the parsed code, name of the main function)

    Raises:
        TranspilerError: If parsing fails
    """
    logger.debug("Parsing shader input")
    tree = None
    effective_main_func = main_func

    # Process dictionary of functions/classes
    if isinstance(shader_input, dict):
        source_lines = []
        for name, obj in shader_input.items():
            try:
                source = textwrap.dedent(inspect.getsource(obj))
                source_lines.append(source)
            except (OSError, TypeError) as e:
                raise TranspilerError(f"Failed to get source for {name}: {e}") from e
        full_source = "\n".join(source_lines)
        tree = ast.parse(full_source)
        if not effective_main_func:
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    effective_main_func = node.name
                    break
    # Process string input (direct Python code)
    else:  # Must be a string based on type annotation
        assert isinstance(shader_input, str), "Input must be a string"
        shader_code = textwrap.dedent(shader_input)
        if not shader_code:
            raise TranspilerError("Empty shader code provided")
        tree = ast.parse(shader_code)
        if not effective_main_func:
            effective_main_func = "shader"

    logger.debug("Parsing complete")
    return tree, effective_main_func
