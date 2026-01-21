"""AST parsing utilities for shader transpilation."""

import ast
import inspect
import textwrap
from typing import Any

from loguru import logger

from py2glsl.transpiler.models import TranspilerError


def get_annotation_type(annotation: ast.AST | None) -> str | None:
    """Extract type name from an AST annotation node."""
    if annotation is None:
        return None
    if isinstance(annotation, ast.Name):
        return annotation.id
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        return annotation.value
    # Handle subscript annotations like input_[vec2] or uniform[float]
    if isinstance(annotation, ast.Subscript):
        return get_annotation_type(annotation.slice)
    return None


def generate_simple_expr(node: ast.AST) -> str:
    """Generate GLSL for simple expressions (constants, vec constructors)."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return "true" if node.value else "false"
        if isinstance(node.value, int | float):
            return str(node.value)
        if isinstance(node.value, str):
            return node.value
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"vec2", "vec3", "vec4"}
    ):
        args = [generate_simple_expr(arg) for arg in node.args]
        return f"{node.func.id}({', '.join(args)})"
    raise TranspilerError("Unsupported expression in global or default value")


def _parse_dict_input(
    shader_input: dict[str, Any], main_func: str | None
) -> tuple[ast.AST, str | None]:
    """Parse a dict of callables into an AST."""
    source_lines = []
    for name, obj in shader_input.items():
        try:
            source_lines.append(textwrap.dedent(inspect.getsource(obj)))
        except (OSError, TypeError) as e:
            raise TranspilerError(f"Failed to get source for {name}: {e}") from e

    tree = ast.parse("\n".join(source_lines))

    if not main_func:
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                main_func = node.name
                break

    return tree, main_func


def _parse_string_input(
    shader_input: str, main_func: str | None
) -> tuple[ast.AST, str | None]:
    """Parse a string of Python code into an AST."""
    shader_code = textwrap.dedent(shader_input)
    if not shader_code:
        raise TranspilerError("Empty shader code provided")
    return ast.parse(shader_code), main_func or "shader"


def parse_shader_code(
    shader_input: str | dict[str, Any], main_func: str | None = None
) -> tuple[ast.AST, str | None]:
    """Parse Python code into an AST for transpilation."""
    logger.debug("Parsing shader input")

    if isinstance(shader_input, dict):
        result = _parse_dict_input(shader_input, main_func)
    else:
        result = _parse_string_input(shader_input, main_func)

    logger.debug("Parsing complete")
    return result
