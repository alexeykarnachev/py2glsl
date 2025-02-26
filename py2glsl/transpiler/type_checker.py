"""
Type checking utilities for the GLSL shader transpiler.

This module provides functions for determining the GLSL types of expressions
and validating type compatibility in operations.
"""

import ast
from typing import Dict

from py2glsl.transpiler.constants import BUILTIN_FUNCTIONS
from py2glsl.transpiler.errors import TranspilerError
from py2glsl.transpiler.models import CollectedInfo


def get_expr_type(
    node: ast.AST, symbols: Dict[str, str], collected: CollectedInfo
) -> str:
    """Determine the GLSL type of an expression.

    Args:
        node: AST node representing an expression
        symbols: Dictionary of variable names to their types
        collected: Information about functions, structs, and globals

    Returns:
        The GLSL type of the expression

    Raises:
        TranspilerError: If the type cannot be determined
    """
    if isinstance(node, ast.Name):
        if node.id in symbols:
            return symbols[node.id]
        raise TranspilerError(f"Undefined variable: {node.id}")

    elif isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return "bool"  # Correctly identify boolean type
        elif isinstance(node.value, int):
            return "int"
        elif isinstance(node.value, float):
            return "float"

    elif isinstance(node, ast.BinOp):
        left_type = get_expr_type(node.left, symbols, collected)
        right_type = get_expr_type(node.right, symbols, collected)

        # Vector-vector operations: same type result
        if left_type == right_type and left_type.startswith("vec"):
            return left_type

        # Vector-scalar operations: vector result
        if left_type.startswith("vec") and right_type in ["float", "int"]:
            return left_type
        if right_type.startswith("vec") and left_type in ["float", "int"]:
            return right_type

        # Numeric operations
        if "float" in (left_type, right_type):
            return "float"
        return "int"

    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id

            # Check built-in functions
            if func_name in BUILTIN_FUNCTIONS:
                return BUILTIN_FUNCTIONS[func_name][0]

            # Check user-defined functions
            elif func_name in collected.functions:
                return collected.functions[func_name].return_type or "vec4"

            # Check struct constructors
            elif func_name in collected.structs:
                return func_name

            raise TranspilerError(f"Unknown function: {func_name}")
        else:
            # Handle method calls or other complex calls
            raise TranspilerError(
                f"Unsupported function call type: {type(node.func).__name__}"
            )

    elif isinstance(node, ast.Attribute):
        value_type = get_expr_type(node.value, symbols, collected)

        # Check struct field access
        if value_type in collected.structs:
            struct_def = collected.structs[value_type]
            for field in struct_def.fields:
                if field.name == node.attr:
                    return field.type_name
            raise TranspilerError(
                f"Unknown field '{node.attr}' in struct '{value_type}'"
            )

        # Handle vector swizzling
        if value_type.startswith("vec"):
            swizzle_len = len(node.attr)
            valid_lengths = {1: "float", 2: "vec2", 3: "vec3", 4: "vec4"}
            vec_dim = int(value_type[-1])

            # Check valid swizzle components based on vector dimension
            valid_components = "xyzw"[:vec_dim] + "rgba"[:vec_dim]

            if (
                swizzle_len not in valid_lengths
                or swizzle_len > vec_dim
                or not all(c in valid_components for c in node.attr)
            ):
                raise TranspilerError(f"Invalid swizzle '{node.attr}' for {value_type}")

            return valid_lengths[swizzle_len]

        raise TranspilerError(f"Cannot determine type for attribute on: {value_type}")

    elif isinstance(node, ast.IfExp):
        true_type = get_expr_type(node.body, symbols, collected)
        false_type = get_expr_type(node.orelse, symbols, collected)

        if true_type != false_type:
            raise TranspilerError(
                f"Ternary expression types mismatch: {true_type} vs {false_type}"
            )

        return true_type

    elif isinstance(node, ast.Compare) or isinstance(node, ast.BoolOp):
        return "bool"

    raise TranspilerError(f"Cannot determine type for: {type(node).__name__}")
