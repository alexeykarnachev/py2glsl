"""
Type checking utilities for the GLSL shader transpiler.

This module provides functions for determining the GLSL types of expressions
and validating type compatibility in operations.
"""

import ast
from collections.abc import Callable

from py2glsl.transpiler.constants import BUILTIN_FUNCTIONS
from py2glsl.transpiler.errors import TranspilerError
from py2glsl.transpiler.models import CollectedInfo


# Implementation of the Visitor pattern for type checking
class ExpressionTypeChecker:
    """Visitor class for determining the type of AST expressions."""

    def __init__(self, symbols: dict[str, str | None], collected: CollectedInfo):
        """Initialize the type checker.

        Args:
            symbols: Dictionary of variable names to their types
            collected: Information about functions, structs, and globals
        """
        self.symbols = symbols
        self.collected = collected

    def visit(self, node: ast.AST) -> str:
        """Visit an AST node and determine its type.

        Args:
            node: The AST node to process

        Returns:
            The GLSL type of the expression

        Raises:
            TranspilerError: If the expression type is not supported or if
                type checking fails
        """
        method_name = f"visit_{type(node).__name__}"
        handler = getattr(self, method_name, self.generic_visit)
        return handler(node)

    def generic_visit(self, node: ast.AST) -> str:
        """Handler for unsupported node types.

        Args:
            node: The AST node

        Raises:
            TranspilerError: Always raised for unsupported nodes
        """
        raise TranspilerError(f"Cannot determine type for: {type(node).__name__}")

    def visit_Name(self, node: ast.Name) -> str:
        """Get the type of a name expression."""
        return _get_name_type(node, self.symbols, self.collected)

    def visit_Constant(self, node: ast.Constant) -> str:
        """Get the type of a constant expression."""
        return _get_constant_type(node, self.symbols, self.collected)

    def visit_BinOp(self, node: ast.BinOp) -> str:
        """Get the type of a binary operation expression."""
        return _get_binop_type(node, self.symbols, self.collected)

    def visit_Compare(self, node: ast.Compare) -> str:
        """Get the type of a comparison expression."""
        return _get_compare_boolop_type(node, self.symbols, self.collected)

    def visit_BoolOp(self, node: ast.BoolOp) -> str:
        """Get the type of a boolean operation expression."""
        return _get_compare_boolop_type(node, self.symbols, self.collected)

    def visit_Call(self, node: ast.Call) -> str:
        """Get the type of a function call expression."""
        return _get_call_type(node, self.symbols, self.collected)

    def visit_Attribute(self, node: ast.Attribute) -> str:
        """Get the type of an attribute access expression."""
        return _get_attribute_type(node, self.symbols, self.collected)

    def visit_IfExp(self, node: ast.IfExp) -> str:
        """Get the type of a conditional expression."""
        return _get_ifexp_type(node, self.symbols, self.collected)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> str:
        """Get the type of a unary operation expression."""
        return _get_unaryop_type(node, self.symbols, self.collected)


def _get_name_type(
    node: ast.Name, symbols: dict[str, str | None], collected: CollectedInfo
) -> str:
    """Determine the type of a name expression.

    Args:
        node: AST name node
        symbols: Dictionary of variable names to their types
        collected: Information about functions, structs, and globals

    Returns:
        The GLSL type of the name

    Raises:
        TranspilerError: If the variable is undefined or has no type
    """
    if node.id in symbols:
        symbol_type = symbols[node.id]
        if symbol_type is None:
            raise TranspilerError(f"Variable has no type: {node.id}")
        return symbol_type
    # Check if it's a global constant
    elif node.id in collected.globals:
        return collected.globals[node.id][0]  # Return the type of the global constant
    raise TranspilerError(f"Undefined variable: {node.id}")


def _get_constant_type(
    node: ast.Constant, symbols: dict[str, str | None], collected: CollectedInfo
) -> str:
    """Determine the type of a constant expression.

    Args:
        node: AST constant node
        symbols: Dictionary of variable names to their types
        collected: Information about functions, structs, and globals

    Returns:
        The GLSL type of the constant
    """
    if isinstance(node.value, bool):
        return "bool"
    elif isinstance(node.value, int):
        return "int"
    elif isinstance(node.value, float):
        return "float"
    raise TranspilerError(f"Unsupported constant type: {type(node.value).__name__}")


def _get_binop_type(
    node: ast.BinOp, symbols: dict[str, str | None], collected: CollectedInfo
) -> str:
    """Determine the type of a binary operation expression.

    Args:
        node: AST binary operation node
        symbols: Dictionary of variable names to their types
        collected: Information about functions, structs, and globals

    Returns:
        The GLSL type of the binary operation
    """
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


def _find_matching_signature(
    func_name: str,
    signatures: list[tuple[str, list[str]]],
    arg_types: list[str],
) -> str:
    """Find a matching function signature for the given argument types.

    Args:
        func_name: Name of the function
        signatures: List of function signatures
        arg_types: Types of the arguments passed to the function

    Returns:
        The return type of the matching signature

    Raises:
        TranspilerError: If no matching signature is found
    """
    for signature in signatures:
        return_type, param_types = signature

        # Skip if argument count doesn't match
        if len(arg_types) != len(param_types):
            continue

        # Check if the arguments match the parameter types
        if all(
            arg_type == param_type
            or (arg_type in ["float", "int"] and param_type in ["float", "int"])
            for arg_type, param_type in zip(arg_types, param_types, strict=False)
        ):
            return return_type

    # If we're here, no matching overload was found
    raise TranspilerError(
        f"No matching overload for function {func_name} with argument types {arg_types}"
    )


def _get_call_type(
    node: ast.Call, symbols: dict[str, str | None], collected: CollectedInfo
) -> str:
    """Determine the type of a function call expression.

    Args:
        node: AST call node
        symbols: Dictionary of variable names to their types
        collected: Information about functions, structs, and globals

    Returns:
        The GLSL type of the function call result

    Raises:
        TranspilerError: If the function is unknown or has no matching signature
    """
    if not isinstance(node.func, ast.Name):
        # Handle method calls or other complex calls
        raise TranspilerError(
            f"Unsupported function call type: {type(node.func).__name__}"
        )

    func_name = node.func.id

    # Check built-in functions
    if func_name in BUILTIN_FUNCTIONS:
        # Get function signatures (single tuple or list of tuples)
        func_signatures = BUILTIN_FUNCTIONS[func_name]

        # If it's a single signature tuple (not overloaded)
        if isinstance(func_signatures, tuple):
            return_type: str = func_signatures[0]  # Return the single return type
            return return_type

        # For overloaded functions, find the matching signature
        arg_types = [get_expr_type(arg, symbols, collected) for arg in node.args]
        return _find_matching_signature(func_name, func_signatures, arg_types)

    # Check user-defined functions
    elif func_name in collected.functions:
        return collected.functions[func_name].return_type or "vec4"

    # Check struct constructors
    elif func_name in collected.structs:
        return func_name

    raise TranspilerError(f"Unknown function: {func_name}")


def _get_attribute_type(
    node: ast.Attribute, symbols: dict[str, str | None], collected: CollectedInfo
) -> str:
    """Determine the type of an attribute access expression.

    Args:
        node: AST attribute node
        symbols: Dictionary of variable names to their types
        collected: Information about functions, structs, and globals

    Returns:
        The GLSL type of the attribute

    Raises:
        TranspilerError: If the attribute is invalid or cannot be determined
    """
    value_type = get_expr_type(node.value, symbols, collected)

    # Check struct field access
    if value_type in collected.structs:
        return _get_struct_field_type(node.attr, value_type, collected)

    # Handle vector swizzling
    if value_type.startswith("vec"):
        return _get_vector_swizzle_type(node.attr, value_type)

    raise TranspilerError(f"Cannot determine type for attribute on: {value_type}")


def _get_struct_field_type(
    field_name: str, struct_name: str, collected: CollectedInfo
) -> str:
    """Get the type of a struct field.

    Args:
        field_name: Name of the field
        struct_name: Name of the struct
        collected: Information about functions, structs, and globals

    Returns:
        The GLSL type of the field

    Raises:
        TranspilerError: If the field is not found in the struct
    """
    struct_def = collected.structs[struct_name]
    for field in struct_def.fields:
        if field.name == field_name:
            return field.type_name
    raise TranspilerError(f"Unknown field '{field_name}' in struct '{struct_name}'")


def _get_vector_swizzle_type(swizzle: str, vector_type: str) -> str:
    """Get the type of a vector swizzle.

    Args:
        swizzle: Swizzle components (e.g., "xyz", "xy", "r")
        vector_type: Type of the vector (e.g., "vec3", "vec4")

    Returns:
        The GLSL type of the swizzle result

    Raises:
        TranspilerError: If the swizzle is invalid
    """
    swizzle_len = len(swizzle)
    valid_lengths = {1: "float", 2: "vec2", 3: "vec3", 4: "vec4"}
    vec_dim = int(vector_type[-1])

    # Check valid swizzle components based on vector dimension
    valid_components = "xyzw"[:vec_dim] + "rgba"[:vec_dim]

    if (
        swizzle_len not in valid_lengths
        or swizzle_len > vec_dim
        or not all(c in valid_components for c in swizzle)
    ):
        raise TranspilerError(f"Invalid swizzle '{swizzle}' for {vector_type}")

    return valid_lengths[swizzle_len]


def _get_ifexp_type(
    node: ast.IfExp, symbols: dict[str, str | None], collected: CollectedInfo
) -> str:
    """Determine the type of a conditional (ternary) expression.

    Args:
        node: AST conditional expression node
        symbols: Dictionary of variable names to their types
        collected: Information about functions, structs, and globals

    Returns:
        The GLSL type of the conditional expression

    Raises:
        TranspilerError: If the types of the branches don't match
    """
    true_type = get_expr_type(node.body, symbols, collected)
    false_type = get_expr_type(node.orelse, symbols, collected)

    if true_type != false_type:
        raise TranspilerError(
            f"Ternary expression types mismatch: {true_type} vs {false_type}"
        )

    return true_type


def _get_compare_boolop_type(
    node: ast.Compare | ast.BoolOp,
    symbols: dict[str, str | None],
    collected: CollectedInfo,
) -> str:
    """Determine the type of a comparison or boolean operation.

    Args:
        node: AST comparison or boolean operation node
        symbols: Dictionary of variable names to their types
        collected: Information about functions, structs, and globals

    Returns:
        The GLSL type of the operation (always "bool")
    """
    return "bool"


def _get_unaryop_type(
    node: ast.UnaryOp, symbols: dict[str, str | None], collected: CollectedInfo
) -> str:
    """Determine the type of a unary operation expression.

    Args:
        node: AST unary operation node
        symbols: Dictionary of variable names to their types
        collected: Information about functions, structs, and globals

    Returns:
        The GLSL type of the unary operation
    """
    operand_type = get_expr_type(node.operand, symbols, collected)

    # For logical not (not x), the result is always bool
    if isinstance(node.op, ast.Not):
        return "bool"

    # For other operations (+x, -x, ~x), the result type is the same as the operand
    # Numeric operations preserve their type
    if operand_type in ["int", "float"] or operand_type.startswith("vec"):
        return operand_type

    raise TranspilerError(f"Unsupported unary operation on type: {operand_type}")


# Type for a generic AST node handler
TypeChecker = Callable[[ast.AST, dict[str, str | None], CollectedInfo], str]


# Create wrapper functions with the right signatures
def _name_type_wrapper(
    node: ast.AST, symbols: dict[str, str | None], collected: CollectedInfo
) -> str:
    if isinstance(node, ast.Name):
        return _get_name_type(node, symbols, collected)
    raise TypeError(f"Expected ast.Name, got {type(node).__name__}")


def _constant_type_wrapper(
    node: ast.AST, symbols: dict[str, str | None], collected: CollectedInfo
) -> str:
    if isinstance(node, ast.Constant):
        return _get_constant_type(node, symbols, collected)
    raise TypeError(f"Expected ast.Constant, got {type(node).__name__}")


def _binop_type_wrapper(
    node: ast.AST, symbols: dict[str, str | None], collected: CollectedInfo
) -> str:
    if isinstance(node, ast.BinOp):
        return _get_binop_type(node, symbols, collected)
    raise TypeError(f"Expected ast.BinOp, got {type(node).__name__}")


def _call_type_wrapper(
    node: ast.AST, symbols: dict[str, str | None], collected: CollectedInfo
) -> str:
    if isinstance(node, ast.Call):
        return _get_call_type(node, symbols, collected)
    raise TypeError(f"Expected ast.Call, got {type(node).__name__}")


def _attribute_type_wrapper(
    node: ast.AST, symbols: dict[str, str | None], collected: CollectedInfo
) -> str:
    if isinstance(node, ast.Attribute):
        return _get_attribute_type(node, symbols, collected)
    raise TypeError(f"Expected ast.Attribute, got {type(node).__name__}")


def _ifexp_type_wrapper(
    node: ast.AST, symbols: dict[str, str | None], collected: CollectedInfo
) -> str:
    if isinstance(node, ast.IfExp):
        return _get_ifexp_type(node, symbols, collected)
    raise TypeError(f"Expected ast.IfExp, got {type(node).__name__}")


def _compare_boolop_type_wrapper(
    node: ast.AST, symbols: dict[str, str | None], collected: CollectedInfo
) -> str:
    if isinstance(node, ast.Compare | ast.BoolOp):
        return _get_compare_boolop_type(node, symbols, collected)
    raise TypeError(f"Expected ast.Compare or ast.BoolOp, got {type(node).__name__}")


def _unaryop_type_wrapper(
    node: ast.AST, symbols: dict[str, str | None], collected: CollectedInfo
) -> str:
    if isinstance(node, ast.UnaryOp):
        return _get_unaryop_type(node, symbols, collected)
    raise TypeError(f"Expected ast.UnaryOp, got {type(node).__name__}")


# Type for type checker functions
TypeCheckerFunc = Callable[[ast.AST, dict[str, str | None], CollectedInfo], str]

# Map of AST node types to their type checker functions with proper typing
_TYPE_CHECKERS: dict[type[ast.AST], TypeCheckerFunc] = {
    ast.Name: _name_type_wrapper,
    ast.Constant: _constant_type_wrapper,
    ast.BinOp: _binop_type_wrapper,
    ast.UnaryOp: _unaryop_type_wrapper,
    ast.Call: _call_type_wrapper,
    ast.Attribute: _attribute_type_wrapper,
    ast.IfExp: _ifexp_type_wrapper,
    ast.Compare: _compare_boolop_type_wrapper,
    ast.BoolOp: _compare_boolop_type_wrapper,
}


def get_expr_type(
    node: ast.AST, symbols: dict[str, str | None], collected: CollectedInfo
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
    # Create a type checker and visit the node
    checker = ExpressionTypeChecker(symbols, collected)
    return checker.visit(node)
