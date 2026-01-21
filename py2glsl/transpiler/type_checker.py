"""Type inference for GLSL expressions."""

import ast

from py2glsl.transpiler.constants import BUILTIN_FUNCTIONS
from py2glsl.transpiler.models import CollectedInfo, TranspilerError

Symbols = dict[str, str | None]


def _get_name_type(node: ast.Name, symbols: Symbols, collected: CollectedInfo) -> str:
    """Get type of a variable reference."""
    if node.id in symbols:
        symbol_type = symbols[node.id]
        if symbol_type is None:
            raise TranspilerError(f"Variable has no type: {node.id}")
        return symbol_type
    if node.id in collected.globals:
        return collected.globals[node.id][0]
    raise TranspilerError(f"Undefined variable: {node.id}")


def _get_constant_type(node: ast.Constant) -> str:
    """Get type of a literal (bool, int, float)."""
    if isinstance(node.value, bool):
        return "bool"
    if isinstance(node.value, int):
        return "int"
    if isinstance(node.value, float):
        return "float"
    raise TranspilerError(f"Unsupported constant type: {type(node.value).__name__}")


def _get_binop_type(node: ast.BinOp, symbols: Symbols, collected: CollectedInfo) -> str:
    """Get type of a binary operation."""
    left_type = get_expr_type(node.left, symbols, collected)
    right_type = get_expr_type(node.right, symbols, collected)

    # Vector-vector: same type result
    if left_type == right_type and left_type.startswith("vec"):
        return left_type

    # Vector-scalar: vector result
    if left_type.startswith("vec") and right_type in ["float", "int"]:
        return left_type
    if right_type.startswith("vec") and left_type in ["float", "int"]:
        return right_type

    # Matrix-vector multiplication: mat * vec -> vec, vec * mat -> vec
    mat_to_vec = {"mat2": "vec2", "mat3": "vec3", "mat4": "vec4"}
    if left_type in mat_to_vec and right_type == mat_to_vec[left_type]:
        return right_type  # mat * vec -> vec
    if right_type in mat_to_vec and left_type == mat_to_vec[right_type]:
        return left_type  # vec * mat -> vec

    # Matrix-matrix: same type result
    if left_type == right_type and left_type.startswith("mat"):
        return left_type

    # Matrix-scalar: matrix result
    if left_type.startswith("mat") and right_type in ["float", "int"]:
        return left_type
    if right_type.startswith("mat") and left_type in ["float", "int"]:
        return right_type

    # Numeric: float if either is float
    if "float" in (left_type, right_type):
        return "float"
    return "int"


def _find_matching_signature(
    func_name: str,
    signatures: list[tuple[str, list[str]]],
    arg_types: list[str],
) -> str:
    """Find matching function signature and return its return type."""
    for return_type, param_types in signatures:
        if len(arg_types) != len(param_types):
            continue
        if all(
            arg_type == param_type
            or (arg_type in ["float", "int"] and param_type in ["float", "int"])
            for arg_type, param_type in zip(arg_types, param_types, strict=False)
        ):
            return return_type

    raise TranspilerError(
        f"No matching overload for function {func_name} with argument types {arg_types}"
    )


def _get_call_type(node: ast.Call, symbols: Symbols, collected: CollectedInfo) -> str:
    """Get type of a function call."""
    if not isinstance(node.func, ast.Name):
        raise TranspilerError(
            f"Unsupported function call type: {type(node.func).__name__}"
        )

    func_name = node.func.id

    # Built-in functions
    if func_name in BUILTIN_FUNCTIONS:
        func_signatures = BUILTIN_FUNCTIONS[func_name]
        if isinstance(func_signatures, tuple):
            return str(func_signatures[0])
        arg_types = [get_expr_type(arg, symbols, collected) for arg in node.args]
        return _find_matching_signature(func_name, func_signatures, arg_types)

    # User-defined functions
    if func_name in collected.functions:
        return collected.functions[func_name].return_type or "vec4"

    # Struct constructors
    if func_name in collected.structs:
        return func_name

    raise TranspilerError(f"Unknown function: {func_name}")


def _get_attribute_type(
    node: ast.Attribute, symbols: Symbols, collected: CollectedInfo
) -> str:
    """Get type of an attribute access (struct field or vector swizzle)."""
    value_type = get_expr_type(node.value, symbols, collected)

    if value_type in collected.structs:
        return _get_struct_field_type(node.attr, value_type, collected)
    if value_type.startswith("vec"):
        return _get_vector_swizzle_type(node.attr, value_type)

    raise TranspilerError(f"Cannot determine type for attribute on: {value_type}")


def _get_struct_field_type(
    field_name: str, struct_name: str, collected: CollectedInfo
) -> str:
    """Get type of a struct field."""
    for field in collected.structs[struct_name].fields:
        if field.name == field_name:
            return field.type_name
    raise TranspilerError(f"Unknown field '{field_name}' in struct '{struct_name}'")


def _get_vector_swizzle_type(swizzle: str, vector_type: str) -> str:
    """Get type of a vector swizzle (e.g., .xyz -> vec3)."""
    swizzle_len = len(swizzle)
    valid_lengths = {1: "float", 2: "vec2", 3: "vec3", 4: "vec4"}
    vec_dim = int(vector_type[-1])
    valid_components = "xyzw"[:vec_dim] + "rgba"[:vec_dim]

    if (
        swizzle_len not in valid_lengths
        or swizzle_len > vec_dim
        or not all(c in valid_components for c in swizzle)
    ):
        raise TranspilerError(f"Invalid swizzle '{swizzle}' for {vector_type}")

    return valid_lengths[swizzle_len]


def _get_ifexp_type(node: ast.IfExp, symbols: Symbols, collected: CollectedInfo) -> str:
    """Get type of a ternary expression (must have matching branch types)."""
    true_type = get_expr_type(node.body, symbols, collected)
    false_type = get_expr_type(node.orelse, symbols, collected)

    if true_type != false_type:
        raise TranspilerError(
            f"Ternary expression types mismatch: {true_type} vs {false_type}"
        )
    return true_type


def _get_compare_boolop_type() -> str:
    """Get type of comparison/boolean op (always bool)."""
    return "bool"


def _get_unaryop_type(
    node: ast.UnaryOp, symbols: Symbols, collected: CollectedInfo
) -> str:
    """Get type of a unary operation (not -> bool, -/+ -> operand type)."""
    if isinstance(node.op, ast.Not):
        return "bool"

    operand_type = get_expr_type(node.operand, symbols, collected)
    if operand_type in ["int", "float"] or operand_type.startswith("vec"):
        return operand_type

    raise TranspilerError(f"Unsupported unary operation on type: {operand_type}")


def _get_subscript_type(
    node: ast.Subscript, symbols: Symbols, collected: CollectedInfo
) -> str:
    """Get type of subscript (mat[i] -> vec, vec[i] -> float, arr[i] -> elem)."""
    value_type = get_expr_type(node.value, symbols, collected)

    # Matrix -> vector
    matrix_to_vector = {"mat2": "vec2", "mat3": "vec3", "mat4": "vec4"}
    if value_type in matrix_to_vector:
        return matrix_to_vector[value_type]

    # Vector -> float
    if value_type in ("vec2", "vec3", "vec4"):
        return "float"

    # Array type "float[3]" -> element type
    if "[" in value_type:
        return value_type.split("[")[0]

    return value_type


def get_expr_type(node: ast.AST, symbols: Symbols, collected: CollectedInfo) -> str:
    """Determine the GLSL type of an expression."""
    match node:
        case ast.Name():
            return _get_name_type(node, symbols, collected)
        case ast.Constant():
            return _get_constant_type(node)
        case ast.BinOp():
            return _get_binop_type(node, symbols, collected)
        case ast.Call():
            return _get_call_type(node, symbols, collected)
        case ast.Attribute():
            return _get_attribute_type(node, symbols, collected)
        case ast.IfExp():
            return _get_ifexp_type(node, symbols, collected)
        case ast.Compare() | ast.BoolOp():
            return _get_compare_boolop_type()
        case ast.UnaryOp():
            return _get_unaryop_type(node, symbols, collected)
        case ast.Subscript():
            return _get_subscript_type(node, symbols, collected)
        case _:
            raise TranspilerError(f"Unsupported expression type: {type(node).__name__}")
