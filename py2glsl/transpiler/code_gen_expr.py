"""
GLSL code generation for expressions.

This module contains functions for generating GLSL code from Python AST expressions,
including names, constants, binary operations, function calls, and more.
"""

import ast
from collections.abc import Callable

from py2glsl.transpiler.constants import BUILTIN_FUNCTIONS, OPERATOR_PRECEDENCE
from py2glsl.transpiler.errors import TranspilerError
from py2glsl.transpiler.models import CollectedInfo


def generate_name_expr(node: ast.Name, symbols: dict[str, str | None]) -> str:
    """Generate GLSL code for a name expression (variable).

    Args:
        node: AST name node
        symbols: Dictionary of variable names to their types

    Returns:
        Generated GLSL code for the name expression
    """
    return node.id


def generate_constant_expr(node: ast.Constant) -> str:
    """Generate GLSL code for a constant expression (literal).

    Args:
        node: AST constant node

    Returns:
        Generated GLSL code for the constant expression

    Raises:
        TranspilerError: If the constant type is not supported
    """
    if isinstance(node.value, bool):
        return "true" if node.value else "false"  # GLSL uses lowercase
    elif isinstance(node.value, int | float):
        return str(node.value)
    raise TranspilerError(f"Unsupported constant type: {type(node.value).__name__}")


def generate_binary_op_expr(
    node: ast.BinOp,
    symbols: dict[str, str | None],
    parent_precedence: int,
    collected: CollectedInfo,
) -> str:
    """Generate GLSL code for a binary operation expression.

    Args:
        node: AST binary operation node
        symbols: Dictionary of variable names to their types
        parent_precedence: Precedence level of the parent operation
        collected: Information about functions, structs, and globals

    Returns:
        Generated GLSL code for the binary operation expression

    Raises:
        TranspilerError: If the operation is not supported
    """
    op_map = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
    op = op_map.get(type(node.op))
    if not op:
        raise TranspilerError(f"Unsupported binary op: {type(node.op).__name__}")

    precedence = OPERATOR_PRECEDENCE[op]
    left = generate_expr(node.left, symbols, precedence, collected)
    right = generate_expr(node.right, symbols, precedence, collected)

    expr = f"{left} {op} {right}"
    return f"({expr})" if precedence < parent_precedence else expr


def generate_compare_expr(
    node: ast.Compare,
    symbols: dict[str, str | None],
    parent_precedence: int,
    collected: CollectedInfo,
) -> str:
    """Generate GLSL code for a comparison expression.

    Args:
        node: AST comparison node
        symbols: Dictionary of variable names to their types
        parent_precedence: Precedence level of the parent operation
        collected: Information about functions, structs, and globals

    Returns:
        Generated GLSL code for the comparison expression

    Raises:
        TranspilerError: If the comparison operation is not supported
    """
    if len(node.ops) == 1 and len(node.comparators) == 1:
        op_map = {
            ast.Lt: "<",
            ast.Gt: ">",
            ast.LtE: "<=",
            ast.GtE: ">=",
            ast.Eq: "==",
            ast.NotEq: "!=",
        }
        op = op_map.get(type(node.ops[0]))
        if not op:
            raise TranspilerError(
                f"Unsupported comparison op: {type(node.ops[0]).__name__}"
            )

        precedence = OPERATOR_PRECEDENCE[op]
        left = generate_expr(node.left, symbols, precedence, collected)
        right = generate_expr(node.comparators[0], symbols, precedence, collected)

        expr = f"{left} {op} {right}"
        return f"({expr})" if precedence <= parent_precedence else expr
    raise TranspilerError("Multiple comparisons not supported")


def generate_bool_op_expr(
    node: ast.BoolOp,
    symbols: dict[str, str | None],
    parent_precedence: int,
    collected: CollectedInfo,
) -> str:
    """Generate GLSL code for a boolean operation expression.

    Args:
        node: AST boolean operation node
        symbols: Dictionary of variable names to their types
        parent_precedence: Precedence level of the parent operation
        collected: Information about functions, structs, and globals

    Returns:
        Generated GLSL code for the boolean operation expression

    Raises:
        TranspilerError: If the boolean operation is not supported
    """
    op_map = {ast.And: "&&", ast.Or: "||"}
    op = op_map.get(type(node.op))
    if not op:
        raise TranspilerError(f"Unsupported boolean op: {type(node.op).__name__}")

    precedence = OPERATOR_PRECEDENCE[op]
    values = [generate_expr(val, symbols, precedence, collected) for val in node.values]

    expr = f" {op} ".join(values)

    return f"({expr})" if precedence < parent_precedence else expr


def generate_attribute_expr(
    node: ast.Attribute,
    symbols: dict[str, str | None],
    parent_precedence: int,
    collected: CollectedInfo,
) -> str:
    """Generate GLSL code for an attribute access expression.

    Args:
        node: AST attribute node
        symbols: Dictionary of variable names to their types
        parent_precedence: Precedence level of the parent operation
        collected: Information about functions, structs, and globals

    Returns:
        Generated GLSL code for the attribute access expression
    """
    value = generate_expr(node.value, symbols, OPERATOR_PRECEDENCE["member"], collected)
    return f"{value}.{node.attr}"


def generate_if_expr(
    node: ast.IfExp,
    symbols: dict[str, str | None],
    parent_precedence: int,
    collected: CollectedInfo,
) -> str:
    """Generate GLSL code for a ternary/conditional expression.

    Args:
        node: AST conditional expression node
        symbols: Dictionary of variable names to their types
        parent_precedence: Precedence level of the parent operation
        collected: Information about functions, structs, and globals

    Returns:
        Generated GLSL code for the conditional expression
    """
    precedence = OPERATOR_PRECEDENCE["?"]
    condition = generate_expr(node.test, symbols, 0, collected)
    true_expr = generate_expr(node.body, symbols, 0, collected)  # Changed to 0
    false_expr = generate_expr(node.orelse, symbols, 0, collected)  # Changed to 0
    expr = f"{condition} ? {true_expr} : {false_expr}"

    return f"({expr})" if precedence < parent_precedence else expr


def generate_struct_constructor(
    struct_name: str,
    node: ast.Call,
    symbols: dict[str, str | None],
    collected: CollectedInfo,
) -> str:
    """Generate GLSL code for a struct constructor.

    Args:
        struct_name: Name of the struct being constructed
        node: AST call node representing the constructor
        symbols: Dictionary of variable names to their types
        collected: Information about functions, structs, and globals

    Returns:
        Generated GLSL code for the struct constructor

    Raises:
        TranspilerError: If the struct initialization is invalid
    """
    struct_def = collected.structs[struct_name]
    field_map = {f.name: i for i, f in enumerate(struct_def.fields)}

    if node.keywords:
        values: list[str] = [""] * len(struct_def.fields)
        provided_fields = set()

        for kw in node.keywords:
            if kw.arg not in field_map:
                raise TranspilerError(
                    f"Unknown field '{kw.arg}' in struct '{struct_name}'"
                )
            values[field_map[kw.arg]] = generate_expr(kw.value, symbols, 0, collected)
            provided_fields.add(kw.arg)

        missing_fields = [
            field.name
            for field in struct_def.fields
            if field.default_value is None and field.name not in provided_fields
        ]
        if missing_fields:
            raise TranspilerError(
                f"Missing required fields in struct {struct_name}: "
                f"{', '.join(missing_fields)}"
            )

        for i, field in enumerate(struct_def.fields):
            if not values[i] and field.default_value is not None:
                values[i] = field.default_value
            elif not values[i]:
                values[i] = "0.0"

        return f"{struct_name}({', '.join(values)})"

    elif node.args:
        if len(node.args) != len(struct_def.fields):
            raise TranspilerError(
                f"Wrong number of arguments for struct {struct_name}: "
                f"expected {len(struct_def.fields)}, got {len(node.args)}"
            )
        args = [generate_expr(arg, symbols, 0, collected) for arg in node.args]
        return f"{struct_name}({', '.join(args)})"

    raise TranspilerError(f"Struct '{struct_name}' initialization requires arguments")


def generate_call_expr(
    node: ast.Call, symbols: dict[str, str | None], collected: CollectedInfo
) -> str:
    """Generate GLSL code for a function call expression.

    Args:
        node: AST call node
        symbols: Dictionary of variable names to their types
        collected: Information about functions, structs, and globals

    Returns:
        Generated GLSL code for the function call expression

    Raises:
        TranspilerError: If the function is unknown or struct initialization is invalid
    """
    func_name = (
        node.func.id
        if isinstance(node.func, ast.Name)
        else generate_expr(node.func, symbols, 0, collected)
    )

    if func_name in collected.functions or func_name in BUILTIN_FUNCTIONS:
        args = [generate_expr(arg, symbols, 0, collected) for arg in node.args]
        return f"{func_name}({', '.join(args)})"
    elif func_name in collected.structs:
        return generate_struct_constructor(func_name, node, symbols, collected)
    raise TranspilerError(f"Unknown function call: {func_name}")


def generate_unary_op_expr(
    node: ast.UnaryOp,
    symbols: dict[str, str | None],
    parent_precedence: int,
    collected: CollectedInfo,
) -> str:
    """Generate GLSL code for a unary operation expression.

    Args:
        node: AST unary operation node
        symbols: Dictionary of variable names to their types
        parent_precedence: Precedence level of the parent operation
        collected: Information about functions, structs, and globals

    Returns:
        Generated GLSL code for the unary operation expression

    Raises:
        TranspilerError: If the unary operation is not supported
    """
    op_map = {ast.USub: "-", ast.Not: "!"}
    op = op_map.get(type(node.op))
    if not op:
        raise TranspilerError(f"Unsupported unary op: {type(node.op).__name__}")

    precedence = OPERATOR_PRECEDENCE["unary"]
    operand = generate_expr(node.operand, symbols, precedence, collected)

    expr = f"{op}{operand}"
    return f"({expr})" if precedence < parent_precedence else expr


# Type for expression generator functions
ExprGenerator = Callable[[ast.AST, dict[str, str | None], int, CollectedInfo], str]


# Create helper functions with the right signatures
def _name_expr_wrapper(
    node: ast.AST,
    symbols: dict[str, str | None],
    parent_precedence: int,
    collected: CollectedInfo,
) -> str:
    if isinstance(node, ast.Name):
        return generate_name_expr(node, symbols)
    raise TypeError(f"Expected ast.Name, got {type(node).__name__}")


def _constant_expr_wrapper(
    node: ast.AST,
    symbols: dict[str, str | None],
    parent_precedence: int,
    collected: CollectedInfo,
) -> str:
    if isinstance(node, ast.Constant):
        return generate_constant_expr(node)
    raise TypeError(f"Expected ast.Constant, got {type(node).__name__}")


def _call_expr_wrapper(
    node: ast.AST,
    symbols: dict[str, str | None],
    parent_precedence: int,
    collected: CollectedInfo,
) -> str:
    if isinstance(node, ast.Call):
        return generate_call_expr(node, symbols, collected)
    raise TypeError(f"Expected ast.Call, got {type(node).__name__}")


# Create wrapper functions for other expressions
def _binop_expr_wrapper(
    node: ast.AST,
    symbols: dict[str, str | None],
    parent_precedence: int,
    collected: CollectedInfo,
) -> str:
    if isinstance(node, ast.BinOp):
        return generate_binary_op_expr(node, symbols, parent_precedence, collected)
    raise TypeError(f"Expected ast.BinOp, got {type(node).__name__}")


def _compare_expr_wrapper(
    node: ast.AST,
    symbols: dict[str, str | None],
    parent_precedence: int,
    collected: CollectedInfo,
) -> str:
    if isinstance(node, ast.Compare):
        return generate_compare_expr(node, symbols, parent_precedence, collected)
    raise TypeError(f"Expected ast.Compare, got {type(node).__name__}")


def _boolop_expr_wrapper(
    node: ast.AST,
    symbols: dict[str, str | None],
    parent_precedence: int,
    collected: CollectedInfo,
) -> str:
    if isinstance(node, ast.BoolOp):
        return generate_bool_op_expr(node, symbols, parent_precedence, collected)
    raise TypeError(f"Expected ast.BoolOp, got {type(node).__name__}")


def _attribute_expr_wrapper(
    node: ast.AST,
    symbols: dict[str, str | None],
    parent_precedence: int,
    collected: CollectedInfo,
) -> str:
    if isinstance(node, ast.Attribute):
        return generate_attribute_expr(node, symbols, parent_precedence, collected)
    raise TypeError(f"Expected ast.Attribute, got {type(node).__name__}")


def _ifexp_expr_wrapper(
    node: ast.AST,
    symbols: dict[str, str | None],
    parent_precedence: int,
    collected: CollectedInfo,
) -> str:
    if isinstance(node, ast.IfExp):
        return generate_if_expr(node, symbols, parent_precedence, collected)
    raise TypeError(f"Expected ast.IfExp, got {type(node).__name__}")


def _unary_op_expr_wrapper(
    node: ast.AST,
    symbols: dict[str, str | None],
    parent_precedence: int,
    collected: CollectedInfo,
) -> str:
    if isinstance(node, ast.UnaryOp):
        return generate_unary_op_expr(node, symbols, parent_precedence, collected)
    raise TypeError(f"Expected ast.UnaryOp, got {type(node).__name__}")


# Dictionary mapping AST node types to handler functions that all accept ast.AST
_EXPR_GENERATORS: dict[type[ast.AST], ExprGenerator] = {
    ast.Name: _name_expr_wrapper,
    ast.Constant: _constant_expr_wrapper,
    ast.BinOp: _binop_expr_wrapper,
    ast.Compare: _compare_expr_wrapper,
    ast.BoolOp: _boolop_expr_wrapper,
    ast.Call: _call_expr_wrapper,
    ast.Attribute: _attribute_expr_wrapper,
    ast.IfExp: _ifexp_expr_wrapper,
    ast.UnaryOp: _unary_op_expr_wrapper,
}


def generate_expr(
    node: ast.AST,
    symbols: dict[str, str | None],
    parent_precedence: int,
    collected: CollectedInfo,
) -> str:
    """Generate GLSL code for an expression.

    Args:
        node: AST node representing an expression
        symbols: Dictionary of variable names to their types
        parent_precedence: Precedence level of the parent operation
        collected: Information about functions, structs, and globals

    Returns:
        Generated GLSL code for the expression

    Raises:
        TranspilerError: If unsupported expressions are encountered
    """
    node_type = type(node)
    if node_type in _EXPR_GENERATORS:
        generator = _EXPR_GENERATORS[node_type]
        return generator(node, symbols, parent_precedence, collected)
    raise TranspilerError(f"Unsupported expression: {node_type.__name__}")
