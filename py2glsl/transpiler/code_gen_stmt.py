"""
GLSL code generation for statements.

This module contains functions for generating GLSL code from Python AST statements,
including assignments, loops, conditionals, and return statements.
"""

import ast
from collections.abc import Callable
from typing import Any, TypeVar, cast

from py2glsl.transpiler.code_gen_expr import generate_attribute_expr, generate_expr
from py2glsl.transpiler.errors import TranspilerError
from py2glsl.transpiler.models import CollectedInfo
from py2glsl.transpiler.type_checker import get_expr_type


def generate_assignment(
    node: ast.Assign,
    symbols: dict[str, str | None],
    indent: str,
    collected: CollectedInfo,
) -> str:
    """Generate GLSL code for an assignment statement.

    This function handles both simple variable assignments and list assignments.
    For list assignments, it delegates to generate_list_declaration.

    Args:
        node: AST assignment node
        symbols: Dictionary of variable names to their types
        indent: Indentation string
        collected: Information about functions, structs, and globals

    Returns:
        Generated GLSL code for the assignment

    Raises:
        TranspilerError: If the assignment target is not supported or if there are
            multiple targets
    """
    if len(node.targets) != 1:
        raise TranspilerError("Multiple assignment targets not supported")
    target = node.targets[0]

    if isinstance(node.value, ast.List):
        return generate_list_declaration(node, symbols, indent, collected)

    value_str = generate_expr(node.value, symbols, 0, collected)

    if isinstance(target, ast.Name):
        target_name = target.id
        if target_name not in symbols:
            inferred_type = get_expr_type(node.value, symbols, collected)
            symbols[target_name] = inferred_type
            return f"{indent}{inferred_type} {target_name} = {value_str};"
        else:
            return f"{indent}{target_name} = {value_str};"
    elif isinstance(target, ast.Attribute):
        target_str = generate_attribute_expr(target, symbols, 0, collected)
        return f"{indent}{target_str} = {value_str};"
    else:
        raise TranspilerError(f"Unsupported assignment target: {type(target).__name__}")


def generate_list_declaration(
    node: ast.Assign,
    symbols: dict[str, str | None],
    indent: str,
    collected: CollectedInfo,
) -> str:
    """Generate GLSL code for list assignment.

    This function handles assignments where the value is a list, creating
    a GLSL array with the appropriate type and size.

    Args:
        node: AST assignment node
        symbols: Dictionary of variable names to their types
        indent: Indentation string
        collected: Information about functions, structs, and globals

    Returns:
        Generated GLSL code for the list assignment

    Raises:
        TranspilerError: If the list elements have mismatched types or if the assignment
            is invalid
    """
    if isinstance(node.value, ast.List):
        elements = node.value.elts
        if isinstance(node.targets[0], ast.Name):
            list_name = node.targets[0].id
        else:
            raise TranspilerError("List assignment target must be a variable name")
        if not elements:
            # Empty list: assume type from context or default to a safe type
            symbol_type = symbols.get(list_name)
            list_type = (
                symbol_type.removeprefix("list[")[:-1]
                if (
                    symbol_type
                    and list_name in symbols
                    and symbol_type.startswith("list[")
                )
                else "vec3"
            )
            size = 0
            symbols[list_name] = f"list[{list_type}]"
            collected.globals[f"{list_name}_size"] = ("int", "0")
            return f"{indent}{list_type} {list_name}[0];"
        else:
            list_type = get_expr_type(elements[0], symbols, collected)
            for elem in elements[1:]:
                if get_expr_type(elem, symbols, collected) != list_type:
                    raise TranspilerError("Type mismatch in list elements")
            size = len(elements)
            symbols[list_name] = f"list[{list_type}]"
            collected.globals[f"{list_name}_size"] = ("int", str(size))
            array_init = ", ".join(
                generate_expr(elem, symbols, 0, collected) for elem in elements
            )
            return (
                f"{indent}{list_type} {list_name}[{size}] = "
                f"{list_type}[{size}]({array_init});"
            )
    else:
        raise TranspilerError(
            "Only list assignments supported in generate_list_declaration"
        )


def generate_annotated_assignment(
    stmt: ast.AnnAssign,
    symbols: dict[str, str | None],
    indent: str,
    collected: CollectedInfo,
) -> str:
    """Generate GLSL code for an annotated assignment.

    Args:
        stmt: AST annotated assignment node
        symbols: Dictionary of variable names to their types
        indent: Indentation string
        collected: Information about functions, structs, and globals

    Returns:
        Generated GLSL code for the annotated assignment

    Raises:
        TranspilerError: If the assignment target is not supported
    """
    if isinstance(stmt.target, ast.Name):
        target = stmt.target.id
        expr_type = get_annotation_type(stmt.annotation)
        expr = generate_expr(stmt.value, symbols, 0, collected) if stmt.value else None
        symbols[target] = expr_type
        return f"{indent}{expr_type} {target}{f' = {expr}' if expr else ''};"
    raise TranspilerError(
        f"Unsupported annotated assignment target: {type(stmt.target).__name__}"
    )


def get_annotation_type(annotation: ast.AST) -> str:
    """Extract the type name from an AST annotation node.

    Args:
        annotation: AST node representing a type annotation

    Returns:
        String representation of the type
    """
    if isinstance(annotation, ast.Name):
        return annotation.id
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        return annotation.value
    raise TranspilerError(f"Unsupported annotation type: {type(annotation).__name__}")


def generate_augmented_assignment(
    stmt: ast.AugAssign,
    symbols: dict[str, str | None],
    indent: str,
    collected: CollectedInfo,
) -> str:
    """Generate GLSL code for an augmented assignment (e.g., +=, -=).

    Args:
        stmt: AST augmented assignment node
        symbols: Dictionary of variable names to their types
        indent: Indentation string
        collected: Information about functions, structs, and globals

    Returns:
        Generated GLSL code for the augmented assignment

    Raises:
        TranspilerError: If the operator is not supported
    """
    target = generate_expr(stmt.target, symbols, 0, collected)
    value = generate_expr(stmt.value, symbols, 0, collected)

    op_map = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
    op = op_map.get(type(stmt.op))

    if op:
        return f"{indent}{target} = {target} {op} {value};"
    raise TranspilerError(f"Unsupported augmented operator: {type(stmt.op).__name__}")


def generate_for_loop(
    stmt: ast.For, symbols: dict[str, str | None], indent: str, collected: CollectedInfo
) -> list[str]:
    """Generate GLSL code for a for loop.

    Supports both list-based iterations and range-based iterations.
    Args:
        stmt: AST for loop node
        symbols: Dictionary of variable names to their types
        indent: Indentation string
        collected: Information about functions, structs, and globals

    Returns:
        List of generated GLSL code lines for the for loop

    Raises:
        TranspilerError: If the loop is not a list or range-based for loop
    """
    code = []
    if isinstance(stmt.iter, ast.Name):  # List iteration, e.g., `for item in some_list`
        list_name = stmt.iter.id
        list_type = symbols.get(list_name, "unknown")
        if list_type and isinstance(list_type, str) and list_type.startswith("list["):
            item_type = list_type[5:-1]  # Extract type, e.g., "vec3" from "list[vec3]"
            index_var = f"i_{list_name}"  # Unique index name
            size_var = f"{list_name}_size"
            code.append(
                f"{indent}for (int {index_var} = 0; {index_var} < {size_var}; "
                f"++{index_var}) {{"
            )
            # Extract the target variable name
            if isinstance(stmt.target, ast.Name):
                target_name = stmt.target.id
            else:
                raise TranspilerError("For loop target must be a variable name")
            code.append(
                f"{indent}    {item_type} {target_name} = {list_name}[{index_var}];"
            )
            body_symbols = symbols.copy()
            body_symbols[target_name] = item_type
            # Type is preserved when copying the symbols dictionary
            for line in generate_body(stmt.body, body_symbols, collected):
                code.append(f"{indent}    {line}")
            code.append(f"{indent}}}")
        else:
            raise TranspilerError(f"Unsupported iterable: {list_type}")
    elif (
        isinstance(stmt.iter, ast.Call)
        and isinstance(stmt.iter.func, ast.Name)
        and stmt.iter.func.id == "range"
    ):
        args = stmt.iter.args
        if isinstance(stmt.target, ast.Name):
            target = stmt.target.id
        else:
            raise TranspilerError("For loop target must be a variable name")
        if len(args) == 1:
            start, end, step = "0", generate_expr(args[0], symbols, 0, collected), "1"
        elif len(args) == 2:  # noqa: PLR2004
            start, end, step = (
                generate_expr(args[0], symbols, 0, collected),
                generate_expr(args[1], symbols, 0, collected),
                "1",
            )
        elif len(args) == 3:  # noqa: PLR2004
            start, end, step = (
                generate_expr(args[0], symbols, 0, collected),
                generate_expr(args[1], symbols, 0, collected),
                generate_expr(args[2], symbols, 0, collected),
            )
        else:
            raise TranspilerError("Range function must have 1 to 3 arguments")
        code.append(
            f"{indent}for (int {target} = {start}; {target} < {end}; "
            f"{target} += {step}) {{"
        )
        body_symbols = symbols.copy()
        body_symbols[target] = "int"
        # Type is preserved when copying the symbols dictionary
        for line in generate_body(stmt.body, body_symbols, collected):
            code.append(f"{indent}    {line}")
        code.append(f"{indent}}}")
    else:
        raise TranspilerError("Only list and range-based for loops are supported")
    return code


def generate_while_loop(
    stmt: ast.While,
    symbols: dict[str, str | None],
    indent: str,
    collected: CollectedInfo,
) -> list[str]:
    """Generate GLSL code for a while loop.

    Args:
        stmt: AST while loop node
        symbols: Dictionary of variable names to their types
        indent: Indentation string
        collected: Information about functions, structs, and globals

    Returns:
        List of generated GLSL code lines for the while loop
    """
    code = []

    condition = generate_expr(stmt.test, symbols, 0, collected)
    # Type is preserved when copying the symbols dictionary
    body_code = generate_body(stmt.body, symbols.copy(), collected)
    inner_lines = [f"{indent}    {line}" for line in body_code]

    code.append(f"{indent}while ({condition}) {{")
    code.extend(inner_lines)
    code.append(f"{indent}}}")
    return code


def generate_if_statement(
    stmt: ast.If, symbols: dict[str, str | None], indent: str, collected: CollectedInfo
) -> list[str]:
    """Generate GLSL code for an if statement.

    Args:
        stmt: AST if statement node
        symbols: Dictionary of variable names to their types
        indent: Indentation string
        collected: Information about functions, structs, and globals

    Returns:
        List of generated GLSL code lines for the if statement
    """
    code = []

    condition = generate_expr(stmt.test, symbols, 0, collected)
    # Type is preserved when copying the symbols dictionary
    body_code = generate_body(stmt.body, symbols.copy(), collected)
    inner_lines = [f"{indent}    {line}" for line in body_code]

    code.append(f"{indent}if ({condition}) {{")
    code.extend(inner_lines)

    if stmt.orelse:
        # Type is preserved when copying the symbols dictionary
        else_code = generate_body(stmt.orelse, symbols.copy(), collected)
        else_lines = [f"{indent}    {line}" for line in else_code]
        code.append(f"{indent}}} else {{")
        code.extend(else_lines)

    code.append(f"{indent}}}")
    return code


def generate_return_statement(
    stmt: ast.Return,
    symbols: dict[str, str | None],
    indent: str,
    collected: CollectedInfo,
) -> str:
    """Generate GLSL code for a return statement.

    Args:
        stmt: AST return statement node
        symbols: Dictionary of variable names to their types
        indent: Indentation string
        collected: Information about functions, structs, and globals

    Returns:
        Generated GLSL code for the return statement
    """
    if stmt.value is None:
        return f"{indent}return;"
    expr = generate_expr(stmt.value, symbols, 0, collected)
    return f"{indent}return {expr};"


# Dictionary mapping AST statement types to handler functions
def _handle_break_stmt(
    stmt: ast.Break,
    symbols: dict[str, str | None],
    indent: str,
    collected: CollectedInfo,
) -> list[str]:
    """Handle break statements."""
    return [f"{indent}break;"]


def _handle_pass_stmt(
    stmt: ast.Pass,
    symbols: dict[str, str | None],
    indent: str,
    collected: CollectedInfo,
) -> list[str]:
    """Handle pass statements (no-op)."""
    return [f"{indent}// Pass statement (no-op)"]


def _handle_expr_stmt(
    stmt: ast.Expr,
    symbols: dict[str, str | None],
    indent: str,
    collected: CollectedInfo,
) -> list[str]:
    """Handle expression statements (ignored in GLSL)."""
    return []


# Type aliases for handler functions
T = TypeVar("T", bound=ast.AST)
StmtHandler = Callable[[Any, dict[str, str | None], str, CollectedInfo], list[str]]
StringStmtHandler = Callable[[Any, dict[str, str | None], str, CollectedInfo], str]


# Using functions rather than lambdas for better readability
def _append_handler(func: StringStmtHandler) -> StmtHandler:
    """Create a handler that converts a single string result to a list.

    Args:
        func: Function that generates a single string of code

    Returns:
        Function that wraps the result in a list
    """
    return lambda stmt, symbols, indent, collected: [
        func(stmt, symbols, indent, collected)
    ]


def _identity_handler(func: StmtHandler) -> StmtHandler:
    """Create a handler that returns the result directly (already a list).

    Args:
        func: Function that generates a list of strings

    Returns:
        The same function (identity operation)
    """
    return func


_STMT_GENERATORS = {
    ast.Assign: _append_handler(generate_assignment),
    ast.AnnAssign: _append_handler(generate_annotated_assignment),
    ast.AugAssign: _append_handler(generate_augmented_assignment),
    ast.For: _identity_handler(generate_for_loop),
    ast.While: _identity_handler(generate_while_loop),
    ast.If: _identity_handler(generate_if_statement),
    ast.Return: _append_handler(generate_return_statement),
    ast.Break: _handle_break_stmt,
    ast.Pass: _handle_pass_stmt,
    ast.Expr: _handle_expr_stmt,
}


def generate_body(
    body: list[ast.stmt], symbols: dict[str, str | None], collected: CollectedInfo
) -> list[str]:
    """Generate GLSL code for a function body.

    Args:
        body: List of AST nodes representing statements in the function body
        symbols: Dictionary of variable names to their types
        collected: Information about functions, structs, and globals

    Returns:
        List of generated GLSL code lines for the function body

    Raises:
        TranspilerError: If unsupported statements are encountered
    """
    code = []
    indent = ""

    # Check for shader functions with only a pass statement in the top-level context
    if len(body) == 1 and isinstance(body[0], ast.Pass):
        return ["// Pass statement (no-op)"]

    for stmt in body:
        stmt_type = type(stmt)
        generator = _STMT_GENERATORS.get(stmt_type)
        if generator:
            # Use cast to help mypy understand the type
            handler = cast(StmtHandler, generator)
            code.extend(handler(stmt, symbols, indent, collected))
        else:
            raise TranspilerError(f"Unsupported statement: {stmt_type.__name__}")

    return code
