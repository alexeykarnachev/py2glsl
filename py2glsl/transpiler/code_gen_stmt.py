"""
GLSL code generation for statements.

This module contains functions for generating GLSL code from Python AST statements,
including assignments, loops, conditionals, and return statements.
"""

import ast
from typing import cast

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
        target_str = generate_attribute_expr(target, symbols, collected)
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


def _generate_list_iteration_loop(
    stmt: ast.For,
    symbols: dict[str, str | None],
    indent: str,
    collected: CollectedInfo,
    list_name: str,
) -> list[str]:
    """Generate GLSL code for a list iteration for loop.

    Args:
        stmt: AST for loop node
        symbols: Dictionary of variable names to their types
        indent: Indentation string
        collected: Information about functions, structs, and globals
        list_name: Name of the list being iterated

    Returns:
        List of generated GLSL code lines for the list iteration loop

    Raises:
        TranspilerError: If the list type is unsupported or target is invalid
    """
    code = []
    list_type = symbols.get(list_name, "unknown")

    if (
        not list_type
        or not isinstance(list_type, str)
        or not list_type.startswith("list[")
    ):
        raise TranspilerError(f"Unsupported iterable: {list_type}")

    # Extract type, e.g., "vec3" from "list[vec3]"
    item_type = list_type[5:-1]
    index_var = f"i_{list_name}"  # Unique index name
    size_var = f"{list_name}_size"

    # Extract the target variable name
    if not isinstance(stmt.target, ast.Name):
        raise TranspilerError("For loop target must be a variable name")

    target_name = stmt.target.id

    # Generate loop header
    code.append(
        f"{indent}for (int {index_var} = 0; {index_var} < {size_var}; ++{index_var}) {{"
    )

    # Generate item assignment
    code.append(f"{indent}    {item_type} {target_name} = {list_name}[{index_var}];")

    # Generate loop body
    body_symbols = symbols.copy()
    body_symbols[target_name] = item_type
    for line in generate_body(stmt.body, body_symbols, collected):
        code.append(f"{indent}    {line}")

    # Close the loop
    code.append(f"{indent}}}")

    return code


def _parse_range_arguments(
    args: list[ast.expr], symbols: dict[str, str | None], collected: CollectedInfo
) -> tuple[str, str, str]:
    """Parse the arguments to a range() call.

    Args:
        args: List of AST nodes representing range arguments
        symbols: Dictionary of variable names to their types
        collected: Information about functions, structs, and globals

    Returns:
        Tuple of (start, end, step) values as strings

    Raises:
        TranspilerError: If the range has an invalid number of arguments
    """
    # Constants for range argument counts
    single_arg = 1
    start_end_args = 2
    full_range_args = 3

    if len(args) == single_arg:
        return "0", generate_expr(args[0], symbols, 0, collected), "1"
    elif len(args) == start_end_args:
        return (
            generate_expr(args[0], symbols, 0, collected),
            generate_expr(args[1], symbols, 0, collected),
            "1",
        )
    elif len(args) == full_range_args:
        return (
            generate_expr(args[0], symbols, 0, collected),
            generate_expr(args[1], symbols, 0, collected),
            generate_expr(args[2], symbols, 0, collected),
        )
    else:
        raise TranspilerError("Range function must have 1 to 3 arguments")


def _generate_range_iteration_loop(
    stmt: ast.For,
    symbols: dict[str, str | None],
    indent: str,
    collected: CollectedInfo,
    range_args: list[ast.expr],
) -> list[str]:
    """Generate GLSL code for a range-based for loop.

    Args:
        stmt: AST for loop node
        symbols: Dictionary of variable names to their types
        indent: Indentation string
        collected: Information about functions, structs, and globals
        range_args: Arguments to the range() call

    Returns:
        List of generated GLSL code lines for the range-based loop

    Raises:
        TranspilerError: If the target is invalid
    """
    code = []

    # Extract the target variable name
    if not isinstance(stmt.target, ast.Name):
        raise TranspilerError("For loop target must be a variable name")

    target = stmt.target.id

    # Parse range arguments
    start, end, step = _parse_range_arguments(range_args, symbols, collected)

    # Generate loop header
    code.append(
        f"{indent}for (int {target} = {start}; {target} < {end}; {target} += {step}) {{"
    )

    # Generate loop body
    body_symbols = symbols.copy()
    body_symbols[target] = "int"
    for line in generate_body(stmt.body, body_symbols, collected):
        code.append(f"{indent}    {line}")

    # Close the loop
    code.append(f"{indent}}}")

    return code


def _is_range_call(node: ast.AST) -> bool:
    """Check if a node is a call to the range() function.

    Args:
        node: AST node to check

    Returns:
        True if the node is a call to range(), False otherwise
    """
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "range"
    )


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
    # List iteration, e.g., `for item in some_list`
    if isinstance(stmt.iter, ast.Name):
        list_name = stmt.iter.id
        return _generate_list_iteration_loop(
            stmt, symbols, indent, collected, list_name
        )

    # Range-based iteration, e.g., `for i in range(10)`
    elif _is_range_call(stmt.iter):
        range_args = cast(ast.Call, stmt.iter).args
        return _generate_range_iteration_loop(
            stmt, symbols, indent, collected, range_args
        )

    # Unsupported iteration type
    else:
        raise TranspilerError("Only list and range-based for loops are supported")


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


# Simplified code structure - removed handlers and mapping dictionary


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
        if isinstance(stmt, ast.Assign):
            code.append(generate_assignment(stmt, symbols, indent, collected))
        elif isinstance(stmt, ast.AnnAssign):
            code.append(generate_annotated_assignment(stmt, symbols, indent, collected))
        elif isinstance(stmt, ast.AugAssign):
            code.append(generate_augmented_assignment(stmt, symbols, indent, collected))
        elif isinstance(stmt, ast.For):
            code.extend(generate_for_loop(stmt, symbols, indent, collected))
        elif isinstance(stmt, ast.While):
            code.extend(generate_while_loop(stmt, symbols, indent, collected))
        elif isinstance(stmt, ast.If):
            code.extend(generate_if_statement(stmt, symbols, indent, collected))
        elif isinstance(stmt, ast.Return):
            code.append(generate_return_statement(stmt, symbols, indent, collected))
        elif isinstance(stmt, ast.Break):
            code.append(f"{indent}break;")
        elif isinstance(stmt, ast.Continue):
            code.append(f"{indent}continue;")
        elif isinstance(stmt, ast.Pass):
            code.append(f"{indent}// Pass statement (no-op)")
        elif isinstance(stmt, ast.Expr):
            # Expression statements are ignored in GLSL
            pass
        else:
            raise TranspilerError(f"Unsupported statement: {type(stmt).__name__}")

    return code
