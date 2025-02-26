"""
GLSL code generation for statements.

This module contains functions for generating GLSL code from Python AST statements,
including assignments, loops, conditionals, and return statements.
"""

import ast
from typing import Dict, List

from py2glsl.transpiler.code_gen_expr import generate_expr
from py2glsl.transpiler.errors import TranspilerError
from py2glsl.transpiler.models import CollectedInfo
from py2glsl.transpiler.type_checker import get_expr_type


def generate_assignment(
    stmt: ast.Assign, symbols: Dict[str, str], indent: str, collected: CollectedInfo
) -> str:
    """Generate GLSL code for a variable assignment.

    Args:
        stmt: AST assignment node
        symbols: Dictionary of variable names to their types
        indent: Indentation string
        collected: Information about functions, structs, and globals

    Returns:
        Generated GLSL code for the assignment

    Raises:
        TranspilerError: If the assignment target is not supported
    """
    target = stmt.targets[0]
    expr = generate_expr(stmt.value, symbols, 0, collected)

    if isinstance(target, ast.Name):
        target_name = target.id
        expr_type = get_expr_type(stmt.value, symbols, collected)
        if target_name not in symbols:
            symbols[target_name] = expr_type
            return f"{indent}{expr_type} {target_name} = {expr};"
        return f"{indent}{target_name} = {expr};"
    elif isinstance(target, ast.Attribute):
        target_expr = generate_expr(target, symbols, 0, collected)
        return f"{indent}{target_expr} = {expr};"
    raise TranspilerError(f"Unsupported assignment target: {type(target).__name__}")


def generate_annotated_assignment(
    stmt: ast.AnnAssign,
    symbols: Dict[str, str],
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
    stmt: ast.AugAssign, symbols: Dict[str, str], indent: str, collected: CollectedInfo
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
    stmt: ast.For,
    symbols: Dict[str, str],
    indent: str,
    collected: CollectedInfo,
) -> List[str]:
    """Generate GLSL code for a for loop.

    Args:
        stmt: AST for loop node
        symbols: Dictionary of variable names to their types
        indent: Indentation string
        collected: Information about functions, structs, and globals

    Returns:
        List of generated GLSL code lines for the for loop

    Raises:
        TranspilerError: If the loop is not a range-based for loop
    """
    code = []

    if (
        isinstance(stmt.iter, ast.Call)
        and isinstance(stmt.iter.func, ast.Name)
        and stmt.iter.func.id == "range"
    ):
        target = stmt.target.id
        args = stmt.iter.args

        start = "0" if len(args) == 1 else generate_expr(args[0], symbols, 0, collected)
        end = generate_expr(args[1 if len(args) > 1 else 0], symbols, 0, collected)
        step = "1" if len(args) <= 2 else generate_expr(args[2], symbols, 0, collected)

        symbols[target] = "int"
        body_symbols = symbols.copy()

        # Special handling for for loops with just a Pass statement
        if len(stmt.body) == 1 and isinstance(stmt.body[0], ast.Pass):
            inner_lines = [f"{indent}    // Pass statement (no-op)"]
        else:
            body_code = generate_body(stmt.body, body_symbols, collected)
            inner_lines = [f"{indent}    {line}" for line in body_code]

        code.append(
            f"{indent}for (int {target} = {start}; {target} < {end}; {target} += {step}) {{"
        )
        code.extend(inner_lines)
        code.append(f"{indent}}}")
    else:
        raise TranspilerError("Only range-based for loops are supported")
    return code


def generate_while_loop(
    stmt: ast.While, symbols: Dict[str, str], indent: str, collected: CollectedInfo
) -> List[str]:
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
    body_code = generate_body(stmt.body, symbols.copy(), collected)
    inner_lines = [f"{indent}    {line}" for line in body_code]

    code.append(f"{indent}while ({condition}) {{")
    code.extend(inner_lines)
    code.append(f"{indent}}}")
    return code


def generate_if_statement(
    stmt: ast.If, symbols: Dict[str, str], indent: str, collected: CollectedInfo
) -> List[str]:
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
    body_code = generate_body(stmt.body, symbols.copy(), collected)
    inner_lines = [f"{indent}    {line}" for line in body_code]

    code.append(f"{indent}if ({condition}) {{")
    code.extend(inner_lines)

    if stmt.orelse:
        else_code = generate_body(stmt.orelse, symbols.copy(), collected)
        else_lines = [f"{indent}    {line}" for line in else_code]
        code.append(f"{indent}}} else {{")
        code.extend(else_lines)

    code.append(f"{indent}}}")
    return code


def generate_return_statement(
    stmt: ast.Return, symbols: Dict[str, str], indent: str, collected: CollectedInfo
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
    expr = generate_expr(stmt.value, symbols, 0, collected)
    return f"{indent}return {expr};"


def generate_body(
    body: List[ast.AST], symbols: Dict[str, str], collected: CollectedInfo
) -> List[str]:
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

    # Keep this check at the top to reject shader functions with only a pass statement
    if len(body) == 1 and isinstance(body[0], ast.Pass):
        raise TranspilerError("Pass statements are not supported in GLSL")

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
        elif isinstance(stmt, ast.Pass):
            # For non-top-level Pass (e.g., within a loop in test_generate_for_loop),
            # This case is not reachable for top-level Pass statements due to the check above
            code.append(f"{indent}// Pass statement (no-op)")
        elif isinstance(stmt, ast.Expr):
            # Ignore expression statements
            pass
        else:
            raise TranspilerError(f"Unsupported statement: {type(stmt).__name__}")

    return code
