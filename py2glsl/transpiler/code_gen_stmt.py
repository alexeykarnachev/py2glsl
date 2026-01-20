"""
GLSL code generation for statements.

This module contains functions for generating GLSL code from Python AST statements,
including assignments, loops, conditionals, and return statements.
"""

import ast
from typing import cast

from py2glsl.transpiler.ast_parser import get_annotation_type
from py2glsl.transpiler.code_gen_expr import generate_attribute_expr, generate_expr
from py2glsl.transpiler.errors import TranspilerError
from py2glsl.transpiler.models import CollectedInfo
from py2glsl.transpiler.type_checker import get_expr_type

# Type aliases for common parameter types
Symbols = dict[str, str | None]


def _indent_lines(lines: list[str]) -> list[str]:
    """Add one level of indentation to each line."""
    return [f"    {line}" for line in lines]


def generate_assignment(
    node: ast.Assign,
    symbols: Symbols,
    collected: CollectedInfo,
) -> str:
    """Generate GLSL code for an assignment statement."""
    if len(node.targets) != 1:
        raise TranspilerError("Multiple assignment targets not supported")
    target = node.targets[0]

    if isinstance(node.value, ast.List):
        return generate_list_declaration(node, symbols, collected)

    value_str = generate_expr(node.value, symbols, 0, collected)

    if isinstance(target, ast.Name):
        target_name = target.id
        if target_name not in symbols:
            inferred_type = get_expr_type(node.value, symbols, collected)
            symbols[target_name] = inferred_type
            return f"{inferred_type} {target_name} = {value_str};"
        return f"{target_name} = {value_str};"

    if isinstance(target, ast.Attribute):
        target_str = generate_attribute_expr(target, symbols, collected)
        return f"{target_str} = {value_str};"

    raise TranspilerError(f"Unsupported assignment target: {type(target).__name__}")


def generate_list_declaration(
    node: ast.Assign,
    symbols: Symbols,
    collected: CollectedInfo,
) -> str:
    """Generate GLSL array declaration from a Python list assignment."""
    if not isinstance(node.value, ast.List):
        raise TranspilerError(
            "Only list assignments supported in generate_list_declaration"
        )

    elements = node.value.elts
    if not isinstance(node.targets[0], ast.Name):
        raise TranspilerError("List assignment target must be a variable name")

    list_name = node.targets[0].id

    if not elements:
        # Empty list: assume type from context or default to vec3
        symbol_type = symbols.get(list_name)
        if symbol_type and symbol_type.startswith("list["):
            list_type = symbol_type[5:-1]  # Extract type from "list[type]"
        else:
            list_type = "vec3"
        symbols[list_name] = f"list[{list_type}]"
        collected.globals[f"{list_name}_size"] = ("int", "0")
        return f"{list_type} {list_name}[0];"

    # Non-empty list: infer type from first element and validate all elements match
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
    return f"{list_type} {list_name}[{size}] = {list_type}[{size}]({array_init});"


def generate_annotated_assignment(
    stmt: ast.AnnAssign,
    symbols: Symbols,
    collected: CollectedInfo,
) -> str:
    """Generate GLSL code for an annotated assignment (e.g., x: float = 1.0)."""
    if not isinstance(stmt.target, ast.Name):
        raise TranspilerError(
            f"Unsupported annotated assignment target: {type(stmt.target).__name__}"
        )

    target = stmt.target.id
    expr_type = get_annotation_type(stmt.annotation)
    symbols[target] = expr_type

    if stmt.value:
        expr = generate_expr(stmt.value, symbols, 0, collected)
        return f"{expr_type} {target} = {expr};"
    return f"{expr_type} {target};"


def generate_augmented_assignment(
    stmt: ast.AugAssign,
    symbols: Symbols,
    collected: CollectedInfo,
) -> str:
    """Generate GLSL code for an augmented assignment (e.g., x += 1)."""
    target = generate_expr(stmt.target, symbols, 0, collected)
    value = generate_expr(stmt.value, symbols, 0, collected)

    op_map = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
    op = op_map.get(type(stmt.op))

    if not op:
        raise TranspilerError(
            f"Unsupported augmented operator: {type(stmt.op).__name__}"
        )
    return f"{target} = {target} {op} {value};"


def _get_loop_target_name(stmt: ast.For) -> str:
    """Extract and validate the loop target variable name."""
    if not isinstance(stmt.target, ast.Name):
        raise TranspilerError("For loop target must be a variable name")
    return stmt.target.id


def _generate_list_iteration_loop(
    stmt: ast.For,
    symbols: Symbols,
    collected: CollectedInfo,
    list_name: str,
) -> list[str]:
    """Generate GLSL for loop iterating over a list (for item in list)."""
    list_type = symbols.get(list_name, "unknown")

    if (
        not list_type
        or not isinstance(list_type, str)
        or not list_type.startswith("list[")
    ):
        raise TranspilerError(f"Unsupported iterable: {list_type}")

    item_type = list_type[5:-1]  # Extract "vec3" from "list[vec3]"
    index_var = f"i_{list_name}"
    size_var = f"{list_name}_size"
    target_name = _get_loop_target_name(stmt)

    # Generate loop body with target in scope
    body_symbols = symbols.copy()
    body_symbols[target_name] = item_type
    body_lines = generate_body(stmt.body, body_symbols, collected)

    return [
        f"for (int {index_var} = 0; {index_var} < {size_var}; ++{index_var}) {{",
        f"    {item_type} {target_name} = {list_name}[{index_var}];",
        *_indent_lines(body_lines),
        "}",
    ]


def _parse_range_arguments(
    args: list[ast.expr], symbols: Symbols, collected: CollectedInfo
) -> tuple[str, str, str]:
    """Parse range() arguments and return (start, end, step) as GLSL strings."""

    def gen(node: ast.expr) -> str:
        return generate_expr(node, symbols, 0, collected)

    if len(args) == 1:
        return "0", gen(args[0]), "1"
    if len(args) == 2:
        return gen(args[0]), gen(args[1]), "1"
    if len(args) == 3:
        return gen(args[0]), gen(args[1]), gen(args[2])
    raise TranspilerError("Range function must have 1 to 3 arguments")


def _generate_range_iteration_loop(
    stmt: ast.For,
    symbols: Symbols,
    collected: CollectedInfo,
    range_args: list[ast.expr],
) -> list[str]:
    """Generate GLSL for loop from range() (for i in range(n))."""
    target = _get_loop_target_name(stmt)
    start, end, step = _parse_range_arguments(range_args, symbols, collected)

    body_symbols = symbols.copy()
    body_symbols[target] = "int"
    body_lines = generate_body(stmt.body, body_symbols, collected)

    return [
        f"for (int {target} = {start}; {target} < {end}; {target} += {step}) {{",
        *_indent_lines(body_lines),
        "}",
    ]


def _is_range_call(node: ast.AST) -> bool:
    """Check if a node is a call to range()."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "range"
    )


def generate_for_loop(
    stmt: ast.For, symbols: Symbols, collected: CollectedInfo
) -> list[str]:
    """Generate GLSL for loop from list iteration or range()."""
    # List iteration: `for item in some_list`
    if isinstance(stmt.iter, ast.Name):
        return _generate_list_iteration_loop(stmt, symbols, collected, stmt.iter.id)

    # Range-based iteration: `for i in range(10)`
    if _is_range_call(stmt.iter):
        return _generate_range_iteration_loop(
            stmt, symbols, collected, cast(ast.Call, stmt.iter).args
        )

    raise TranspilerError("Only list and range-based for loops are supported")


def generate_while_loop(
    stmt: ast.While,
    symbols: Symbols,
    collected: CollectedInfo,
) -> list[str]:
    """Generate GLSL while loop."""
    condition = generate_expr(stmt.test, symbols, 0, collected)
    body_lines = generate_body(stmt.body, symbols.copy(), collected)

    return [
        f"while ({condition}) {{",
        *_indent_lines(body_lines),
        "}",
    ]


def _find_assignments_in_body(body: list[ast.stmt]) -> set[str]:
    """Find all variable names assigned in a body of statements."""
    assigned_vars = set()

    for stmt in body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    assigned_vars.add(target.id)
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            assigned_vars.add(stmt.target.id)
        elif isinstance(stmt, ast.If):
            # Recursively check nested if statements
            assigned_vars.update(_find_assignments_in_body(stmt.body))
            if stmt.orelse:
                assigned_vars.update(_find_assignments_in_body(stmt.orelse))

    return assigned_vars


def _collect_all_branches(stmt: ast.If) -> list[list[ast.stmt]]:
    """Collect all branch bodies (if, elif, else) from an if statement."""
    branches = [stmt.body]

    # Check if there's an else/elif
    current = stmt
    while current.orelse:
        # If orelse is a single If statement, it's an elif
        if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
            current = current.orelse[0]
            branches.append(current.body)
        else:
            # It's an else clause
            branches.append(current.orelse)
            break

    return branches


def generate_if_statement(
    stmt: ast.If, symbols: Symbols, collected: CollectedInfo
) -> list[str]:
    """Generate GLSL if/else statement with variable hoisting."""
    code: list[str] = []

    # Hoist variables that are assigned in all branches
    branches = _collect_all_branches(stmt)
    if len(branches) > 1:
        assigned_in_branches = [_find_assignments_in_body(b) for b in branches]
        common_vars = (
            set.intersection(*assigned_in_branches) if assigned_in_branches else set()
        )
        vars_to_hoist = {var for var in common_vars if var not in symbols}

        for var in vars_to_hoist:
            var_type = _infer_hoisted_var_type(var, branches[0], symbols, collected)
            if var_type:
                code.append(f"{var_type} {var};")
                symbols[var] = var_type

    condition = generate_expr(stmt.test, symbols, 0, collected)
    body_lines = generate_body(stmt.body, symbols.copy(), collected)

    code.append(f"if ({condition}) {{")
    code.extend(_indent_lines(body_lines))

    if stmt.orelse:
        else_lines = generate_body(stmt.orelse, symbols.copy(), collected)
        code.append("} else {")
        code.extend(_indent_lines(else_lines))

    code.append("}")
    return code


def _infer_hoisted_var_type(
    var_name: str,
    body: list[ast.stmt],
    symbols: Symbols,
    collected: CollectedInfo,
) -> str | None:
    """Infer a variable's type from its first assignment in a body."""
    for stmt in body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == var_name:
                    return get_expr_type(stmt.value, symbols, collected)
        elif (
            isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and stmt.target.id == var_name
        ):
            if stmt.value:
                return get_expr_type(stmt.value, symbols, collected)
            # If no value, get type from annotation
            return get_annotation_type(stmt.annotation)

    return None


def generate_return_statement(
    stmt: ast.Return,
    symbols: Symbols,
    collected: CollectedInfo,
) -> str:
    """Generate GLSL return statement."""
    if stmt.value is None:
        return "return;"
    expr = generate_expr(stmt.value, symbols, 0, collected)
    return f"return {expr};"


def generate_body(
    body: list[ast.stmt], symbols: Symbols, collected: CollectedInfo
) -> list[str]:
    """Generate GLSL code for a function body."""
    if len(body) == 1 and isinstance(body[0], ast.Pass):
        return ["// Pass statement (no-op)"]

    code: list[str] = []

    for stmt in body:
        if isinstance(stmt, ast.Assign):
            code.append(generate_assignment(stmt, symbols, collected))
        elif isinstance(stmt, ast.AnnAssign):
            code.append(generate_annotated_assignment(stmt, symbols, collected))
        elif isinstance(stmt, ast.AugAssign):
            code.append(generate_augmented_assignment(stmt, symbols, collected))
        elif isinstance(stmt, ast.For):
            code.extend(generate_for_loop(stmt, symbols, collected))
        elif isinstance(stmt, ast.While):
            code.extend(generate_while_loop(stmt, symbols, collected))
        elif isinstance(stmt, ast.If):
            code.extend(generate_if_statement(stmt, symbols, collected))
        elif isinstance(stmt, ast.Return):
            code.append(generate_return_statement(stmt, symbols, collected))
        elif isinstance(stmt, ast.Break):
            code.append("break;")
        elif isinstance(stmt, ast.Continue):
            code.append("continue;")
        elif isinstance(stmt, ast.Pass):
            code.append("// Pass statement (no-op)")
        elif isinstance(stmt, ast.Expr):
            pass  # Expression statements are ignored in GLSL
        else:
            raise TranspilerError(f"Unsupported statement: {type(stmt).__name__}")

    return code
