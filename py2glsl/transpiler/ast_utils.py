"""AST manipulation utilities for the transpiler."""

import ast


def substitute_name(node: ast.expr, name: str, value: int) -> ast.expr:
    """Substitute a name with a constant value in an AST expression.

    Used for unrolling list comprehensions at compile time.

    Args:
        node: The AST expression to transform
        name: The variable name to substitute
        value: The integer constant to substitute in

    Returns:
        A new AST expression with the substitution applied
    """
    if isinstance(node, ast.Name) and node.id == name:
        return ast.Constant(value=value)
    if isinstance(node, ast.BinOp):
        return ast.BinOp(
            left=substitute_name(node.left, name, value),
            op=node.op,
            right=substitute_name(node.right, name, value),
        )
    if isinstance(node, ast.UnaryOp):
        return ast.UnaryOp(
            op=node.op,
            operand=substitute_name(node.operand, name, value),
        )
    if isinstance(node, ast.Call):
        return ast.Call(
            func=node.func,
            args=[substitute_name(a, name, value) for a in node.args],
            keywords=[
                ast.keyword(
                    arg=kw.arg,
                    value=substitute_name(kw.value, name, value),
                )
                for kw in node.keywords
            ],
        )
    if isinstance(node, ast.IfExp):
        return ast.IfExp(
            test=substitute_name(node.test, name, value),
            body=substitute_name(node.body, name, value),
            orelse=substitute_name(node.orelse, name, value),
        )
    if isinstance(node, ast.Compare):
        return ast.Compare(
            left=substitute_name(node.left, name, value),
            ops=node.ops,
            comparators=[substitute_name(c, name, value) for c in node.comparators],
        )
    if isinstance(node, ast.Subscript):
        return ast.Subscript(
            value=substitute_name(node.value, name, value),
            slice=substitute_name(node.slice, name, value),
            ctx=node.ctx,
        )
    if isinstance(node, ast.Attribute):
        return ast.Attribute(
            value=substitute_name(node.value, name, value),
            attr=node.attr,
            ctx=node.ctx,
        )
    if isinstance(node, ast.Tuple):
        return ast.Tuple(
            elts=[substitute_name(e, name, value) for e in node.elts],
            ctx=node.ctx,
        )
    if isinstance(node, ast.List):
        return ast.List(
            elts=[substitute_name(e, name, value) for e in node.elts],
            ctx=node.ctx,
        )
    # For constants and other nodes, return as-is
    return node


def eval_constant(
    node: ast.expr, globals_dict: dict[str, tuple[str, str]]
) -> int | None:
    """Evaluate an AST node as a compile-time constant integer.

    Supports:
    - Integer literals
    - References to global int constants
    - Unary minus
    - Binary operations (+, -, *, //)

    Args:
        node: The AST expression to evaluate
        globals_dict: Dict mapping name -> (type_str, value_str) for global constants

    Returns:
        The integer value if constant, None otherwise
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.Name) and node.id in globals_dict:
        type_str, value_str = globals_dict[node.id]
        if type_str == "int":
            return int(value_str)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = eval_constant(node.operand, globals_dict)
        if inner is not None:
            return -inner
    if isinstance(node, ast.BinOp):
        left = eval_constant(node.left, globals_dict)
        right = eval_constant(node.right, globals_dict)
        if left is not None and right is not None:
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
    return None


# Type alias for constant evaluation results
ConstValue = float | int | bool


def eval_const_expr(
    node: ast.expr, globals_dict: dict[str, tuple[str, str]]
) -> tuple[str, ConstValue] | None:
    """Evaluate a constant expression returning type and value.

    More general than eval_constant - handles floats, ints, and bools.
    Used for collecting module-level constants that may reference each other.

    Args:
        node: The AST expression to evaluate
        globals_dict: Dict mapping name -> (type_str, value_str) for global constants

    Returns:
        Tuple of (type_str, value) or None if not evaluable
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return ("bool", node.value)
        if isinstance(node.value, float):
            return ("float", node.value)
        if isinstance(node.value, int):
            return ("int", node.value)
        return None

    if isinstance(node, ast.Name):
        # Reference to another constant
        if node.id in globals_dict:
            type_str, value_str = globals_dict[node.id]
            try:
                if type_str == "int":
                    return (type_str, int(value_str))
                if type_str == "float":
                    return (type_str, float(value_str))
                if type_str == "bool":
                    return (type_str, value_str == "true")
            except ValueError:
                pass
        return None

    if isinstance(node, ast.BinOp):
        left_result = eval_const_expr(node.left, globals_dict)
        right_result = eval_const_expr(node.right, globals_dict)
        if left_result is None or right_result is None:
            return None

        left_type, left_val = left_result
        right_type, right_val = right_result

        # Determine result type (float if either is float)
        result_type = "float" if "float" in (left_type, right_type) else left_type

        # Evaluate the operation
        op = node.op
        result_val: ConstValue
        if isinstance(op, ast.Add):
            result_val = left_val + right_val
        elif isinstance(op, ast.Sub):
            result_val = left_val - right_val
        elif isinstance(op, ast.Mult):
            result_val = left_val * right_val
        elif isinstance(op, ast.Div):
            result_val = left_val / right_val
        elif isinstance(op, ast.FloorDiv):
            result_val = left_val // right_val
        elif isinstance(op, ast.Mod):
            result_val = left_val % right_val
        elif isinstance(op, ast.Pow):
            result_val = left_val**right_val
        else:
            return None

        return (result_type, result_val)

    if isinstance(node, ast.UnaryOp):
        operand_result = eval_const_expr(node.operand, globals_dict)
        if operand_result is None:
            return None

        operand_type, operand_val = operand_result
        if isinstance(node.op, ast.USub):
            return (operand_type, -operand_val)
        if isinstance(node.op, ast.UAdd):
            return (operand_type, operand_val)

    return None


def infer_literal_glsl_type(value: ast.Constant) -> tuple[str, str] | None:
    """Infer GLSL type and value string from a Python literal.

    Args:
        value: An AST Constant node

    Returns:
        Tuple of (glsl_type, value_string) or None if not a supported literal
    """
    if isinstance(value.value, bool):
        return ("bool", "true" if value.value else "false")
    if isinstance(value.value, float):
        return ("float", str(value.value))
    if isinstance(value.value, int):
        return ("int", str(value.value))
    return None
