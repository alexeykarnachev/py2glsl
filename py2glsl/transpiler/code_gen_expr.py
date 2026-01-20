"""GLSL code generation for expressions."""

import ast

from py2glsl.transpiler.constants import BUILTIN_FUNCTIONS, OPERATOR_PRECEDENCE
from py2glsl.transpiler.errors import TranspilerError
from py2glsl.transpiler.models import CollectedInfo

Symbols = dict[str, str | None]


def generate_name_expr(node: ast.Name) -> str:
    """Generate GLSL variable reference."""
    return node.id


def generate_constant_expr(node: ast.Constant) -> str:
    """Generate GLSL literal (bool, int, float)."""
    if isinstance(node.value, bool):
        return "true" if node.value else "false"
    if isinstance(node.value, int | float):
        return str(node.value)
    raise TranspilerError(f"Unsupported constant type: {type(node.value).__name__}")


def generate_binary_op_expr(
    node: ast.BinOp,
    symbols: Symbols,
    parent_precedence: int,
    collected: CollectedInfo,
) -> str:
    """Generate GLSL binary operation (+, -, *, /, **)."""
    # Power operator -> pow() function
    if isinstance(node.op, ast.Pow):
        left = generate_expr(node.left, symbols, 0, collected)
        right = generate_expr(node.right, symbols, 0, collected)
        return f"pow({left}, {right})"

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
    symbols: Symbols,
    parent_precedence: int,
    collected: CollectedInfo,
) -> str:
    """Generate GLSL comparison (<, >, <=, >=, ==, !=)."""
    if len(node.ops) != 1 or len(node.comparators) != 1:
        raise TranspilerError("Multiple comparisons not supported")

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


def generate_bool_op_expr(
    node: ast.BoolOp,
    symbols: Symbols,
    parent_precedence: int,
    collected: CollectedInfo,
) -> str:
    """Generate GLSL boolean operation (&&, ||)."""
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
    symbols: Symbols,
    collected: CollectedInfo,
) -> str:
    """Generate GLSL attribute access (struct field or vector swizzle)."""
    value = generate_expr(node.value, symbols, OPERATOR_PRECEDENCE["member"], collected)
    return f"{value}.{node.attr}"


def generate_if_expr(
    node: ast.IfExp,
    symbols: Symbols,
    parent_precedence: int,
    collected: CollectedInfo,
) -> str:
    """Generate GLSL ternary expression (cond ? a : b)."""
    precedence = OPERATOR_PRECEDENCE["?"]
    condition = generate_expr(node.test, symbols, 0, collected)
    true_expr = generate_expr(node.body, symbols, 0, collected)
    false_expr = generate_expr(node.orelse, symbols, 0, collected)
    expr = f"{condition} ? {true_expr} : {false_expr}"
    return f"({expr})" if precedence < parent_precedence else expr


def generate_struct_constructor(
    struct_name: str,
    node: ast.Call,
    symbols: Symbols,
    collected: CollectedInfo,
) -> str:
    """Generate GLSL struct constructor call."""
    struct_def = collected.structs[struct_name]
    field_map = {f.name: i for i, f in enumerate(struct_def.fields)}

    # Keyword arguments: MyStruct(x=1, y=2)
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
            f.name
            for f in struct_def.fields
            if f.default_value is None and f.name not in provided_fields
        ]
        if missing_fields:
            raise TranspilerError(
                f"Missing required fields in struct {struct_name}: "
                f"{', '.join(missing_fields)}"
            )

        for i, field in enumerate(struct_def.fields):
            if not values[i]:
                values[i] = field.default_value if field.default_value else "0.0"

        return f"{struct_name}({', '.join(values)})"

    # Positional arguments: MyStruct(1, 2)
    if node.args:
        if len(node.args) != len(struct_def.fields):
            raise TranspilerError(
                f"Wrong number of arguments for struct {struct_name}: "
                f"expected {len(struct_def.fields)}, got {len(node.args)}"
            )
        args = [generate_expr(arg, symbols, 0, collected) for arg in node.args]
        return f"{struct_name}({', '.join(args)})"

    raise TranspilerError(f"Struct '{struct_name}' initialization requires arguments")


def generate_call_expr(
    node: ast.Call, symbols: Symbols, collected: CollectedInfo
) -> str:
    """Generate GLSL function or struct constructor call."""
    func_name = (
        node.func.id
        if isinstance(node.func, ast.Name)
        else generate_expr(node.func, symbols, 0, collected)
    )

    # Struct constructor
    if func_name in collected.structs:
        return generate_struct_constructor(func_name, node, symbols, collected)

    # Built-in or user-defined function
    if func_name in collected.functions or func_name in BUILTIN_FUNCTIONS:
        args = [generate_expr(arg, symbols, 0, collected) for arg in node.args]
        return f"{func_name}({', '.join(args)})"

    raise TranspilerError(f"Unknown function call: {func_name}")


def generate_unary_op_expr(
    node: ast.UnaryOp,
    symbols: Symbols,
    parent_precedence: int,
    collected: CollectedInfo,
) -> str:
    """Generate GLSL unary operation (-, !)."""
    op_map = {ast.USub: "-", ast.Not: "!"}
    op = op_map.get(type(node.op))
    if not op:
        raise TranspilerError(f"Unsupported unary op: {type(node.op).__name__}")

    precedence = OPERATOR_PRECEDENCE["unary"]
    operand = generate_expr(node.operand, symbols, precedence, collected)
    expr = f"{op}{operand}"
    return f"({expr})" if precedence < parent_precedence else expr


def generate_subscript_expr(
    node: ast.Subscript,
    symbols: Symbols,
    collected: CollectedInfo,
) -> str:
    """Generate GLSL array/matrix indexing (arr[i])."""
    value = generate_expr(node.value, symbols, OPERATOR_PRECEDENCE["member"], collected)
    index = (
        str(node.slice.value)
        if isinstance(node.slice, ast.Constant)
        else generate_expr(node.slice, symbols, 0, collected)
    )
    return f"{value}[{index}]"


def generate_expr(
    node: ast.AST,
    symbols: Symbols,
    parent_precedence: int,
    collected: CollectedInfo,
) -> str:
    """Generate GLSL code for an expression."""
    match node:
        case ast.Name():
            return generate_name_expr(node)
        case ast.Constant():
            return generate_constant_expr(node)
        case ast.BinOp():
            return generate_binary_op_expr(node, symbols, parent_precedence, collected)
        case ast.Compare():
            return generate_compare_expr(node, symbols, parent_precedence, collected)
        case ast.BoolOp():
            return generate_bool_op_expr(node, symbols, parent_precedence, collected)
        case ast.Call():
            return generate_call_expr(node, symbols, collected)
        case ast.Attribute():
            return generate_attribute_expr(node, symbols, collected)
        case ast.IfExp():
            return generate_if_expr(node, symbols, parent_precedence, collected)
        case ast.UnaryOp():
            return generate_unary_op_expr(node, symbols, parent_precedence, collected)
        case ast.Subscript():
            return generate_subscript_expr(node, symbols, collected)
        case _:
            raise TranspilerError(f"Unsupported expression: {type(node).__name__}")
