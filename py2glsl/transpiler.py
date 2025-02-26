"""
GLSL Shader Transpiler for Python.

This module provides functionality to convert Python code into GLSL shader code for use with OpenGL.
It parses Python functions, dataclasses, and other constructs and generates equivalent GLSL code.

The main entry point is the `transpile` function, which takes Python code (as a string or callable)
and returns the equivalent GLSL code along with a set of used uniforms.

Example:
    ```python
    from py2glsl import transpile, vec2, vec4
    
    def shader(vs_uv: vec2, u_time: float) -> vec4:
        return vec4(vs_uv.x, vs_uv.y, 0.0, 1.0)
    
    glsl_code, uniforms = transpile(shader)
    print(glsl_code)
    ```
"""

import ast
import inspect
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from loguru import logger


class TranspilerError(Exception):
    """Exception raised for errors during shader code transpilation."""

    pass


@dataclass
class StructField:
    """Field definition in a GLSL struct.

    Attributes:
        name: Field name
        type_name: GLSL type of the field
        default_value: Optional default value as a string
    """

    name: str
    type_name: str
    default_value: Optional[str] = None


@dataclass
class StructDefinition:
    """Representation of a GLSL struct definition.

    Attributes:
        name: Name of the struct
        fields: List of field definitions
    """

    name: str
    fields: List[StructField]


@dataclass
class FunctionInfo:
    """Information about a function to be transpiled to GLSL.

    Attributes:
        name: Function name
        return_type: Return type or None if not specified
        param_types: List of parameter types
        node: AST node for the function
    """

    name: str
    return_type: Optional[str]
    param_types: List[Optional[str]]
    node: ast.FunctionDef


@dataclass
class CollectedInfo:
    """Information collected from Python code to be transpiled.

    Attributes:
        functions: Dictionary mapping function names to function information
        structs: Dictionary mapping struct names to struct definitions
        globals: Dictionary mapping global variable names to (type, value)
    """

    functions: Dict[str, FunctionInfo] = field(default_factory=dict)
    structs: Dict[str, StructDefinition] = field(default_factory=dict)
    globals: Dict[str, Tuple[str, str]] = field(default_factory=dict)


# Dictionary of built-in GLSL functions with return types and parameter types
BUILTIN_FUNCTIONS: Dict[str, Tuple[str, List[str]]] = {
    "sin": ("float", ["float"]),
    "cos": ("float", ["float"]),
    "tan": ("float", ["float"]),
    "abs": ("float", ["float"]),
    "length": ("float", ["vec2"]),
    "max": ("float", ["float", "float"]),
    "min": ("float", ["float", "float"]),
    "mix": ("vec3", ["vec3", "vec3", "float"]),
    "normalize": ("vec3", ["vec3"]),
    "cross": ("vec3", ["vec3", "vec3"]),
    "radians": ("float", ["float"]),
    "float": ("float", ["int"]),
    "vec2": ("vec2", ["float", "float"]),
    "vec3": ("vec3", ["float", "float", "float"]),
    "vec4": ("vec4", ["float", "float", "float", "float"]),
}

# Operator precedence for generating correct expressions
OPERATOR_PRECEDENCE: Dict[str, int] = {
    "=": 1,
    "||": 2,
    "&&": 3,
    "==": 4,
    "!=": 4,
    "<": 5,
    ">": 5,
    "<=": 5,
    ">=": 5,
    "+": 6,
    "-": 6,
    "*": 7,
    "/": 7,
    "%": 7,
    "unary": 8,
    "call": 9,
    "member": 10,
}


def get_annotation_type(annotation: Optional[ast.AST]) -> Optional[str]:
    """Extract the type name from an AST annotation node.

    Args:
        annotation: AST node representing a type annotation

    Returns:
        String representation of the type or None if no valid annotation
    """
    if annotation is None:
        return None
    if isinstance(annotation, ast.Name):
        return annotation.id
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        return annotation.value
    return None


def generate_simple_expr(node: ast.AST) -> str:
    """Generate GLSL code for simple expressions used in global constants or defaults.

    Args:
        node: AST node for a simple expression

    Returns:
        String representation of the expression in GLSL

    Raises:
        TranspilerError: If the expression is not supported for globals/defaults
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return str(node.value)
        elif isinstance(node.value, bool):
            return "true" if node.value else "false"
        elif isinstance(node.value, str):
            return node.value
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in {
            "vec2",
            "vec3",
            "vec4",
        }:
            args = [generate_simple_expr(arg) for arg in node.args]
            return f"{node.func.id}({', '.join(args)})"
    raise TranspilerError("Unsupported expression in global or default value")


def collect_info(tree: ast.AST) -> CollectedInfo:
    """Collect information about functions, structs, and globals from the AST.

    Args:
        tree: AST of the Python code to be transpiled

    Returns:
        CollectedInfo containing functions, structs, and global variables
    """
    collected = CollectedInfo()

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            param_types = [
                get_annotation_type(arg.annotation) for arg in node.args.args
            ]
            return_type = get_annotation_type(node.returns)
            collected.functions[node.name] = FunctionInfo(
                name=node.name,
                return_type=return_type,
                param_types=param_types,
                node=node,
            )
            logger.debug(
                f"Collected function: {node.name}, return_type: {return_type}, params: {param_types}"
            )
            self.generic_visit(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            is_dataclass = any(
                isinstance(d, ast.Name) and d.id == "dataclass"
                for d in node.decorator_list
            )
            if is_dataclass:
                fields = []
                for stmt in node.body:
                    if isinstance(stmt, ast.AnnAssign) and isinstance(
                        stmt.target, ast.Name
                    ):
                        field_type = get_annotation_type(stmt.annotation)
                        default_value = None
                        if stmt.value:
                            default_value = generate_simple_expr(stmt.value)
                        fields.append(
                            StructField(
                                name=stmt.target.id,
                                type_name=field_type,
                                default_value=default_value,
                            )
                        )
                collected.structs[node.name] = StructDefinition(
                    name=node.name, fields=fields
                )
                logger.debug(
                    f"Collected struct: {node.name}, fields: {[(f.name, f.type_name) for f in fields]}"
                )
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if isinstance(node.target, ast.Name) and node.value:
                expr_type = get_annotation_type(node.annotation)
                try:
                    value = generate_simple_expr(node.value)
                    collected.globals[node.target.id] = (expr_type, value)
                    logger.debug(
                        f"Collected global: {node.target.id}, type: {expr_type}, value: {value}"
                    )
                except TranspilerError:
                    pass
            self.generic_visit(node)

    visitor = Visitor()
    visitor.visit(tree)
    return collected


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
        if isinstance(node.value, (int, float)):
            return "float"
        elif isinstance(node.value, bool):
            return "bool"
    elif isinstance(node, ast.BinOp):
        left_type = get_expr_type(node.left, symbols, collected)
        right_type = get_expr_type(node.right, symbols, collected)
        if left_type == right_type and left_type.startswith("vec"):
            return left_type
        if left_type.startswith("vec") and right_type in ["float", "int"]:
            return left_type
        if right_type.startswith("vec") and left_type in ["float", "int"]:
            return right_type
        if "float" in (left_type, right_type):
            return "float"
        return "int"
    elif isinstance(node, ast.Call):
        func_name = node.func.id if isinstance(node.func, ast.Name) else None
        if func_name in BUILTIN_FUNCTIONS:
            return BUILTIN_FUNCTIONS[func_name][0]
        elif func_name in collected.functions:
            return collected.functions[func_name].return_type or "vec4"
        elif func_name in collected.structs:
            return func_name
        raise TranspilerError(f"Unknown function: {func_name}")
    elif isinstance(node, ast.Attribute):
        value_type = get_expr_type(node.value, symbols, collected)
        if value_type in collected.structs:
            struct_def = collected.structs[value_type]
            for field in struct_def.fields:
                if field.name == node.attr:
                    return field.type_name
            raise TranspilerError(
                f"Unknown field '{node.attr}' in struct '{value_type}'"
            )
        if value_type.startswith("vec"):
            swizzle_len = len(node.attr)
            valid_lengths = {1: "float", 2: "vec2", 3: "vec3", 4: "vec4"}
            vec_dim = int(value_type[-1])
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


def generate_name_expr(node: ast.Name, symbols: Dict[str, str]) -> str:
    """Generate GLSL code for a name expression (variable).

    Args:
        node: AST name node
        symbols: Dictionary of variable names to their types

    Returns:
        Generated GLSL code for the name expression
    """
    return node.id if node.id in symbols else node.id


def generate_constant_expr(node: ast.Constant) -> str:
    """Generate GLSL code for a constant expression (literal).

    Args:
        node: AST constant node

    Returns:
        Generated GLSL code for the constant expression

    Raises:
        TranspilerError: If the constant type is not supported
    """
    if isinstance(node.value, (int, float)):
        return str(node.value)
    elif isinstance(node.value, bool):
        # Use lowercase for GLSL boolean literals
        return "true" if node.value else "false"

    raise TranspilerError(f"Unsupported constant type: {type(node.value).__name__}")


def generate_binary_op_expr(
    node: ast.BinOp,
    symbols: Dict[str, str],
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
    symbols: Dict[str, str],
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
        return f"({expr})" if precedence < parent_precedence else expr

    raise TranspilerError("Multiple comparisons not supported")


def generate_bool_op_expr(
    node: ast.BoolOp,
    symbols: Dict[str, str],
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


def generate_struct_constructor(
    struct_name: str, node: ast.Call, symbols: Dict[str, str], collected: CollectedInfo
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
    logger.debug(f"Found {struct_name} as struct constructor")
    struct_def = collected.structs[struct_name]
    field_map = {f.name: i for i, f in enumerate(struct_def.fields)}
    required_fields = sum(1 for f in struct_def.fields if f.default_value is None)

    if node.keywords:
        # Keyword arguments constructor
        values = [None] * len(struct_def.fields)
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
                f"Wrong number of arguments for struct {struct_name}: expected {len(struct_def.fields)}, "
                f"got {len(provided_fields)} (missing required fields: {', '.join(missing_fields)})"
            )

        for i, field in enumerate(struct_def.fields):
            if values[i] is None:
                values[i] = (
                    field.default_value if field.default_value is not None else "0.0"
                )

        return f"{struct_name}({', '.join(values)})"

    elif node.args:
        # Positional arguments constructor
        if len(node.args) != len(struct_def.fields):
            raise TranspilerError(
                f"Wrong number of arguments for struct {struct_name}: expected {len(struct_def.fields)}, got {len(node.args)}"
            )

        args = [generate_expr(arg, symbols, 0, collected) for arg in node.args]
        return f"{struct_name}({', '.join(args)})"

    raise TranspilerError(f"Struct '{struct_name}' initialization requires arguments")


def generate_call_expr(
    node: ast.Call, symbols: Dict[str, str], collected: CollectedInfo
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
    logger.debug(f"Processing function call: {func_name}")

    if func_name in collected.functions:
        # Regular function call
        logger.debug(f"Found {func_name} in collected functions")
        args = [generate_expr(arg, symbols, 0, collected) for arg in node.args]
        return f"{func_name}({', '.join(args)})"

    elif func_name in BUILTIN_FUNCTIONS:
        # Built-in function call
        logger.debug(f"Found {func_name} in builtins")
        args = [generate_expr(arg, symbols, 0, collected) for arg in node.args]
        return f"{func_name}({', '.join(args)})"

    elif func_name in collected.structs:
        # Struct constructor
        return generate_struct_constructor(func_name, node, symbols, collected)

    else:
        logger.error(f"Unknown function call detected: {func_name}")
        raise TranspilerError(f"Unknown function call: {func_name}")


def generate_attribute_expr(
    node: ast.Attribute,
    symbols: Dict[str, str],
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
    symbols: Dict[str, str],
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
    condition = generate_expr(node.test, symbols, 3, collected)
    true_expr = generate_expr(node.body, symbols, 3, collected)
    false_expr = generate_expr(node.orelse, symbols, 3, collected)

    expr = f"{condition} ? {true_expr} : {false_expr}"
    return f"({expr})" if parent_precedence <= 3 else expr


def generate_expr(
    node: ast.AST,
    symbols: Dict[str, str],
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
    # Handle different expression types by delegating to specialized functions
    if isinstance(node, ast.Name):
        return generate_name_expr(node, symbols)
    elif isinstance(node, ast.Constant):
        return generate_constant_expr(node)
    elif isinstance(node, ast.BinOp):
        return generate_binary_op_expr(node, symbols, parent_precedence, collected)
    elif isinstance(node, ast.Compare):
        return generate_compare_expr(node, symbols, parent_precedence, collected)
    elif isinstance(node, ast.BoolOp):
        return generate_bool_op_expr(node, symbols, parent_precedence, collected)
    elif isinstance(node, ast.Call):
        return generate_call_expr(node, symbols, collected)
    elif isinstance(node, ast.Attribute):
        return generate_attribute_expr(node, symbols, parent_precedence, collected)
    elif isinstance(node, ast.IfExp):
        return generate_if_expr(node, symbols, parent_precedence, collected)

    raise TranspilerError(f"Unsupported expression: {type(node).__name__}")


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
        else:
            return f"{indent}{target_name} = {expr};"
    elif isinstance(target, ast.Attribute):
        target_expr = generate_expr(target, symbols, 0, collected)
        return f"{indent}{target_expr} = {expr};"

    raise TranspilerError(f"Unsupported assignment target: {type(target).__name__}")


def generate_annotated_assignment(
    stmt: ast.AnnAssign, symbols: Dict[str, str], indent: str
) -> str:
    """Generate GLSL code for an annotated assignment.

    Args:
        stmt: AST annotated assignment node
        symbols: Dictionary of variable names to their types
        indent: Indentation string

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
    else:
        raise TranspilerError(
            f"Unsupported augmented operator: {type(stmt.op).__name__}"
        )


def generate_for_loop(
    stmt: ast.For, symbols: Dict[str, str], indent: str, collected: CollectedInfo
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

        # Parse range parameters
        start = "0" if len(args) == 1 else generate_expr(args[0], symbols, 0, collected)
        end = generate_expr(args[1 if len(args) > 1 else 0], symbols, 0, collected)
        step = "1" if len(args) <= 2 else generate_expr(args[2], symbols, 0, collected)

        # Add target to symbols
        symbols[target] = "int"

        # Generate loop body
        body_code = generate_body(stmt.body, symbols.copy(), collected)

        # Compose the for loop
        code.append(
            f"{indent}for (int {target} = {start}; {target} < {end}; {target} += {step}) {{"
        )
        code.extend(body_code.splitlines())
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

    code.append(f"{indent}while ({condition}) {{")
    code.extend(body_code.splitlines())
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

    code.append(f"{indent}if ({condition}) {{")
    code.extend(generate_body(stmt.body, symbols.copy(), collected).splitlines())

    if stmt.orelse:
        code.append(f"{indent}}} else {{")
        code.extend(generate_body(stmt.orelse, symbols.copy(), collected).splitlines())

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
) -> str:
    """Generate GLSL code for a function body.

    Args:
        body: List of AST nodes representing statements in the function body
        symbols: Dictionary of variable names to their types
        collected: Information about functions, structs, and globals

    Returns:
        Generated GLSL code for the function body

    Raises:
        TranspilerError: If unsupported statements are encountered
    """
    logger.debug(f"Generating body with initial symbols: {symbols}")
    code = []
    indent = "    "

    for stmt in body:
        if isinstance(stmt, ast.Assign):
            code.append(generate_assignment(stmt, symbols, indent, collected))
        elif isinstance(stmt, ast.AnnAssign):
            code.append(generate_annotated_assignment(stmt, symbols, indent))
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
            raise TranspilerError("Pass statements are not supported in GLSL")

    return "\n".join(code)


def generate_glsl(collected: CollectedInfo, main_func: str) -> Tuple[str, Set[str]]:
    """Generate GLSL code from the collected information.

    Args:
        collected: Information about functions, structs, and globals
        main_func: Name of the main function to use as shader entry point

    Returns:
        Tuple of (generated GLSL code, set of used uniform variables)

    Raises:
        TranspilerError: If there are issues generating valid GLSL
    """
    logger.debug(f"Starting GLSL generation for main function: {main_func}")
    lines = []
    used_uniforms = set()

    main_func_info = collected.functions[main_func]
    main_func_node = main_func_info.node

    if not main_func_node.body or isinstance(main_func_node.body[0], ast.Pass):
        raise TranspilerError("Pass statements are not supported in GLSL")

    lines.append("#version 460 core\n")

    for arg in main_func_node.args.args:
        if arg.arg != "vs_uv":
            param_type = main_func_info.param_types[main_func_node.args.args.index(arg)]
            lines.append(f"uniform {param_type} {arg.arg};")
            used_uniforms.add(arg.arg)

    for name, (type_name, value) in collected.globals.items():
        lines.append(f"const {type_name} {name} = {value};")

    for struct_name, struct_def in collected.structs.items():
        lines.append(f"struct {struct_name} {{")
        for field in struct_def.fields:
            lines.append(f"    {field.type_name} {field.name};")
        lines.append("};\n")

    for func_name, func_info in collected.functions.items():
        is_main = func_name == main_func
        effective_return_type = "vec4" if is_main else func_info.return_type
        if not is_main and func_info.return_type is None:
            raise TranspilerError(
                f"Helper function '{func_name}' lacks return type annotation"
            )

        node = func_info.node
        param_str = ", ".join(
            f"{p_type} {arg.arg}"
            for p_type, arg in zip(func_info.param_types, node.args.args)
        )

        # Create initial symbols dictionary from function parameters
        symbols = {
            arg.arg: p_type
            for arg, p_type in zip(node.args.args, func_info.param_types)
        }

        body = generate_body(node.body, symbols, collected)

        lines.append(f"{effective_return_type} {func_name}({param_str}) {{")
        lines.extend(body.splitlines())
        lines.append("}\n")
        logger.debug(f"Generated GLSL for function: {func_name}")

    lines.append("in vec2 vs_uv;\nout vec4 fragColor;\n\nvoid main() {")
    main_call_args = [arg.arg for arg in main_func_node.args.args]
    main_call_str = ", ".join(main_call_args)
    lines.append(f"    fragColor = {main_func}({main_call_str});")
    lines.append("}")

    glsl_code = "\n".join(lines)

    # Ensure boolean literals are lowercase in GLSL
    glsl_code = glsl_code.replace(" True", " true").replace(" False", " false")
    glsl_code = glsl_code.replace("(True", "(true").replace("(False", "(false")
    glsl_code = glsl_code.replace(",True", ",true").replace(",False", ",false")
    glsl_code = glsl_code.replace("=True", "=true").replace("=False", "=false")

    logger.debug(f"GLSL generation complete. Used uniforms: {used_uniforms}")
    return glsl_code, used_uniforms


def parse_shader_code(
    shader_input: Union[str, Dict[str, Callable]], main_func: Optional[str] = None
) -> Tuple[ast.AST, Optional[str]]:
    """Parse the input Python code into an AST.

    Args:
        shader_input: The input Python code (string or dict of callables)
        main_func: Name of the main function to use as shader entry point

    Returns:
        Tuple of (AST of the parsed code, name of the main function)

    Raises:
        TranspilerError: If parsing fails
    """
    logger.debug("Parsing shader input")
    tree = None
    effective_main_func = main_func

    if isinstance(shader_input, dict):
        # Parse all items in context together to include globals
        source_lines = []
        for name, obj in shader_input.items():
            try:
                source = textwrap.dedent(inspect.getsource(obj))
                source_lines.append(source)
            except (OSError, TypeError) as e:
                raise TranspilerError(f"Failed to get source for {name}: {e}")
        full_source = "\n".join(source_lines)
        tree = ast.parse(full_source)
        if not effective_main_func:
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    effective_main_func = node.name
                    break
    elif isinstance(shader_input, str):
        shader_code = textwrap.dedent(shader_input)
        if not shader_code:
            raise TranspilerError("Empty shader code provided")
        tree = ast.parse(shader_code)
        if not effective_main_func:
            effective_main_func = "shader"
    else:
        raise TranspilerError("Shader input must be a string or context dictionary")

    logger.debug("Parsing complete")
    return tree, effective_main_func


def transpile(
    *args: Union[str, Callable, Type, object],
    main_func: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[str, Set[str]]:
    """Transpile Python code to GLSL shader code.

    This is the main entry point for the transpiler. It accepts various forms of input:
    - A string containing Python code
    - A function or class to transpile
    - Multiple functions or classes to include in the transpilation

    Args:
        *args: The Python code or callables to transpile
        main_func: Name of the main function to use as shader entry point
        **kwargs: Additional keyword arguments:
            - Additional functions/classes to include
            - Global constants to include in the shader

    Returns:
        Tuple of (generated GLSL code, set of used uniform variables)

    Raises:
        TranspilerError: If transpilation fails

    Examples:
        ```python
        # Transpile a single function
        glsl_code, uniforms = transpile(my_shader_func)

        # Transpile with multiple functions/structs
        glsl_code, uniforms = transpile(my_struct, my_helper_func, my_shader_func)

        # Specify the main function
        glsl_code, uniforms = transpile(my_struct, my_helper_func, my_shader_func,
                                        main_func="my_shader_func")

        # Include global constants
        glsl_code, uniforms = transpile(my_shader_func, PI=3.14159, MAX_STEPS=100)
        ```
    """
    logger.debug(
        f"Transpiling with args: {args}, main_func: {main_func}, kwargs: {kwargs}"
    )

    # Handle global constants passed as kwargs
    global_constants = {}
    for name, value in kwargs.items():
        if name != "main_func" and not callable(value) and not isinstance(value, type):
            if isinstance(value, (int, float, bool)):
                global_constants[name] = value

    # Prepare the shader input
    shader_input = None
    effective_main_func = main_func

    if len(args) == 1:
        # Single argument case
        if isinstance(args[0], str):
            shader_input = args[0]
        elif inspect.ismodule(args[0]):
            module = args[0]
            if hasattr(module, "__all__"):
                context = {name: getattr(module, name) for name in module.__all__}
            else:
                context = {
                    name: obj
                    for name, obj in module.__dict__.items()
                    if inspect.isfunction(obj)
                    or (inspect.isclass(obj) and hasattr(obj, "__dataclass_fields__"))
                }
            shader_input = context
            effective_main_func = main_func or "shader"
        else:
            main_item = args[0]
            if main_item.__name__.startswith("test_"):
                raise TranspilerError(
                    f"Main function '{main_item.__name__}' excluded due to 'test_' prefix. "
                    "Please rename it to avoid conflicts with test function naming conventions."
                )
            context = {main_item.__name__: main_item}
            shader_input = context
            effective_main_func = main_func or main_item.__name__
    else:
        # Multiple arguments case
        context = {}
        for item in args:
            if inspect.isfunction(item):
                context[item.__name__] = item
            elif inspect.isclass(item) and hasattr(item, "__dataclass_fields__"):
                context[item.__name__] = item
            else:
                raise TranspilerError(
                    f"Unsupported item type in transpile args: {type(item)}"
                )
        shader_input = context
        effective_main_func = main_func

    # Parse the code
    tree, parsed_main_func = parse_shader_code(shader_input, effective_main_func)
    effective_main_func = parsed_main_func

    # Collect information from the AST
    collected = collect_info(tree)

    # Add explicitly passed globals to the collected info
    for name, value in global_constants.items():
        if isinstance(value, (int, float)):
            type_name = "float" if isinstance(value, float) else "int"
            collected.globals[name] = (type_name, str(value))
        elif isinstance(value, bool):
            # Ensure booleans are properly converted to GLSL's lowercase true/false
            collected.globals[name] = ("bool", "true" if value else "false")

    # Verify main function exists
    if effective_main_func not in collected.functions:
        raise TranspilerError(f"Main function '{effective_main_func}' not found")

    # Generate GLSL code
    return generate_glsl(collected, effective_main_func)
