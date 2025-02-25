import ast
import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Set, Tuple, Union

from loguru import logger

# Built-ins for GLSL transpilation
builtins = {
    "abs": ("float", ["float"]),
    "cos": ("float", ["float"]),
    "fract": ("vec2", ["vec2"]),
    "length": ("float", ["vec2"]),
    "sin": ("float", ["float"]),
    "sqrt": ("float", ["float"]),
    "tan": ("float", ["float"]),
    "radians": ("float", ["float"]),
    "mix": ("vec3", ["vec3", "vec3", "float"]),
    "normalize": ("vec3", ["vec3"]),
    "cross": ("vec3", ["vec3", "vec3"]),
    "max": ("float", ["float", "float"]),
    "min": ("float", ["float", "float"]),
    "vec2": ("vec2", ["float", "float"]),
    "vec3": ("vec3", ["float", "float", "float"]),
    "vec4": ("vec4", ["float", "float", "float", "float"]),
    "distance": ("float", ["vec3", "vec3"]),
    "smoothstep": ("float", ["float", "float", "float"]),
}


# Default uniforms
@dataclass
class Interface:
    u_time: "float"
    u_aspect: "float"
    u_resolution: "vec2"
    u_mouse_pos: "vec2"
    u_mouse_uv: "vec2"


default_uniforms = {
    "u_time": "float",
    "u_aspect": "float",
    "u_resolution": "vec2",
    "u_mouse_pos": "vec2",
    "u_mouse_uv": "vec2",
}


def pytype_to_glsl(pytype: str) -> str:
    """Convert Python type annotations to GLSL types."""
    return pytype


class StructDefinition:
    """Helper class to represent GLSL structs in Python."""

    def __init__(self, name: str, fields: List[Tuple[str, str]]):
        self.name = name
        self.fields = fields  # List of (name, type) tuples


class FunctionCollector(ast.NodeVisitor):
    """Collect function, struct definitions, and module-level globals from the AST."""

    def __init__(self):
        self.functions: Dict[str, Tuple[str, List[str], ast.FunctionDef]] = {}
        self.structs: Dict[str, StructDefinition] = {}
        self.globals: Dict[str, Tuple[str, str]] = {}
        self.current_context: List[str] = (
            []
        )  # Track context: 'module', 'class', or 'function'

    def visit_Module(self, node: ast.Module) -> None:
        """Set module context and visit children."""
        self.current_context.append("module")
        self.generic_visit(node)
        self.current_context.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Handle struct definitions via Python classes."""
        self.current_context.append("class")

        # Check for @dataclass decorator
        is_dataclass = False
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == "dataclass":
                is_dataclass = True
                break
            elif isinstance(decorator, ast.Attribute) and decorator.attr == "dataclass":
                is_dataclass = True
                break

        # Handle both traditional structs and dataclasses
        if node.name.endswith("Struct") or is_dataclass:
            struct_name = node.name
            if node.name.endswith("Struct"):
                struct_name = node.name.replace("Struct", "")

            fields = []
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(
                    stmt.target, ast.Name
                ):
                    field_type = None
                    if isinstance(stmt.annotation, ast.Constant):
                        field_type = stmt.annotation.value
                    elif hasattr(stmt.annotation, "id"):
                        field_type = stmt.annotation.id

                    if field_type:
                        fields.append((stmt.target.id, field_type))

            self.structs[struct_name] = StructDefinition(struct_name, fields)

        self.generic_visit(node)
        self.current_context.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Collect function definitions."""
        self.current_context.append("function")
        func_name = node.name

        params = []
        for arg in node.args.posonlyargs + node.args.args:
            if isinstance(arg.annotation, ast.Constant):
                params.append(arg.annotation.value)
            elif hasattr(arg.annotation, "id"):
                params.append(arg.annotation.id)

        return_type = None
        if node.returns:
            if isinstance(node.returns, ast.Constant):
                return_type = node.returns.value
            elif hasattr(node.returns, "id"):
                return_type = node.returns.id

        if return_type is None:
            logger.error(f"Function '{func_name}' missing return type")
            raise ValueError(
                f"Function '{func_name}' must have a return type annotation"
            )
        self.functions[func_name] = (return_type, params, node)
        self.generic_visit(node)
        self.current_context.pop()

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Handle global variable declarations only at module level."""
        if (
            self.current_context
            and self.current_context[-1] == "module"
            and isinstance(node.target, ast.Name)
        ):
            target = node.target.id
            expr_type = None
            if isinstance(node.annotation, ast.Constant):
                expr_type = node.annotation.value
            elif hasattr(node.annotation, "id"):
                expr_type = node.annotation.id

            value = self._generate_expr(node.value) if node.value else None
            if expr_type:
                self.globals[target] = (expr_type, value)
        self.generic_visit(node)

    def _generate_expr(self, node: ast.expr) -> str:
        """Generate expression strings for globals."""
        if isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.BinOp):
            left = self._generate_expr(node.left)
            right = self._generate_expr(node.right)
            op_map = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
            op = op_map.get(type(node.op))
            if op is None:
                raise ValueError(
                    f"Unsupported binary operation in global: {type(node.op).__name__}"
                )
            return f"({left} {op} {right})"
        raise ValueError(
            f"Unsupported expression type in global: {type(node).__name__}"
        )


class GLSLGenerator:
    """Generate GLSL code from Python AST."""

    def __init__(
        self,
        functions: Dict[str, Tuple[str, List[str], ast.FunctionDef]],
        structs: Dict[str, StructDefinition],
        globals: Dict[str, Tuple[str, str]],
        builtins: Dict[str, Tuple[str, List[str]]],
        default_uniforms: Dict[str, str],
    ):
        self.functions = functions
        self.structs = structs
        self.globals = globals
        self.builtins = builtins
        self.default_uniforms = default_uniforms
        self.code = ""

    def generate(self, func_name: str, node: ast.FunctionDef) -> None:
        logger.info(f"Generating GLSL for function: {func_name}")
        params = []
        for arg in node.args.posonlyargs + node.args.args:
            if isinstance(arg.annotation, ast.Constant):
                params.append(f"{pytype_to_glsl(arg.annotation.value)} {arg.arg}")
            elif hasattr(arg.annotation, "id"):
                params.append(f"{pytype_to_glsl(arg.annotation.id)} {arg.arg}")

        return_type = None
        if isinstance(node.returns, ast.Constant):
            return_type = pytype_to_glsl(node.returns.value)
        elif hasattr(node.returns, "id"):
            return_type = pytype_to_glsl(node.returns.id)

        signature = f"{return_type} {func_name}({', '.join(params)})"

        symbols = {}
        for arg in node.args.posonlyargs + node.args.args:
            if isinstance(arg.annotation, ast.Constant):
                symbols[arg.arg] = pytype_to_glsl(arg.annotation.value)
            elif hasattr(arg.annotation, "id"):
                symbols[arg.arg] = pytype_to_glsl(arg.annotation.id)

        body_code = self.generate_body(node.body, symbols)
        self.code += f"{signature} {{\n{body_code}}}\n"

    def generate_body(self, body: List[ast.stmt], symbols: Dict[str, str]) -> str:
        """Generate GLSL code for a function body."""
        code = ""
        for stmt in body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                target = stmt.target.id
                expr_type = None
                if isinstance(stmt.annotation, ast.Constant):
                    expr_type = stmt.annotation.value
                elif hasattr(stmt.annotation, "id"):
                    expr_type = stmt.annotation.id

                if stmt.value:
                    expr_code = self.generate_expr(stmt.value, symbols, 0)
                    if expr_type in self.structs and isinstance(stmt.value, ast.Call):
                        struct_name = expr_type
                        expr_code = expr_code.replace(
                            f"{struct_name}Struct(", f"{struct_name}("
                        )
                    code += f"    {expr_type} {target} = {expr_code};\n"
                else:
                    code += f"    {expr_type} {target};\n"
                symbols[target] = expr_type
            elif isinstance(stmt, ast.Assign):
                if isinstance(stmt.targets[0], ast.Name):
                    target = stmt.targets[0].id
                    expr_code = self.generate_expr(stmt.value, symbols, 0)
                    expr_type = self.get_expr_type(stmt.value, symbols)
                    if target not in symbols:
                        symbols[target] = expr_type
                        code += f"    {expr_type} {target} = {expr_code};\n"
                    else:
                        code += f"    {target} = {expr_code};\n"
                elif isinstance(stmt.targets[0], ast.Attribute):
                    target = self.generate_expr(stmt.targets[0], symbols, 0)
                    expr_code = self.generate_expr(stmt.value, symbols, 0)
                    code += f"    {target} = {expr_code};\n"
                else:
                    logger.error(
                        f"Unsupported assignment target type: {type(stmt.targets[0])}"
                    )
                    raise ValueError(
                        f"Unsupported assignment target type: {type(stmt.targets[0]).__name__}"
                    )
            elif isinstance(stmt, ast.AugAssign):
                target = self.generate_expr(stmt.target, symbols, 0)
                value_code = self.generate_expr(stmt.value, symbols, 0)
                op_map = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
                op = op_map.get(type(stmt.op))
                if op is None:
                    logger.error(
                        f"Unsupported augmented assignment operation: {type(stmt.op).__name__}"
                    )
                    raise ValueError(
                        f"Unsupported augmented assignment operation: {type(stmt.op).__name__}"
                    )
                # For augmented assignment, use a simpler form
                code += f"    {target} = {target} {op} {value_code};\n"
            elif isinstance(stmt, ast.While):
                condition = self.generate_expr(stmt.test, symbols, 0)
                body_code = self.generate_body(stmt.body, symbols.copy())
                code += f"    while ({condition}) {{\n{body_code}    }}\n"
            elif isinstance(stmt, ast.For):
                if (
                    isinstance(stmt.target, ast.Name)
                    and isinstance(stmt.iter, ast.Call)
                    and hasattr(stmt.iter.func, "id")
                    and stmt.iter.func.id == "range"
                ):
                    target = stmt.target.id
                    args = stmt.iter.args
                    start = (
                        self.generate_expr(args[0], symbols, 0)
                        if len(args) > 1
                        else "0"
                    )
                    end = self.generate_expr(
                        args[1 if len(args) > 1 else 0], symbols, 0
                    )
                    step = (
                        self.generate_expr(args[2], symbols, 0)
                        if len(args) > 2
                        else "1"
                    )
                    symbols[target] = "int"
                    body_code = self.generate_body(stmt.body, symbols.copy())
                    code += f"    for (int {target} = {start}; {target} < {end}; {target} += {step}) {{\n{body_code}    }}\n"
                else:
                    logger.error(
                        "Only for loops with range() are supported in GLSL transpilation"
                    )
                    raise ValueError(
                        "Only for loops with range() are supported in GLSL transpilation"
                    )
            elif isinstance(stmt, ast.If):
                condition = self.generate_expr(stmt.test, symbols, 0)
                if_body = self.generate_body(stmt.body, symbols.copy())
                code += f"    if ({condition}) {{\n{if_body}    }}\n"
                if stmt.orelse:
                    else_body = self.generate_body(stmt.orelse, symbols.copy())
                    code += f"    else {{\n{else_body}    }}\n"
            elif isinstance(stmt, ast.Break):
                code += "    break;\n"
            elif isinstance(stmt, ast.Return):
                expr_code = self.generate_expr(stmt.value, symbols, 0)
                code += f"    return {expr_code};\n"
            elif isinstance(stmt, ast.Pass):
                # Pass statements should raise an error in shader code
                logger.error("Pass statements are not supported in GLSL")
                raise ValueError("Unsupported statement type: Pass")
            elif isinstance(stmt, ast.Expr):
                # Skip standalone expressions (like docstrings or comments)
                # These don't impact the GLSL output
                logger.debug(f"Skipping expression statement: {ast.dump(stmt)}")
                pass
            else:
                logger.error(f"Unsupported statement type: {type(stmt)}")
                raise ValueError(f"Unsupported statement type: {type(stmt).__name__}")
        return code

    def generate_expr(
        self,
        node: ast.expr,
        symbols: Dict[str, str],
        parent_precedence: int = 0,
    ) -> str:
        """
        Generate GLSL code for an expression with proper operator precedence.

        Args:
            node: The AST node to generate code for
            symbols: Dictionary mapping variable names to their types
            parent_precedence: Operator precedence of the parent expression (0 = lowest)

        Returns:
            Generated GLSL code
        """
        # Define operator precedence (higher number = higher precedence)
        # Based on GLSL operator precedence
        PRECEDENCE = {
            # Assignment has lowest precedence
            "=": 1,
            # Logical OR
            "||": 2,
            # Logical AND
            "&&": 3,
            # Equality operators
            "==": 4,
            "!=": 4,
            # Relational operators
            "<": 5,
            ">": 5,
            "<=": 5,
            ">=": 5,
            # Additive operators
            "+": 6,
            "-": 6,
            # Multiplicative operators
            "*": 7,
            "/": 7,
            "%": 7,
            # Unary operators have high precedence
            "unary": 8,
            # Function calls and member access have highest precedence
            "call": 9,
            "member": 9,
        }

        if isinstance(node, ast.Constant):
            if isinstance(node.value, float):
                # Ensure float literals have a decimal point
                return (
                    f"{node.value:.1f}"
                    if node.value == int(node.value)
                    else str(node.value)
                )
            elif isinstance(node.value, bool):
                # Convert Python booleans to GLSL booleans
                return "true" if node.value else "false"
            return str(node.value)

        elif isinstance(node, ast.Name):
            # Convert Python's True/False to GLSL's true/false
            if node.id == "True":
                return "true"
            elif node.id == "False":
                return "false"
            return node.id

        elif isinstance(node, ast.BinOp):
            # Map Python operators to GLSL operators and their precedence
            op_map = {
                ast.Add: ("+", PRECEDENCE["+"]),
                ast.Sub: ("-", PRECEDENCE["-"]),
                ast.Mult: ("*", PRECEDENCE["*"]),
                ast.Div: ("/", PRECEDENCE["/"]),
                ast.Mod: ("%", PRECEDENCE["%"]),
                ast.Pow: ("pow", PRECEDENCE["call"]),  # pow is a function call
            }

            op_info = op_map.get(type(node.op))
            if op_info is None:
                logger.error(f"Unsupported binary operation: {type(node.op).__name__}")
                raise ValueError(
                    f"Unsupported expression type: {type(node.op).__name__}"
                )

            op, precedence = op_info

            # Handle modulo operation separately to check for integer operands
            if isinstance(node.op, ast.Mod):
                left_type = self.get_expr_type(node.left, symbols)
                right_type = self.get_expr_type(node.right, symbols)
                if left_type != "int" or right_type != "int":
                    logger.error("Modulo operation requires integer operands in GLSL")
                    raise ValueError(
                        "Unsupported expression type: Modulo with non-integer operands"
                    )

            # Generate code for operands, passing current precedence
            left = self.generate_expr(node.left, symbols, precedence)
            right = self.generate_expr(node.right, symbols, precedence)

            # Special case for pow which is a function call
            if op == "pow":
                return f"pow({left}, {right})"

            # Determine if we need parentheses based on precedence
            expr = f"{left} {op} {right}"
            if precedence < parent_precedence:
                return f"({expr})"
            return expr

        elif isinstance(node, ast.BoolOp):
            op_map = {
                ast.And: ("&&", PRECEDENCE["&&"]),
                ast.Or: ("||", PRECEDENCE["||"]),
            }
            op_info = op_map.get(type(node.op))
            if op_info is None:
                logger.error(f"Unsupported boolean operation: {type(node.op).__name__}")
                raise ValueError(
                    f"Unsupported boolean operation: {type(node.op).__name__}"
                )

            op, precedence = op_info

            # Generate code for values with appropriate precedence
            values = []
            for val in node.values:
                val_code = self.generate_expr(val, symbols, precedence)
                # Add parentheses around operands if they have lower precedence
                if (
                    isinstance(val, (ast.BoolOp, ast.Compare))
                    and self._get_precedence(val) < precedence
                ):
                    val_code = f"({val_code})"
                values.append(val_code)

            # Join with operator
            expr = f" {op} ".join(values)

            # Add parentheses if needed based on parent precedence
            if precedence < parent_precedence:
                return f"({expr})"
            return expr

        elif isinstance(node, ast.Compare):
            # For comparison operations
            if len(node.ops) == 1:
                # Simple case with one comparison
                op_map = {
                    ast.Eq: ("==", PRECEDENCE["=="]),
                    ast.NotEq: ("!=", PRECEDENCE["!="]),
                    ast.Lt: ("<", PRECEDENCE["<"]),
                    ast.LtE: ("<=", PRECEDENCE["<="]),
                    ast.Gt: (">", PRECEDENCE[">"]),
                    ast.GtE: (">=", PRECEDENCE[">="]),
                }

                op_info = op_map.get(type(node.ops[0]))
                if op_info is None:
                    logger.error(
                        f"Unsupported comparison operation: {type(node.ops[0]).__name__}"
                    )
                    raise ValueError(
                        f"Unsupported comparison operation: {type(node.ops[0]).__name__}"
                    )

                op, precedence = op_info
                left = self.generate_expr(node.left, symbols, precedence)
                right = self.generate_expr(node.comparators[0], symbols, precedence)

                expr = f"{left} {op} {right}"
                if precedence < parent_precedence:
                    return f"({expr})"
                return expr
            else:
                # Multiple comparisons (e.g., a < b < c) - chain with &&
                ops = []
                current_left = self.generate_expr(node.left, symbols, 0)

                for i, op in enumerate(node.ops):
                    op_map = {
                        ast.Eq: "==",
                        ast.NotEq: "!=",
                        ast.Lt: "<",
                        ast.LtE: "<=",
                        ast.Gt: ">",
                        ast.GtE: ">=",
                    }

                    op_str = op_map.get(type(op))
                    if op_str is None:
                        logger.error(
                            f"Unsupported comparison operation: {type(op).__name__}"
                        )
                        raise ValueError(
                            f"Unsupported comparison operation: {type(op).__name__}"
                        )

                    right = self.generate_expr(node.comparators[i], symbols, 0)
                    ops.append(f"{current_left} {op_str} {right}")
                    current_left = right

                # Join with && and add parentheses if needed
                expr = " && ".join(ops)
                if PRECEDENCE["&&"] < parent_precedence:
                    return f"({expr})"
                return expr

        elif isinstance(node, ast.Call):
            func_name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else self.generate_expr(node.func, symbols, 0)
            )
            args = [self.generate_expr(arg, symbols, 0) for arg in node.args]

            # Handle different types of function calls
            if func_name in ["float", "int", "bool"]:
                # Type casting functions
                return f"{func_name}({', '.join(args)})"
            elif func_name in self.builtins:
                return f"{func_name}({', '.join(args)})"
            elif func_name in self.functions:
                return f"{func_name}({', '.join(args)})"
            elif func_name in self.structs:
                # Direct struct constructor
                struct_name = func_name

                # If no arguments provided, initialize all fields to default values
                if not args:
                    field_inits = []
                    # Initialize each field with appropriate default value based on type
                    for field_name, field_type in self.structs[struct_name].fields:
                        if field_type == "int":
                            field_inits.append("0")
                        elif field_type == "float":
                            field_inits.append("0.0")
                        elif field_type == "bool":
                            field_inits.append("false")
                        elif field_type.startswith("vec"):
                            # Get the dimension from the type (e.g., vec3 -> 3)
                            dim = int(field_type[-1])
                            field_inits.append(f"{field_type}(0.0)")
                        else:
                            # For other types, default to 0
                            field_inits.append("0")

                    return f"{struct_name}({', '.join(field_inits)})"
                else:
                    return f"{struct_name}({', '.join(args)})"

            elif func_name.endswith("Struct"):
                # Handle struct constructors with the "Struct" suffix
                struct_name = func_name.replace("Struct", "")
                if struct_name in self.structs:
                    return f"{struct_name}({', '.join(args)})"
                else:
                    logger.error(f"Unknown struct: {struct_name}")
                    raise ValueError(f"Unknown struct: {struct_name}")
            else:
                # Special handling for potential dataclass calls
                # In GLSL, we treat dataclasses as structs
                if func_name in symbols or any(
                    s.startswith(func_name) for s in self.structs.keys()
                ):
                    # If we've seen this type in symbols or it looks like a struct name
                    # Assume it's a struct constructor
                    return f"{func_name}({', '.join(args)})"

                # Check if this might be a struct type that we haven't processed yet
                for struct_name in self.structs.keys():
                    if struct_name.lower() == func_name.lower():
                        return f"{struct_name}({', '.join(args)})"

                logger.error(f"Unknown function: {func_name}")
                raise ValueError(f"Unknown function: {func_name}")

        elif isinstance(node, ast.Attribute):
            # Member access has high precedence
            value = self.generate_expr(node.value, symbols, PRECEDENCE["member"])
            return f"{value}.{node.attr}"

        elif isinstance(node, ast.UnaryOp):
            op_map = {ast.USub: "-", ast.UAdd: "+", ast.Not: "!"}
            op = op_map.get(type(node.op))
            if op is None:
                logger.error(f"Unsupported unary operation: {type(node.op).__name__}")
                raise ValueError(
                    f"Unsupported unary operation: {type(node.op).__name__}"
                )

            # Unary operations have high precedence
            operand = self.generate_expr(node.operand, symbols, PRECEDENCE["unary"])

            # Add parentheses for clarity with complex operands
            if isinstance(node.operand, (ast.BinOp, ast.BoolOp, ast.Compare)):
                return f"{op}({operand})"
            return f"{op}{operand}"

        elif isinstance(node, ast.Subscript):
            # Array access has high precedence
            value = self.generate_expr(node.value, symbols, 0)
            if isinstance(node.slice, ast.Index):
                index = self.generate_expr(node.slice.value, symbols, 0)
            else:
                index = self.generate_expr(node.slice, symbols, 0)
            return f"{value}[{index}]"

        else:
            logger.error(f"Unsupported expression type: {type(node)}")
            raise ValueError(f"Unsupported expression type: {type(node).__name__}")

    def _get_precedence(self, node: ast.expr) -> int:
        """Helper method to get the precedence of an expression."""
        PRECEDENCE = {
            # Assignment has lowest precedence
            "=": 1,
            # Logical OR
            "||": 2,
            # Logical AND
            "&&": 3,
            # Equality operators
            "==": 4,
            "!=": 4,
            # Relational operators
            "<": 5,
            ">": 5,
            "<=": 5,
            ">=": 5,
            # Additive operators
            "+": 6,
            "-": 6,
            # Multiplicative operators
            "*": 7,
            "/": 7,
            "%": 7,
            # Unary operators have high precedence
            "unary": 8,
            # Function calls and member access have highest precedence
            "call": 9,
            "member": 9,
        }

        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.Or):
                return PRECEDENCE["||"]
            elif isinstance(node.op, ast.And):
                return PRECEDENCE["&&"]
        elif isinstance(node, ast.Compare):
            # Use the lowest precedence of any comparison operator
            return min(
                PRECEDENCE["=="], PRECEDENCE["<"]
            )  # They're all the same in our table
        elif isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Add) or isinstance(node.op, ast.Sub):
                return PRECEDENCE["+"]  # Same as "-"
            elif (
                isinstance(node.op, ast.Mult)
                or isinstance(node.op, ast.Div)
                or isinstance(node.op, ast.Mod)
            ):
                return PRECEDENCE["*"]  # Same as "/" and "%"

        # Default to high precedence for other expressions
        return 10

    def get_expr_type(self, node: ast.expr, symbols: Dict[str, str]) -> str:
        """Determine the GLSL type of an expression."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, int):
                return "int"
            elif isinstance(node.value, float):
                return "float"
            elif isinstance(node.value, bool):
                return "bool"
            else:
                logger.error(f"Unsupported constant type: {type(node.value)}")
                raise ValueError(
                    f"Unsupported constant type: {type(node.value).__name__}"
                )
        elif isinstance(node, ast.Name):
            if node.id in symbols:
                return symbols[node.id]
            elif node.id in self.globals:
                return self.globals[node.id][0]
            else:
                logger.error(f"Unknown variable: {node.id}")
                raise ValueError(f"Unknown variable: {node.id}")
        elif isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else None

            # Handle type casting functions
            if func_name in ["float", "int", "bool"]:
                return func_name
            elif func_name in self.builtins:
                return self.builtins[func_name][0]
            elif func_name in self.functions:
                return self.functions[func_name][0]
            elif func_name in self.structs:
                return func_name
            elif func_name and func_name.endswith("Struct"):
                # Handle struct constructors with the "Struct" suffix
                struct_name = func_name.replace("Struct", "")
                if struct_name in self.structs:
                    return struct_name
                else:
                    logger.error(f"Unknown struct: {struct_name}")
                    raise ValueError(f"Unknown struct: {struct_name}")
            else:
                # Check if this might be a dataclass/struct type
                # Use case-insensitive match for more robust detection
                if func_name:
                    for struct_name in self.structs.keys():
                        if struct_name.lower() == func_name.lower():
                            return struct_name

                logger.error(f"Unknown function: {func_name}")
                raise ValueError(f"Unknown function: {func_name}")
        elif isinstance(node, ast.BinOp):
            left_type = self.get_expr_type(node.left, symbols)
            right_type = self.get_expr_type(node.right, symbols)

            # Type promotion rules for binary operations
            if isinstance(node.op, ast.Mod):
                # Modulo operation is only valid for integers in GLSL
                if left_type != "int" or right_type != "int":
                    logger.error("Modulo operation requires integer operands in GLSL")
                    raise ValueError(
                        "Unsupported expression type: Modulo with non-integer operands"
                    )
                return "int"

            # Vector-scalar operations preserve the vector type
            if left_type.startswith("vec") and right_type in ["float", "int"]:
                return left_type
            elif right_type.startswith("vec") and left_type in ["float", "int"]:
                return right_type
            # Vector-vector operations of the same dimension preserve the type
            elif (
                left_type.startswith("vec")
                and right_type.startswith("vec")
                and left_type == right_type
            ):
                return left_type
            # Numeric type promotion
            elif left_type == "float" or right_type == "float":
                return "float"
            elif left_type == "int" and right_type == "int":
                return "int"
            else:
                return left_type
        elif isinstance(node, ast.Attribute):
            # Handle vector swizzling
            value_type = self.get_expr_type(node.value, symbols)
            attr = node.attr

            if value_type.startswith("vec"):
                vec_dim = int(value_type[-1])
                # Check for valid swizzle patterns
                if all(c in "xyzw"[:vec_dim] for c in attr):
                    if len(attr) == 1:
                        return "float"
                    elif len(attr) == 2:
                        return "vec2"
                    elif len(attr) == 3:
                        return "vec3"
                    elif len(attr) == 4:
                        return "vec4"
                elif all(c in "rgba"[:vec_dim] for c in attr):
                    if len(attr) == 1:
                        return "float"
                    elif len(attr) == 2:
                        return "vec2"
                    elif len(attr) == 3:
                        return "vec3"
                    elif len(attr) == 4:
                        return "vec4"

                logger.error(
                    f"Cannot infer type for attribute '{attr}' of '{value_type}'"
                )
                raise ValueError(
                    f"Cannot infer type for attribute '{attr}' of '{value_type}'"
                )
            else:
                # For struct field access
                if value_type in self.structs:
                    for field_name, field_type in self.structs[value_type].fields:
                        if field_name == attr:
                            return field_type

                logger.error(f"Cannot determine type of attribute: {node.attr}")
                raise ValueError(f"Cannot determine type of attribute: {node.attr}")
        else:
            logger.error(f"Cannot determine type of expression: {type(node)}")
            raise ValueError(
                f"Cannot determine type of expression: {type(node).__name__}"
            )


def transpile(
    shader_input: Union[str, Callable],
    main_func: str = "main_shader",
    version: str = "460 core",
) -> Tuple[str, Set[str]]:
    """Transpile Python shader code to GLSL.

    Args:
        shader_input: Either a string containing Python code or a callable function
        main_func: Name of the main shader function (default: "main_shader")
        version: GLSL version to use (default: "460 core")

    Returns:
        Tuple of (GLSL code, set of used uniform names)
    """
    # If a function object was passed, get its source code from the whole module
    if callable(shader_input):
        try:
            # Get the module where the function is defined
            module = inspect.getmodule(shader_input)
            if module is not None:
                # Get the entire module source code to include all helper functions
                shader_code = inspect.getsource(module)
            else:
                # Fallback to just the function source if module can't be determined
                shader_code = inspect.getsource(shader_input)

            # If the function isn't the main one, we need to set it as main_func
            if shader_input.__name__ != main_func:
                main_func = shader_input.__name__
        except (TypeError, OSError) as e:
            raise ValueError(f"Could not extract source code from function: {e}")
    else:
        shader_code = shader_input

    if not shader_code.strip():
        logger.error("Empty shader code")
        raise ValueError(f"Main shader function '{main_func}' not found")

    tree = ast.parse(shader_code)
    collector = FunctionCollector()
    collector.visit(tree)

    # Check if main_shader function exists
    if main_func not in collector.functions:
        logger.error(f"Main shader function '{main_func}' not found")
        raise ValueError(f"Main shader function '{main_func}' not found")

    # Check if the main shader function has a body
    main_node = collector.functions[main_func][2]
    if not main_node.body:
        logger.error(f"Main shader function '{main_func}' has no body")
        raise ValueError(f"Main shader function '{main_func}' has no body")

    # Generate GLSL code
    generator = GLSLGenerator(
        collector.functions,
        collector.structs,
        collector.globals,
        builtins,
        default_uniforms,
    )

    # Generate struct definitions
    struct_code = ""
    for struct_name, struct_def in collector.structs.items():
        struct_code += f"struct {struct_name} {{\n"
        for field_name, field_type in struct_def.fields:
            struct_code += f"    {field_type} {field_name};\n"
        struct_code += "};\n\n"

    # Extract used uniforms from function parameters
    used_uniforms = set()
    custom_uniforms = {}

    # Get parameters from main shader function
    main_func_params = []
    main_func_node = collector.functions[main_func][2]
    for arg in main_func_node.args.posonlyargs + main_func_node.args.args:
        param_name = arg.arg
        param_type = None
        if isinstance(arg.annotation, ast.Constant):
            param_type = arg.annotation.value
        elif hasattr(arg.annotation, "id"):
            param_type = arg.annotation.id

        if param_name != "vs_uv":  # vs_uv is not a uniform
            if param_name in default_uniforms:
                used_uniforms.add(param_name)
            else:
                custom_uniforms[param_name] = param_type
                used_uniforms.add(param_name)  # Add custom uniforms to used_uniforms

    # Generate uniform declarations
    uniform_code = ""
    # Only include default uniforms that are actually used
    for name, type_name in default_uniforms.items():
        if name in used_uniforms:
            uniform_code += f"uniform {type_name} {name};\n"

    # Add custom uniforms
    for name, type_name in custom_uniforms.items():
        uniform_code += f"uniform {type_name} {name};\n"

    # Generate global variable declarations
    global_code = ""
    for name, (type_name, value) in collector.globals.items():
        if value:
            global_code += f"{type_name} {name} = {value};\n"
        else:
            global_code += f"{type_name} {name};\n"

    # Generate function definitions
    for func_name, (return_type, params, node) in collector.functions.items():
        generator.generate(func_name, node)

    # Generate main function with appropriate parameters
    main_params = []
    for arg in main_func_node.args.posonlyargs + main_func_node.args.args:
        param_name = arg.arg
        if param_name == "vs_uv":
            main_params.append("gl_FragCoord.xy / u_resolution")
        elif param_name in used_uniforms:
            main_params.append(param_name)

    main_code = (
        """
out vec4 fragColor;

void main() {
    fragColor = main_shader("""
        + ", ".join(main_params)
        + """);
}
"""
    )

    # Combine all parts
    glsl_code = f"#version {version}\n\n{uniform_code}\n{global_code}\n{struct_code}\n{generator.code}\n{main_code}"

    # Log the generated code for debugging
    logger.debug(f"Generated GLSL code:\n{glsl_code}")

    # Return the generated GLSL code and the used uniforms
    return glsl_code, used_uniforms
