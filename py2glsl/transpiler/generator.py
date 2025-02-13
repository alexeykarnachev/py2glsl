"""GLSL code generator with type system integration."""

import ast
from dataclasses import dataclass

from loguru import logger

from py2glsl.types import (
    BOOL,
    BVEC2,
    BVEC3,
    BVEC4,
    FLOAT,
    INT,
    IVEC2,
    IVEC3,
    IVEC4,
    VEC2,
    VEC3,
    VEC4,
    GLSLSwizzleError,
    GLSLType,
    GLSLTypeError,
    can_convert_to,
    validate_operation,
)

from .analyzer import ShaderAnalysis
from .constants import BUILTIN_UNIFORMS, GLSL_VERSION, VERTEX_SHADER


@dataclass
class GeneratedShader:
    """Result of shader generation."""

    vertex_source: str
    fragment_source: str
    uniforms: dict[str, GLSLType]


class GLSLGenerator:
    """Generates formatted GLSL code with type validation."""

    def __init__(self, analysis: ShaderAnalysis):
        """Initialize generator with analysis results."""
        self.analysis = analysis
        self.indent_level = 0
        self.lines: list[str] = []
        self.current_scope = "global"
        self.scope_stack: list[str] = []
        self.declared_vars: dict[str, set[str]] = {"global": set()}
        self.vertex_source = VERTEX_SHADER

    def indent(self) -> str:
        """Get current indentation."""
        return "    " * self.indent_level

    def add_line(self, line: str = "") -> None:
        """Add line with proper indentation and formatting."""
        if not line:
            if not self.lines or self.lines[-1]:
                self.lines.append("")
            return

        if line.startswith(("#", "layout", "in ", "out ", "uniform ")):
            self.lines.append(line)
            return

        self.lines.append(f"{self.indent()}{line}")

    def enter_scope(self, name: str) -> None:
        """Enter a new scope."""
        self.scope_stack.append(self.current_scope)
        self.current_scope = name
        if name not in self.declared_vars:
            self.declared_vars[name] = set()

    def exit_scope(self) -> None:
        """Exit current scope."""
        if self.scope_stack:
            self.current_scope = self.scope_stack.pop()
        else:
            self.current_scope = "global"

    def begin_block(self) -> None:
        """Begin a new block."""
        self.add_line("{")
        self.indent_level += 1

    def end_block(self) -> None:
        """End current block."""
        self.indent_level -= 1
        self.add_line("}")

    def get_type(self, node: ast.AST) -> GLSLType:
        """Get GLSL type for node with validation."""
        if isinstance(node, ast.Call):
            # Handle math module functions
            if isinstance(node.func, ast.Attribute) and isinstance(
                node.func.value, ast.Name
            ):
                if node.func.value.id == "math":
                    # All math functions return float
                    return FLOAT

            if isinstance(node.func, ast.Name):
                func_name = node.func.id.lower()

                # Basic type constructors
                type_constructors = {
                    "float": FLOAT,
                    "int": INT,
                    "bool": BOOL,
                }
                if func_name in type_constructors:
                    if len(node.args) != 1:
                        raise GLSLTypeError(
                            f"{func_name} constructor requires exactly one argument"
                        )
                    arg_type = self.get_type(node.args[0])
                    if not can_convert_to(arg_type, type_constructors[func_name]):
                        raise GLSLTypeError(
                            f"Cannot convert type {arg_type} to {type_constructors[func_name]}"
                        )
                    return type_constructors[func_name]

                # Vector constructors
                vector_types = {
                    "vec2": (VEC2, 2),
                    "vec3": (VEC3, 3),
                    "vec4": (VEC4, 4),
                    "ivec2": (IVEC2, 2),
                    "ivec3": (IVEC3, 3),
                    "ivec4": (IVEC4, 4),
                    "bvec2": (BVEC2, 2),
                    "bvec3": (BVEC3, 3),
                    "bvec4": (BVEC4, 4),
                }

                if func_name in vector_types:
                    target_type, size = vector_types[func_name]
                    return target_type

                # Built-in functions
                builtin_types = {
                    "length": FLOAT,
                    "distance": FLOAT,
                    "dot": FLOAT,
                    "cross": VEC3,
                    "normalize": None,  # Returns same as input
                    "faceforward": None,  # Returns same as input
                    "reflect": None,  # Returns same as input
                    "refract": None,  # Returns same as input
                    "pow": FLOAT,
                    "exp": FLOAT,
                    "log": FLOAT,
                    "exp2": FLOAT,
                    "log2": FLOAT,
                    "sqrt": FLOAT,
                    "inversesqrt": FLOAT,
                    "abs": None,  # Returns same as input
                    "sign": None,  # Returns same as input
                    "floor": None,  # Returns same as input
                    "ceil": None,  # Returns same as input
                    "fract": None,  # Returns same as input
                    "mod": None,  # Returns same as input
                    "min": None,  # Returns same as input
                    "max": None,  # Returns same as input
                    "clamp": None,  # Returns same as input
                    "mix": None,  # Returns same as input
                    "step": None,  # Returns same as input
                    "smoothstep": None,  # Returns same as input
                    "sin": FLOAT,
                    "cos": FLOAT,
                    "tan": FLOAT,
                    "asin": FLOAT,
                    "acos": FLOAT,
                    "atan": FLOAT,
                }

                if func_name in builtin_types:
                    return_type = builtin_types[func_name]
                    if return_type is None:
                        # Function returns same type as input
                        return self.get_type(node.args[0])
                    return return_type

        elif isinstance(node, ast.Name):
            if node.id in self.analysis.var_types[self.current_scope]:
                return self.analysis.var_types[self.current_scope][node.id]
            if node.id in self.analysis.uniforms:
                return self.analysis.uniforms[node.id]
            if node.id == "vs_uv":
                return VEC2
            raise GLSLTypeError(f"Undefined variable: {node.id}")

        elif isinstance(node, ast.Attribute):
            value_type = self.get_type(node.value)
            if value_type.is_vector:
                try:
                    return value_type.validate_swizzle(node.attr)
                except GLSLSwizzleError as e:
                    raise GLSLTypeError(f"Invalid swizzle operation: {e}")
            raise GLSLTypeError(f"Cannot access attribute of type {value_type}")

        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return BOOL
            elif isinstance(node.value, int):
                return INT
            elif isinstance(node.value, float):
                return FLOAT
            raise GLSLTypeError(f"Unsupported constant type: {type(node.value)}")

        elif isinstance(node, ast.Compare):
            # All comparison operations return bool
            return BOOL

        elif isinstance(node, ast.UnaryOp):
            # Handle unary operations
            operand_type = self.get_type(node.operand)

            if isinstance(node.op, ast.USub):
                # Negation preserves type
                return operand_type
            elif isinstance(node.op, ast.UAdd):
                # Positive preserves type
                return operand_type
            elif isinstance(node.op, ast.Not):
                # Logical not returns bool
                if operand_type == BOOL or operand_type.is_bool_vector:
                    return operand_type
                raise GLSLTypeError(f"Cannot apply logical not to type {operand_type}")

            raise GLSLTypeError(f"Unsupported unary operator: {type(node.op)}")

        elif isinstance(node, ast.BinOp):
            left_type = self.get_type(node.left)
            right_type = self.get_type(node.right)
            operators = {
                ast.Add: "+",
                ast.Sub: "-",
                ast.Mult: "*",
                ast.Div: "/",
                ast.Mod: "%",
            }
            if type(node.op) not in operators:
                raise GLSLTypeError(f"Unsupported binary operator: {type(node.op)}")

            op = operators[type(node.op)]
            result_type = validate_operation(left_type, op, right_type)
            if result_type is None:
                raise GLSLTypeError(
                    f"Invalid operation {op} between types {left_type} and {right_type}"
                )
            return result_type

        raise GLSLTypeError(f"Cannot determine type for: {ast.dump(node)}")

    def generate_expression(self, node: ast.AST) -> str:
        """Generate formatted GLSL expression with type validation."""
        match node:
            case ast.Name():
                # Validate variable exists
                _ = self.get_type(node)
                return node.id

            case ast.Constant():
                return self._format_constant(node.value)

            case ast.Call():
                # Handle math module functions
                if isinstance(node.func, ast.Attribute) and isinstance(
                    node.func.value, ast.Name
                ):
                    if node.func.value.id == "math":
                        # Convert math.func() to func()
                        args = [self.generate_expression(arg) for arg in node.args]
                        return f"{node.func.attr}({', '.join(args)})"

                if isinstance(node.func, ast.Name):
                    return self._generate_call(node)

                raise GLSLTypeError(f"Invalid function call: {ast.dump(node)}")

            case ast.BinOp():
                return self._generate_binary_op(node)

            case ast.Compare():
                return self._generate_comparison(node)

            case ast.UnaryOp():
                return self._generate_unary_op(node)

            case ast.Attribute():
                return self._generate_attribute(node)

            case _:
                raise GLSLTypeError(f"Unsupported expression: {ast.dump(node)}")

    def _format_constant(self, value: object) -> str:
        """Format constant value for GLSL."""
        if isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, int):
            return str(value)
        elif isinstance(value, float):
            s = f"{value:g}"
            return f"{s}.0" if "." not in s and "e" not in s else s
        raise GLSLTypeError(f"Unsupported constant type: {type(value)}")

    def _generate_call(self, node: ast.Call) -> str:
        """Generate formatted function call with type validation."""
        func_name = node.func.id.lower()
        logger.debug(f"Generating call for function: {func_name}")

        # Type conversion functions
        type_conversions = {
            "float": "float",
            "int": "int",
            "bool": "bool",
        }
        if func_name in type_conversions:
            if len(node.args) != 1:
                raise GLSLTypeError(f"{func_name} requires exactly one argument")
            arg = self.generate_expression(node.args[0])
            return f"{type_conversions[func_name]}({arg})"

        # Vector constructor validation
        vector_types = {
            "vec2": (VEC2, 2),
            "vec3": (VEC3, 3),
            "vec4": (VEC4, 4),
            "ivec2": (IVEC2, 2),
            "ivec3": (IVEC3, 3),
            "ivec4": (IVEC4, 4),
            "bvec2": (BVEC2, 2),
            "bvec3": (BVEC3, 3),
            "bvec4": (BVEC4, 4),
        }

        if func_name in vector_types:
            target_type, expected_components = vector_types[func_name]
            total_components = 0

            # First, validate the constructor pattern
            if expected_components == 4:
                # Special validation for vec4 constructors
                if len(node.args) == 1:
                    # Single argument: must be scalar or vec4
                    arg_type = self.get_type(node.args[0])
                    if arg_type.is_vector and arg_type.vector_size() != 4:
                        raise GLSLTypeError(
                            f"Invalid number of components for {func_name} constructor: "
                            f"expected {expected_components}, got {arg_type.vector_size()}"
                        )
                elif len(node.args) == 2:
                    # Two arguments: must be vec3 + scalar
                    arg_type = self.get_type(node.args[0])
                    if not (arg_type.is_vector and arg_type.vector_size() == 3):
                        raise GLSLTypeError(
                            f"Invalid number of components for {func_name} constructor"
                        )
                elif len(node.args) == 3:
                    # Three arguments: must be vec2 + scalar + scalar
                    arg_type = self.get_type(node.args[0])
                    if not (arg_type.is_vector and arg_type.vector_size() == 2):
                        raise GLSLTypeError(
                            f"Invalid number of components for {func_name} constructor"
                        )
                elif len(node.args) == 4:
                    # Four arguments: all must be scalars
                    for arg in node.args:
                        arg_type = self.get_type(arg)
                        if arg_type.is_vector:
                            raise GLSLTypeError(
                                f"Invalid number of components for {func_name} constructor"
                            )
                else:
                    raise GLSLTypeError(
                        f"Invalid number of components for {func_name} constructor"
                    )

            # Count total components
            for arg in node.args:
                arg_type = self.get_type(arg)
                if arg_type.is_vector:
                    size = arg_type.vector_size()
                    if size is None:
                        raise GLSLTypeError(f"Invalid vector type: {arg_type}")
                    total_components += size
                else:
                    total_components += 1

            if total_components != expected_components:
                raise GLSLTypeError(
                    f"Invalid number of components for {func_name} constructor: "
                    f"expected {expected_components}, got {total_components}"
                )

        # Generate expressions after validation
        args = [self.generate_expression(arg) for arg in node.args]

        if func_name in vector_types:
            return f"{func_name}({', '.join(args)})"

        # Built-in function validation
        builtin_functions = {
            "length",
            "distance",
            "dot",
            "cross",
            "normalize",
            "sin",
            "cos",
            "tan",
            "asin",
            "acos",
            "atan",
            "pow",
            "exp",
            "log",
            "sqrt",
            "abs",
            "sign",
            "floor",
            "ceil",
            "fract",
            "min",
            "max",
            "clamp",
            "mix",
            "step",
            "smoothstep",
            "mod",
        }

        if func_name in builtin_functions:
            # This will validate the function call and raise if invalid
            _ = self.get_type(node)
            return f"{func_name}({', '.join(args)})"

        raise GLSLTypeError(f"Unknown function: {func_name}")

    def _generate_binary_op(self, node: ast.BinOp) -> str:
        """Generate formatted binary operation with type validation."""
        left_type = self.get_type(node.left)
        right_type = self.get_type(node.right)

        operators = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.Mod: "%",
        }

        if type(node.op) not in operators:
            raise GLSLTypeError(f"Unsupported binary operator: {type(node.op)}")

        op = operators[type(node.op)]

        # Validate operation using type system
        result_type = validate_operation(left_type, op, right_type)
        if result_type is None:
            raise GLSLTypeError(
                f"Invalid operation {op} between types {left_type} and {right_type}"
            )

        left = self.generate_expression(node.left)
        right = self.generate_expression(node.right)

        return f"{left} {op} {right}"

    def _generate_comparison(self, node: ast.Compare) -> str:
        """Generate formatted comparison with type validation."""
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise GLSLTypeError("Only simple comparisons are supported")

        operators = {
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
        }

        left_type = self.get_type(node.left)
        right_type = self.get_type(node.comparators[0])
        op = operators[type(node.ops[0])]

        # Validate comparison using type system
        result_type = validate_operation(left_type, op, right_type)
        if result_type != BOOL:
            raise GLSLTypeError(
                f"Invalid comparison {op} between types {left_type} and {right_type}"
            )

        left = self.generate_expression(node.left)
        right = self.generate_expression(node.comparators[0])

        return f"{left} {op} {right}"

    def _generate_unary_op(self, node: ast.UnaryOp) -> str:
        """Generate formatted unary operation with type validation."""
        operators = {
            ast.UAdd: "+",
            ast.USub: "-",
            ast.Not: "!",
        }

        if type(node.op) not in operators:
            raise GLSLTypeError(f"Unsupported unary operator: {type(node.op)}")

        op = operators[type(node.op)]
        operand_type = self.get_type(node.operand)

        # Validate unary operation
        if isinstance(node.op, ast.Not) and not (
            operand_type == BOOL or operand_type.is_bool_vector
        ):
            raise GLSLTypeError(f"Cannot apply logical not to type {operand_type}")

        if isinstance(node.op, (ast.UAdd, ast.USub)) and not operand_type.is_numeric:
            raise GLSLTypeError(f"Cannot apply unary {op} to type {operand_type}")

        operand = self.generate_expression(node.operand)
        return f"{op}{operand}"

    def _generate_attribute(self, node: ast.Attribute) -> str:
        """Generate formatted attribute access with swizzle validation."""
        value_type = self.get_type(node.value)

        if value_type.is_vector:
            try:
                _ = value_type.validate_swizzle(node.attr)
            except GLSLSwizzleError as e:
                raise GLSLTypeError(f"Invalid swizzle operation: {e}")
        else:
            raise GLSLTypeError(f"Cannot access attribute of type {value_type}")

        value = self.generate_expression(node.value)
        return f"{value}.{node.attr}"

    def _generate_assignment(self, node: ast.Assign) -> None:
        """Generate formatted assignment with type validation."""
        value_type = self.get_type(node.value)
        value = self.generate_expression(node.value)

        for target in node.targets:
            if not isinstance(target, ast.Name):
                raise GLSLTypeError("Only simple assignments are supported")

            target_type = self.analysis.var_types[self.current_scope].get(target.id)
            if not target_type:
                raise GLSLTypeError(f"Unknown variable: {target.id}")

            # Validate assignment compatibility
            if not can_convert_to(value_type, target_type):
                raise GLSLTypeError(
                    f"Cannot assign value of type {value_type} to variable of type {target_type}"
                )

            if target.id not in self.declared_vars[self.current_scope]:
                self.add_line(f"{target_type!s} {target.id} = {value};")
                self.declared_vars[self.current_scope].add(target.id)
            else:
                self.add_line(f"{target.id} = {value};")

    def _generate_annotated_assignment(self, node: ast.AnnAssign) -> None:
        """Generate annotated assignment with type validation."""
        if not isinstance(node.target, ast.Name):
            raise GLSLTypeError("Only simple assignments are supported")

        target_type = self.analysis.var_types[self.current_scope].get(node.target.id)
        if not target_type:
            raise GLSLTypeError(f"Unknown variable: {node.target.id}")

        if node.value:
            value_type = self.get_type(node.value)
            value = self.generate_expression(node.value)

            # Validate assignment compatibility
            if not can_convert_to(value_type, target_type):
                raise GLSLTypeError(
                    f"Cannot assign value of type {value_type} to variable of type {target_type}"
                )

            if node.target.id not in self.declared_vars[self.current_scope]:
                self.add_line(f"{target_type!s} {node.target.id} = {value};")
                self.declared_vars[self.current_scope].add(node.target.id)
            else:
                self.add_line(f"{node.target.id} = {value};")
        else:
            self.add_line(f"{target_type!s} {node.target.id};")
            self.declared_vars[self.current_scope].add(node.target.id)

    def generate_function(self, node: ast.FunctionDef) -> None:
        """Generate formatted function with type validation."""
        self.enter_scope(node.name)

        try:
            # Get return type and validate it exists
            return_type = self.analysis.var_types["global"].get(node.name)
            if not return_type:
                raise GLSLTypeError(f"Unknown return type for function: {node.name}")

            # Process arguments with type validation
            args = []
            for arg in node.args.args:
                arg_type = self.analysis.var_types[node.name].get(arg.arg)
                if not arg_type:
                    raise GLSLTypeError(f"Unknown type for argument: {arg.arg}")
                args.append(f"{arg_type!s} {arg.arg}")
                self.declared_vars[self.current_scope].add(arg.arg)

            # Generate function declaration
            self.add_line(f"{return_type!s} {node.name}({', '.join(args)})")
            self.begin_block()

            # Generate all variable declarations at the start
            hoisted = sorted(
                var
                for var in self.analysis.hoisted_vars[node.name]
                if var not in {arg.arg for arg in node.args.args}
            )

            if hoisted:
                for var_name in hoisted:
                    var_type = self.analysis.var_types[node.name].get(var_name)
                    if not var_type:
                        raise GLSLTypeError(f"Unknown type for variable: {var_name}")
                    self.add_line(f"{var_type!s} {var_name};")
                    self.declared_vars[self.current_scope].add(var_name)
                self.add_line("")

            # Generate function body
            for stmt in node.body:
                if isinstance(stmt, ast.FunctionDef):
                    # Nested functions are not supported in GLSL
                    raise GLSLTypeError("Nested functions are not supported in GLSL")
                self.generate_statement(stmt)

            self.end_block()

        finally:
            self.exit_scope()

    def generate_statement(self, node: ast.AST) -> None:
        """Generate formatted GLSL statement with type validation."""
        match node:
            case ast.Assign():
                self._generate_assignment(node)

            case ast.AnnAssign():
                self._generate_annotated_assignment(node)

            case ast.AugAssign():
                self._generate_aug_assignment(node)

            case ast.If():
                self._generate_if_statement(node)

            case ast.For():
                self._generate_for_loop(node)

            case ast.Return():
                self._generate_return(node)

            case ast.Break():
                self.add_line("break;")

            case ast.Continue():
                self.add_line("continue;")

            case ast.Expr():
                expr = self.generate_expression(node.value)
                self.add_line(f"{expr};")

            case _:
                raise GLSLTypeError(f"Unsupported statement: {type(node)}")

    def _generate_if_statement(self, node: ast.If) -> None:
        """Generate if statement with type validation."""
        # Validate condition type
        condition_type = self.get_type(node.test)
        if condition_type != BOOL:
            raise GLSLTypeError(f"If condition must be boolean, got {condition_type}")

        # Generate condition
        condition = self.generate_expression(node.test)
        self.add_line(f"if ({condition})")

        # Generate if body
        self.begin_block()
        for stmt in node.body:
            self.generate_statement(stmt)
        self.end_block()

        # Generate else block if present
        if node.orelse:
            self.add_line("else")
            self.begin_block()
            for stmt in node.orelse:
                self.generate_statement(stmt)
            self.end_block()

    def _generate_for_loop(self, node: ast.For) -> None:
        """Generate for loop with type validation."""
        if not isinstance(node.target, ast.Name):
            raise GLSLTypeError("Only simple loop variables are supported")

        if not (
            isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
        ):
            raise GLSLTypeError("Only range-based for loops are supported")

        # Validate range arguments
        for arg in node.iter.args:
            arg_type = self.get_type(arg)
            if not can_convert_to(arg_type, INT):
                raise GLSLTypeError(f"Range argument must be integer, got {arg_type}")

        # Handle range arguments
        if len(node.iter.args) == 1:
            end = self.generate_expression(node.iter.args[0])
            init = f"int {node.target.id} = 0"
            cond = f"{node.target.id} < {end}"
        elif len(node.iter.args) == 2:
            start = self.generate_expression(node.iter.args[0])
            end = self.generate_expression(node.iter.args[1])
            init = f"int {node.target.id} = {start}"
            cond = f"{node.target.id} < {end}"
        else:
            raise GLSLTypeError("Range with step not supported in GLSL")

        # Register loop variable in current scope
        self.analysis.var_types[self.current_scope][node.target.id] = INT
        self.declared_vars[self.current_scope].add(node.target.id)

        # Generate for loop
        self.add_line(f"for ({init}; {cond}; {node.target.id}++)")

        # Generate loop body
        self.begin_block()
        for stmt in node.body:
            self.generate_statement(stmt)
        self.end_block()

    def _generate_return(self, node: ast.Return) -> None:
        """Generate return statement with type validation."""
        if node.value is None:
            raise GLSLTypeError("Return value is required")

        return_type = self.analysis.var_types["global"][self.current_scope]
        value_type = self.get_type(node.value)

        # Validate return type compatibility
        if not can_convert_to(value_type, return_type):
            raise GLSLTypeError(
                f"Cannot return value of type {value_type} from function returning {return_type}"
            )

        value = self.generate_expression(node.value)
        self.add_line(f"return {value};")

    def _generate_aug_assignment(self, node: ast.AugAssign) -> None:
        """Generate augmented assignment with type validation."""
        target_type = self.get_type(node.target)
        value_type = self.get_type(node.value)

        operators = {
            ast.Add: "+=",
            ast.Sub: "-=",
            ast.Mult: "*=",
            ast.Div: "/=",
            ast.Mod: "%=",
        }

        if type(node.op) not in operators:
            raise GLSLTypeError(
                f"Unsupported augmented assignment operator: {type(node.op)}"
            )

        op = operators[type(node.op)]
        base_op = op[0]  # Get the basic operator without '='

        # Validate operation using type system
        result_type = validate_operation(target_type, base_op, value_type)
        if result_type is None or not can_convert_to(result_type, target_type):
            raise GLSLTypeError(
                f"Invalid augmented assignment {op} between types {target_type} and {value_type}"
            )

        target = self.generate_expression(node.target)
        value = self.generate_expression(node.value)
        self.add_line(f"{target} {op} {value};")

    def generate(self) -> GeneratedShader:
        """Generate complete formatted GLSL shader code."""
        logger.debug("Starting GLSL code generation")

        # Version declaration
        self.add_line(f"#version {GLSL_VERSION}")
        self.add_line()

        # Input from vertex shader
        self.add_line("in vec2 vs_uv;")  # Input from vertex shader
        self.add_line("out vec4 fs_color;")
        self.add_line()

        # Ensure built-in uniforms are included
        all_uniforms = {
            **BUILTIN_UNIFORMS,  # Add built-in uniforms first
            **self.analysis.uniforms,  # Then add shader-specific uniforms
        }

        # Uniform declarations (sorted for consistent output)
        for name, glsl_type in sorted(all_uniforms.items()):
            uniform_type = GLSLType(
                kind=glsl_type.kind, is_uniform=True, array_size=glsl_type.array_size
            )
            self.add_line(f"uniform {uniform_type.name} {name};")
        if all_uniforms:
            self.add_line()

        # Generate functions
        for func in self.analysis.functions:
            self.generate_function(func)
            self.add_line()

        # Generate main shader function if exists
        if self.analysis.main_function:
            self.generate_function(self.analysis.main_function)
            self.add_line()

            # Generate main function
            self.add_line("void main()")
            self.begin_block()
            self.add_line(f"fs_color = {self.analysis.main_function.name}(vs_uv);")
            self.end_block()

        return GeneratedShader(
            vertex_source=VERTEX_SHADER,
            fragment_source="\n".join(self.lines),
            uniforms=all_uniforms,  # Return all uniforms including built-ins
        )
