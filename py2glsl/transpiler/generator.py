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
from .operators import (
    AUGASSIGN_OPERATORS,
    BINARY_OPERATORS,
    COMPARISON_OPERATORS,
    UNARY_OPERATORS,
)
from .type_mappings import (
    BUILTIN_FUNCTIONS,
    BUILTIN_FUNCTIONS_ARGS,
    BUILTIN_TYPES,
    MATRIX_CONSTRUCTORS,
    TYPE_CONSTRUCTORS,
    VALID_VECTOR_COMBINATIONS,
    VECTOR_CONSTRUCTORS,
    VECTOR_TYPES,
)


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
        if isinstance(node, ast.Name):
            # Check current scope first
            if node.id in self.analysis.var_types[self.current_scope]:
                return self.analysis.var_types[self.current_scope][node.id]

            # Check parent scopes
            for scope in reversed(self.scope_stack):
                if node.id in self.analysis.var_types[scope]:
                    return self.analysis.var_types[scope][node.id]

            # Check uniforms and built-ins
            if node.id in self.analysis.uniforms:
                return self.analysis.uniforms[node.id]
            if node.id == "vs_uv":
                return VEC2

            raise GLSLTypeError(f"Undefined variable: {node.id}")

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise GLSLTypeError("Only simple function calls are supported")

            func_name = node.func.id.lower()

            if func_name in VECTOR_TYPES:
                target_type, size = VECTOR_TYPES[func_name]
                # Count total components for all arguments
                total_components = 0
                for arg in node.args:
                    arg_type = self.get_type(arg)
                    if arg_type.is_vector:
                        total_components += arg_type.vector_size()
                    else:
                        total_components += 1

                # Special case: single scalar argument - expands to fill all components
                if len(node.args) == 1 and not self.get_type(node.args[0]).is_vector:
                    return target_type

                if total_components != size:
                    raise GLSLTypeError(
                        f"Invalid number of components for {func_name} constructor: "
                        f"expected {size}, got {total_components}"
                    )

                return target_type

            if func_name in TYPE_CONSTRUCTORS:
                if len(node.args) != 1:
                    raise GLSLTypeError(
                        f"{func_name} constructor requires exactly one argument"
                    )
                arg_type = self.get_type(node.args[0])
                if not can_convert_to(arg_type, TYPE_CONSTRUCTORS[func_name]):
                    raise GLSLTypeError(
                        f"Cannot convert type {arg_type} to {TYPE_CONSTRUCTORS[func_name]}"
                    )
                return TYPE_CONSTRUCTORS[func_name]

            if func_name in BUILTIN_TYPES:
                return_type = BUILTIN_TYPES[func_name]
                if return_type is None:
                    # Function returns same type as input
                    return self.get_type(node.args[0])
                return return_type

            # Handle math module functions
            if isinstance(node.func, ast.Attribute) and isinstance(
                node.func.value, ast.Name
            ):
                if node.func.value.id == "math":
                    # All math functions return float
                    return FLOAT

            raise GLSLTypeError(f"Unknown function: {func_name}")

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

        elif isinstance(node, ast.BinOp):
            left_type = self.get_type(node.left)
            right_type = self.get_type(node.right)
            if type(node.op) not in BINARY_OPERATORS:
                raise GLSLTypeError(f"Unsupported binary operator: {type(node.op)}")

            op = BINARY_OPERATORS[type(node.op)]
            result_type = validate_operation(left_type, op, right_type)
            if result_type is None:
                raise GLSLTypeError(
                    f"Invalid operation {op} between types {left_type} and {right_type}"
                )
            return result_type

        elif isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                # Validate all values are boolean
                for value in node.values:
                    value_type = self.get_type(value)
                    if value_type != BOOL:
                        raise GLSLTypeError(
                            f"Boolean operation requires bool operands, got {value_type}"
                        )
                return BOOL
            raise GLSLTypeError("Only 'and' operations are supported")

        elif isinstance(node, ast.UnaryOp):
            operand_type = self.get_type(node.operand)
            if isinstance(node.op, ast.UAdd | ast.USub):
                return operand_type
            elif isinstance(node.op, ast.Not):
                if operand_type == BOOL or operand_type.is_bool_vector:
                    return operand_type
                raise GLSLTypeError(f"Cannot apply logical not to type {operand_type}")
            raise GLSLTypeError(f"Unsupported unary operator: {type(node.op)}")

        elif isinstance(node, ast.Attribute):
            value_type = self.get_type(node.value)
            if value_type.is_vector:
                try:
                    return value_type.validate_swizzle(node.attr)
                except GLSLSwizzleError as e:
                    raise GLSLTypeError(f"Invalid swizzle operation: {e}")
            raise GLSLTypeError(f"Cannot access attribute of type {value_type}")

        raise GLSLTypeError(f"Unsupported node type: {type(node)}")

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

            case ast.BoolOp():
                if isinstance(node.op, ast.And):
                    # Generate each condition and join with &&
                    conditions = [
                        self.generate_expression(value) for value in node.values
                    ]
                    return f"({' && '.join(conditions)})"
                raise GLSLTypeError("Only 'and' operations are supported")

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
        if not isinstance(node.func, ast.Name):
            raise GLSLTypeError("Only simple function calls are supported")

        func_name = node.func.id.lower()

        if func_name in VECTOR_CONSTRUCTORS:
            size = VECTOR_CONSTRUCTORS[func_name]

            # Special case: single scalar argument fills all components
            if len(node.args) == 1:
                arg_type = self.get_type(node.args[0])
                if not arg_type.is_vector:
                    arg = self.generate_expression(node.args[0])
                    return f"{func_name}({arg})"
                # Single vector argument must match size exactly
                if arg_type.vector_size() != size:
                    raise GLSLTypeError(f"Cannot construct {func_name} from {arg_type}")

            # Get component sizes for all arguments
            arg_sizes = []
            for arg in node.args:
                arg_type = self.get_type(arg)
                if arg_type.is_vector:
                    arg_sizes.append(arg_type.vector_size())
                else:
                    arg_sizes.append(1)

            # Check if argument combination is valid
            if tuple(arg_sizes) not in VALID_VECTOR_COMBINATIONS[size]:
                raise GLSLTypeError(
                    f"Invalid arguments for {func_name} constructor: "
                    f"cannot construct from components {arg_sizes}"
                )

            # Generate constructor call
            args = [self.generate_expression(arg) for arg in node.args]
            return f"{func_name}({', '.join(args)})"

        # Handle type conversions
        if func_name in ("int", "float", "bool"):
            if len(node.args) != 1:
                raise GLSLTypeError(f"{func_name} requires exactly one argument")
            arg = self.generate_expression(node.args[0])
            return f"{func_name}({arg})"

        # Handle built-in functions
        if func_name in BUILTIN_FUNCTIONS:
            args = [self.generate_expression(arg) for arg in node.args]

            # Validate argument count using BUILTIN_FUNCTIONS_ARGS
            if func_name in BUILTIN_FUNCTIONS_ARGS:
                expected_args = BUILTIN_FUNCTIONS_ARGS[func_name]
                if len(args) != expected_args:
                    raise GLSLTypeError(
                        f"{func_name} requires exactly {expected_args} arguments"
                    )

            return f"{func_name}({', '.join(args)})"

        # Handle matrix constructors
        if func_name in MATRIX_CONSTRUCTORS:
            size = MATRIX_CONSTRUCTORS[func_name]
            args = [self.generate_expression(arg) for arg in node.args]

            # Count total components
            total_components = 0
            for arg in node.args:
                arg_type = self.get_type(arg)
                if arg_type.is_matrix:
                    total_components += arg_type.matrix_size() * arg_type.matrix_size()
                elif arg_type.is_vector:
                    total_components += arg_type.vector_size()
                else:
                    total_components += 1

            if total_components != size:
                raise GLSLTypeError(
                    f"Invalid number of components for {func_name} constructor: "
                    f"expected {size}, got {total_components}"
                )

            return f"{func_name}({', '.join(args)})"

        # Check if it's a user-defined function
        if func_name in self.analysis.var_types["global"]:
            args = [self.generate_expression(arg) for arg in node.args]
            return f"{func_name}({', '.join(args)})"

        raise GLSLTypeError(f"Unknown function: {func_name}")

    def _generate_binary_op(self, node: ast.BinOp) -> str:
        """Generate formatted binary operation with type validation."""
        left_type = self.get_type(node.left)
        right_type = self.get_type(node.right)

        if type(node.op) not in BINARY_OPERATORS:
            raise GLSLTypeError(f"Unsupported binary operator: {type(node.op)}")

        op = BINARY_OPERATORS[type(node.op)]

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

        left_type = self.get_type(node.left)
        right_type = self.get_type(node.comparators[0])
        op = COMPARISON_OPERATORS[type(node.ops[0])]

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

        if type(node.op) not in UNARY_OPERATORS:
            raise GLSLTypeError(f"Unsupported unary operator: {type(node.op)}")

        op = UNARY_OPERATORS[type(node.op)]
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
        try:
            value_type = self.get_type(node.value)
            value = self.generate_expression(node.value)

            for target in node.targets:
                if not isinstance(target, ast.Name):
                    raise GLSLTypeError("Only simple assignments are supported")

                # Check if variable exists in any parent scope
                target_type = None
                current = self.current_scope
                while current:
                    if target.id in self.analysis.var_types[current]:
                        target_type = self.analysis.var_types[current][target.id]
                        break
                    current = self.scope_stack[-1] if self.scope_stack else None

                # If variable doesn't exist anywhere, declare it
                if not target_type:
                    target_type = value_type
                    self.analysis.var_types[self.current_scope][target.id] = value_type

                # Validate assignment compatibility
                if not can_convert_to(value_type, target_type):
                    raise GLSLTypeError(
                        f"Cannot assign value of type {value_type} to variable of type {target_type}"
                    )

                # Generate declaration or assignment
                if target.id not in self.declared_vars[self.current_scope]:
                    self.add_line(f"{target_type!s} {target.id} = {value};")
                    self.declared_vars[self.current_scope].add(target.id)
                else:
                    self.add_line(f"{target.id} = {value};")

        except Exception as e:
            raise GLSLTypeError(f"Error in assignment to {target.id}: {e!s}") from e

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
        """Generate GLSL function with proper scope handling."""
        # Enter new scope
        self.enter_scope(node.name)

        try:
            # Get return type
            return_type = self.analysis.var_types["global"].get(node.name)
            if not return_type:
                raise GLSLTypeError(f"Unknown return type for function: {node.name}")

            # Process arguments
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

            # Generate variable declarations for hoisted variables
            hoisted = sorted(
                var
                for var in self.analysis.hoisted_vars[node.name]
                if var not in {arg.arg for arg in node.args.args}
            )

            # Declare all variables at the start of function
            for var_name in hoisted:
                var_type = self.analysis.var_types[node.name].get(var_name)
                if var_type:
                    self.add_line(f"{var_type!s} {var_name};")
                    self.declared_vars[self.current_scope].add(var_name)

            if hoisted:
                self.add_line("")

            # Generate function body
            for stmt in node.body:
                self.generate_statement(stmt)

            self.end_block()
            self.add_line("")

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

            case ast.Pass():
                # Skip pass statements - they have no GLSL equivalent
                return

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

        # Validate range arguments are integer literals
        for arg in node.iter.args:
            if isinstance(arg, ast.Constant):
                if isinstance(arg.value, float):
                    raise GLSLTypeError(
                        f"Range argument must be integer, got float {arg.value}"
                    )
            # For non-constants, check type compatibility
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

        if type(node.op) not in AUGASSIGN_OPERATORS:
            raise GLSLTypeError(
                f"Unsupported augmented assignment operator: {type(node.op)}"
            )

        op = AUGASSIGN_OPERATORS[type(node.op)]
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
            logger.debug(f"Generating function: {func.name}")
            self.generate_function(func)
            self.add_line()

        # Generate main shader function if exists
        if self.analysis.main_function:
            logger.debug("Generating main shader function")
            self.generate_function(self.analysis.main_function)
            self.add_line()

            logger.debug("Generating main() wrapper")
            self.add_line("void main()")
            self.begin_block()
            self.add_line(f"fs_color = {self.analysis.main_function.name}(vs_uv);")
            self.end_block()

        return GeneratedShader(
            vertex_source=VERTEX_SHADER,
            fragment_source="\n".join(self.lines),
            uniforms=all_uniforms,  # Return all uniforms including built-ins
        )
