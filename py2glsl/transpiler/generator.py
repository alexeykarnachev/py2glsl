"""GLSL code generator with type system integration."""

import ast
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from py2glsl.transpiler.analyzer import GLSLContext, ShaderAnalysis
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
    VOID,
    GLSLSwizzleError,
    GLSLType,
    GLSLTypeError,
    can_convert_to,
    validate_operation,
)

# Vertex shader template
VERTEX_SHADER = """#version 460
layout(location = 0) in vec2 in_pos;
layout(location = 1) in vec2 in_uv;
out vec2 vs_uv;

void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    vs_uv = in_uv;
}
"""


@dataclass
class GeneratedShader:
    """Result of shader generation."""

    vertex_source: str
    fragment_source: str


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
            if node.id in self.analysis.var_types[self.current_scope]:
                return self.analysis.var_types[self.current_scope][node.id]
            if node.id in self.analysis.uniforms:
                return self.analysis.uniforms[node.id]
            if node.id == "vs_uv":
                return VEC2
            raise GLSLTypeError(f"Undefined variable: {node.id}")

        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return BOOL
            elif isinstance(node.value, int):
                return INT
            elif isinstance(node.value, float):
                return FLOAT
            raise GLSLTypeError(f"Unsupported constant type: {type(node.value)}")

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

            case ast.Call() if isinstance(node.func, ast.Name):
                return self._generate_call(node)

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
        func_name = node.func.id
        arg_types = [self.get_type(arg) for arg in node.args]
        args = [self.generate_expression(arg) for arg in node.args]

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

        if func_name.lower() in vector_types:
            target_type, size = vector_types[func_name.lower()]

            # Validate vector construction
            total_components = sum(
                arg_type.vector_size() if arg_type.is_vector else 1
                for arg_type in arg_types
            )
            if total_components != size:
                raise GLSLTypeError(
                    f"Invalid number of components for {func_name} construction"
                )

            return f"{func_name.lower()}({', '.join(args)})"

        # Built-in function validation
        if func_name in self.analysis.var_types["global"]:
            expected_type = self.analysis.var_types["global"][func_name]
            # Validate argument types here if needed
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

            # Generate hoisted variable declarations with type validation
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
        self.add_line("#version 460")
        self.add_line()

        # Input/output declarations
        self.add_line("in vec2 vs_uv;")
        self.add_line("out vec4 fs_color;")
        self.add_line()

        # Uniform declarations
        for name, glsl_type in sorted(self.analysis.uniforms.items()):
            self.add_line(f"uniform {glsl_type!s} {name};")
        if self.analysis.uniforms:
            self.add_line()

        # Generate functions
        for func in self.analysis.functions:
            self.generate_function(func)
            self.add_line()

        # Generate main shader function
        main_func = self.analysis.main_function
        self.generate_function(main_func)
        self.add_line()

        # Generate main function
        self.add_line("void main()")
        self.begin_block()
        self.add_line(f"fs_color = {main_func.name}(vs_uv);")
        self.end_block()

        return GeneratedShader(
            vertex_source=VERTEX_SHADER, fragment_source="\n".join(self.lines)
        )
