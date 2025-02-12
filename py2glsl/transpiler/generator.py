"""GLSL code generator."""

import ast

from loguru import logger

from py2glsl.transpiler.analyzer import GLSLContext, ShaderAnalysis
from py2glsl.transpiler.formatter import GLSLFormatter
from py2glsl.types import BOOL, FLOAT, INT, VEC2, GLSLType


class GLSLGenerator:
    """Generates GLSL code from analyzed shader."""

    def __init__(self, analysis: ShaderAnalysis):
        """Initialize generator with analysis results."""
        self.analysis = analysis
        self.indent_level = 0
        self.lines: list[str] = []
        self.current_scope = "global"
        self.scope_stack: list[str] = []
        self.current_context = GLSLContext.DEFAULT
        self.context_stack: list[GLSLContext] = []
        self.declared_vars: dict[str, set[str]] = {
            "global": set()
        }  # Initialize with global scope

    def push_context(self, ctx: GLSLContext) -> None:
        """Push new context."""
        self.context_stack.append(self.current_context)
        self.current_context = ctx

    def pop_context(self) -> None:
        """Pop context."""
        if self.context_stack:
            self.current_context = self.context_stack.pop()
        else:
            self.current_context = GLSLContext.DEFAULT

    def enter_scope(self, name: str) -> None:
        """Enter a new scope."""
        self.scope_stack.append(self.current_scope)
        self.current_scope = name
        if name not in self.declared_vars:
            self.declared_vars[name] = set()  # Create new set for this scope

    def exit_scope(self) -> None:
        """Exit current scope."""
        if self.scope_stack:
            self.current_scope = self.scope_stack.pop()
        else:
            self.current_scope = "global"

    def indent(self) -> str:
        """Get current indentation."""
        return "    " * self.indent_level

    def add_line(self, line: str = "") -> None:
        """Add line with proper indentation."""
        if line.strip():
            # Don't indent version, in/out declarations, and uniforms
            if any(
                line.startswith(prefix)
                for prefix in ["#version", "in ", "out ", "uniform "]
            ):
                self.lines.append(line)
            else:
                self.lines.append(f"{self.indent()}{line}")
        else:
            self.lines.append("")

    def get_type(self, node: ast.AST) -> GLSLType:
        """Get GLSL type for node."""
        if isinstance(node, ast.Name):
            # Check variable type in current scope
            if node.id in self.analysis.var_types[self.current_scope]:
                return self.analysis.var_types[self.current_scope][node.id]
            # Check uniforms
            if node.id in self.analysis.uniforms:
                return self.analysis.uniforms[node.id]
            # Special case for vs_uv
            if node.id == "vs_uv":
                return VEC2
        elif isinstance(node, ast.Num):
            return INT if isinstance(node.n, int) else FLOAT
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return BOOL
            elif isinstance(node.value, int):
                return INT
            elif isinstance(node.value, float):
                return FLOAT
        return FLOAT

    def generate_expression(self, node: ast.AST, parenthesize: bool = False) -> str:
        """Generate GLSL expression with improved pattern matching."""
        match node:
            case ast.Name():
                return node.id

            case ast.Constant():
                return self._generate_constant(node, self.current_context)

            case ast.Call() if isinstance(node.func, ast.Name):
                return self._generate_call(node)

            case ast.BinOp():
                return self._generate_binary_op(node, parenthesize)

            case ast.Compare():
                return self._generate_compare(node, parenthesize)

            case ast.Attribute():
                return self._generate_attribute(node)

            case ast.UnaryOp():
                return self._generate_unary_op(node, parenthesize)

            case _:
                raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    def _generate_constant(self, node: ast.Constant, context: GLSLContext) -> str:
        """Generate GLSL code for constants."""
        match node.value:
            case bool():
                return str(node.value).lower()
            case int():
                return (
                    str(node.value)
                    if context == GLSLContext.LOOP
                    else f"{float(node.value)}"
                )
            case float():
                return f"{node.value}"
            case _:
                return "0.0"

    def _generate_expression_legacy(self, node: ast.AST, parenthesize: bool) -> str:
        """Legacy expression generation for non-refactored cases."""
        if isinstance(node, ast.Compare):
            left = self.generate_expression(node.left, True)
            right = self.generate_expression(node.comparators[0], True)
            op = {
                ast.Lt: "<",
                ast.LtE: "<=",
                ast.Gt: ">",
                ast.GtE: ">=",
                ast.Eq: "==",
                ast.NotEq: "!=",
            }[type(node.ops[0])]
            expr = f"{left} {op} {right}"
            return f"({expr})" if parenthesize else expr

        elif isinstance(node, ast.Attribute):
            value = self.generate_expression(node.value)
            return f"{value}.{node.attr}"

        elif isinstance(node, ast.UnaryOp):
            op = {ast.USub: "-", ast.UAdd: "+"}[type(node.op)]
            operand = self.generate_expression(node.operand, True)
            expr = f"{op}{operand}"
            return f"({expr})" if parenthesize else expr

        raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    def _generate_call(self, node: ast.Call) -> str:
        """Generate GLSL code for function calls."""
        args = [self.generate_expression(arg) for arg in node.args]

        # Handle vector constructors
        if node.func.id.lower() in {"vec2", "vec3", "vec4"}:
            return f"{node.func.id.lower()}({', '.join(args)})"

        return f"{node.func.id}({', '.join(args)})"

    def _generate_compare(self, node: ast.Compare, parenthesize: bool) -> str:
        """Generate GLSL code for comparison operations."""
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ValueError("Only simple comparisons are supported")

        operators = {
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
            ast.Eq: "==",
            ast.NotEq: "!=",
        }

        left = self.generate_expression(node.left, True)
        right = self.generate_expression(node.comparators[0], True)
        op = operators[type(node.ops[0])]

        expr = f"{left} {op} {right}"
        return f"({expr})" if parenthesize else expr

    def _generate_attribute(self, node: ast.Attribute) -> str:
        """Generate GLSL code for attribute access (e.g., swizzling)."""
        value = self.generate_expression(node.value)
        value_type = self.get_type(node.value)

        if value_type.is_vector():
            valid_components = {
                "x",
                "y",
                "z",
                "w",
                "r",
                "g",
                "b",
                "a",
                "s",
                "t",
                "p",
                "q",
            }
            if not all(c in valid_components for c in node.attr):
                raise ValueError(f"Invalid vector component access: {node.attr}")

        return f"{value}.{node.attr}"

    def _generate_unary_op(self, node: ast.UnaryOp, parenthesize: bool) -> str:
        """Generate GLSL code for unary operations."""
        operators = {
            ast.USub: "-",
            ast.UAdd: "+",
            ast.Not: "!",
        }

        if type(node.op) not in operators:
            raise ValueError(f"Unsupported unary operator: {type(node.op)}")

        op = operators[type(node.op)]
        operand = self.generate_expression(node.operand, True)
        expr = f"{op}{operand}"

        return f"({expr})" if parenthesize else expr

    def _validate_type_compatibility(
        self, expected: GLSLType, actual: GLSLType, context: str
    ) -> None:
        """Validate type compatibility for operations."""
        if not actual.can_convert_to(expected):
            raise TypeError(
                f"Type mismatch in {context}: "
                f"expected {expected.name}, got {actual.name}"
            )

    def _generate_binary_op(self, node: ast.BinOp, parenthesize: bool) -> str:
        """Generate GLSL code for binary operations."""
        operators = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
        }
        left = self.generate_expression(node.left, True)
        right = self.generate_expression(node.right, True)
        op = operators[type(node.op)]
        expr = f"{left} {op} {right}"
        return f"({expr})" if parenthesize else expr

    def generate_statement(self, node: ast.AST) -> None:
        """Generate GLSL statement with improved pattern matching."""
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
                value = self.generate_expression(node.value)
                self.add_line(f"return {value};")

            case ast.Break():
                self.add_line("break;")

            case ast.Continue():
                self.add_line("continue;")

            case ast.Expr():
                value = self.generate_expression(node.value)
                self.add_line(f"{value};")

            case ast.FunctionDef():
                self.generate_function(node)

            case _:
                raise ValueError(f"Unsupported statement: {type(node)}")

    def _generate_assignment(self, node: ast.Assign) -> None:
        """Generate GLSL assignment statement."""
        value = self.generate_expression(node.value)

        for target in node.targets:
            if not isinstance(target, ast.Name):
                raise ValueError("Only simple assignments are supported")

            var_type = self.analysis.var_types[self.current_scope].get(target.id)

            if target.id not in self.declared_vars[self.current_scope]:
                # Declaration with initialization
                if isinstance(node.value, ast.Call) and isinstance(
                    node.value.func, ast.Name
                ):
                    self.add_line(f"{var_type!s} {target.id} = {value};")
                else:
                    # Separate declaration and initialization
                    self.add_line(f"{var_type!s} {target.id};")
                    self.add_line(f"{target.id} = {value};")
                self.declared_vars[self.current_scope].add(target.id)
            else:
                # Simple assignment
                self.add_line(f"{target.id} = {value};")

    def _generate_aug_assignment(self, node: ast.AugAssign) -> None:
        """Generate GLSL augmented assignment statement."""
        operators = {
            ast.Add: "+=",
            ast.Sub: "-=",
            ast.Mult: "*=",
            ast.Div: "/=",
        }

        # Generate target and value expressions
        target = self.generate_expression(node.target)
        value = self.generate_expression(node.value)

        # Get operator
        if type(node.op) not in operators:
            raise ValueError(
                f"Unsupported augmented assignment operator: {type(node.op)}"
            )
        op = operators[type(node.op)]

        # Generate the augmented assignment
        self.add_line(f"{target} {op} {value};")

    def _generate_if_statement(self, node: ast.If) -> None:
        """Generate GLSL if statement."""
        cond = self.generate_expression(node.test)
        self.add_line(f"if ({cond})")

        self.begin_block()
        for stmt in node.body:
            self.generate_statement(stmt)
        self.end_block()

        if node.orelse:
            self.add_line("else")
            self.begin_block()
            for stmt in node.orelse:
                self.generate_statement(stmt)
            self.end_block()

    def _generate_for_loop(self, node: ast.For) -> None:
        """Generate GLSL for loop."""
        self.push_context(GLSLContext.LOOP)

        try:
            if not (
                isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"
            ):
                raise ValueError("Only range-based for loops are supported")

            if not isinstance(node.target, ast.Name):
                raise ValueError("Only simple loop variables are supported")

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
                raise ValueError("Range with step not supported in GLSL")

            self.add_line(f"for ({init}; {cond}; {node.target.id}++)")

            self.begin_block()
            for stmt in node.body:
                self.generate_statement(stmt)
            self.end_block()

        finally:
            self.pop_context()

    def begin_block(self) -> None:
        """Begin a new block with increased indentation."""
        self.add_line("{")
        self.indent_level += 1

    def end_block(self) -> None:
        """End current block with decreased indentation."""
        self.indent_level -= 1
        self.add_line("}")

    def generate_function(self, node: ast.FunctionDef) -> None:
        """Generate GLSL function."""
        self.enter_scope(node.name)
        return_type = self.analysis.type_registry.get_type(node.returns.id)
        args = []
        param_names = set()

        # Track parameters first
        for arg in node.args.args:
            arg_type = self.analysis.var_types[node.name].get(arg.arg)
            if arg_type:
                args.append(f"{arg_type!s} {arg.arg}")
                param_names.add(arg.arg)
                # Add to declared vars to prevent redeclaration
                self.declared_vars[self.current_scope].add(arg.arg)

        # Generate function declaration
        self.add_line(f"{return_type!s} {node.name}({', '.join(args)})")
        self.begin_block()

        # Declare all variables at the start
        hoisted = sorted(
            var
            for var in self.analysis.hoisted_vars[node.name]
            if var not in param_names
        )
        if hoisted:
            for var_name in hoisted:
                var_type = self.analysis.var_types[node.name][var_name]
                self.add_line(f"{var_type!s} {var_name};")
                self.declared_vars[self.current_scope].add(var_name)
            self.add_line("")

        # Generate function body
        for stmt in node.body:
            if not isinstance(stmt, ast.FunctionDef):  # Skip nested functions
                self.generate_statement(stmt)

        self.end_block()
        self.exit_scope()

    def generate(self) -> str:
        """Generate complete GLSL shader code."""
        self.declared_vars = {"global": set()}
        raw_code = self._generate_raw()
        formatter = GLSLFormatter()
        return formatter.format_code(raw_code)

    def _generate_raw(self) -> str:
        """Generate complete GLSL shader code."""
        logger.debug("Starting GLSL code generation")
        self.add_line("#version 460")
        self.add_line()

        # Input/output declarations
        logger.debug("Generating input/output declarations")
        self.add_line("in vec2 vs_uv;")
        self.add_line("out vec4 fs_color;")
        self.add_line()

        # Uniform declarations
        logger.debug("Generating uniform declarations")
        for name, glsl_type in sorted(self.analysis.uniforms.items()):
            self.add_line(f"uniform {glsl_type.name} {name};")
        if self.analysis.uniforms:
            self.add_line()

        # Generate all functions at top level
        logger.debug("Generating functions")
        for func in self.analysis.functions:
            self.generate_function(func)
            self.add_line()

        # Generate main shader function
        logger.debug("Generating main shader function")
        main_func = self.analysis.main_function
        self.generate_function(main_func)
        self.add_line()

        # Generate main function
        logger.debug("Generating main function")
        self.add_line("void main()")
        self.begin_block()
        self.add_line(f"fs_color = {main_func.name}(vs_uv);")
        self.end_block()

        final_code = "\n".join(self.lines)
        logger.debug(f"Generated GLSL code:\n{final_code}")
        return final_code
