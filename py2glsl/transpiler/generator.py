"""GLSL code generator."""

import ast
from typing import Dict, List, Optional, Set, Union

from .analyzer import ShaderAnalysis
from .constants import FLOAT_TYPE, INT_TYPE, VEC2_TYPE, VEC3_TYPE, VEC4_TYPE
from .formatter import GLSLFormatter
from .types import GLSLContext, GLSLType, TypeInference


class GLSLGenerator:
    """Generates GLSL code from analyzed shader."""

    def __init__(self, analysis: ShaderAnalysis) -> None:
        """Initialize generator with analysis results."""
        self.analysis = analysis
        self.formatter = GLSLFormatter()
        self.current_scope = "global"
        self.scope_stack: List[str] = []
        self.declared_vars: Dict[str, Set[str]] = {}
        self.context_stack: List[GLSLContext] = [GLSLContext.DEFAULT]

    def _enter_scope(self, name: str) -> None:
        """Enter a new scope."""
        self.scope_stack.append(self.current_scope)
        self.current_scope = name
        if name not in self.declared_vars:
            self.declared_vars[name] = set()

    def _exit_scope(self) -> None:
        """Exit current scope."""
        self.current_scope = self.scope_stack.pop()

    def _push_context(self, ctx: GLSLContext) -> None:
        """Push new context onto stack."""
        self.context_stack.append(ctx)

    def _pop_context(self) -> None:
        """Pop context from stack."""
        self.context_stack.pop()

    @property
    def current_context(self) -> GLSLContext:
        """Get current context."""
        return self.context_stack[-1]

    def _generate_hoisted_declarations(self, scope: str) -> List[str]:
        """Generate hoisted variable declarations for scope."""
        declarations = []
        vars_to_hoist = self.analysis.hoisted_vars.get(scope, set())
        for var in sorted(vars_to_hoist):
            if var not in self.declared_vars.get(scope, set()):
                glsl_type = self.analysis.type_info.get_type(
                    ast.Name(id=var, ctx=ast.Load())
                )
                declarations.append(f"{str(glsl_type)} {var};")
                self.declared_vars[scope].add(var)
        return declarations

    def _convert_expression(self, node: ast.AST) -> str:
        """Convert Python expression to GLSL."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return str(node.value).lower()
            if isinstance(node.value, (int, float)):
                if self.current_context == GLSLContext.LOOP_BOUND:
                    if isinstance(node.value, int):
                        return str(node.value)
                    raise ValueError("Loop bounds must be integers")
                return f"{float(node.value)}"
            return "0.0"  # Default for other constants

        elif isinstance(node, ast.Name):
            return node.id

        elif isinstance(node, ast.Attribute):
            base = self._convert_expression(node.value)
            return f"{base}.{node.attr}"

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                args = [self._convert_expression(arg) for arg in node.args]
                return self.formatter.format_function_call(node.func.id, args)

        elif isinstance(node, ast.BinOp):
            left = self._convert_expression(node.left)
            right = self._convert_expression(node.right)
            op = {
                ast.Add: "+",
                ast.Sub: "-",
                ast.Mult: "*",
                ast.Div: "/",
            }[type(node.op)]
            return self.formatter.format_binary_op(left, op, right, True)

        elif isinstance(node, ast.Compare):
            left = self._convert_expression(node.left)
            right = self._convert_expression(node.comparators[0])
            op = {
                ast.Lt: "<",
                ast.LtE: "<=",
                ast.Gt: ">",
                ast.GtE: ">=",
                ast.Eq: "==",
                ast.NotEq: "!=",
            }[type(node.ops[0])]
            return self.formatter.format_binary_op(left, op, right, True)

        elif isinstance(node, ast.UnaryOp):
            op = {ast.USub: "-", ast.UAdd: "+"}[type(node.op)]
            operand = self._convert_expression(node.operand)
            return f"{op}{operand}"

        elif isinstance(node, ast.BoolOp):
            op = {
                ast.And: "&&",
                ast.Or: "||",
            }[type(node.op)]
            values = [self._convert_expression(val) for val in node.values]
            return f" {op} ".join(f"({val})" for val in values)

        raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    def _convert_statement(self, node: ast.AST) -> List[str]:
        """Convert Python statement to GLSL."""
        if isinstance(node, ast.Assign):
            lines = []
            value = self._convert_expression(node.value)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    var_type = self.analysis.type_info.get_type(node.value)
                    if target.id not in self.declared_vars.get(
                        self.current_scope, set()
                    ):
                        lines.append(f"{str(var_type)} {target.id} = {value};")
                        self.declared_vars[self.current_scope].add(target.id)
                    else:
                        lines.append(f"{target.id} = {value};")
            return lines

        elif isinstance(node, ast.If):
            condition = self._convert_expression(node.test)
            body = []
            for stmt in node.body:
                body.extend(self._convert_statement(stmt))

            else_body = []
            if node.orelse:
                for stmt in node.orelse:
                    else_body.extend(self._convert_statement(stmt))

            lines = []
            lines.append(f"if ({condition})")
            lines.append("{")
            lines.extend(f"    {line}" for line in body)
            lines.append("}")

            if else_body:
                lines.append("else")
                lines.append("{")
                lines.extend(f"    {line}" for line in else_body)
                lines.append("}")

            return lines

        elif isinstance(node, ast.For):
            if (
                isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"
            ):
                self._push_context(GLSLContext.LOOP_BOUND)
                if len(node.iter.args) == 1:
                    end = self._convert_expression(node.iter.args[0])
                    init = f"int {node.target.id} = 0"
                    condition = f"{node.target.id} < {end}"
                else:
                    start = self._convert_expression(node.iter.args[0])
                    end = self._convert_expression(node.iter.args[1])
                    init = f"int {node.target.id} = {start}"
                    condition = f"{node.target.id} < {end}"
                self._pop_context()

                increment = f"{node.target.id}++"

                body = []
                for stmt in node.body:
                    body.extend(self._convert_statement(stmt))

                lines = []
                lines.append(f"for ({init}; {condition}; {increment})")
                lines.append("{")
                lines.extend(f"    {line}" for line in body)
                lines.append("}")
                return lines

        elif isinstance(node, ast.Return):
            return [f"return {self._convert_expression(node.value)};"]

        elif isinstance(node, ast.Expr):
            return [f"{self._convert_expression(node.value)};"]

        return []

    def _convert_function(
        self, node: ast.FunctionDef, is_main: bool = False
    ) -> List[str]:
        """Convert Python function to GLSL."""
        self._enter_scope(node.name)

        # Generate hoisted declarations
        lines = self._generate_hoisted_declarations(node.name)

        # Convert function body
        body_lines = []
        for stmt in node.body:
            body_lines.extend(self._convert_statement(stmt))

        self._exit_scope()

        # Get return type and function name
        return_type = (
            self.analysis.type_info.get_type(node.returns)
            if node.returns
            else GLSLType("void")
        )
        func_name = "shader" if is_main else node.name

        # Build argument list
        args = []
        for arg in node.args.args:
            arg_type = self.analysis.type_info.get_type(arg.annotation)
            args.append(f"{str(arg_type)} {arg.arg}")

        # Format function
        result = [
            f"{str(return_type)} {func_name}({', '.join(args)})",
            "{",
        ]

        # Add declarations and body with proper indentation
        if lines:
            result.extend(f"    {line}" for line in lines)
        result.extend(f"    {line}" for line in body_lines)

        result.append("}")
        return result

    def generate(self) -> str:
        """Generate complete GLSL shader code."""
        # Generate uniform declarations
        uniforms = [
            f"uniform {str(glsl_type)} {name};"
            for name, glsl_type in self.analysis.uniforms.items()
        ]

        # Generate nested functions
        functions = []
        for func in self.analysis.functions:
            functions.extend(self._convert_function(func))
            functions.append("")

        # Generate shader function
        shader_func = self._convert_function(self.analysis.main_function, is_main=False)

        # Format complete shader with proper indentation and structure
        lines = [
            "#version 460",
            "",
            "in vec2 vs_uv;",
            "out vec4 fs_color;",
            "",
        ]

        # Add uniforms
        if uniforms:
            lines.extend(uniforms)
            lines.append("")

        # Add functions
        if functions:
            lines.extend(functions)

        # Add shader function
        lines.extend(shader_func)
        lines.append("")

        # Add main function
        lines.extend(["void main()", "{", "    fs_color = shader(vs_uv);", "}"])

        return "\n".join(lines)
