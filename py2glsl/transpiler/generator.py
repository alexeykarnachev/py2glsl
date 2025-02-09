"""GLSL code generator."""

import ast
from typing import Dict, List, Optional, Set, Union

from py2glsl.transpiler.analyzer import ShaderAnalysis
from py2glsl.transpiler.constants import (
    FLOAT_TYPE,
    GLSL_VERSION,
    INT_TYPE,
    VEC2_TYPE,
    VEC4_TYPE,
)
from py2glsl.transpiler.types import GLSLContext, GLSLType


class GLSLGenerator:
    """Generates GLSL code from analyzed shader."""

    def __init__(self, analysis: ShaderAnalysis) -> None:
        """Initialize generator with analysis results."""
        self.analysis = analysis
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

    def _convert_expression(self, node: ast.AST) -> str:
        """Convert Python expression to GLSL."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return str(node.value).lower()
            if isinstance(node.value, (int, float)):
                return f"{float(node.value)}"
            return "0.0"

        elif isinstance(node, ast.Name):
            return node.id

        elif isinstance(node, ast.Attribute):
            base = self._convert_expression(node.value)
            return f"{base}.{node.attr}"

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                args = [self._convert_expression(arg) for arg in node.args]
                return f"{node.func.id}({', '.join(args)})"

        elif isinstance(node, ast.BinOp):
            left = self._convert_expression(node.left)
            right = self._convert_expression(node.right)
            op = {
                ast.Add: "+",
                ast.Sub: "-",
                ast.Mult: "*",
                ast.Div: "/",
            }[type(node.op)]
            return f"({left} {op} {right})"

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
            return f"({left} {op} {right})"

        elif isinstance(node, ast.UnaryOp):
            op = {ast.USub: "-", ast.UAdd: "+"}[type(node.op)]
            operand = self._convert_expression(node.operand)
            return f"{op}{operand}"

        raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    def _convert_statement(self, node: ast.AST) -> List[str]:
        """Convert Python statement to GLSL."""
        if isinstance(node, ast.Assign):
            value = self._convert_expression(node.value)
            lines = []
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

        elif isinstance(node, ast.AugAssign):
            target = self._convert_expression(node.target)
            value = self._convert_expression(node.value)
            op = {
                ast.Add: "+=",
                ast.Sub: "-=",
                ast.Mult: "*=",
                ast.Div: "/=",
            }[type(node.op)]
            return [f"{target} {op} {value};"]

        elif isinstance(node, ast.If):
            condition = self._convert_expression(node.test)
            body = []
            for stmt in node.body:
                body.extend(self._convert_statement(stmt))

            else_body = []
            if node.orelse:
                for stmt in node.orelse:
                    else_body.extend(self._convert_statement(stmt))

            lines = [f"if ({condition})"]
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

                lines = [f"for ({init}; {condition}; {increment})"]
                lines.append("{")
                lines.extend(f"    {line}" for line in body)
                lines.append("}")
                return lines

        elif isinstance(node, ast.Return):
            value = self._convert_expression(node.value)
            return [f"return {value};"]

        elif isinstance(node, ast.Expr):
            value = self._convert_expression(node.value)
            return [f"{value};"]

        return []

    def _convert_function(
        self, node: ast.FunctionDef, is_main: bool = False
    ) -> List[str]:
        """Convert Python function to GLSL."""
        self._enter_scope(node.name)

        # Get correct return type
        if node.name == self.analysis.main_function.name:
            return_type = GLSLType(VEC4_TYPE)
        else:
            return_type = self.analysis.type_info.get_type(node.returns)

        # Build argument list with correct types
        args = []
        for arg in node.args.args:
            if arg.arg == "vs_uv":
                args.append(f"vec2 {arg.arg}")  # Always vec2 for vs_uv
            else:
                arg_type = self.analysis.type_info.get_type(arg)
                args.append(f"{str(arg_type)} {arg.arg}")

        # Format function
        lines = [f"{str(return_type)} {node.name}({', '.join(args)})"]
        lines.append("{")

        # Convert body with proper variable declarations
        body_lines = []
        for stmt in node.body:
            body_lines.extend(self._convert_statement(stmt))

        lines.extend(f"    {line}" for line in body_lines)
        lines.append("}")

        self._exit_scope()
        return lines

    def generate(self) -> str:
        """Generate complete GLSL shader code."""
        lines = [
            GLSL_VERSION,
            "",
            "in vec2 vs_uv;",
            "out vec4 fs_color;",
            "",
        ]

        # Add uniforms (only once, at global scope)
        for name, glsl_type in self.analysis.uniforms.items():
            lines.append(f"uniform {glsl_type.name} {name};")
        if self.analysis.uniforms:
            lines.append("")

        # Add functions
        for func in self.analysis.functions:
            lines.extend(self._convert_function(func))
            lines.append("")

        # Add shader function
        lines.extend(self._convert_function(self.analysis.main_function))
        lines.append("")

        # Add main function
        lines.extend(
            [
                "void main()",
                "{",
                f"    fs_color = {self.analysis.main_function.name}(vs_uv);",
                "}",
            ]
        )

        return "\n".join(lines)
