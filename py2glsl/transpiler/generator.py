"""GLSL code generator."""

import ast
from typing import List, Optional, Set

from loguru import logger

from py2glsl.transpiler.analyzer import GLSLContext, ShaderAnalysis
from py2glsl.transpiler.types import BOOL, FLOAT, INT, VEC2, VEC3, VEC4, GLSLType


class GLSLGenerator:
    """Generates GLSL code from analyzed shader."""

    def __init__(self, analysis: ShaderAnalysis):
        """Initialize generator with analysis results."""
        self.analysis = analysis
        self.indent_level = 0
        self.lines: List[str] = []
        self.current_scope = "global"
        self.scope_stack: List[str] = []

    def enter_scope(self, name: str) -> None:
        """Enter a new scope."""
        self.scope_stack.append(self.current_scope)
        self.current_scope = name

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
        """Generate GLSL expression."""
        if isinstance(node, ast.Name):
            return node.id

        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return str(node.value).lower()
            elif isinstance(node.value, (int, float)):
                # Ensure float literals have decimal point
                return f"{float(node.value)}"
            return "0.0"

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                args = [self.generate_expression(arg) for arg in node.args]
                # Vector constructors
                if node.func.id in ("vec2", "Vec2", "vec3", "Vec3", "vec4", "Vec4"):
                    return f"{node.func.id.lower()}({', '.join(args)})"
                # Regular function call
                return f"{node.func.id}({', '.join(args)})"

        elif isinstance(node, ast.BinOp):
            left = self.generate_expression(node.left, True)
            right = self.generate_expression(node.right, True)
            op = {
                ast.Add: "+",
                ast.Sub: "-",
                ast.Mult: "*",
                ast.Div: "/",
            }[type(node.op)]
            expr = f"{left} {op} {right}"
            return f"({expr})" if parenthesize else expr

        elif isinstance(node, ast.Compare):
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

    def generate_statement(self, node: ast.AST) -> None:
        """Generate GLSL statement."""
        if isinstance(node, ast.Assign):
            value = self.generate_expression(node.value)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    var_type = self.analysis.var_types[self.current_scope].get(
                        target.id
                    )
                    if var_type:
                        self.add_line(f"{str(var_type)} {target.id} = {value};")
                    else:
                        self.add_line(f"{target.id} = {value};")

        elif isinstance(node, ast.AugAssign):
            target = self.generate_expression(node.target)
            value = self.generate_expression(node.value)
            op = {
                ast.Add: "+=",
                ast.Sub: "-=",
                ast.Mult: "*=",
                ast.Div: "/=",
            }[type(node.op)]
            self.add_line(f"{target} {op} {value};")

        elif isinstance(node, ast.If):
            cond = self.generate_expression(node.test)
            self.add_line(f"if ({cond})")
            self.add_line("{")
            self.indent_level += 1
            for stmt in node.body:
                self.generate_statement(stmt)
            self.indent_level -= 1
            self.add_line("}")
            if node.orelse:
                self.add_line("else")
                self.add_line("{")
                self.indent_level += 1
                for stmt in node.orelse:
                    self.generate_statement(stmt)
                self.indent_level -= 1
                self.add_line("}")

        elif isinstance(node, ast.For):
            if (
                isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"
            ):
                if len(node.iter.args) == 1:
                    end = self.generate_expression(node.iter.args[0])
                    init = f"int {node.target.id} = 0"
                    cond = f"{node.target.id} < {end}"
                else:
                    start = self.generate_expression(node.iter.args[0])
                    end = self.generate_expression(node.iter.args[1])
                    init = f"int {node.target.id} = {start}"
                    cond = f"{node.target.id} < {end}"
                self.add_line(f"for ({init}; {cond}; {node.target.id}++)")
                self.add_line("{")
                self.indent_level += 1
                for stmt in node.body:
                    self.generate_statement(stmt)
                self.indent_level -= 1
                self.add_line("}")

        elif isinstance(node, ast.Return):
            value = self.generate_expression(node.value)
            self.add_line(f"return {value};")

        elif isinstance(node, ast.Expr):
            value = self.generate_expression(node.value)
            self.add_line(f"{value};")

    def generate_function(self, node: ast.FunctionDef) -> None:
        """Generate GLSL function."""
        self.enter_scope(node.name)

        # Get return type from annotation
        return_type = self.analysis.type_registry.get_type(node.returns.id)

        # Build argument list and track parameter names
        args = []
        param_names = set()
        for arg in node.args.args:
            arg_type = self.analysis.var_types[node.name].get(arg.arg)
            if arg_type:
                args.append(f"{str(arg_type)} {arg.arg}")
                param_names.add(arg.arg)  # Track parameter names

        # Generate function declaration
        self.add_line(f"{str(return_type)} {node.name}({', '.join(args)})")
        self.add_line("{")
        self.indent_level += 1

        # Generate hoisted variable declarations, excluding parameters
        if node.name in self.analysis.hoisted_vars:
            hoisted_declarations = []
            for var_name in sorted(self.analysis.hoisted_vars[node.name]):
                if var_name not in param_names:  # Skip parameters
                    var_type = self.analysis.var_types[node.name][var_name]
                    hoisted_declarations.append(f"{str(var_type)} {var_name};")

            if hoisted_declarations:
                for decl in hoisted_declarations:
                    self.add_line(decl)
                self.add_line("")

        # Generate function body
        for stmt in node.body:
            self.generate_statement(stmt)

        self.indent_level -= 1
        self.add_line("}")

        self.exit_scope()

    def generate(self) -> str:
        """Generate complete GLSL shader code."""
        # Version declaration
        self.add_line("#version 460")
        self.add_line()

        # Input/output declarations
        self.add_line("in vec2 vs_uv;")
        self.add_line("out vec4 fs_color;")
        self.add_line()

        # Uniform declarations
        for name, glsl_type in sorted(self.analysis.uniforms.items()):
            # Explicitly add uniform keyword here instead of relying on GLSLType.__str__
            self.add_line(f"uniform {glsl_type.name} {name};")
        if self.analysis.uniforms:
            self.add_line()

        # Generate functions
        for func in self.analysis.functions:
            self.generate_function(func)
            self.add_line()

        # Generate main shader function
        self.generate_function(self.analysis.main_function)
        self.add_line()

        # Generate main function
        self.add_line("void main()")
        self.add_line("{")
        self.indent_level += 1
        self.add_line(f"fs_color = {self.analysis.main_function.name}(vs_uv);")
        self.indent_level -= 1
        self.add_line("}")

        return "\n".join(self.lines)
