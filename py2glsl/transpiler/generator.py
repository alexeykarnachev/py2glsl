"""GLSL shader code generator."""

import ast
from dataclasses import dataclass
from typing import Dict

from loguru import logger

from py2glsl.transpiler.analyzer import ShaderAnalysis
from py2glsl.transpiler.constants import GLSL_VERSION, VERTEX_SHADER
from py2glsl.transpiler.formatting import CodeBlock, ExprFormatter, StatementFormatter
from py2glsl.types import GLSLType


@dataclass
class GeneratedShader:
    """Result of shader generation."""

    vertex_source: str
    fragment_source: str
    uniforms: Dict[str, GLSLType]


class GLSLGenerator:
    """Generates GLSL code from analyzed shader."""

    def __init__(self, analysis: ShaderAnalysis):
        """Initialize generator with analysis results."""
        self.analysis = analysis
        self.code = CodeBlock()
        self.expr = ExprFormatter(var_types=analysis.var_types["global"])
        self.stmt = StatementFormatter(code=self.code, expr=self.expr)
        self.current_scope = "global"
        self.scope_stack: list[str] = []

    def enter_scope(self, name: str) -> None:
        """Enter a new scope."""
        self.scope_stack.append(self.current_scope)
        self.current_scope = name
        self.expr.var_types = self.analysis.var_types[name]

    def exit_scope(self) -> None:
        """Exit current scope."""
        if self.scope_stack:
            self.current_scope = self.scope_stack.pop()
            self.expr.var_types = self.analysis.var_types[self.current_scope]
        else:
            self.current_scope = "global"
            self.expr.var_types = self.analysis.var_types["global"]

    def generate(self) -> GeneratedShader:
        """Generate complete GLSL shader code."""
        logger.debug("Starting GLSL code generation")

        # Version declaration
        self.code.add_line(f"#version {GLSL_VERSION}")
        self.code.add_line()

        # Input from vertex shader
        self.code.add_line("in vec2 vs_uv;")  # Input from vertex shader
        self.code.add_line("out vec4 fs_color;")
        self.code.add_line()

        # Uniform declarations (sorted for consistent output)
        for name, glsl_type in sorted(self.analysis.uniforms.items()):
            self.code.add_line(f"uniform {glsl_type} {name};")
        if self.analysis.uniforms:
            self.code.add_line()

        # Generate nested functions first
        for func in self.analysis.functions:
            if func != self.analysis.main_function:
                logger.debug(f"Generating nested function: {func.name}")
                self.generate_function(func)
                self.code.add_line()

        # Generate main shader function if exists
        if self.analysis.main_function:
            logger.debug("Generating main shader function")
            self.generate_function(self.analysis.main_function)
            self.code.add_line()

            logger.debug("Generating main() wrapper")
            self.code.add_line("void main()")
            self.code.begin_block()
            self.code.add_line(f"fs_color = {self.analysis.main_function.name}(vs_uv);")
            self.code.end_block()

        return GeneratedShader(
            vertex_source=VERTEX_SHADER,
            fragment_source=self.code.get_code(),
            uniforms=self.analysis.uniforms,
        )

    def generate_function(self, node: ast.FunctionDef) -> None:
        """Generate GLSL function."""
        self.enter_scope(node.name)

        try:
            # Get return type
            return_type = self.analysis.var_types["global"][node.name]

            # Process arguments
            args = []
            for arg in node.args.args:
                arg_type = self.analysis.var_types[node.name][arg.arg]
                args.append(f"{arg_type} {arg.arg}")

            # Generate function declaration
            self.code.add_line(f"{return_type} {node.name}({', '.join(args)})")
            self.code.begin_block()

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
                    self.code.add_line(f"{var_type} {var_name};")

            if hoisted:
                self.code.add_line()

            # Generate function body
            for stmt in node.body:
                self.stmt.format_statement(stmt)

            self.code.end_block()

        finally:
            self.exit_scope()
