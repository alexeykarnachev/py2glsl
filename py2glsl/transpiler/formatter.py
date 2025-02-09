"""GLSL code formatting utilities."""

from typing import List, Optional

from .constants import GLSL_VERSION, INDENT


class GLSLFormatter:
    """Handles GLSL code formatting."""

    def __init__(self):
        self.indent_level = 0
        self.lines: List[str] = []
        self._indent_str = "    "  # Fixed 4-space indentation

    def _indent(self) -> str:
        """Get current indentation string."""
        return INDENT * self.indent_level

    def format_line(self, line: str) -> str:
        """Format a single line with proper indentation."""
        if not line.strip():
            return ""

        # Don't indent preprocessor directives, in/out declarations, and uniforms
        if any(
            line.startswith(prefix)
            for prefix in ["#", "in ", "out ", "uniform ", "void main", "layout"]
        ):
            return line.strip()

        # Don't indent closing braces
        if line.strip() == "}":
            self.indent_level = max(0, self.indent_level - 1)

        result = f"{self._indent_str * self.indent_level}{line.strip()}"

        # Increase indent after opening brace
        if line.strip().endswith("{"):
            self.indent_level += 1

        return result

    def format_code(self, code: str) -> str:
        """Format complete GLSL code."""
        self.indent_level = 0
        formatted_lines = []

        for line in code.split("\n"):
            if line.strip():
                formatted_lines.append(self.format_line(line))
            else:
                formatted_lines.append("")

        return "\n".join(formatted_lines)

    def add_line(self, line: str = "") -> None:
        """Add a line of code with proper indentation."""
        if line.strip():
            # Don't indent version, in/out declarations, and uniforms
            if any(
                line.startswith(prefix)
                for prefix in ["#version", "in ", "out ", "uniform "]
            ):
                self.lines.append(line)
            # Don't indent function declarations and closing braces
            elif line.endswith(("{", "}", "};")) or line.startswith(
                ("void main", "vec", "float", "int", "bool")
            ):
                self.lines.append(line)
            else:
                self.lines.append(f"{self._indent()}{line}")
        else:
            self.lines.append("")

    def begin_block(self) -> None:
        """Begin a new block with increased indentation."""
        self.add_line("{")
        self.indent_level += 1

    def end_block(self) -> None:
        """End current block with decreased indentation."""
        self.indent_level -= 1
        self.add_line("}")

    def format_function(
        self, name: str, args: List[str], return_type: str, body: List[str]
    ) -> None:
        """Format a function definition."""
        signature = f"{return_type} {name}({', '.join(args)})"
        self.add_line(signature)
        self.begin_block()
        for line in body:
            self.add_line(line)
        self.end_block()

    def format_if(
        self, condition: str, body: List[str], else_body: Optional[List[str]] = None
    ) -> None:
        """Format an if statement."""
        self.add_line(f"if ({condition})")
        self.begin_block()
        for line in body:
            self.add_line(line)
        self.end_block()

        if else_body:
            self.add_line("else")
            self.begin_block()
            for line in else_body:
                self.add_line(line)
            self.end_block()

    def format_for(
        self, init: str, condition: str, increment: str, body: List[str]
    ) -> None:
        """Format a for loop."""
        self.add_line(f"for ({init}; {condition}; {increment})")
        self.begin_block()
        for line in body:
            self.add_line(line)
        self.end_block()

    def format_variable_declaration(
        self, type_name: str, name: str, value: Optional[str] = None
    ) -> None:
        """Format a variable declaration."""
        if value is not None:
            self.add_line(f"{type_name} {name} = {value};")
        else:
            self.add_line(f"{type_name} {name};")

    def format_assignment(self, target: str, value: str) -> None:
        """Format an assignment statement."""
        self.add_line(f"{target} = {value};")

    def format_return(self, value: str) -> None:
        """Format a return statement."""
        self.add_line(f"return {value};")

    def format_shader(
        self, uniforms: List[str], functions: List[str], main_body: List[str]
    ) -> str:
        """Format complete shader code."""
        # Clear any existing content
        self.lines = []
        self.indent_level = 0

        # Add version
        self.add_line(GLSL_VERSION)
        self.add_line()

        # Add in/out declarations
        self.add_line("in vec2 vs_uv;")
        self.add_line("out vec4 fs_color;")
        self.add_line()

        # Add uniforms
        for uniform in uniforms:
            self.add_line(uniform)
        if uniforms:
            self.add_line()

        # Add functions
        for func in functions:
            self.add_line(func)
            self.add_line()

        # Add main function
        self.add_line("void main()")
        self.begin_block()
        for line in main_body:
            self.add_line(line)
        self.end_block()

        return "\n".join(self.lines)

    def format_expression(self, expr: str, parenthesize: bool = False) -> str:
        """Format an expression."""
        if parenthesize:
            return f"({expr})"
        return expr

    def format_vector_constructor(self, type_name: str, components: List[str]) -> str:
        """Format a vector constructor."""
        return f"{type_name}({', '.join(components)})"

    def format_function_call(self, name: str, args: List[str]) -> str:
        """Format a function call."""
        return f"{name}({', '.join(args)})"

    def format_binary_op(
        self, left: str, op: str, right: str, parenthesize: bool = False
    ) -> str:
        """Format a binary operation."""
        expr = f"{left} {op} {right}"
        if parenthesize:
            return f"({expr})"
        return expr

    def format_swizzle(self, base: str, components: str) -> str:
        """Format a swizzle operation."""
        return f"{base}.{components}"
