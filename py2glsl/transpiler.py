import ast
import inspect
import textwrap
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Iterator, List, Optional, Set, Union

from py2glsl.types import Vec2, Vec3, Vec4, vec2, vec3, vec4

# Basic vertex shader that just passes UV coords
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
class ShaderAnalysis:
    """Result of shader analysis."""

    uniforms: Dict[str, str]
    functions: List[ast.FunctionDef]
    main_function: ast.FunctionDef


@dataclass
class ShaderResult:
    """Result of shader transformation."""

    fragment_source: str
    uniforms: Dict[str, str]
    vertex_source: str = VERTEX_SHADER


class GLSLContext(Enum):
    DEFAULT = auto()
    LOOP_BOUND = auto()  # For loop bounds and indices
    ARITHMETIC = auto()  # For general math expressions
    VECTOR_INIT = auto()  # Keep existing vector context


class ShaderTranspiler:
    """Transform Python shader functions to GLSL."""

    def __init__(self) -> None:
        """Initialize transpiler."""
        self.uniforms: Dict[str, str] = {}
        self.functions: List[ast.FunctionDef] = []
        self.indent_level: int = 0
        self.declared_vars: Dict[str, str] = {}
        self.context_stack: List[GLSLContext] = [GLSLContext.DEFAULT]

    def _convert_assignment(self, target: str, value: Any) -> str:
        """Convert Python assignment to GLSL."""
        # Handle AST nodes directly instead of using isinstance
        if isinstance(value, ast.Constant):
            if isinstance(value.value, bool):
                return f"bool {target} = {str(value.value).lower()};"
            elif isinstance(value.value, (int, float)):
                return f"float {target} = {value.value}.0;"
        elif isinstance(value, ast.Call):
            if isinstance(value.func, ast.Name):
                if value.func.id in ("vec2", "Vec2"):
                    return f"vec2 {target} = vec2({', '.join(self._convert_expr(arg) for arg in value.args)});"
                elif value.func.id in ("vec3", "Vec3"):
                    return f"vec3 {target} = vec3({', '.join(self._convert_expr(arg) for arg in value.args)});"
                elif value.func.id in ("vec4", "Vec4"):
                    return f"vec4 {target} = vec4({', '.join(self._convert_expr(arg) for arg in value.args)});"
        elif isinstance(value, ast.BinOp):
            return f"{self._infer_type(value)} {target} = {self._convert_expr(value)};"
        elif isinstance(value, ast.Attribute):
            return f"{self._infer_type(value)} {target} = {self._convert_expr(value)};"

        return f"{self._infer_type(value)} {target} = {self._convert_expr(value)};"

    @contextmanager
    def context(self, ctx: GLSLContext) -> Iterator[None]:
        """Context manager for GLSL context tracking."""
        self.context_stack.append(ctx)
        try:
            yield
        finally:
            self.context_stack.pop()

    @property
    def current_context(self) -> GLSLContext:
        """Get current GLSL context."""
        return self.context_stack[-1]

    def _indent(self, text: str) -> str:
        """Add proper indentation to text."""
        # Don't indent version, in/out declarations, and uniforms
        if text.startswith(("#version", "in ", "out ", "uniform ")):
            return text
        # Don't indent function declarations and main
        if text.endswith(("{", "}", "};")) or text.startswith(
            ("void main", "vec", "float")
        ):
            return text
        # Indent everything else with 4 spaces
        return "    " + text

    def _is_vec2_type(self, type_: Any) -> bool:
        """Check if type is vec2."""
        return (
            isinstance(type_, ast.Name) and type_.id in ("Vec2", "vec2")
        ) or type_ in (Vec2, vec2)

    def _is_vec3_type(self, type_: Any) -> bool:
        """Check if type is vec3."""
        return (
            isinstance(type_, ast.Name) and type_.id in ("Vec3", "vec3")
        ) or type_ in (Vec3, vec3)

    def _is_vec4_type(self, type_: Any) -> bool:
        """Check if type is vec4."""
        return (
            isinstance(type_, ast.Name) and type_.id in ("Vec4", "vec4")
        ) or type_ in (Vec4, vec4)

    def _get_glsl_type(self, type_: Any) -> str:
        """Convert Python type to GLSL type."""
        if isinstance(type_, ast.Name):
            type_name = type_.id
            if type_name in ("float", "Float"):
                return "float"
            elif type_name in ("int", "Int"):
                return "int"
            elif type_name in ("Vec2", "vec2"):
                return "vec2"
            elif type_name in ("Vec3", "vec3"):
                return "vec3"
            elif type_name in ("Vec4", "vec4"):
                return "vec4"
        elif type_ == float:
            return "float"
        elif type_ == int:
            return "int"
        elif self._is_vec2_type(type_):
            return "vec2"
        elif self._is_vec3_type(type_):
            return "vec3"
        elif self._is_vec4_type(type_):
            return "vec4"

        raise TypeError(f"Unsupported type: {type_}")

    def _infer_type(self, node: ast.AST) -> str:
        """Infer GLSL type from AST node."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return "bool"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                # Vector constructors
                if node.func.id in ("vec2", "Vec2"):
                    return "vec2"
                elif node.func.id in ("vec3", "Vec3"):
                    return "vec3"
                elif node.func.id in ("vec4", "Vec4"):
                    return "vec4"
                # Functions that preserve input type
                elif node.func.id == "normalize":
                    arg_type = self._infer_type(node.args[0])
                    if arg_type.startswith("vec"):
                        return arg_type
                    return "float"
                # Functions that return float
                elif node.func.id in ("length", "dot"):
                    return "float"
                # Mix returns the type of its first argument
                elif node.func.id == "mix":
                    return self._infer_type(node.args[0])
                # Other functions
                elif node.func.id in [f.name for f in self.functions]:
                    for func in self.functions:
                        if func.name == node.func.id:
                            return self._get_glsl_type(func.returns)
        elif isinstance(node, ast.BinOp):
            left_type = self._infer_type(node.left)
            right_type = self._infer_type(node.right)
            if "vec" in left_type or "vec" in right_type:
                if "vec4" in (left_type, right_type):
                    return "vec4"
                elif "vec3" in (left_type, right_type):
                    return "vec3"
                elif "vec2" in (left_type, right_type):
                    return "vec2"
            return "float"
        elif isinstance(node, ast.Name):
            # Check uniforms first
            if node.id in self.uniforms:
                return self.uniforms[node.id]
            # Then check local variables
            if node.id in self.declared_vars:
                return self.declared_vars[node.id]
        elif isinstance(node, ast.Attribute):
            value_type = self._infer_type(node.value)
            if value_type.startswith("vec"):
                components = len(node.attr)
                return f"vec{components}" if components > 1 else "float"
        return "float"

    def _convert_function(self, node: ast.FunctionDef, is_main: bool = False) -> str:
        """Convert Python function to GLSL function."""
        self.declared_vars = {}

        return_type = self._get_glsl_type(node.returns)
        args = []
        for arg in node.args.args:
            arg_type = self._get_glsl_type(arg.annotation)
            args.append(f"{arg_type} {arg.arg}")
            self.declared_vars[arg.arg] = arg_type

        func_name = "shader" if is_main else node.name
        lines = []
        lines.append(f"{return_type} {func_name}({', '.join(args)})")
        lines.append("{")
        self.indent_level += 1
        lines.extend(self._convert_body(node.body))
        self.indent_level -= 1
        lines.append("}")
        return "\n".join(lines)

    def _convert_body(self, body: List[ast.stmt]) -> List[str]:
        """Convert Python statements to GLSL statements."""
        lines = []
        for node in body:
            if isinstance(node, ast.Return):
                lines.append(f"return {self._convert_expr(node.value)};")

            elif isinstance(node, ast.Assign):
                # Handle chained assignments (x = y = z = 1.0)
                value = self._convert_expr(node.value)
                value_type = self._infer_type(node.value)

                # Process targets in reverse to maintain correct assignment order
                for target in reversed(node.targets):
                    if isinstance(target, ast.Name):
                        if target.id not in self.declared_vars:
                            lines.append(f"{value_type} {target.id} = {value};")
                            self.declared_vars[target.id] = value_type
                        else:
                            lines.append(f"{target.id} = {value};")

            elif isinstance(node, ast.AugAssign):
                target = self._convert_expr(node.target)
                value = self._convert_expr(node.value)
                op = {
                    ast.Add: "+=",
                    ast.Sub: "-=",
                    ast.Mult: "*=",
                    ast.Div: "/=",
                }[type(node.op)]
                lines.append(f"{target} {op} {value};")

            elif isinstance(node, ast.If):
                lines.append(f"if ({self._convert_expr(node.test)})")
                lines.append("{")
                lines.extend(self._convert_body(node.body))
                lines.append("}")
                if node.orelse:
                    if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                        # elif case
                        elif_node = node.orelse[0]
                        lines.append(f"else if ({self._convert_expr(elif_node.test)})")
                        lines.append("{")
                        lines.extend(self._convert_body(elif_node.body))
                        lines.append("}")
                        if elif_node.orelse:
                            lines.append("else")
                            lines.append("{")
                            lines.extend(self._convert_body(elif_node.orelse))
                            lines.append("}")
                    else:
                        # else case
                        lines.append("else")
                        lines.append("{")
                        lines.extend(self._convert_body(node.orelse))
                        lines.append("}")

            elif isinstance(node, ast.For):
                if (
                    isinstance(node.iter, ast.Call)
                    and isinstance(node.iter.func, ast.Name)
                    and node.iter.func.id == "range"
                ):
                    with self.context(GLSLContext.LOOP_BOUND):
                        if len(node.iter.args) == 1:
                            end = self._convert_expr(node.iter.args[0])
                            lines.append(
                                f"for (int {node.target.id} = 0; {node.target.id} < {end}; {node.target.id}++)"
                            )
                        else:
                            start = self._convert_expr(node.iter.args[0])
                            end = self._convert_expr(node.iter.args[1])
                            lines.append(
                                f"for (int {node.target.id} = {start}; {node.target.id} < {end}; {node.target.id}++)"
                            )
                    lines.append("{")
                    lines.extend(self._convert_body(node.body))
                    lines.append("}")
                else:
                    raise ValueError("Only range-based for loops are supported")

            elif isinstance(node, ast.Expr):
                # Handle expression statements (like function calls)
                lines.append(f"{self._convert_expr(node.value)};")

        return lines

    def _convert_if(self, node: ast.If) -> List[str]:
        """Convert if statement to GLSL."""
        lines = []
        lines.append(self._indent(f"if ({self._convert_expr(node.test)})"))
        lines.append(self._indent("{"))
        self.indent_level += 1
        lines.extend(self._convert_body(node.body))
        self.indent_level -= 1
        lines.append(self._indent("}"))

        if node.orelse:
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                elif_node = node.orelse[0]
                lines.append(
                    self._indent(f"else if ({self._convert_expr(elif_node.test)})")
                )
                lines.append(self._indent("{"))
                self.indent_level += 1
                lines.extend(self._convert_body(elif_node.body))
                self.indent_level -= 1
                lines.append(self._indent("}"))
                if elif_node.orelse:
                    lines.append(self._indent("else"))
                    lines.append(self._indent("{"))
                    self.indent_level += 1
                    lines.extend(self._convert_body(elif_node.orelse))
                    self.indent_level -= 1
                    lines.append(self._indent("}"))
            else:
                lines.append(self._indent("else"))
                lines.append(self._indent("{"))
                self.indent_level += 1
                lines.extend(self._convert_body(node.orelse))
                self.indent_level -= 1
                lines.append(self._indent("}"))

        return lines

    def _convert_for(self, node: ast.For) -> List[str]:
        """Convert for loop to GLSL."""

        if (
            isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
        ):
            var = node.target.id

            if len(node.iter.args) == 1:
                end = self._convert_expr(node.iter.args[0])
                lines = [f"for (int {var} = 0; {var} < {end}; {var}++)"]
            else:
                start = self._convert_expr(node.iter.args[0])
                end = self._convert_expr(node.iter.args[1])
                lines = [f"for (int {var} = {start}; {var} < {end}; {var}++)"]

            lines.append("{")
            self.indent_level += 1
            lines.extend(self._convert_body(node.body))
            self.indent_level -= 1
            lines.append("}")
            return lines

        raise ValueError("Only range-based for loops are supported")

    def _convert_expr(self, node: ast.expr) -> str:
        """Convert Python expression to GLSL expression."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return "true" if node.value else "false"
            if isinstance(node.value, (int, float)):
                if self.current_context == GLSLContext.LOOP_BOUND:
                    if isinstance(node.value, int):
                        return str(node.value)
                    raise ValueError("Loop bounds must be integers")
                return f"{float(node.value)}"

        elif isinstance(node, ast.Name):
            return node.id

        elif isinstance(node, ast.Attribute):
            base = self._convert_expr(node.value)
            return f"{base}.{node.attr}"

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                args = [self._convert_expr(arg) for arg in node.args]
                return f"{node.func.id}({', '.join(args)})"

        elif isinstance(node, ast.BinOp):
            left = self._convert_expr(node.left)
            right = self._convert_expr(node.right)
            if isinstance(node.left, ast.BinOp):
                left = f"({left})"
            if isinstance(node.right, ast.BinOp):
                right = f"({right})"
            op = {
                ast.Add: "+",
                ast.Sub: "-",
                ast.Mult: "*",
                ast.Div: "/",
            }[type(node.op)]
            return f"{left} {op} {right}"

        elif isinstance(node, ast.Compare):
            left = self._convert_expr(node.left)
            right = self._convert_expr(node.comparators[0])
            op = {
                ast.Lt: "<",
                ast.LtE: "<=",
                ast.Gt: ">",
                ast.GtE: ">=",
                ast.Eq: "==",
                ast.NotEq: "!=",
            }[type(node.ops[0])]
            return f"{left} {op} {right}"

        elif isinstance(node, ast.UnaryOp):
            op = {ast.USub: "-", ast.UAdd: "+"}[type(node.op)]
            operand = self._convert_expr(node.operand)
            return f"{op}{operand}"

        elif isinstance(node, ast.BoolOp):
            op = {
                ast.And: "&&",
                ast.Or: "||",
            }[type(node.op)]
            values = [self._convert_expr(val) for val in node.values]
            return f" {op} ".join(f"({val})" for val in values)

        error = f"Unsupported expression: {ast.dump(node)}"
        raise ValueError(error)

    def analyze(self, func: Any) -> ShaderAnalysis:
        """Analyze Python shader function and extract structure."""
        source = inspect.getsource(func)
        source = textwrap.dedent(source)
        tree = ast.parse(source)

        # Check if shader uses u_resolution
        uses_resolution = "u_resolution" in source
        if uses_resolution:
            self.uniforms["u_resolution"] = "vec2"

        signature = inspect.signature(func)

        # Validate all arguments have type hints
        params = list(signature.parameters.items())
        if not params:
            raise TypeError("Shader must have at least one argument (vs_uv)")

        # Validate first argument is vs_uv: vec2
        if params[0][0] != "vs_uv":
            raise TypeError("First argument must be vs_uv")

        first_param = params[0][1]
        if first_param.annotation == first_param.empty:
            raise TypeError("All arguments must have type hints")
        if not self._is_vec2_type(first_param.annotation):
            raise TypeError("First argument must be vec2")

        # Extract uniforms
        for name, param in params[1:]:
            if not param.kind == param.KEYWORD_ONLY:
                raise TypeError("All arguments except vs_uv must be uniforms")
            if param.annotation == param.empty:
                raise TypeError("All arguments must have type hints")
            self.uniforms[name] = self._get_glsl_type(param.annotation)

        # Extract nested functions
        self.functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name != func.__name__:
                self.functions.append(node)

        # Validate return type
        if signature.return_annotation == signature.empty:
            raise TypeError("Shader must have return type annotation")
        if not self._is_vec4_type(signature.return_annotation):
            raise TypeError("Shader must return vec4")

        return ShaderAnalysis(
            uniforms=self.uniforms, functions=self.functions, main_function=tree.body[0]
        )

    def transform(self, analysis: ShaderAnalysis) -> ShaderResult:
        """Transform Python shader to GLSL."""
        lines = ["#version 460", ""]
        lines.extend(["in vec2 vs_uv;", "out vec4 fs_color;"])

        # Add uniforms
        for name, type_ in analysis.uniforms.items():
            lines.append(f"uniform {type_} {name};")

        if analysis.uniforms:
            lines.append("")

        # Add nested functions
        for func in analysis.functions:
            lines.append(self._convert_function(func))
            lines.append("")

        # Add main shader function
        lines.append(self._convert_function(analysis.main_function, is_main=True))
        lines.append("")

        # Add main function with proper indentation
        lines.extend(
            [
                "void main()",
                "{",
                "    fs_color = shader(vs_uv);",  # Add 4 spaces indentation
                "}",
            ]
        )

        return ShaderResult(
            fragment_source="\n".join(lines),
            vertex_source=VERTEX_SHADER,
            uniforms=self.uniforms,
        )


def py2glsl(func: Any) -> ShaderResult:
    """Transform Python shader function to GLSL."""
    transpiler = ShaderTranspiler()
    analysis = transpiler.analyze(func)
    return transpiler.transform(analysis)
