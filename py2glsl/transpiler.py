import ast
import inspect
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Union

from py2glsl.types import Vec2, Vec3, Vec4, vec2, vec3, vec4


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


class ShaderTranspiler:
    """Transform Python shader functions to GLSL."""

    def __init__(self) -> None:
        """Initialize transpiler."""
        self.uniforms: Dict[str, str] = {}
        self.functions: List[ast.FunctionDef] = []
        self.indent_level: int = 0
        self.declared_vars: Dict[str, str] = {}

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
            elif type_name in ("Vec2", "vec2"):
                return "vec2"
            elif type_name in ("Vec3", "vec3"):
                return "vec3"
            elif type_name in ("Vec4", "vec4"):
                return "vec4"
        elif type_ == float:
            return "float"
        elif self._is_vec2_type(type_):
            return "vec2"
        elif self._is_vec3_type(type_):
            return "vec3"
        elif self._is_vec4_type(type_):
            return "vec4"

        raise TypeError(f"Unsupported type: {type_}")

    def _infer_type(self, node: ast.AST) -> str:
        """Infer GLSL type from AST node."""
        if isinstance(node, ast.Call):
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
                value = self._convert_expr(node.value)
                inferred_type = self._infer_type(node.value)

                # Preserve vector types through operations
                if isinstance(node.value, ast.Call):
                    if isinstance(node.value.func, ast.Name):
                        if node.value.func.id == "normalize":
                            inferred_type = self._infer_type(node.value.args[0])
                        elif node.value.func.id == "mix":
                            inferred_type = self._infer_type(node.value.args[0])

                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id not in self.declared_vars:
                            self.declared_vars[target.id] = inferred_type
                            lines.append(f"{inferred_type} {target.id} = {value};")
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
                lines = [self._indent(f"for (int {var} = 0; {var} < {end}; {var}++)")]
            else:
                start = self._convert_expr(node.iter.args[0])
                end = self._convert_expr(node.iter.args[1])
                lines = [
                    self._indent(f"for (int {var} = {start}; {var} < {end}; {var}++)")
                ]

            lines.append(self._indent("{"))
            self.indent_level += 1
            lines.extend(self._convert_body(node.body))
            self.indent_level -= 1
            lines.append(self._indent("}"))
            return lines

        raise ValueError("Only range-based for loops are supported")

    def _convert_expr(self, node: ast.expr) -> str:
        """Convert Python expression to GLSL expression."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return f"{float(node.value)}"
            raise ValueError(f"Unsupported constant type: {type(node.value)}")

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

        raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    def analyze(self, func: Any) -> ShaderAnalysis:
        """Analyze Python shader function and extract structure."""
        source = inspect.getsource(func)
        source = textwrap.dedent(source)
        tree = ast.parse(source)

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

        # Add declarations without indentation
        lines.extend(["in vec2 vs_uv;", "out vec4 fs_color;"])

        # Add uniforms without indentation
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

        # Add main function without indentation
        lines.append("void main()")
        lines.append("{")
        lines.append("fs_color = shader(vs_uv);")
        lines.append("}")

        return ShaderResult(fragment_source="\n".join(lines), uniforms=self.uniforms)


def py2glsl(func: Any) -> ShaderResult:
    """Transform Python shader function to GLSL."""
    transpiler = ShaderTranspiler()
    analysis = transpiler.analyze(func)
    return transpiler.transform(analysis)
