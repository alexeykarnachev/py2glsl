import ast
import inspect
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

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
        self.declared_vars: Set[str] = set()

    def _indent(self, text: str) -> str:
        """Add proper indentation to text."""
        return "    " * self.indent_level + text

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
                if node.func.id in ("vec2", "Vec2"):
                    return "vec2"
                elif node.func.id in ("vec3", "Vec3"):
                    return "vec3"
                elif node.func.id in ("vec4", "Vec4"):
                    return "vec4"
                elif node.func.id in ("normalize", "length"):
                    return self._infer_type(node.args[0])
                elif node.func.id == "dot":
                    return "float"
                elif node.func.id == "mix":
                    return self._infer_type(node.args[0])
        elif isinstance(node, ast.BinOp):
            left_type = self._infer_type(node.left)
            right_type = self._infer_type(node.right)
            if "vec" in left_type and "vec" in right_type:
                return max(left_type, right_type, key=len)
            elif "vec" in left_type:
                return left_type
            elif "vec" in right_type:
                return right_type
            return "float"
        elif isinstance(node, ast.Attribute):
            base_type = self._infer_type(node.value)
            if node.attr in ("xy", "yz", "xz"):
                return "vec2"
            elif node.attr in ("xyz", "rgb"):
                return "vec3"
            elif node.attr in ("x", "y", "z", "w"):
                return "float"
            return base_type
        elif isinstance(node, ast.Name):
            if node.id in self.declared_vars:
                return "float"  # Default to float for now
        return "float"

    def _convert_function(self, node: ast.FunctionDef, is_main: bool = False) -> str:
        """Convert Python function to GLSL function."""
        self.declared_vars = set()

        return_type = self._get_glsl_type(node.returns)

        args = []
        for arg in node.args.args:
            arg_type = self._get_glsl_type(arg.annotation)
            args.append(f"{arg_type} {arg.arg}")
            self.declared_vars.add(arg.arg)

        func_name = "shader" if is_main else node.name

        lines = [f"{return_type} {func_name}({', '.join(args)}) {{"]
        self.indent_level += 1
        lines.extend(self._convert_body(node.body))
        self.indent_level -= 1
        lines.append("}")

        return "\n".join(self._indent(line) for line in lines)

    def _convert_body(self, body: List[ast.stmt]) -> List[str]:
        """Convert Python statements to GLSL statements."""
        lines = []
        for node in body:
            if isinstance(node, ast.Return):
                lines.append(f"return {self._convert_expr(node.value)};")
            elif isinstance(node, ast.Assign):
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    value = self._convert_expr(node.value)
                    if target.id not in self.declared_vars:
                        var_type = self._infer_type(node.value)
                        lines.append(f"{var_type} {target.id} = {value};")
                        self.declared_vars.add(target.id)
                    else:
                        lines.append(f"{target.id} = {value};")
            elif isinstance(node, ast.If):
                lines.extend(self._convert_if(node))
            elif isinstance(node, ast.For):
                lines.extend(self._convert_for(node))
            elif isinstance(node, ast.Break):
                lines.append("break;")
        return lines

    def _convert_if(self, node: ast.If) -> List[str]:
        """Convert if statement to GLSL."""
        lines = []
        lines.append(f"if ({self._convert_expr(node.test)}) {{")
        self.indent_level += 1
        lines.extend(self._convert_body(node.body))
        self.indent_level -= 1

        if node.orelse:
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                elif_node = node.orelse[0]
                lines.append("} else if (" + self._convert_expr(elif_node.test) + ") {")
                self.indent_level += 1
                lines.extend(self._convert_body(elif_node.body))
                self.indent_level -= 1
            else:
                lines.append("} else {")
                self.indent_level += 1
                lines.extend(self._convert_body(node.orelse))
                self.indent_level -= 1

        lines.append("}")
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
                lines = [f"for (int {var} = 0; {var} < {end}; {var}++) {{"]
            else:
                start = self._convert_expr(node.iter.args[0])
                end = self._convert_expr(node.iter.args[1])
                lines = [f"for (int {var} = {start}; {var} < {end}; {var}++) {{"]

            self.indent_level += 1
            lines.extend(self._convert_body(node.body))
            self.indent_level -= 1
            lines.append("}")
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
            raise TypeError("First argument must be named 'vs_uv'")

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

        # Add main function
        lines.extend(["void main() {", "    fs_color = shader(vs_uv);", "}"])

        return ShaderResult(
            fragment_source="\n".join(lines), uniforms=analysis.uniforms
        )


def py2glsl(shader_func: Any) -> ShaderResult:
    """Transform Python shader function to GLSL."""
    transpiler = ShaderTranspiler()
    analysis = transpiler.analyze(shader_func)
    return transpiler.transform(analysis)
