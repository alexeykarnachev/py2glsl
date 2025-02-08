import ast
import inspect
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from py2glsl.types import Vec2, Vec3, Vec4, vec2, vec3, vec4


@dataclass
class ShaderAnalysis:
    uniforms: Dict[str, str]
    functions: List[ast.FunctionDef]
    main_function: ast.FunctionDef


@dataclass
class ShaderResult:
    fragment_source: str
    uniforms: Dict[str, str]


class ShaderTranspiler:
    """Transpiles Python shader functions using NumPy to GLSL."""

    NUMPY_TO_GLSL = {
        np.dtype("float32"): "float",
        (np.dtype("float32"), 2): "vec2",
        (np.dtype("float32"), 3): "vec3",
        (np.dtype("float32"), 4): "vec4",
    }

    PYTHON_TO_GLSL = {
        "float": "float",
        "Vec2": "vec2",
        "Vec3": "vec3",
        "Vec4": "vec4",
        "vec2": "vec2",
        "vec3": "vec3",
        "vec4": "vec4",
    }

    # GLSL built-in functions that we can use directly
    BUILTIN_FUNCTIONS = {
        "length",
        "normalize",
        "distance",
        "dot",
        "cross",
        "mix",
        "smoothstep",
        "step",
        "clamp",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "pow",
        "exp",
        "log",
        "exp2",
        "log2",
        "sqrt",
        "inversesqrt",
        "abs",
        "sign",
        "floor",
        "ceil",
        "fract",
        "mod",
        "min",
        "max",
    }

    def __init__(self) -> None:
        self.uniforms: Dict[str, str] = {}
        self.functions: List[ast.FunctionDef] = []

    def _is_vec2_type(self, annotation: Any) -> bool:
        """Check if type annotation is vec2."""
        return annotation in (Vec2, vec2)

    def _is_vec4_type(self, annotation: Any) -> bool:
        """Check if type annotation is vec4."""
        return annotation in (Vec4, vec4)

    def analyze(self, func: Any) -> ShaderAnalysis:
        """Analyze Python shader function and extract structure."""
        # Get function source and dedent it
        source = inspect.getsource(func)
        source = textwrap.dedent(source)
        tree = ast.parse(source)

        # Get function signature
        signature = inspect.signature(func)

        # Validate first argument is vec2
        params = list(signature.parameters.items())
        if not params or params[0][0] != "vs_uv":
            raise TypeError("First argument must be vec2 named 'vs_uv'")

        # Validate first argument type
        first_param = params[0][1]
        if not self._is_vec2_type(first_param.annotation):
            raise TypeError("First argument must be vec2")

        # Extract uniforms (arguments with u_ prefix)
        for name, param in params[1:]:
            if not name.startswith("u_"):
                raise TypeError(f"Uniform parameter '{name}' must start with 'u_'")
            self.uniforms[name] = self._get_glsl_type(param.annotation)

        # Extract nested functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name != func.__name__:
                self.functions.append(node)

        # Validate return type
        return_annotation = signature.return_annotation
        if not self._is_vec4_type(return_annotation):
            raise TypeError("Shader must return vec4")

        return ShaderAnalysis(
            uniforms=self.uniforms, functions=self.functions, main_function=tree.body[0]
        )

    def transform(self, analysis: ShaderAnalysis) -> ShaderResult:
        """Transform Python shader to GLSL."""
        glsl_parts = [
            "#version 460",
            "",
            "in vec2 vs_uv;",
            "out vec4 fs_color;",
        ]

        # Only uniforms need declaration
        for name, type_ in analysis.uniforms.items():
            glsl_parts.append(f"uniform {type_} {name};")

        # User-defined functions
        if self.functions:
            glsl_parts.append("\n// User-defined functions")
            for func in self.functions:
                glsl_parts.append(self._convert_function(func))

        # Main shader function
        glsl_parts.extend(
            [
                "",
                "void main() {",
                "    fs_color = "
                + self._convert_body(analysis.main_function.body)
                + ";",
                "}",
            ]
        )

        return ShaderResult(
            fragment_source="\n".join(glsl_parts), uniforms=analysis.uniforms
        )

    def _get_glsl_type(self, py_type: Any) -> str:
        """Convert Python/NumPy type annotation to GLSL type."""
        if isinstance(py_type, np.ndarray):
            key = (py_type.dtype, py_type.shape[0])
            return self.NUMPY_TO_GLSL.get(key, "float")

        type_name = str(py_type)
        for py_name, glsl_name in self.PYTHON_TO_GLSL.items():
            if py_name in type_name:
                return glsl_name
        return "float"

    def _convert_function(self, node: ast.FunctionDef) -> str:
        """Convert a Python function to GLSL."""
        return_type = self._get_glsl_type(node.returns) if node.returns else "void"
        args = []
        for arg in node.args.args:
            arg_type = (
                self._get_glsl_type(arg.annotation) if arg.annotation else "float"
            )
            args.append(f"{arg_type} {arg.arg}")

        body = self._convert_body(node.body)
        return f"{return_type} {node.name}({', '.join(args)}) {{\n    {body}\n}}"

    def _convert_body(self, body: List[ast.stmt]) -> str:
        """Convert Python function body to GLSL."""
        lines = []
        for node in body:
            if isinstance(node, ast.Return):
                return self._convert_expr(node.value)
            elif isinstance(node, ast.Assign):
                target = self._convert_expr(node.targets[0])
                value = self._convert_expr(node.value)
                lines.append(f"{target} = {value};")
            elif isinstance(node, ast.If):
                lines.append(f"if ({self._convert_expr(node.test)}) {{")
                lines.extend(self._convert_body(node.body).splitlines())
                if node.orelse:
                    lines.append("} else {")
                    lines.extend(self._convert_body(node.orelse).splitlines())
                lines.append("}")
            elif isinstance(node, ast.For):
                if isinstance(node.iter, ast.Call) and isinstance(
                    node.iter.func, ast.Name
                ):
                    if node.iter.func.id == "range":
                        args = [self._convert_expr(arg) for arg in node.iter.args]
                        if len(args) == 1:
                            start, stop = "0", args[0]
                        else:
                            start, stop = args[:2]
                        lines.append(
                            f"for (int {node.target.id} = {start}; {node.target.id} < {stop}; {node.target.id}++) {{"
                        )
                        lines.extend(self._convert_body(node.body).splitlines())
                        lines.append("}")

        return "\n    ".join(lines)

    def _convert_expr(self, node: ast.expr) -> str:
        """Convert Python expression to GLSL."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return f"{float(node.value):.1f}"
            return str(node.value)
        elif isinstance(node, ast.Call):
            func_name = self._convert_expr(node.func)
            args = [self._convert_expr(arg) for arg in node.args]

            # Direct mapping for GLSL built-ins
            if func_name in self.BUILTIN_FUNCTIONS:
                return f"{func_name}({', '.join(args)})"

            # Vector constructors
            if func_name in ["vec2", "vec3", "vec4"]:
                return f"{func_name}({', '.join(args)})"

            return f"{func_name}({', '.join(args)})"
        elif isinstance(node, ast.BinOp):
            left = self._convert_expr(node.left)
            right = self._convert_expr(node.right)
            op = self._convert_operator(node.op)
            return f"{left} {op} {right}"
        elif isinstance(node, ast.Compare):
            left = self._convert_expr(node.left)
            ops = [self._convert_operator(op) for op in node.ops]
            comparators = [self._convert_expr(comp) for comp in node.comparators]
            return f"{left} {' '.join(f'{op} {comp}' for op, comp in zip(ops, comparators))}"
        elif isinstance(node, ast.Attribute):
            value = self._convert_expr(node.value)
            return f"{value}.{node.attr}"
        return str(node)

    def _convert_operator(self, op: ast.operator) -> str:
        """Convert Python operator to GLSL operator."""
        op_map = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.Mod: "%",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
            ast.Eq: "==",
            ast.NotEq: "!=",
        }
        return op_map.get(type(op), "?")


def py2glsl(shader_func: Any) -> ShaderResult:
    """Convert Python shader function using NumPy to GLSL."""
    transpiler = ShaderTranspiler()
    analysis = transpiler.analyze(shader_func)
    return transpiler.transform(analysis)
