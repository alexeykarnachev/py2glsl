import ast
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from py2glsl.types import vec2, vec3, vec4


@dataclass
class ShaderAnalysis:
    uniforms: Dict[str, str]  # name -> glsl type
    functions: List[ast.FunctionDef]
    main_function: ast.FunctionDef


@dataclass
class ShaderResult:
    fragment_source: str
    uniforms: Dict[str, str]


class ShaderTranspiler:
    GLSL_TYPES = {
        "float": "float",
        "vec2": "vec2",
        "vec3": "vec3",
        "vec4": "vec4",
    }

    def __init__(self) -> None:
        self.uniforms: Dict[str, str] = {}
        self.functions: List[ast.FunctionDef] = []

    def analyze(self, func: Any) -> ShaderAnalysis:
        """Analyze Python shader function and extract structure."""
        tree = ast.parse(func.__code__.co_code)

        # Get function signature
        signature = ast.get_source_segment(func.__code__.co_code, tree)
        if not signature:
            raise ValueError("Could not get function signature")

        # Validate first argument is vec2
        args = func.__code__.co_varnames
        if not args or args[0] != "vs_uv":
            raise TypeError("First argument must be vec2 named 'vs_uv'")

        # Extract uniforms (arguments with * prefix)
        for name, param in func.__annotations__.items():
            if name != "vs_uv" and name != "return":
                if name.startswith("u_"):
                    self.uniforms[name] = self._get_glsl_type(param)
                else:
                    raise TypeError(f"Uniform parameter '{name}' must start with 'u_'")

        # Validate return type is vec4
        return_type = func.__annotations__.get("return")
        if return_type != vec4:
            raise TypeError("Shader must return vec4")

        return ShaderAnalysis(
            uniforms=self.uniforms, functions=self.functions, main_function=tree.body[0]
        )

    def transform(self, analysis: ShaderAnalysis) -> ShaderResult:
        """Transform Python shader to GLSL."""
        # Generate uniform declarations
        uniform_decls = self._generate_uniforms(analysis.uniforms)

        # Generate function definitions
        function_defs = self._generate_functions(analysis.functions)

        # Generate main shader function
        shader_code = self._generate_shader(analysis.main_function)

        # Combine everything into final GLSL
        glsl = f"""#version 460

in vec2 vs_uv;
out vec4 fs_color;

{uniform_decls}

{function_defs}

void main() {{
    fs_color = {shader_code};
}}
"""
        return ShaderResult(fragment_source=glsl.strip(), uniforms=analysis.uniforms)

    def _get_glsl_type(self, py_type: Any) -> str:
        """Convert Python type to GLSL type."""
        type_name = py_type.__name__
        if type_name not in self.GLSL_TYPES:
            raise TypeError(f"Unsupported type: {type_name}")
        return self.GLSL_TYPES[type_name]

    def _generate_uniforms(self, uniforms: Dict[str, str]) -> str:
        """Generate GLSL uniform declarations."""
        return "\n".join(f"uniform {type_} {name};" for name, type_ in uniforms.items())

    def _generate_functions(self, functions: List[ast.FunctionDef]) -> str:
        """Generate GLSL function definitions."""
        result = []
        for func in functions:
            # Convert Python function to GLSL
            glsl_func = self._convert_function(func)
            result.append(glsl_func)
        return "\n\n".join(result)

    def _convert_function(self, func: ast.FunctionDef) -> str:
        """Convert a Python function to GLSL."""
        # Basic implementation - will need to be expanded
        return_type = self._get_glsl_type(func.returns)
        args = [
            f"{self._get_glsl_type(arg.annotation)} {arg.arg}" for arg in func.args.args
        ]

        body = self._convert_body(func.body)

        return f"{return_type} {func.name}({', '.join(args)}) {{\n{body}\n}}"

    def _convert_body(self, body: List[ast.stmt]) -> str:
        """Convert Python function body to GLSL."""
        lines = []
        for node in body:
            if isinstance(node, ast.Return):
                lines.append(f"return {self._convert_expr(node.value)};")
            elif isinstance(node, ast.Assign):
                target = self._convert_expr(node.targets[0])
                value = self._convert_expr(node.value)
                lines.append(f"{target} = {value};")
            # Add more statement types as needed
        return "\n".join(f"    {line}" for line in lines)

    def _convert_expr(self, node: ast.expr) -> str:
        """Convert Python expression to GLSL."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Num):
            return str(float(node.n))
        elif isinstance(node, ast.Call):
            func_name = self._convert_expr(node.func)
            args = [self._convert_expr(arg) for arg in node.args]
            return f"{func_name}({', '.join(args)})"
        elif isinstance(node, ast.BinOp):
            left = self._convert_expr(node.left)
            right = self._convert_expr(node.right)
            op = self._convert_operator(node.op)
            return f"{left} {op} {right}"
        # Add more expression types as needed
        return str(node)

    def _convert_operator(self, op: ast.operator) -> str:
        """Convert Python operator to GLSL operator."""
        op_map = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
        }
        return op_map.get(type(op), "?")


def py2glsl(shader_func: Any) -> ShaderResult:
    """Convert Python shader function to GLSL."""
    transpiler = ShaderTranspiler()
    analysis = transpiler.analyze(shader_func)
    return transpiler.transform(analysis)
