import ast
import inspect
import re
from dataclasses import dataclass
from inspect import Parameter, signature
from textwrap import dedent
from typing import Callable, Dict, List, Set, Tuple

from loguru import logger

from py2glsl.glsl.types import mat3, mat4, vec2, vec3, vec4
from py2glsl.transpiler.glsl_builder import GLSLBuilder, GLSLCodeError
from py2glsl.transpiler.type_system import TypeInferer, TypeInfo

from .glsl_builder import GLSLBuilder, GLSLCodeError, TranspilerResult
from .utils import extract_function_body


class GLSLTypeError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


def transpile(func: Callable) -> TranspilerResult:
    validate_function_signature(func)
    source = inspect.getsource(func)
    tree = ast.parse(dedent(source))
    func_node = tree.body[0]
    func_body_ast = func_node.body

    # 1. Find all called functions
    called_functions = _find_called_functions(
        func_body_ast,
        func.__globals__,
        func.__name__,
    )

    # 2. Create type inferer with dependency info
    type_inferer = TypeInferer(called_functions=called_functions)

    # 3. Detect interface and populate type info
    uniforms, attributes = detect_interface(func, type_inferer)

    # 4. Generate GLSL body with full type info
    glsl_body = extract_function_body(func_node, type_inferer)

    # 5. Build final shader code
    builder = GLSLBuilder()
    builder.configure_shader_transpiler(
        uniforms=uniforms,
        attributes=attributes,
        func_name=func.__name__,
        shader_body=glsl_body,
        called_functions=called_functions,
    )
    return builder.build()


def _find_called_functions(
    body: list[ast.stmt],
    globals_dict: dict,
    entry_point: str,
) -> dict[str, Callable]:
    visitor = FunctionCallVisitor(globals_dict)
    for node in body:
        visitor.visit(node)

    ordered = []
    seen = set()

    def add_func(name: str):
        if name in seen or name not in visitor.valid_functions:
            return
        seen.add(name)
        for dep in visitor.call_graph.get(name, []):
            add_func(dep)
        ordered.append(visitor.valid_functions[name])

    # Start with the entry point function
    add_func(entry_point)

    return {func.__name__: func for func in ordered}


def detect_interface(
    func: Callable, inferer: TypeInferer
) -> Tuple[Dict[str, str], Dict[str, str]]:
    sig = signature(func)
    uniforms = {}
    attributes = {}

    for param in sig.parameters.values():
        # Parameter classification logic
        if param.default != Parameter.empty:
            uniforms[param.name] = get_glsl_type(param.annotation)
        elif param.kind == Parameter.POSITIONAL_ONLY:
            attributes[param.name] = get_glsl_type(param.annotation)
        else:
            uniforms[param.name] = get_glsl_type(param.annotation)

        # Populate type inferer
        py_type = param.annotation
        inferer.symbols[param.name] = TypeInfo.from_pytype(py_type)

    return uniforms, attributes


class FunctionCallVisitor(ast.NodeVisitor):
    def __init__(self, globals_dict: dict):
        self.globals = globals_dict
        self.valid_functions: dict[str, Callable] = {}
        self.call_graph: dict[str, list[str]] = {}
        self.top_level_calls: set[str] = set()
        self.current_func: str | None = None

    def visit_FunctionDef(self, node: ast.FunctionDef):
        prev_func = self.current_func
        self.current_func = node.name
        self.generic_visit(node)
        self.current_func = prev_func

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            self._process_call(node.func.id)
        self.generic_visit(node)

    def _process_call(self, func_name: str):
        func = self.globals.get(func_name)
        if not func or not inspect.isfunction(func):
            return

        try:
            # Get full function AST instead of splitting lines
            func_ast = ast.parse(inspect.getsource(func))
            func_node = func_ast.body[0]

            # Process the function body nodes directly
            for stmt in func_node.body:
                self.visit(stmt)

        except IndentationError as e:
            raise GLSLCodeError(f"Indentation error in {func_name}: {str(e)}")
        except Exception as e:
            raise GLSLCodeError(f"Error processing {func_name}: {str(e)}")


@dataclass
class FunctionRegistry:
    functions: dict[str, Callable]
    dependencies: dict[str, list[str]]

    @classmethod
    def from_ast(cls, body: list[ast.stmt], globals_dict: dict) -> "FunctionRegistry":
        visitor = FunctionCallVisitor(globals_dict)
        for node in body:
            visitor.visit(node)

        ordered = []
        seen = set()

        def add_func(name: str):
            if name in seen or name not in visitor.valid_functions:
                return
            seen.add(name)
            for dep in visitor.call_graph.get(name, []):
                add_func(dep)
            ordered.append(visitor.valid_functions[name])

        for name in visitor.top_level_calls:
            add_func(name)

        return cls(
            functions={f.__name__: f for f in ordered}, dependencies=visitor.call_graph
        )

    def get_return_type(self, func_name: str) -> TypeInfo:
        func = self.functions[func_name]
        return TypeInfo.from_pytype(func.return_annotation)


def validate_function_signature(func: Callable) -> None:
    """Validate the shader function signature"""
    sig = signature(func)
    if sig.return_annotation != vec4:
        raise GLSLTypeError("Shader must return vec4")

    pos_params = [
        p for p in sig.parameters.values() if p.kind == Parameter.POSITIONAL_ONLY
    ]
    if len(pos_params) != 1 or pos_params[0].annotation != vec2:
        raise GLSLTypeError("Must have one positional-only vec2 parameter")


def get_glsl_type(py_type: type) -> str:
    """Map Python type to GLSL type string"""
    type_map = {
        vec2: "vec2",
        vec3: "vec3",
        vec4: "vec4",
        mat3: "mat3",
        mat4: "mat4",
        float: "float",
        int: "int",
        bool: "bool",
    }
    if py_type not in type_map:
        raise GLSLTypeError(f"Unsupported type: {py_type.__name__}")
    return type_map[py_type]
