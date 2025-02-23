import ast
import inspect
import re
from dataclasses import dataclass
from inspect import Parameter, signature
from textwrap import dedent
from typing import Callable, Dict, List, Tuple

from loguru import logger

from py2glsl.glsl.types import mat3, mat4, vec2, vec3, vec4
from py2glsl.transpiler.glsl_builder import GLSLBuilder, GLSLCodeError
from py2glsl.transpiler.type_system import TypeInferer, TypeInfo


class GLSLTypeError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


@dataclass
class TranspilerResult:
    vertex_src: str
    fragment_src: str


def transpile(func: Callable) -> TranspilerResult:
    """Transpile a Python function to GLSL shaders"""
    validate_function_signature(func)
    uniforms, attributes = detect_interface(func)

    builder = GLSLBuilder()
    builder.configure_shader_transpiler(
        uniforms=uniforms,
        attributes=attributes,
        func_name=func.__name__,
        shader_body=extract_function_body(func),
    )

    # Explicitly remove main shader function from vertex stage
    builder.vertex_functions = [
        f for f in builder.vertex_functions if not f.startswith(f"vec4 {func.__name__}")
    ]

    result = TranspilerResult(
        vertex_src=builder.build_vertex_shader(),
        fragment_src=builder.build_fragment_shader(),
    )

    # Log generated shaders
    logger.debug("Generated Vertex Shader:\n{}", result.vertex_src)
    logger.debug("Generated Fragment Shader:\n{}", result.fragment_src)

    return result


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


def detect_interface(func: Callable) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Detect uniforms and attributes from function parameters"""
    sig = signature(func)
    uniforms = {}
    attributes = {}

    for param in sig.parameters.values():
        # Handle parameters with default values
        if param.default != Parameter.empty:
            uniforms[param.name] = get_glsl_type(param.annotation)
        elif param.kind == Parameter.POSITIONAL_ONLY:
            attributes[param.name] = get_glsl_type(param.annotation)
        else:
            uniforms[param.name] = get_glsl_type(param.annotation)

    return uniforms, attributes


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


def extract_function_body(func: Callable) -> list[str]:
    """Extract function body with proper GLSL type declarations"""
    source = inspect.getsource(func)
    tree = ast.parse(dedent(source))

    # Run type inference first
    inferer = TypeInferer()
    inferer.visit(tree)

    processed = []
    for node in tree.body[0].body:
        if isinstance(node, ast.AnnAssign):
            # Handle typed assignments: "var: type = value"
            decl = f"{inferer.symbols[node.target.id].glsl_name} {ast.unparse(node)}"
            processed.append(decl.replace("=", "="))
        elif isinstance(node, ast.Assign):
            # Handle regular assignments with type inference
            target = node.targets[0].id
            var_type = inferer.symbols.get(target, TypeInfo.FLOAT).glsl_name
            value = ast.unparse(node.value)
            processed.append(f"{var_type} {target} = {value};")
        elif isinstance(node, ast.Expr):
            # Skip comments/docstrings
            continue
        else:
            # Preserve other statements
            processed.append(ast.unparse(node))

    return processed
