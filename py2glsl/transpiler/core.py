import inspect
from dataclasses import dataclass
from inspect import Parameter, signature
from typing import Callable, Dict, List, Tuple

from py2glsl.glsl.types import vec2, vec4
from py2glsl.transpiler.glsl_builder import GLSLBuilder, GLSLCodeError


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

    return TranspilerResult(
        vertex_src=builder.build_vertex_shader(),
        fragment_src=builder.build_fragment_shader(),
    )


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
        target = attributes if param.kind == Parameter.POSITIONAL_ONLY else uniforms
        target[param.name] = get_glsl_type(param.annotation)

    return uniforms, attributes


def get_glsl_type(py_type: type) -> str:
    """Map Python type to GLSL type string"""
    type_map = {vec2: "vec2", vec4: "vec4", float: "float", int: "int", bool: "bool"}
    if py_type not in type_map:
        raise GLSLTypeError(f"Unsupported type: {py_type.__name__}")
    return type_map[py_type]


def extract_function_body(func: Callable) -> List[str]:
    """Extract and clean the function body lines"""
    source = inspect.getsource(func)
    lines = source.split("\n")

    # Find function definition
    try:
        def_line = next(
            i for i, line in enumerate(lines) if line.strip().startswith("def ")
        )
    except StopIteration:
        raise GLSLCodeError("Function definition not found")

    # Find body start
    body_start = None
    for i in range(def_line, len(lines)):
        if "-> vec4:" in lines[i]:
            body_start = i + 1
            break
    if body_start is None:
        raise GLSLCodeError("Missing return type annotation '-> vec4'")

    # Determine indentation
    indent = len(lines[body_start]) - len(lines[body_start].lstrip())
    if indent == 0:
        raise GLSLCodeError("Invalid indentation in function body")

    # Extract body lines
    body_lines = []
    for line in lines[body_start:]:
        clean_line = line[indent:]
        if clean_line.lstrip().startswith("def "):
            break
        if clean_line.strip():
            body_lines.append(clean_line.rstrip())

    return body_lines
