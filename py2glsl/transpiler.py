import argparse
import ast
import inspect
import sys
from dataclasses import dataclass

from loguru import logger

from py2glsl.builtins import *

# Configure Loguru for color-coded logging
logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{message}</cyan>",
)

# Built-ins for GLSL transpilation
builtins = {
    "abs": ("float", ["float"]),
    "cos": ("float", ["float"]),
    "fract": ("vec2", ["vec2"]),
    "length": ("float", ["vec2"]),
    "sin": ("float", ["float"]),
    "sqrt": ("float", ["float"]),
    "tan": ("float", ["float"]),
    "radians": ("float", ["float"]),
    "mix": ("vec3", ["vec3", "vec3", "float"]),
    "normalize": ("vec3", ["vec3"]),
    "cross": ("vec3", ["vec3", "vec3"]),
    "max": ("float", ["float", "float"]),
    "min": ("float", ["float", "float"]),
    "vec2": ("vec2", ["float", "float"]),
    "vec3": ("vec3", ["float", "float", "float"]),
    "vec4": ("vec4", ["float", "float", "float", "float"]),
    "distance": ("float", ["vec3", "vec3"]),  # Added distance
}


# Default uniforms
@dataclass
class Interface:
    u_time: "float"
    u_aspect: "float"
    u_resolution: "vec2"
    u_mouse_pos: "vec2"
    u_mouse_uv: "vec2"


default_uniforms = {
    "u_time": "float",
    "u_aspect": "float",
    "u_resolution": "vec2",
    "u_mouse_pos": "vec2",
    "u_mouse_uv": "vec2",
}


def pytype_to_glsl(pytype):
    """Convert Python type annotations to GLSL types."""
    return pytype


class StructDefinition:
    """Helper class to represent GLSL structs in Python."""

    def __init__(self, name, fields):
        self.name = name
        self.fields = fields  # List of (name, type) tuples


class FunctionCollector(ast.NodeVisitor):
    # [Same as before, no changes needed]
    ...


class GLSLGenerator:
    # [Same as before, no changes needed]
    ...


def transpile(shader_input, main_func_name="main_shader"):
    """Transpile Python shader code or function to GLSL."""
    if isinstance(shader_input, str):
        module_ast = ast.parse(shader_input)
    elif callable(shader_input):
        source = inspect.getsource(shader_input)
        # Since inspect.getsource gets only the function, we need the full module context
        # For simplicity, assume the function's module is imported and parse the whole file
        module_source = inspect.getsource(inspect.getmodule(shader_input))
        module_ast = ast.parse(module_source)
    else:
        raise ValueError("Input must be a string or a callable function")

    collector = FunctionCollector()
    collector.visit(module_ast)
    generator = GLSLGenerator(
        collector.functions,
        collector.structs,
        collector.globals,
        builtins,
        default_uniforms,
    )

    # Generate global variable declarations
    for var_name, (var_type, var_value) in collector.globals.items():
        if var_value:
            generator.code += f"{var_type} {var_name} = {var_value};\n"
        else:
            generator.code += f"{var_type} {var_name};\n"

    # Generate struct definitions
    for struct_name, struct_def in collector.structs.items():
        fields = [
            f"    {field_type} {field_name};"
            for field_name, field_type in struct_def.fields
        ]
        generator.code += f"struct {struct_name} {{\n" + "\n".join(fields) + "\n};\n"

    # Generate all functions
    for func_name, (_, _, node) in collector.functions.items():
        generator.generate(func_name, node)

    # Validate main function
    if main_func_name not in collector.functions:
        raise ValueError(f"Main shader function '{main_func_name}' not found")
    main_params = collector.functions[main_func_name][1]
    main_param_names = [
        arg.arg
        for arg in (
            collector.functions[main_func_name][2].args.posonlyargs
            + collector.functions[main_func_name][2].args.args
        )
    ]

    # Check which parameters are uniforms (all u_* parameters)
    used_uniforms = {
        param_name for param_name in main_param_names if param_name.startswith("u_")
    }

    # Build fragment shader
    fragment_shader = "#version 460 core\n"
    fragment_shader += f"in {main_params[0]} {main_param_names[0]};\n"  # vs_uv as input
    for param_type, param_name in zip(main_params[1:], main_param_names[1:]):
        if param_name in used_uniforms:  # All u_* parameters are uniforms
            fragment_shader += f"uniform {param_type} {param_name};\n"
    fragment_shader += "out vec4 fragColor;\n"
    fragment_shader += generator.code
    fragment_shader += "void main() {\n"
    fragment_shader += (
        f"    fragColor = {main_func_name}({', '.join(main_param_names)});\n"
    )
    fragment_shader += "}\n"

    logger.opt(colors=True).info(
        f"<yellow>Fragment Shader GLSL:\n{fragment_shader}</yellow>"
    )
    return fragment_shader, used_uniforms
