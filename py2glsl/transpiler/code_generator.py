"""
GLSL code generation for complete shader programs.

This module handles generation of GLSL code from the collected information,
combining function bodies, struct definitions, and global constants.
"""

from typing import Set, Tuple

from loguru import logger

from py2glsl.transpiler.code_gen_stmt import generate_body
from py2glsl.transpiler.errors import TranspilerError
from py2glsl.transpiler.models import CollectedInfo


def generate_glsl(collected: CollectedInfo, main_func: str) -> Tuple[str, Set[str]]:
    """Generate GLSL code from the collected information.

    Args:
        collected: Information about functions, structs, and globals
        main_func: Name of the main function to use as shader entry point

    Returns:
        Tuple of (generated GLSL code, set of used uniform variables)

    Raises:
        TranspilerError: If there are issues generating valid GLSL
    """
    logger.debug(f"Starting GLSL generation for main function: {main_func}")
    lines = []
    used_uniforms = set()

    main_func_info = collected.functions[main_func]
    main_func_node = main_func_info.node

    if not main_func_node.body:
        raise TranspilerError("Empty function body not supported in GLSL")

    # Version directive must be the first line
    lines.append("#version 460 core\n")

    # Collect uniform variables from main function params
    for i, arg in enumerate(main_func_node.args.args):
        if arg.arg != "vs_uv":  # vs_uv is a special input variable, not a uniform
            param_type = main_func_info.param_types[i]
            lines.append(f"uniform {param_type} {arg.arg};")
            used_uniforms.add(arg.arg)

    # Add global constants
    if collected.globals:
        lines.append("")  # Add blank line for readability
        for name, (type_name, value) in collected.globals.items():
            lines.append(f"const {type_name} {name} = {value};")

    # Add struct definitions
    if collected.structs:
        lines.append("")  # Add blank line for readability
        for struct_name, struct_def in collected.structs.items():
            lines.append(f"struct {struct_name} {{")
            for field in struct_def.fields:
                lines.append(f"    {field.type_name} {field.name};")
            lines.append("};")

    # Add function definitions
    lines.append("")  # Add blank line for readability
    for func_name, func_info in collected.functions.items():
        # Check if function has a return type annotation
        if not func_info.return_type and func_name != main_func:
            raise TranspilerError(
                f"Helper function '{func_name}' lacks return type annotation"
            )

        is_main = func_name == main_func
        effective_return_type = (
            "vec4" if is_main and not func_info.return_type else func_info.return_type
        )

        node = func_info.node
        param_str = ", ".join(
            f"{p_type} {arg.arg}"
            for p_type, arg in zip(func_info.param_types, node.args.args)
        )

        # Initialize symbols with function parameters and global constants
        symbols = {
            arg.arg: p_type
            for arg, p_type in zip(node.args.args, func_info.param_types)
        }
        
        # Add global constants to the symbols table
        for name, (type_name, value) in collected.globals.items():
            if type_name is not None:
                symbols[name] = type_name
            
        # Since body is List[ast.stmt] and not List[ast.AST], this is compatible with generate_body
        body_lines = generate_body(node.body, symbols, collected)

        lines.append(f"{effective_return_type} {func_name}({param_str}) {{")
        for line in body_lines:
            lines.append(f"    {line}")
        lines.append("}")

    # Add main function
    lines.append("\nin vec2 vs_uv;\nout vec4 fragColor;\n\nvoid main() {")
    main_call_args = [arg.arg for arg in main_func_node.args.args]
    main_call_str = ", ".join(main_call_args)
    lines.append(f"    fragColor = {main_func}({main_call_str});")
    lines.append("}")

    glsl_code = "\n".join(lines)
    return glsl_code, used_uniforms
