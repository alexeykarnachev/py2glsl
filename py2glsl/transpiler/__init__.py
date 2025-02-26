"""
Python to GLSL shader transpiler.

This package provides functionality to convert Python code into GLSL shader code for use with OpenGL.
It parses Python functions, dataclasses, and other constructs and generates equivalent GLSL code.

The main entry point is the "transpile" function, which takes Python code (as a string or callable)
and returns the equivalent GLSL code along with a set of used uniforms.
"""

import ast
import inspect
import textwrap
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union

from loguru import logger

from py2glsl.transpiler.ast_parser import parse_shader_code
from py2glsl.transpiler.code_generator import generate_glsl
from py2glsl.transpiler.collector import collect_info
from py2glsl.transpiler.errors import TranspilerError


def transpile(
    *args: Union[str, Callable, Type, object],
    main_func: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[str, Set[str]]:
    """Transpile Python code to GLSL shader code.

    This is the main entry point for the transpiler. It accepts various forms of input:
    - A string containing Python code
    - A function or class to transpile
    - Multiple functions or classes to include in the transpilation

    Args:
        *args: The Python code or callables to transpile
        main_func: Name of the main function to use as shader entry point
        **kwargs: Additional keyword arguments:
            - Additional functions/classes to include
            - Global constants to include in the shader

    Returns:
        Tuple of (generated GLSL code, set of used uniform variables)

    Raises:
        TranspilerError: If transpilation fails

    Examples:
        # Transpile a single function
        glsl_code, uniforms = transpile(my_shader_func)

        # Transpile with multiple functions/structs
        glsl_code, uniforms = transpile(my_struct, my_helper_func, my_shader_func)

        # Specify the main function
        glsl_code, uniforms = transpile(my_struct, my_helper_func, my_shader_func,
                                        main_func="my_shader_func")

        # Include global constants
        glsl_code, uniforms = transpile(my_shader_func, PI=3.14159, MAX_STEPS=100)
    """
    logger.debug(
        f"Transpiling with args: {args}, main_func: {main_func}, kwargs: {kwargs}"
    )

    global_constants = {}
    for name, value in kwargs.items():
        if name != "main_func" and not callable(value) and not isinstance(value, type):
            if isinstance(value, (int, float, bool)):
                global_constants[name] = value

    shader_input = None
    effective_main_func = main_func

    if len(args) == 1:
        if isinstance(args[0], str):
            shader_input = args[0]
        elif inspect.ismodule(args[0]):
            module = args[0]
            if hasattr(module, "__all__"):
                context = {name: getattr(module, name) for name in module.__all__}
            else:
                context = {
                    name: obj
                    for name, obj in module.__dict__.items()
                    if inspect.isfunction(obj)
                    or (inspect.isclass(obj) and hasattr(obj, "__dataclass_fields__"))
                }
            shader_input = context
            effective_main_func = main_func or "shader"
        else:
            main_item = args[0]
            if main_item.__name__.startswith("test_"):
                raise TranspilerError(
                    f"Main function '{main_item.__name__}' excluded due to 'test_' prefix"
                )
            context = {main_item.__name__: main_item}
            shader_input = context
            effective_main_func = main_func or main_item.__name__
    else:
        context = {}
        for item in args:
            if inspect.isfunction(item):
                context[item.__name__] = item
            elif inspect.isclass(item) and hasattr(item, "__dataclass_fields__"):
                context[item.__name__] = item
            else:
                raise TranspilerError(
                    f"Unsupported item type in transpile args: {type(item)}"
                )
        shader_input = context
        effective_main_func = main_func

    tree, parsed_main_func = parse_shader_code(shader_input, effective_main_func)
    effective_main_func = parsed_main_func

    collected = collect_info(tree)

    for name, value in global_constants.items():
        if isinstance(value, (int, float)):
            type_name = "float" if isinstance(value, float) else "int"
            collected.globals[name] = (type_name, str(value))
        elif isinstance(value, bool):
            collected.globals[name] = ("bool", "true" if value else "false")

    if effective_main_func not in collected.functions:
        raise TranspilerError(f"Main function '{effective_main_func}' not found")

    return generate_glsl(collected, effective_main_func)


__all__ = ["transpile", "TranspilerError"]
