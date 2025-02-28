"""
GLSL code generation for complete shader programs.

This module provides the top-level interface for transpiling Python code to GLSL
shaders.
"""

import inspect
from collections.abc import Callable
from typing import Any, cast

from loguru import logger

from py2glsl.transpiler.ast_parser import ShaderFunction, parse_shader_code
from py2glsl.transpiler.code_generator import generate_glsl
from py2glsl.transpiler.collector import collect_info
from py2glsl.transpiler.errors import TranspilerError


def transpile(
    *args: str | Callable[..., Any] | type[Any] | object,
    main_func: str | None = None,
    **kwargs: Any,
) -> tuple[str, set[str]]:
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
        if (
            name != "main_func"
            and not callable(value)
            and not isinstance(value, type)
            and isinstance(value, int | float | bool)
        ):
            global_constants[name] = value

    shader_input: str | dict[str, Callable[..., Any] | type[Any]] | None = None
    effective_main_func: str | None = main_func

    if len(args) == 1:
        if isinstance(args[0], str):
            shader_input = args[0]
        elif inspect.ismodule(args[0]):
            module = args[0]
            context: dict[str, Callable[..., Any] | type[Any]] = {}
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
            if callable(main_item) or isinstance(main_item, type):
                if hasattr(main_item, "__name__"):
                    # Check for test functions
                    if main_item.__name__.startswith("test_"):
                        raise TranspilerError(
                            "Test functions/classes are not supported in transpilation"
                        )
                    shader_input = {main_item.__name__: main_item}
                else:
                    raise TranspilerError("Item must have a __name__ attribute")
            else:
                raise TranspilerError("Unsupported item type")
    elif len(args) > 1:
        shader_input_dict: dict[str, Callable[..., Any] | type[Any]] = {}
        for item in args:
            if callable(item) or isinstance(item, type):
                if hasattr(item, "__name__"):
                    shader_input_dict[item.__name__] = item
                else:
                    raise TranspilerError("Item must have a __name__ attribute")
            else:
                raise TranspilerError(
                    f"Unsupported argument type: {type(item).__name__}"
                )
        shader_input = shader_input_dict

    if shader_input is None:
        raise TranspilerError("No shader input provided")

    # We need to handle the different types that could come through
    # to parse_shader_code correctly
    tree, effective_main_func = parse_shader_code(shader_input, effective_main_func)

    collected = collect_info(tree)
    for name, value in global_constants.items():
        value_str = str(value).lower() if isinstance(value, bool) else str(value)
        collected.globals[name] = (
            "float" if isinstance(value, float) else "int",
            value_str,
        )

    if effective_main_func not in collected.functions:
        raise TranspilerError(
            f"Main function '{effective_main_func}' not found in collected functions"
        )

    glsl_code, uniforms = generate_glsl(collected, effective_main_func)
    return glsl_code, uniforms
