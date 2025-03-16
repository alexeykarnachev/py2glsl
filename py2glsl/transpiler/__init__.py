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
from py2glsl.transpiler.collector import collect_info
from py2glsl.transpiler.errors import TranspilerError
from py2glsl.transpiler.models import CollectedInfo


def _extract_global_constants(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Extract global constants from keyword arguments.

    Args:
        kwargs: Keyword arguments passed to transpile

    Returns:
        Dictionary of global constants
    """
    global_constants = {}
    for name, value in kwargs.items():
        if (
            name != "main_func"
            and not callable(value)
            and not isinstance(value, type)
            and isinstance(value, int | float | bool)
        ):
            global_constants[name] = value
    return global_constants


def _process_single_arg(
    arg: Any, main_func: str | None
) -> tuple[str | dict[str, Callable[..., Any] | type[Any]], str | None]:
    """Process a single argument to transpile.

    Args:
        arg: Single argument to process
        main_func: Name of the main function provided by the user

    Returns:
        Tuple of (shader input, effective main function name)

    Raises:
        TranspilerError: If the argument is not valid for transpilation
    """
    if isinstance(arg, str):
        return arg, main_func
    elif inspect.ismodule(arg):
        return _process_module(arg, main_func)
    else:
        return _process_callable_or_type(arg)


def _process_module(
    module: Any, main_func: str | None
) -> tuple[dict[str, Callable[..., Any] | type[Any]], str]:
    """Process a module argument to transpile.

    Args:
        module: Module to process
        main_func: Name of the main function provided by the user

    Returns:
        Tuple of (shader input dictionary, effective main function name)
    """
    if hasattr(module, "__all__"):
        context = {name: getattr(module, name) for name in module.__all__}
    else:
        context = {
            name: obj
            for name, obj in module.__dict__.items()
            if inspect.isfunction(obj)
            or (inspect.isclass(obj) and hasattr(obj, "__dataclass_fields__"))
        }
    return context, main_func or "shader"


def _process_callable_or_type(
    item: Any,
) -> tuple[dict[str, Callable[..., Any] | type[Any]], None]:
    """Process a callable or type argument to transpile.

    Args:
        item: Callable or type to process

    Returns:
        Tuple of (shader input dictionary, None for main function)

    Raises:
        TranspilerError: If the item is not valid for transpilation
    """
    if callable(item) or isinstance(item, type):
        if not hasattr(item, "__name__"):
            raise TranspilerError("Item must have a __name__ attribute")

        # Check for test functions
        if item.__name__.startswith("test_"):
            raise TranspilerError(
                "Test functions/classes are not supported in transpilation"
            )

        return {item.__name__: item}, None
    else:
        raise TranspilerError("Unsupported item type")


def _process_multiple_args(
    args: tuple[Any, ...],
) -> dict[str, Callable[..., Any] | type[Any]]:
    """Process multiple arguments to transpile.

    Args:
        args: Tuple of arguments to process

    Returns:
        Dictionary mapping names to callables or types

    Raises:
        TranspilerError: If any argument is not valid for transpilation
    """
    shader_input_dict: dict[str, Callable[..., Any] | type[Any]] = {}
    for item in args:
        if callable(item) or isinstance(item, type):
            if not hasattr(item, "__name__"):
                raise TranspilerError("Item must have a __name__ attribute")
            shader_input_dict[item.__name__] = item
        else:
            raise TranspilerError(f"Unsupported argument type: {type(item).__name__}")
    return shader_input_dict


def _add_globals_to_collected(
    collected: CollectedInfo, global_constants: dict[str, Any]
) -> None:
    """Add global constants to collected information.

    Args:
        collected: Collected information about the shader
        global_constants: Dictionary of global constants
    """
    for name, value in global_constants.items():
        value_str = str(value).lower() if isinstance(value, bool) else str(value)
        collected.globals[name] = (
            "float" if isinstance(value, float) else "int",
            value_str,
        )


def _determine_shader_input(
    args: tuple[Any, ...], main_func: str | None
) -> tuple[str | dict[str, Callable[..., Any] | type[Any]], str | None]:
    """Determine the shader input from args.

    Args:
        args: Arguments passed to transpile
        main_func: Name of the main function provided by the user

    Returns:
        Tuple of (shader input, effective main function name)

    Raises:
        TranspilerError: If no valid shader input is provided
    """
    if not args:
        raise TranspilerError("No shader input provided")

    if len(args) == 1:
        return _process_single_arg(args[0], main_func)
    else:
        return _process_multiple_args(args), main_func


def transpile(
    *args: str | Callable[..., Any] | type[Any] | object,
    main_func: str | None = None,
    backend_type: Any = None,
    backend_options: dict[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[str, set[str]]:
    """Transpile Python code to shader code.

    This is the main entry point for the transpiler. It accepts various forms of input:
    - A string containing Python code
    - A function or class to transpile
    - Multiple functions or classes to include in the transpilation

    Args:
        *args: The Python code or callables to transpile
        main_func: Name of the main function to use as shader entry point
        backend_type: The backend type to use for code generation (default: STANDARD)
        backend_options: Options to pass to the backend
        **kwargs: Additional keyword arguments:
            - Additional functions/classes to include
            - Global constants to include in the shader

    Returns:
        Tuple of (generated shader code, set of used uniform variables)

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

        # Use Shadertoy backend
        from py2glsl.transpiler.backends.models import BackendType
        glsl_code, uniforms = transpile(
            my_shader_func, backend_type=BackendType.SHADERTOY
        )
    """
    # Lazy import backends to avoid circular imports
    from py2glsl.transpiler.backends import create_backend
    from py2glsl.transpiler.backends.models import BackendType

    if backend_type is None:
        backend_type = BackendType.STANDARD

    logger.debug(
        f"Transpiling with args: {args}, main_func: {main_func}, "
        f"backend_type: {backend_type}, kwargs: {kwargs}"
    )

    # Extract global constants from kwargs
    global_constants = _extract_global_constants(kwargs)

    # Determine shader input and main function
    shader_input, effective_main_func = _determine_shader_input(args, main_func)

    # Parse shader code
    tree, effective_main_func = parse_shader_code(shader_input, effective_main_func)

    # Collect shader information
    collected = collect_info(tree)

    # Add global constants
    _add_globals_to_collected(collected, global_constants)

    # Validate main function exists
    if effective_main_func not in collected.functions:
        raise TranspilerError(
            f"Main function '{effective_main_func}' not found in collected functions"
        )

    # Create the appropriate backend
    backend = create_backend(backend_type)

    # Generate shader code using the backend
    shader_code, uniforms = backend.generate_code(collected, effective_main_func)
    return shader_code, uniforms
