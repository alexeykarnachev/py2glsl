"""Transpiler entry point for Python to shader code."""

import inspect
from collections.abc import Callable
from typing import Any

from loguru import logger

from py2glsl.transpiler.ast_parser import parse_shader_code
from py2glsl.transpiler.collector import collect_info
from py2glsl.transpiler.core.interfaces import TargetLanguageType
from py2glsl.transpiler.errors import TranspilerError
from py2glsl.transpiler.models import CollectedInfo, StructDefinition, StructField
from py2glsl.transpiler.target import create_target

_RESERVED_KWARGS = {"main_func", "target_type", "shadertoy"}

ShaderInput = str | dict[str, Callable[..., Any] | type[Any]]


def _extract_global_constants(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Extract numeric/boolean constants from kwargs."""
    return {
        name: value
        for name, value in kwargs.items()
        if name not in _RESERVED_KWARGS
        and not callable(value)
        and not isinstance(value, type)
        and isinstance(value, int | float | bool)
    }


def _process_single_arg(
    arg: Any, main_func: str | None
) -> tuple[ShaderInput, str | None]:
    """Process a single transpile argument (string, module, or callable)."""
    if isinstance(arg, str):
        return arg, main_func
    if inspect.ismodule(arg):
        return _process_module(arg, main_func)
    return _process_callable_or_type(arg)


def _process_module(
    module: Any, main_func: str | None
) -> tuple[dict[str, Callable[..., Any] | type[Any]], str]:
    """Extract functions and dataclasses from a module."""
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
    """Wrap a callable/type in a dict for transpilation."""
    if not (callable(item) or isinstance(item, type)):
        raise TranspilerError("Unsupported item type")
    if not hasattr(item, "__name__"):
        raise TranspilerError("Item must have a __name__ attribute")
    if item.__name__.startswith("test_"):
        raise TranspilerError(
            "Test functions/classes are not supported in transpilation"
        )
    return {item.__name__: item}, None


def _process_multiple_args(
    args: tuple[Any, ...],
) -> dict[str, Callable[..., Any] | type[Any]]:
    """Process multiple callables/types into a shader input dict."""
    result: dict[str, Callable[..., Any] | type[Any]] = {}
    for item in args:
        if not (callable(item) or isinstance(item, type)):
            raise TranspilerError(f"Unsupported argument type: {type(item).__name__}")
        if not hasattr(item, "__name__"):
            raise TranspilerError("Item must have a __name__ attribute")
        result[item.__name__] = item
    return result


def _extract_structs_from_kwargs(
    collected: CollectedInfo, kwargs: dict[str, Any]
) -> None:
    """Extract dataclass structs from kwargs and add to collected info."""
    for name, value in kwargs.items():
        if name in _RESERVED_KWARGS or name in collected.structs:
            continue
        if not hasattr(value, "__dataclass_fields__"):
            continue

        logger.debug(f"Processing dataclass from kwargs: {name}")
        fields = []
        for field_name, field_info in value.__dataclass_fields__.items():
            field_type = getattr(field_info, "type", None)
            type_name = getattr(field_type, "__name__", None) if field_type else None
            if type_name:
                fields.append(StructField(name=field_name, type_name=type_name))

        if fields:
            collected.structs[name] = StructDefinition(name=name, fields=fields)
            logger.debug(f"Added struct: {name} fields={[f.name for f in fields]}")


def _add_globals_to_collected(
    collected: CollectedInfo, global_constants: dict[str, Any]
) -> None:
    """Add global constants to collected info."""
    for name, value in global_constants.items():
        value_str = str(value).lower() if isinstance(value, bool) else str(value)
        type_str = "float" if isinstance(value, float) else "int"
        collected.globals[name] = (type_str, value_str)


def _determine_shader_input(
    args: tuple[Any, ...], main_func: str | None
) -> tuple[ShaderInput, str | None]:
    """Determine shader input from args."""
    if not args:
        raise TranspilerError("No shader input provided")
    if len(args) == 1:
        return _process_single_arg(args[0], main_func)
    return _process_multiple_args(args), main_func


def transpile(
    *args: str | Callable[..., Any] | type[Any] | object,
    main_func: str | None = None,
    target_type: TargetLanguageType = TargetLanguageType.GLSL,
    shadertoy: bool = False,
    **kwargs: Any,
) -> tuple[str, set[str]]:
    """Transpile Python code to shader code.

    Args:
        *args: Python code string, function(s), or class(es) to transpile
        main_func: Entry point function name (auto-detected if not provided)
        target_type: Target language (GLSL or SHADERTOY)
        shadertoy: Deprecated, use target_type=TargetLanguageType.SHADERTOY
        **kwargs: Global constants (int/float/bool) or dataclass structs

    Returns:
        Tuple of (shader code, set of uniform names)
    """
    if shadertoy and target_type == TargetLanguageType.GLSL:
        target_type = TargetLanguageType.SHADERTOY

    logger.debug(f"Transpiling: args={args}, main={main_func}, target={target_type}")

    global_constants = _extract_global_constants(kwargs)
    shader_input, effective_main_func = _determine_shader_input(args, main_func)
    tree, effective_main_func = parse_shader_code(shader_input, effective_main_func)
    collected = collect_info(tree)

    _add_globals_to_collected(collected, global_constants)
    _extract_structs_from_kwargs(collected, kwargs)

    if effective_main_func not in collected.functions:
        raise TranspilerError(
            f"Main function '{effective_main_func}' not found in collected functions"
        )

    for func_name, func_info in collected.functions.items():
        if func_name != effective_main_func and func_info.return_type is None:
            raise TranspilerError(
                f"Helper function '{func_name}' lacks return type annotation"
            )

    language, _, _ = create_target(target_type)
    return language.generate_code(collected, effective_main_func)
