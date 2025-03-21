"""Compatibility layer between new architecture and old backends.

This module provides adapters that allow the new architecture to work with
the existing codebase without breaking compatibility.
"""

from typing import Any

from py2glsl.transpiler.backends.models import BackendType
from py2glsl.transpiler.core.interfaces import TargetLanguageType
from py2glsl.transpiler.target import create_glsl_target


def backend_type_to_target_type(backend_type: BackendType) -> TargetLanguageType:
    """Convert a BackendType to a TargetLanguageType.

    Args:
        backend_type: Old backend type

    Returns:
        Corresponding new target type
    """
    if backend_type == BackendType.STANDARD:
        return TargetLanguageType.GLSL
    elif backend_type == BackendType.SHADERTOY:
        return TargetLanguageType.GLSL  # Still GLSL, but with Shadertoy dialect
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}")


def transpile_with_target(
    collected: Any, main_func: str, backend_type: BackendType
) -> tuple[str, set[str]]:
    """Transpile using the new target architecture.

    Args:
        collected: Collected information about functions, etc.
        main_func: Name of the main function
        backend_type: Old backend type to use

    Returns:
        Tuple of (generated code, used uniforms)
    """
    # First, validate all functions have proper return types (to match old behavior)
    for func_name, func_info in collected.functions.items():
        if func_name != main_func and func_info.return_type is None:
            from py2glsl.transpiler.errors import TranspilerError
            raise TranspilerError(
                f"Helper function '{func_name}' lacks return type annotation"
            )

    # Convert old backend type to new target parameters
    is_shadertoy = backend_type == BackendType.SHADERTOY

    # Create appropriate target
    language, _, _ = create_glsl_target(shadertoy=is_shadertoy)

    # Generate code using the target language
    return language.generate_code(collected, main_func)