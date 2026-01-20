"""Backend module for shader code generation."""

from enum import Enum, auto

from py2glsl.transpiler.backends.base import Backend
from py2glsl.transpiler.backends.glsl.shadertoy import create_shadertoy_backend
from py2glsl.transpiler.backends.glsl.standard import create_standard_backend


class BackendType(Enum):
    """Supported GLSL backend types."""

    STANDARD = auto()
    SHADERTOY = auto()


def create_backend(backend_type: BackendType = BackendType.STANDARD) -> Backend:
    """Create a backend instance based on type."""
    if backend_type == BackendType.STANDARD:
        return create_standard_backend()
    elif backend_type == BackendType.SHADERTOY:
        return create_shadertoy_backend()
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}")
