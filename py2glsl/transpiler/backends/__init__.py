from py2glsl.transpiler.backends.base import Backend
from py2glsl.transpiler.backends.glsl.shadertoy import create_shadertoy_backend
from py2glsl.transpiler.backends.glsl.standard import create_standard_backend
from py2glsl.transpiler.backends.models import BackendType


def create_backend(backend_type: BackendType = BackendType.STANDARD) -> Backend:
    """Create a backend instance based on type.

    Args:
        backend_type: The type of backend to create

    Returns:
        An instance of the requested backend

    Raises:
        ValueError: If the backend type is not supported
    """
    if backend_type == BackendType.STANDARD:
        return create_standard_backend()
    elif backend_type == BackendType.SHADERTOY:
        return create_shadertoy_backend()
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}")
