from py2glsl.transpiler.backends.base import Backend, GLSLBackend
from py2glsl.transpiler.backends.glsl.shadertoy import create_shadertoy_backend
from py2glsl.transpiler.backends.glsl.standard import create_standard_backend
from py2glsl.transpiler.backends.models import BackendType
from py2glsl.transpiler.backends.render import (
    BaseRenderBackend,
    RenderBackend,
    UniformProvider,
)
from py2glsl.transpiler.backends.render_factory import (
    create_render_backend,
    register_render_backend,
)


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
