"""Factory for creating render backends."""


from py2glsl.transpiler.backends.glsl.shadertoy_render import ShadertoyRenderBackend
from py2glsl.transpiler.backends.glsl.standard_render import StandardGLSLRenderBackend
from py2glsl.transpiler.backends.models import BackendType
from py2glsl.transpiler.backends.render import RenderBackend

# Registry of backend types to their render backend implementations
_BACKEND_REGISTRY: dict[BackendType, type[RenderBackend]] = {
    BackendType.STANDARD: StandardGLSLRenderBackend,
    BackendType.SHADERTOY: ShadertoyRenderBackend,
}


def create_render_backend(
    backend_type: BackendType = BackendType.STANDARD,
) -> RenderBackend:
    """Create a render backend for the given backend type.

    Args:
        backend_type: The type of backend to create a render backend for

    Returns:
        A render backend instance

    Raises:
        ValueError: If the backend type is not supported
    """
    if backend_type not in _BACKEND_REGISTRY:
        raise ValueError(f"Unsupported backend type: {backend_type}")

    backend_class = _BACKEND_REGISTRY[backend_type]
    return backend_class()


def register_render_backend(
    backend_type: BackendType, backend_class: type[RenderBackend]
) -> None:
    """Register a new render backend implementation.

    Args:
        backend_type: The backend type to register
        backend_class: The render backend class to associate with the type
    """
    _BACKEND_REGISTRY[backend_type] = backend_class
