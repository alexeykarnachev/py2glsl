"""Core interfaces for the transpiler system."""

from abc import ABC, abstractmethod
from typing import Any


class RenderInterface(ABC):
    """Abstract interface for rendering backends."""

    @abstractmethod
    def get_vertex_code(self) -> str:
        """Get vertex shader code."""
        ...

    @abstractmethod
    def setup_uniforms(self, params: dict[str, Any]) -> dict[str, Any]:
        """Transform uniform values to backend-specific format."""
        ...

    @abstractmethod
    def get_render_requirements(self) -> dict[str, Any]:
        """Get renderer requirements (version, profile, etc.)."""
        ...
