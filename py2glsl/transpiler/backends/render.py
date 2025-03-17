"""Rendering backend interfaces for py2glsl.

This module provides the interfaces that connect shader backends with rendering.
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol


class UniformProvider(Protocol):
    """Interface for providing uniform values for a backend."""

    def get_uniform_type_mapping(self) -> dict[str, str]:
        """
        Get the mapping of uniform names to their GLSL types.

        Returns:
            A dictionary mapping uniform names to their GLSL types
        """
        ...

    def get_uniform_name_mapping(self) -> dict[str, str]:
        """
        Get the mapping of standard uniform names to backend-specific names.

        Returns:
            A dictionary mapping standard uniform names to backend-specific names
        """
        ...


class RenderBackend(ABC):
    """Interface for backend-specific rendering operations."""

    @abstractmethod
    def get_vertex_shader(self) -> str:
        """
        Get the vertex shader code for this backend.

        Returns:
            The vertex shader code as a string
        """
        pass

    @abstractmethod
    def get_opengl_version(self) -> tuple[int, int]:
        """
        Get the required OpenGL version for this backend.

        Returns:
            A tuple of (major, minor) version numbers
        """
        pass

    @abstractmethod
    def get_opengl_profile(self) -> str:
        """
        Get the required OpenGL profile for this backend.

        Returns:
            The OpenGL profile as a string
        """
        pass

    @abstractmethod
    def setup_uniforms(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Transform standard uniform values to backend-specific values.

        Args:
            params: Dictionary of standard uniform parameters

        Returns:
            Dictionary of backend-specific uniform parameters
        """
        pass


class BaseRenderBackend(RenderBackend):
    """Base implementation of RenderBackend with common functionality."""

    def __init__(self, uniform_provider: UniformProvider):
        """Initialize with a uniform provider."""
        self.uniform_provider = uniform_provider

    def setup_uniforms(self, params: dict[str, Any]) -> dict[str, Any]:
        """Transform standard uniforms to backend-specific uniforms."""
        mapping = self.uniform_provider.get_uniform_name_mapping()
        result = {}

        # Copy all params that don't need mapping
        for name, value in params.items():
            if name not in mapping:
                result[name] = value

        # Map params that need transformation
        for std_name, backend_name in mapping.items():
            if std_name in params:
                result[backend_name] = params[std_name]

        return result
