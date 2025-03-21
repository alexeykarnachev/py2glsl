"""Core interfaces for the transpiler system.

This module defines the core interfaces used throughout the transpiling process,
providing language-agnostic abstractions for the various components.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Protocol

from py2glsl.transpiler.models import CollectedInfo


class TargetLanguageType(Enum):
    """Supported target language types."""

    GLSL = auto()
    HLSL = auto()
    WGSL = auto()
    # Add more languages as needed


@dataclass
class TypeMapping:
    """Mapping between Python and target language types."""

    python_type: str
    target_type: str
    default_value: str


@dataclass
class LanguageConfig:
    """Configuration for a target language."""

    name: str
    file_extension: str
    version: str
    type_mappings: dict[str, TypeMapping]
    # Add more language-specific configuration as needed


class SymbolMapper(Protocol):
    """Interface for mapping Python symbols to target language symbols."""

    def map_type(self, python_type: str) -> str:
        """Map a Python type to a target language type.

        Args:
            python_type: Python type name

        Returns:
            Equivalent type in the target language
        """
        ...

    def map_function(self, python_function: str) -> str:
        """Map a Python function name to a target language function.

        Args:
            python_function: Python function name

        Returns:
            Equivalent function in the target language
        """
        ...

    def map_operator(self, python_operator: str) -> str:
        """Map a Python operator to a target language operator.

        Args:
            python_operator: Python operator

        Returns:
            Equivalent operator in the target language
        """
        ...


class TargetLanguage(ABC):
    """Abstract interface for target shader languages."""

    @abstractmethod
    def get_config(self) -> LanguageConfig:
        """Get the language configuration.

        Returns:
            Language configuration
        """
        pass

    @abstractmethod
    def get_symbol_mapper(self) -> SymbolMapper:
        """Get the symbol mapper for this language.

        Returns:
            Symbol mapper implementation
        """
        pass

    @abstractmethod
    def generate_code(
        self, collected: CollectedInfo, main_func: str
    ) -> tuple[str, set[str]]:
        """Generate code in the target language.

        Args:
            collected: Information about functions, structs, and globals
            main_func: Name of the main function to use as entry point

        Returns:
            Tuple of (generated code, set of used uniform variables)
        """
        pass


class RenderInterface(ABC):
    """Abstract interface for rendering backends."""

    @abstractmethod
    def get_vertex_code(self) -> str:
        """Get the vertex code for this renderer.

        Returns:
            The vertex code as a string
        """
        pass

    @abstractmethod
    def setup_uniforms(self, params: dict[str, Any]) -> dict[str, Any]:
        """Transform standard uniform values to backend-specific values.

        Args:
            params: Dictionary of standard uniform parameters

        Returns:
            Dictionary of backend-specific uniform parameters
        """
        pass

    @abstractmethod
    def get_render_requirements(self) -> dict[str, Any]:
        """Get the requirements for this renderer.

        Returns:
            Dictionary of renderer requirements (version, profile, etc.)
        """
        pass


class LanguageAdapter(ABC):
    """Adapter between a target language and a rendering backend."""

    def __init__(self, language: TargetLanguage, renderer: RenderInterface):
        """Initialize with a language and renderer.

        Args:
            language: Target language implementation
            renderer: Renderer implementation
        """
        self.language = language
        self.renderer = renderer

    @abstractmethod
    def adapt(self, code: str, uniforms: set[str]) -> tuple[str, dict[str, Any]]:
        """Adapt the generated code to the rendering backend.

        Args:
            code: Generated code in the target language
            uniforms: Set of uniform names used in the code

        Returns:
            Tuple of (adapted code, uniform mapping)
        """
        pass