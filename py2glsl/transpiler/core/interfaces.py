"""Core interfaces for the transpiler system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Protocol

from py2glsl.transpiler.models import CollectedInfo


class TargetLanguageType(Enum):
    """Supported target language types."""

    GLSL = auto()
    SHADERTOY = auto()
    HLSL = auto()
    WGSL = auto()


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


class SymbolMapper(Protocol):
    """Maps Python symbols to target language symbols."""

    def map_type(self, python_type: str) -> str:
        """Map a Python type to target language type."""
        ...

    def map_function(self, python_function: str) -> str:
        """Map a Python function to target language function."""
        ...

    def map_operator(self, python_operator: str) -> str:
        """Map a Python operator to target language operator."""
        ...


class TargetLanguage(ABC):
    """Abstract interface for target shader languages."""

    @abstractmethod
    def get_config(self) -> LanguageConfig:
        """Get language configuration."""
        ...

    @abstractmethod
    def get_symbol_mapper(self) -> SymbolMapper:
        """Get symbol mapper for this language."""
        ...

    @abstractmethod
    def generate_code(
        self, collected: CollectedInfo, main_func: str
    ) -> tuple[str, set[str]]:
        """Generate code. Returns (code, uniforms)."""
        ...


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


class LanguageAdapter(ABC):
    """Adapter between target language and rendering backend."""

    def __init__(self, language: TargetLanguage, renderer: RenderInterface):
        self.language = language
        self.renderer = renderer

    @abstractmethod
    def adapt(self, code: str, uniforms: set[str]) -> tuple[str, dict[str, Any]]:
        """Adapt generated code to rendering backend."""
        ...
