"""Core interfaces and utilities for the transpiler system."""

import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Protocol, TypeVar

from py2glsl.transpiler.models import CollectedInfo

T = TypeVar("T")


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


class DependencyResolver:
    """Resolves function dependencies for correct ordering."""

    def __init__(self, collected: CollectedInfo) -> None:
        self.collected = collected
        self.dependencies: dict[str, set[str]] = {}
        self._build_dependency_graph()

    def _build_dependency_graph(self) -> None:
        """Build dependency graph for all functions."""
        for func_name, func_info in self.collected.functions.items():
            self.dependencies[func_name] = self._find_function_calls(func_info.node)

    def _find_function_calls(self, ast_node: ast.AST) -> set[str]:
        """Find all user-defined function calls within an AST node."""
        called_functions: set[str] = set()
        collected_functions = self.collected.functions

        class CallVisitor(ast.NodeVisitor):
            def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
                func = node.func
                if isinstance(func, ast.Name) and func.id in collected_functions:
                    called_functions.add(func.id)
                self.generic_visit(node)

        CallVisitor().visit(ast_node)
        return called_functions

    def get_ordered_functions(self, main_func: str) -> list[str]:
        """Get functions ordered by dependencies (dependencies first)."""
        ordered: list[str] = []
        visited: set[str] = set()

        def visit(func_name: str) -> None:
            if func_name in visited:
                return
            for dep in self.dependencies.get(func_name, set()):
                visit(dep)
            ordered.append(func_name)
            visited.add(func_name)

        visit(main_func)
        for func_name in self.collected.functions:
            visit(func_name)

        return ordered


class SymbolTable:
    """Symbol table for tracking variable types during code generation."""

    def __init__(self) -> None:
        self.scopes: list[dict[str, Any]] = [{}]

    def enter_scope(self) -> None:
        """Enter a new scope."""
        self.scopes.append({})

    def exit_scope(self) -> None:
        """Exit current scope."""
        if len(self.scopes) > 1:
            self.scopes.pop()

    def add_symbol(self, name: str, value: T) -> None:
        """Add symbol to current scope."""
        self.scopes[-1][name] = value

    def lookup(self, name: str) -> Any | None:
        """Look up symbol from inner to outer scope."""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def current_scope(self) -> dict[str, Any]:
        """Get current scope."""
        return self.scopes[-1]
