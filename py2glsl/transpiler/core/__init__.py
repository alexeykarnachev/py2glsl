"""Core transpiler functionality.

This module defines the core interfaces and base classes for the transpiler
that are independent of specific target languages.
"""

# Re-export core interfaces and processing classes
from py2glsl.transpiler.core.ast_processor import DependencyResolver, SymbolTable
from py2glsl.transpiler.core.interfaces import (
    LanguageAdapter,
    LanguageConfig,
    RenderInterface,
    SymbolMapper,
    TargetLanguage,
    TargetLanguageType,
    TypeMapping,
)

# Define what's available for import from this package
__all__ = [
    "DependencyResolver",
    "SymbolTable",
    "LanguageAdapter",
    "LanguageConfig",
    "RenderInterface",
    "SymbolMapper",
    "TargetLanguage",
    "TargetLanguageType",
    "TypeMapping",
]
