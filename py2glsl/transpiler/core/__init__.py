"""Core transpiler functionality.

This module defines the core interfaces and base classes for the transpiler
that are independent of specific target languages.
"""

# Re-export core interfaces
from py2glsl.transpiler.core.interfaces import (
    LanguageAdapter as LanguageAdapter,
    LanguageConfig as LanguageConfig,
    RenderInterface as RenderInterface,
    SymbolMapper as SymbolMapper,
    TargetLanguage as TargetLanguage,
    TargetLanguageType as TargetLanguageType,
    TypeMapping as TypeMapping,
)

# Re-export core processing classes
from py2glsl.transpiler.core.ast_processor import (
    DependencyResolver as DependencyResolver,
    SymbolTable as SymbolTable,
)
