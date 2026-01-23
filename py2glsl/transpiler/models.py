"""Data models and exceptions for the shader transpiler."""

import ast
from dataclasses import dataclass, field


class TranspilerError(Exception):
    """Exception raised during shader code transpilation."""

    pass


@dataclass
class StructField:
    """Field definition in a struct."""

    name: str
    type_name: str
    default_value: str | None = None


@dataclass
class StructDefinition:
    """Struct definition."""

    name: str
    fields: list[StructField]


@dataclass
class FunctionInfo:
    """Function information for transpilation."""

    name: str
    return_type: str | None
    param_types: list[str | None]
    node: ast.FunctionDef
    # Default values for parameters (as GLSL expression strings)
    # Indexed from the END of parameters (like Python's defaults)
    param_defaults: list[str] = field(default_factory=list)


@dataclass
class CollectedInfo:
    """Collected information from Python code."""

    functions: dict[str, FunctionInfo] = field(default_factory=dict)
    structs: dict[str, StructDefinition] = field(default_factory=dict)
    globals: dict[str, tuple[str, str]] = field(default_factory=dict)
    # Globals that are reassigned inside functions (not true constants)
    mutable_globals: set[str] = field(default_factory=set)
