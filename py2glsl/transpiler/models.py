"""
Data models and structures for the GLSL shader transpiler.

This module contains the dataclass definitions used throughout the transpiler
to represent functions, structs, fields, and collected information.
"""

import ast
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class StructField:
    """Field definition in a GLSL struct.

    Attributes:
        name: Field name
        type_name: GLSL type of the field
        default_value: Optional default value as a string
    """

    name: str
    type_name: str
    default_value: Optional[str] = None


@dataclass
class StructDefinition:
    """Representation of a GLSL struct definition.

    Attributes:
        name: Name of the struct
        fields: List of field definitions
    """

    name: str
    fields: List[StructField]


@dataclass
class FunctionInfo:
    """Information about a function to be transpiled to GLSL.

    Attributes:
        name: Function name
        return_type: Return type or None if not specified
        param_types: List of parameter types
        node: AST node for the function
    """

    name: str
    return_type: Optional[str]
    param_types: List[Optional[str]]
    node: ast.FunctionDef


@dataclass
class CollectedInfo:
    """Information collected from Python code to be transpiled.

    Attributes:
        functions: Dictionary mapping function names to function information
        structs: Dictionary mapping struct names to struct definitions
        globals: Dictionary mapping global variable names to (type, value)
    """

    functions: Dict[str, FunctionInfo] = field(default_factory=dict)
    structs: Dict[str, StructDefinition] = field(default_factory=dict)
    globals: Dict[str, Tuple[str, str]] = field(default_factory=dict)
