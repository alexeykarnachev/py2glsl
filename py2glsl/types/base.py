"""GLSL Type System - Core Definitions (Combined)"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Union

# ======================================================================================
#                                      CORE TYPES
# ======================================================================================


class TypeKind(Enum):
    """Enumeration of fundamental GLSL type categories"""

    VOID = auto()
    FLOAT = auto()
    INT = auto()
    BOOL = auto()
    VECTOR = auto()
    MATRIX = auto()
    FUNCTION = auto()
    STRUCT = auto()


class GLSLType:
    """Base class for all GLSL types with common properties"""

    def __init__(
        self,
        kind: TypeKind,
        size: Optional[int] = None,
        component_type: Optional["GLSLType"] = None,
        members: Optional[Dict[str, "GLSLType"]] = None,
    ):
        self.kind = kind
        self.size = size  # For vectors/matrices/arrays
        self.component_type = component_type  # For vectors/arrays
        self.members = members or {}  # For structs

    @property
    def is_numeric(self) -> bool:
        return self.kind in {TypeKind.FLOAT, TypeKind.INT}

    @property
    def is_vector(self) -> bool:
        return self.kind == TypeKind.VECTOR

    # ... other properties from original base.py ...

    def __eq__(self, other):
        return (
            isinstance(other, GLSLType)
            and self.kind == other.kind
            and self.size == other.size
            and self.component_type == other.component_type
        )


# ======================================================================================
#                                    FUNCTION TYPES
# ======================================================================================


@dataclass
class FunctionSignature:
    """Represents a function's type signature"""

    parameters: List[GLSLType]
    return_type: GLSLType
    is_builtin: bool = False

    def __str__(self):
        params = ", ".join(str(p) for p in self.parameters)
        return f"({params}) -> {self.return_type}"


# ======================================================================================
#                                 TYPE CONTEXT & RULES
# ======================================================================================


@dataclass
class TypeContext:
    """Tracks typing information within a specific scope"""

    variables: Dict[str, GLSLType]
    functions: Dict[str, List[FunctionSignature]]
    structs: Dict[str, GLSLType]
    current_return: Optional[GLSLType] = None

    def child_context(self):
        """Create a new nested context"""
        return TypeContext(
            variables=self.variables.copy(),
            functions=self.functions.copy(),
            structs=self.structs.copy(),
            current_return=self.current_return,
        )


@dataclass
class TypePromotionRule:
    """Defines allowed type conversions"""

    source: GLSLType
    target: GLSLType
    implicit: bool  # Whether conversion happens automatically


# ======================================================================================
#                                 PREDEFINED TYPE INSTANCES
# ======================================================================================

VOID = GLSLType(TypeKind.VOID)
FLOAT = GLSLType(TypeKind.FLOAT)
INT = GLSLType(TypeKind.INT)
BOOL = GLSLType(TypeKind.BOOL)
VEC2 = GLSLType(TypeKind.VECTOR, size=2, component_type=FLOAT)
VEC3 = GLSLType(TypeKind.VECTOR, size=3, component_type=FLOAT)
VEC4 = GLSLType(TypeKind.VECTOR, size=4, component_type=FLOAT)
MAT4 = GLSLType(TypeKind.MATRIX, size=4)

__all__ = [
    "TypeKind",
    "GLSLType",
    "FunctionSignature",
    "TypeContext",
    "TypePromotionRule",
    "VOID",
    "FLOAT",
    "INT",
    "BOOL",
    "VEC2",
    "VEC3",
    "VEC4",
    "MAT4",
]
