"""GLSL type system and type operations."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Set, Tuple, Union

import numpy as np


class TypeKind(Enum):
    """Kind of GLSL type."""

    VOID = auto()
    BOOL = auto()
    INT = auto()
    FLOAT = auto()
    VEC2 = auto()
    VEC3 = auto()
    VEC4 = auto()
    MAT2 = auto()
    MAT3 = auto()
    MAT4 = auto()
    SAMPLER2D = auto()


@dataclass(frozen=True)
class GLSLType:
    """Represents a GLSL type with qualifiers and metadata."""

    kind: TypeKind
    is_uniform: bool = False
    is_const: bool = False
    is_attribute: bool = False
    array_size: Optional[int] = None

    @property
    def name(self) -> str:
        """Get GLSL type name."""
        base = {
            TypeKind.VOID: "void",
            TypeKind.BOOL: "bool",
            TypeKind.INT: "int",
            TypeKind.FLOAT: "float",
            TypeKind.VEC2: "vec2",
            TypeKind.VEC3: "vec3",
            TypeKind.VEC4: "vec4",
            TypeKind.MAT2: "mat2",
            TypeKind.MAT3: "mat3",
            TypeKind.MAT4: "mat4",
            TypeKind.SAMPLER2D: "sampler2D",
        }[self.kind]

        if self.array_size is not None:
            return f"{base}[{self.array_size}]"
        return base

    def __str__(self) -> str:
        """Convert to GLSL declaration."""
        parts = []
        if self.is_uniform:
            parts.append("uniform")
        if self.is_const:
            parts.append("const")
        if self.is_attribute:
            parts.append("attribute")
        parts.append(self.name)
        return " ".join(parts)

    def is_numeric(self) -> bool:
        """Check if type is numeric (int/float/vector)."""
        return self.kind in {
            TypeKind.INT,
            TypeKind.FLOAT,
            TypeKind.VEC2,
            TypeKind.VEC3,
            TypeKind.VEC4,
        }

    def is_scalar(self) -> bool:
        """Check if type is scalar."""
        return self.kind in {TypeKind.INT, TypeKind.FLOAT, TypeKind.BOOL}

    def is_vector(self) -> bool:
        """Check if type is vector."""
        return self.kind in {TypeKind.VEC2, TypeKind.VEC3, TypeKind.VEC4}

    def is_matrix(self) -> bool:
        """Check if type is matrix."""
        return self.kind in {TypeKind.MAT2, TypeKind.MAT3, TypeKind.MAT4}

    def vector_size(self) -> int:
        """Get vector size if vector type."""
        if not self.is_vector():
            raise TypeError("Not a vector type")
        return {
            TypeKind.VEC2: 2,
            TypeKind.VEC3: 3,
            TypeKind.VEC4: 4,
        }[self.kind]

    def component_type(self) -> "GLSLType":
        """Get component type for vectors/matrices."""
        if self.is_vector():
            return FLOAT
        if self.is_matrix():
            return FLOAT
        raise TypeError("Type has no components")

    def can_convert_to(self, target: "GLSLType") -> bool:
        """Check if this type can be converted to target."""
        # Same type always works
        if self == target:
            return True

        # Numeric conversions
        if self.is_numeric() and target.is_numeric():
            # Scalar to vector promotion
            if self.is_scalar() and target.is_vector():
                return True
            # Vector size must match
            if self.is_vector() and target.is_vector():
                return self.vector_size() == target.vector_size()
            # Int to float conversion
            if self.kind == TypeKind.INT and target.kind == TypeKind.FLOAT:
                return True

        return False

    def common_type(self, other: "GLSLType") -> Optional["GLSLType"]:
        """Find common type for operation between two types."""
        if self == other:
            return self

        if self.is_numeric() and other.is_numeric():
            # Vector takes precedence
            if self.is_vector() and other.is_vector():
                if self.vector_size() == other.vector_size():
                    return self
            elif self.is_vector():
                return self
            elif other.is_vector():
                return other

            # Float takes precedence over int
            if TypeKind.FLOAT in (self.kind, other.kind):
                return FLOAT
            return INT

        return None


# Singleton instances for built-in types
VOID = GLSLType(TypeKind.VOID)
BOOL = GLSLType(TypeKind.BOOL)
INT = GLSLType(TypeKind.INT)
FLOAT = GLSLType(TypeKind.FLOAT)
VEC2 = GLSLType(TypeKind.VEC2)
VEC3 = GLSLType(TypeKind.VEC3)
VEC4 = GLSLType(TypeKind.VEC4)
MAT2 = GLSLType(TypeKind.MAT2)
MAT3 = GLSLType(TypeKind.MAT3)
MAT4 = GLSLType(TypeKind.MAT4)
SAMPLER2D = GLSLType(TypeKind.SAMPLER2D)


class TypeRegistry:
    """Registry of GLSL types and operations."""

    def __init__(self):
        """Initialize registry."""
        self.types: Set[GLSLType] = {
            VOID,
            BOOL,
            INT,
            FLOAT,
            VEC2,
            VEC3,
            VEC4,
            MAT2,
            MAT3,
            MAT4,
            SAMPLER2D,
        }

    def get_type(self, name: str) -> GLSLType:
        """Get type by name."""
        for t in self.types:
            if t.name == name:
                return t
        raise KeyError(f"Unknown type: {name}")

    def validate_operation(
        self, op: str, left: GLSLType, right: GLSLType
    ) -> Optional[GLSLType]:
        """Validate operation between types and return result type."""
        if not (left in self.types and right in self.types):
            return None

        # Get common numeric type
        result = left.common_type(right)
        if result is None:
            return None

        # Validate operation
        if op in ("+", "-", "*", "/"):
            return result
        elif op in ("==", "!=", "<", "<=", ">", ">="):
            return BOOL

        return None


# NumPy array type aliases
Vec2 = np.ndarray  # shape (2,)
Vec3 = np.ndarray  # shape (3,)
Vec4 = np.ndarray  # shape (4,)


def vec2(x: float, y: float) -> Vec2:
    """Create 2D vector."""
    return np.array([x, y], dtype=np.float32)


def vec3(x: float, y: float, z: float) -> Vec3:
    """Create 3D vector."""
    return np.array([x, y, z], dtype=np.float32)


def vec4(x: float, y: float, z: float, w: float) -> Vec4:
    """Create 4D vector."""
    return np.array([x, y, z, w], dtype=np.float32)
