"""Base GLSL type definitions."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional

from loguru import logger


class GLSLError(Exception):
    """Base error for GLSL type system."""


class GLSLTypeError(GLSLError):
    """Error related to type compatibility."""


class GLSLOperationError(GLSLError):
    """Error in GLSL operations."""


class GLSLSwizzleError(GLSLError):
    """Error in vector swizzling."""


class TypeKind(Enum):
    """GLSL type kinds with validation rules."""

    VOID = auto()
    BOOL = auto()
    INT = auto()
    FLOAT = auto()
    VEC2 = auto()
    VEC3 = auto()
    VEC4 = auto()
    IVEC2 = auto()
    IVEC3 = auto()
    IVEC4 = auto()
    BVEC2 = auto()
    BVEC3 = auto()
    BVEC4 = auto()
    MAT2 = auto()
    MAT3 = auto()
    MAT4 = auto()

    @property
    def is_numeric(self) -> bool:
        """Check if type is numeric."""
        return self in {
            TypeKind.INT,
            TypeKind.FLOAT,
            TypeKind.VEC2,
            TypeKind.VEC3,
            TypeKind.VEC4,
            TypeKind.IVEC2,
            TypeKind.IVEC3,
            TypeKind.IVEC4,
            TypeKind.MAT2,
            TypeKind.MAT3,
            TypeKind.MAT4,
        }

    @property
    def is_vector(self) -> bool:
        """Check if type is a vector type."""
        return self in {
            TypeKind.VEC2,
            TypeKind.VEC3,
            TypeKind.VEC4,
            TypeKind.IVEC2,
            TypeKind.IVEC3,
            TypeKind.IVEC4,
            TypeKind.BVEC2,
            TypeKind.BVEC3,
            TypeKind.BVEC4,
        }

    @property
    def is_matrix(self) -> bool:
        """Check if type is a matrix type."""
        return self in {TypeKind.MAT2, TypeKind.MAT3, TypeKind.MAT4}

    @property
    def vector_size(self) -> Optional[int]:
        """Get vector size if applicable."""
        return {
            TypeKind.VEC2: 2,
            TypeKind.VEC3: 3,
            TypeKind.VEC4: 4,
            TypeKind.IVEC2: 2,
            TypeKind.IVEC3: 3,
            TypeKind.IVEC4: 4,
            TypeKind.BVEC2: 2,
            TypeKind.BVEC3: 3,
            TypeKind.BVEC4: 4,
        }.get(self)

    @property
    def matrix_size(self) -> Optional[int]:
        """Get matrix size if applicable."""
        return {
            TypeKind.MAT2: 2,
            TypeKind.MAT3: 3,
            TypeKind.MAT4: 4,
        }.get(self)


@dataclass(frozen=True)
class GLSLType:
    """GLSL type with core properties."""

    kind: TypeKind
    is_uniform: bool = False
    is_const: bool = False
    is_attribute: bool = False
    array_size: Optional[int] = None

    # Cache for singleton instances
    _instances = {}

    def __new__(cls, kind, **kwargs):
        """Implement singleton pattern."""
        key = (kind, *sorted(kwargs.items()))
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

    def __post_init__(self) -> None:
        """Validate type configuration."""
        logger.debug(f"Validating GLSLType: {self}")

        if self.array_size is not None:
            self._validate_array_size(self.array_size)

        if self.is_uniform and self.is_attribute:
            msg = "Type cannot be both uniform and attribute"
            logger.error(msg)
            raise GLSLTypeError(msg)

        if self.kind == TypeKind.VOID:
            if self.is_uniform or self.is_attribute or self.array_size is not None:
                msg = "Void type cannot have storage qualifiers or be an array"
                logger.error(msg)
                raise GLSLTypeError(msg)

    def _validate_array_size(self, size: Any) -> None:
        """Validate array size."""
        if not isinstance(size, int) or isinstance(size, bool):
            msg = f"Array size must be an integer, got {type(size)}"
            logger.error(msg)
            raise GLSLTypeError(msg)
        if size <= 0:
            msg = f"Array size must be positive, got {size}"
            logger.error(msg)
            raise GLSLTypeError(msg)

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
            TypeKind.IVEC2: "ivec2",
            TypeKind.IVEC3: "ivec3",
            TypeKind.IVEC4: "ivec4",
            TypeKind.BVEC2: "bvec2",
            TypeKind.BVEC3: "bvec3",
            TypeKind.BVEC4: "bvec4",
            TypeKind.MAT2: "mat2",
            TypeKind.MAT3: "mat3",
            TypeKind.MAT4: "mat4",
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

    @property
    def is_numeric(self) -> bool:
        """Check if type is numeric."""
        return self.kind.is_numeric

    @property
    def is_vector(self) -> bool:
        """Check if type is vector."""
        return self.kind.is_vector

    @property
    def is_matrix(self) -> bool:
        """Check if type is matrix."""
        return self.kind.is_matrix

    @property
    def is_bool_vector(self) -> bool:
        """Check if type is a boolean vector."""
        return self.kind in {TypeKind.BVEC2, TypeKind.BVEC3, TypeKind.BVEC4}

    @property
    def is_int_vector(self) -> bool:
        """Check if type is an integer vector."""
        return self.kind in {TypeKind.IVEC2, TypeKind.IVEC3, TypeKind.IVEC4}

    def vector_size(self) -> Optional[int]:
        """Get vector size if vector type."""
        return self.kind.vector_size

    def matrix_size(self) -> Optional[int]:
        """Get matrix size if matrix type."""
        return self.kind.matrix_size

    def validate_swizzle(self, components: str) -> Optional["GLSLType"]:
        """Validate swizzle operation and return resulting type."""
        from .singleton_types import (  # Import here to avoid circular dependency
            FLOAT,
            VEC2,
            VEC3,
            VEC4,
        )

        logger.debug(f"Validating swizzle: {self}.{components}")

        if not self.is_vector:
            msg = f"Cannot swizzle non-vector type {self.name}"
            logger.error(msg)
            raise GLSLSwizzleError(msg)

        if not components:
            msg = "Empty swizzle mask"
            logger.error(msg)
            raise GLSLSwizzleError(msg)

        # Split valid components into sets
        position_components = {"x", "y", "z", "w"}
        color_components = {"r", "g", "b", "a"}
        texture_components = {"s", "t", "p", "q"}

        component_set = set(components)
        logger.debug(f"Component set: {component_set}")

        # Check if components are from a single valid set
        if not (
            component_set.issubset(position_components)
            or component_set.issubset(color_components)
            or component_set.issubset(texture_components)
        ):
            msg = f"Invalid swizzle components: {components}"
            logger.error(msg)
            raise GLSLSwizzleError(msg)

        size = len(components)
        if size > 4:
            msg = "Swizzle mask too long"
            logger.error(msg)
            raise GLSLSwizzleError(msg)

        # Check if components are valid for this vector's size
        max_component_idx = max(
            (
                "xyzw".index(c)
                if c in "xyzw"
                else "rgba".index(c) if c in "rgba" else "stpq".index(c)
            )
            for c in components
        )
        if max_component_idx >= self.vector_size():
            msg = f"Component index {max_component_idx} out of range for {self.name}"
            logger.error(msg)
            raise GLSLSwizzleError(msg)

        return {1: FLOAT, 2: VEC2, 3: VEC3, 4: VEC4}[size]
