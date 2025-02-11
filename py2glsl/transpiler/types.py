"""GLSL type system with validation and operations."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Literal, Optional, TypeAlias

import numpy as np

# Runtime type aliases
Vec2: TypeAlias = np.ndarray  # shape (2,)
Vec3: TypeAlias = np.ndarray  # shape (3,)
Vec4: TypeAlias = np.ndarray  # shape (4,)
Mat2: TypeAlias = np.ndarray  # shape (2, 2)
Mat3: TypeAlias = np.ndarray  # shape (3, 3)
Mat4: TypeAlias = np.ndarray  # shape (4, 4)
IVec2: TypeAlias = np.ndarray  # shape (2,) dtype=int32
IVec3: TypeAlias = np.ndarray  # shape (3,) dtype=int32
IVec4: TypeAlias = np.ndarray  # shape (4,) dtype=int32
BVec2: TypeAlias = np.ndarray  # shape (2,) dtype=bool
BVec3: TypeAlias = np.ndarray  # shape (3,) dtype=bool
BVec4: TypeAlias = np.ndarray  # shape (4,) dtype=bool

# Operation types
GLSLOperator = Literal[
    "+",
    "-",
    "*",
    "/",
    "%",  # Arithmetic
    "&&",
    "||",  # Logical
    "==",
    "!=",
    "<",
    ">",
    "<=",
    ">=",  # Comparison
]


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
    def vector_size(self) -> int | None:
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
    def matrix_size(self) -> int | None:
        """Get matrix size if applicable."""
        return {
            TypeKind.MAT2: 2,
            TypeKind.MAT3: 3,
            TypeKind.MAT4: 4,
        }.get(self)


@dataclass(frozen=True)
class GLSLType:
    """GLSL type with validation and operations."""

    kind: TypeKind
    is_uniform: bool = False
    is_const: bool = False
    is_attribute: bool = False
    array_size: int | None = None

    def __post_init__(self) -> None:
        """Validate type configuration."""
        if self.array_size is not None:
            self._validate_array_size(self.array_size)

        if self.is_uniform and self.is_attribute:
            raise GLSLTypeError("Type cannot be both uniform and attribute")

        if self.kind == TypeKind.VOID:
            if self.is_uniform or self.is_attribute:
                raise GLSLTypeError("Void type cannot have storage qualifiers")

    def _validate_array_size(self, size: Any) -> None:
        """Validate array size."""
        if isinstance(size, GLSLType):
            raise GLSLTypeError("Arrays of arrays are not allowed in GLSL")
        if isinstance(size, (float, bool, str)):
            raise GLSLTypeError("Array size must be an integer")
        if hasattr(size, "__array_interface__"):
            raise GLSLTypeError("Array size cannot be a numpy array")
        if hasattr(size, "array_size"):
            raise GLSLTypeError("Arrays of arrays are not allowed in GLSL")
        if not isinstance(size, int):
            raise GLSLTypeError("Array size must be an integer")
        if size <= 0:
            raise GLSLTypeError("Array size must be positive")

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

    def vector_size(self) -> int | None:
        """Get vector size if vector type."""
        return self.kind.vector_size

    def matrix_size(self) -> int | None:
        """Get matrix size if matrix type."""
        return self.kind.matrix_size

    def validate_swizzle(self, components: str) -> Optional["GLSLType"]:
        """Validate swizzle operation and return resulting type."""
        if not self.is_vector:
            raise GLSLSwizzleError(f"Cannot swizzle non-vector type {self.name}")

        if not components:
            raise GLSLSwizzleError("Empty swizzle mask")

        # Split valid components into sets
        position_components = {"x", "y", "z", "w"}
        color_components = {"r", "g", "b", "a"}
        texture_components = {"s", "t", "p", "q"}

        component_set = set(components)

        # Check if components are from a single valid set
        if not (
            component_set.issubset(position_components)
            or component_set.issubset(color_components)
            or component_set.issubset(texture_components)
        ):
            raise GLSLSwizzleError(f"Invalid swizzle components: {components}")

        size = len(components)
        if size > 4:
            raise GLSLSwizzleError("Swizzle mask too long")

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
            raise GLSLSwizzleError(
                f"Component index {max_component_idx} out of range for {self.name}"
            )

        return {1: FLOAT, 2: VEC2, 3: VEC3, 4: VEC4}[size]

    def validate_operation(
        self, op: GLSLOperator, other: "GLSLType"
    ) -> Optional["GLSLType"]:
        """Validate operation between types and return result type."""
        # Handle void type
        if self.kind == TypeKind.VOID or other.kind == TypeKind.VOID:
            return None

        # Arrays support component-wise operations if sizes match
        if self.array_size is not None:
            if other.array_size is not None:
                if self.array_size != other.array_size:
                    return None
                if op in ("==", "!="):
                    return BOOL
                if op in ("+", "-", "*", "/", "%"):
                    return self
            elif other.kind in (TypeKind.FLOAT, TypeKind.INT):
                if op in ("*", "/"):  # Allow scalar operations
                    return self
            return None

        # Boolean operations
        if op in ("&&", "||"):
            if self.kind == TypeKind.BOOL and other.kind == TypeKind.BOOL:
                return BOOL
            if self.is_bool_vector and other.is_bool_vector and self == other:
                return self
            return None

        # Boolean vector operations
        if self.is_bool_vector:
            if op in ("==", "!="):
                return BOOL if self == other else None
            if op in ("&&", "||"):
                return self if self == other else None
            return None

        # Matrix operations
        if self.is_matrix or other.is_matrix:
            if op in ("%", "&&", "||", "<", ">", "<=", ">=", "+", "-"):  # Added +, -
                return None
            if op in ("==", "!="):
                return BOOL if self == other else None
            if op == "*":  # Only allow multiplication
                if self.is_matrix and other.is_matrix:
                    return self if self.matrix_size() == other.matrix_size() else None
                if self.is_matrix and other.kind in (TypeKind.FLOAT, TypeKind.INT):
                    return self
                if other.is_matrix and self.kind in (TypeKind.FLOAT, TypeKind.INT):
                    return other
                if self.is_matrix and other.is_vector:
                    return other if self.matrix_size() == other.vector_size() else None
                if other.is_matrix and self.is_vector:
                    return self if other.matrix_size() == self.vector_size() else None
            return None

        # Vector operations
        if self.is_vector or other.is_vector:
            # Boolean vectors only support logical and comparison operations
            if self.is_bool_vector or other.is_bool_vector:
                if op in ("&&", "||"):
                    return self if self == other else None
                if op in ("==", "!="):
                    return BOOL if self == other else None
                return None

            # Vector-scalar operations
            if self.is_vector and other.kind in (TypeKind.FLOAT, TypeKind.INT):
                if op in ("+", "-", "*", "/"):
                    return self
            if other.is_vector and self.kind in (TypeKind.FLOAT, TypeKind.INT):
                if op in ("+", "-", "*", "/"):
                    return other

            # Vector-vector operations
            if self.is_vector and other.is_vector:
                if op in ("==", "!="):
                    return BOOL if self.is_compatible_with(other) else None
                if op in ("<", ">", "<=", ">="):
                    return None if self.is_bool_vector else BOOL
                if self.is_bool_vector:
                    return None  # Boolean vectors don't support arithmetic
                return self if self.is_compatible_with(other) else None

        # Comparison operations
        if op in ("==", "!="):
            if self.is_bool_vector or other.is_bool_vector:
                return BOOL if self == other else None
            return BOOL if self.is_compatible_with(other) else None

        if op in ("<", ">", "<=", ">="):
            if self.is_bool_vector or other.is_bool_vector:
                return None
            if self.is_vector and other.is_vector:
                return BOOL if self.is_compatible_with(other) else None
            if self.is_numeric and other.is_numeric:
                return BOOL
            return None

        # Basic arithmetic operations
        if op in ("+", "-", "*", "%"):
            if not (self.is_numeric and other.is_numeric):
                return None
            return self.common_type(other)

        # Division always promotes to float
        if op == "/":
            if not (self.is_numeric and other.is_numeric):
                return None
            if self.kind == TypeKind.INT and other.kind == TypeKind.INT:
                return FLOAT
            return self.common_type(other)

        return None

    def is_compatible_with(self, other: "GLSLType") -> bool:
        """Check if types can be used together."""
        if self == other:
            return True

        if self.kind == TypeKind.VOID or other.kind == TypeKind.VOID:
            return False

        if self.array_size is not None or other.array_size is not None:
            return self.array_size == other.array_size and self.kind == other.kind

        # Vector types should only be compatible with same type
        if self.is_vector and other.is_vector:
            return (
                self.vector_size() == other.vector_size()
                and not (self.is_bool_vector or other.is_bool_vector)
                and not (
                    (self.is_int_vector and not other.is_int_vector)
                    or (other.is_int_vector and not self.is_int_vector)
                )
            )

        # Matrix-vector compatibility
        if self.is_matrix and other.is_vector:
            return self.matrix_size() == other.vector_size()
        if other.is_matrix and self.is_vector:
            return other.matrix_size() == self.vector_size()

        # Matrix compatibility
        if self.is_matrix or other.is_matrix:
            if self.is_matrix and other.is_matrix:
                return self.matrix_size() == other.matrix_size()
            return other.kind in (TypeKind.FLOAT, TypeKind.INT) or self.kind in (
                TypeKind.FLOAT,
                TypeKind.INT,
            )

        return self.is_numeric and other.is_numeric

    def can_convert_to(self, target: "GLSLType") -> bool:
        """Check if this type can be implicitly converted to target."""
        if self == target:
            return True
        if self.kind == TypeKind.INT and target.kind == TypeKind.FLOAT:
            return True
        return False

    def common_type(self, other: "GLSLType") -> Optional["GLSLType"]:
        """Find common type for operation between two types."""
        if self == other:
            return self

        if self.array_size is not None or other.array_size is not None:
            if self.array_size == other.array_size and self.kind == other.kind:
                return self
            return None

        # Vector-scalar operations
        if self.is_vector and other.kind in (TypeKind.FLOAT, TypeKind.INT):
            return self
        if other.is_vector and self.kind in (TypeKind.FLOAT, TypeKind.INT):
            return other

        # Matrix-scalar operations
        if self.is_matrix and other.kind in (TypeKind.FLOAT, TypeKind.INT):
            return self
        if other.is_matrix and self.kind in (TypeKind.FLOAT, TypeKind.INT):
            return other

        if self.kind == TypeKind.INT and other.kind == TypeKind.FLOAT:
            return FLOAT
        if self.kind == TypeKind.FLOAT and other.kind == TypeKind.INT:
            return FLOAT

        if (
            self.is_vector
            and other.is_vector
            and self.vector_size() == other.vector_size()
        ):
            if TypeKind.FLOAT in (self.kind, other.kind):
                return {2: VEC2, 3: VEC3, 4: VEC4}[self.vector_size()]
            if TypeKind.INT in (self.kind, other.kind):
                return {2: IVEC2, 3: IVEC3, 4: IVEC4}[self.vector_size()]

        return None


# Singleton instances for all GLSL types
VOID = GLSLType(TypeKind.VOID)
BOOL = GLSLType(TypeKind.BOOL)
INT = GLSLType(TypeKind.INT)
FLOAT = GLSLType(TypeKind.FLOAT)

# Vector types
VEC2 = GLSLType(TypeKind.VEC2)
VEC3 = GLSLType(TypeKind.VEC3)
VEC4 = GLSLType(TypeKind.VEC4)

# Integer vector types
IVEC2 = GLSLType(TypeKind.IVEC2)
IVEC3 = GLSLType(TypeKind.IVEC3)
IVEC4 = GLSLType(TypeKind.IVEC4)

# Boolean vector types
BVEC2 = GLSLType(TypeKind.BVEC2)
BVEC3 = GLSLType(TypeKind.BVEC3)
BVEC4 = GLSLType(TypeKind.BVEC4)

# Matrix types
MAT2 = GLSLType(TypeKind.MAT2)
MAT3 = GLSLType(TypeKind.MAT3)
MAT4 = GLSLType(TypeKind.MAT4)


def vec2(*args) -> Vec2:
    """Create a 2D float vector."""
    if len(args) == 0:
        raise ValueError("vec2 requires at least 1 argument")
    if len(args) == 1:
        val = float(args[0])
        return np.array([val, val], dtype=np.float32)
    if len(args) == 2:
        return np.array([float(args[0]), float(args[1])], dtype=np.float32)
    raise TypeError("vec2 requires 1 or 2 arguments")  # Changed to TypeError


def vec3(*args) -> Vec3:
    """Create a 3D float vector."""
    if len(args) == 0:
        raise ValueError("vec3 requires at least 1 argument")
    if len(args) == 1:
        val = float(args[0])
        return np.array([val, val, val], dtype=np.float32)
    if len(args) == 2 and isinstance(args[0], Vec2):
        return np.array([args[0][0], args[0][1], float(args[1])], dtype=np.float32)
    if len(args) == 3:
        return np.array(
            [float(args[0]), float(args[1]), float(args[2])], dtype=np.float32
        )
    raise TypeError(
        "vec3 requires 1 or 3 arguments, or vec2 and a scalar"
    )  # Changed to TypeError


def vec4(*args) -> Vec4:
    """Create a 4D float vector."""
    if len(args) == 0:
        raise ValueError("vec4 requires at least 1 argument")
    if len(args) == 1:
        val = float(args[0])
        return np.array([val, val, val, val], dtype=np.float32)
    if len(args) == 2 and isinstance(args[0], Vec3):
        if len(args) > 2:  # Added validation
            raise ValueError("Too many components for vec4")
        return np.array(
            [args[0][0], args[0][1], args[0][2], float(args[1])], dtype=np.float32
        )
    if len(args) == 3 and isinstance(args[0], Vec2):
        return np.array(
            [args[0][0], args[0][1], float(args[1]), float(args[2])], dtype=np.float32
        )
    if len(args) == 4:
        return np.array(
            [float(args[0]), float(args[1]), float(args[2]), float(args[3])],
            dtype=np.float32,
        )
    raise ValueError(
        "vec4 requires 1 or 4 arguments, or vec3 and a scalar, or vec2 and two scalars"
    )


def ivec2(*args) -> IVec2:
    """Create a 2D integer vector."""
    if len(args) == 0:
        raise ValueError("ivec2 requires at least 1 argument")
    if len(args) == 1:
        val = int(args[0])
        return np.array([val, val], dtype=np.int32)
    if len(args) == 2:
        return np.array([int(args[0]), int(args[1])], dtype=np.int32)
    raise TypeError("ivec2 requires 1 or 2 arguments")  # Changed to TypeError


def ivec3(*args) -> IVec3:
    """Create a 3D integer vector."""
    if len(args) == 0:
        raise ValueError("ivec3 requires at least 1 argument")
    if len(args) == 1:
        val = int(args[0])
        return np.array([val, val, val], dtype=np.int32)
    if len(args) == 2 and isinstance(args[0], IVec2):
        return np.array([args[0][0], args[0][1], int(args[1])], dtype=np.int32)
    if len(args) == 3:
        return np.array([int(args[0]), int(args[1]), int(args[2])], dtype=np.int32)
    raise TypeError(
        "ivec3 requires 1 or 3 arguments, or ivec2 and a scalar"
    )  # Changed to TypeError


def ivec4(*args) -> IVec4:
    """Create a 4D integer vector."""
    if len(args) == 0:
        raise ValueError("ivec4 requires at least 1 argument")
    if len(args) == 1:
        val = int(args[0])
        return np.array([val, val, val, val], dtype=np.int32)
    if len(args) == 2 and isinstance(args[0], IVec3):
        return np.array(
            [args[0][0], args[0][1], args[0][2], int(args[1])], dtype=np.int32
        )
    if len(args) == 3 and isinstance(args[0], IVec2):
        return np.array(
            [args[0][0], args[0][1], int(args[1]), int(args[2])], dtype=np.int32
        )
    if len(args) == 4:
        return np.array(
            [int(args[0]), int(args[1]), int(args[2]), int(args[3])], dtype=np.int32
        )
    raise TypeError(
        "ivec4 requires 1 or 4 arguments, or ivec3 and a scalar, or ivec2 and two scalars"
    )  # Changed to TypeError


def bvec2(*args) -> BVec2:
    """Create a 2D boolean vector."""
    if len(args) == 0:
        raise ValueError("bvec2 requires at least 1 argument")
    if len(args) == 1:
        val = bool(args[0])
        return np.array([val, val], dtype=bool)
    if len(args) == 2:
        return np.array([bool(args[0]), bool(args[1])], dtype=bool)
    raise TypeError("bvec2 requires 1 or 2 arguments")  # Changed to TypeError


def bvec3(*args) -> BVec3:
    """Create a 3D boolean vector."""
    if len(args) == 0:
        raise ValueError("bvec3 requires at least 1 argument")
    if len(args) == 1:
        val = bool(args[0])
        return np.array([val, val, val], dtype=bool)
    if len(args) == 2 and isinstance(args[0], BVec2):
        return np.array([args[0][0], args[0][1], bool(args[1])], dtype=bool)
    if len(args) == 3:
        return np.array([bool(args[0]), bool(args[1]), bool(args[2])], dtype=bool)
    raise TypeError(
        "bvec3 requires 1 or 3 arguments, or bvec2 and a scalar"
    )  # Changed to TypeError


def bvec4(*args) -> BVec4:
    """Create a 4D boolean vector."""
    if len(args) == 0:
        raise ValueError("bvec4 requires at least 1 argument")
    if len(args) == 1:
        val = bool(args[0])
        return np.array([val, val, val, val], dtype=bool)
    if len(args) == 2 and isinstance(args[0], BVec3):
        return np.array([args[0][0], args[0][1], args[0][2], bool(args[1])], dtype=bool)
    if len(args) == 3 and isinstance(args[0], BVec2):
        return np.array(
            [args[0][0], args[0][1], bool(args[1]), bool(args[2])], dtype=bool
        )
    if len(args) == 4:
        return np.array(
            [bool(args[0]), bool(args[1]), bool(args[2]), bool(args[3])], dtype=bool
        )
    raise TypeError(
        "bvec4 requires 1 or 4 arguments, or bvec3 and a scalar, or bvec2 and two scalars"
    )  # Changed to TypeError
