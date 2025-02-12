"""GLSL type system."""

from .base import (
    GLSLError,
    GLSLOperationError,
    GLSLSwizzleError,
    GLSLType,
    GLSLTypeError,
    TypeKind,
)
from .singleton_types import (
    BOOL,
    BVEC2,
    BVEC3,
    BVEC4,
    FLOAT,
    INT,
    IVEC2,
    IVEC3,
    IVEC4,
    MAT2,
    MAT3,
    MAT4,
    VEC2,
    VEC3,
    VEC4,
    VOID,
)
from .type_constructors import (
    bvec2,
    bvec3,
    bvec4,
    ivec2,
    ivec3,
    ivec4,
    vec2,
    vec3,
    vec4,
)
from .type_validation import (
    can_convert_to,
    common_type,
    is_compatible_with,
    validate_operation,
)

__all__ = [
    # Base types
    "GLSLError",
    "GLSLTypeError",
    "GLSLOperationError",
    "GLSLSwizzleError",
    "GLSLType",
    "TypeKind",
    # Singleton types
    "VOID",
    "BOOL",
    "INT",
    "FLOAT",
    "VEC2",
    "VEC3",
    "VEC4",
    "IVEC2",
    "IVEC3",
    "IVEC4",
    "BVEC2",
    "BVEC3",
    "BVEC4",
    "MAT2",
    "MAT3",
    "MAT4",
    # Constructors
    "vec2",
    "vec3",
    "vec4",
    "ivec2",
    "ivec3",
    "ivec4",
    "bvec2",
    "bvec3",
    "bvec4",
    # Validation
    "validate_operation",
    "is_compatible_with",
    "can_convert_to",
    "common_type",
]
