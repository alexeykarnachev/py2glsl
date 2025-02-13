"""GLSL type system."""

from .base import GLSLType, TypeKind
from .constructors import (
    bvec2,
    bvec3,
    bvec4,
    ivec2,
    ivec3,
    ivec4,
    mat2,
    mat3,
    mat4,
    vec2,
    vec3,
    vec4,
)
from .errors import GLSLError, GLSLOperationError, GLSLSwizzleError, GLSLTypeError
from .singletons import (
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
from .validation import (
    can_convert_to,
    common_type,
    is_compatible_with,
    validate_operation,
)

__all__ = [
    # Base types
    "GLSLType",
    "TypeKind",
    # Errors
    "GLSLError",
    "GLSLTypeError",
    "GLSLOperationError",
    "GLSLSwizzleError",
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
    "mat2",
    "mat3",
    "mat4",
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
