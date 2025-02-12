"""Singleton GLSL type instances."""

from .base import GLSLType, TypeKind

# Singleton types
VOID = GLSLType(TypeKind.VOID)
BOOL = GLSLType(TypeKind.BOOL)
INT = GLSLType(TypeKind.INT)
FLOAT = GLSLType(TypeKind.FLOAT)
VEC2 = GLSLType(TypeKind.VEC2)
VEC3 = GLSLType(TypeKind.VEC3)
VEC4 = GLSLType(TypeKind.VEC4)
IVEC2 = GLSLType(TypeKind.IVEC2)
IVEC3 = GLSLType(TypeKind.IVEC3)
IVEC4 = GLSLType(TypeKind.IVEC4)
BVEC2 = GLSLType(TypeKind.BVEC2)
BVEC3 = GLSLType(TypeKind.BVEC3)
BVEC4 = GLSLType(TypeKind.BVEC4)
MAT2 = GLSLType(TypeKind.MAT2)
MAT3 = GLSLType(TypeKind.MAT3)
MAT4 = GLSLType(TypeKind.MAT4)
