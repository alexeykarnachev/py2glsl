"""GLSL Type System - Type Conversion Rules"""

from typing import Callable, Dict, Optional, Tuple

from .base import (
    BOOL,
    FLOAT,
    INT,
    MAT2,
    MAT3,
    MAT4,
    VEC2,
    VEC3,
    VEC4,
    GLSLType,
    TypeKind,
)
from .errors import TypeConversionError


class TypeConverter:
    """Handles explicit type conversions between GLSL types"""

    # Conversion rules: (source_type, target_type) -> conversion_function
    CONVERSION_RULES: Dict[Tuple[TypeKind, TypeKind], Callable[[str], str]] = {
        # Scalar conversions
        (TypeKind.FLOAT, TypeKind.INT): lambda x: f"int({x})",
        (TypeKind.INT, TypeKind.FLOAT): lambda x: f"float({x})",
        (TypeKind.BOOL, TypeKind.INT): lambda x: f"int({x})",
        (TypeKind.INT, TypeKind.BOOL): lambda x: f"bool({x})",
        # Vector conversions
        (TypeKind.FLOAT, TypeKind.VEC2): lambda x: f"vec2({x})",
        (TypeKind.FLOAT, TypeKind.VEC3): lambda x: f"vec3({x})",
        (TypeKind.FLOAT, TypeKind.VEC4): lambda x: f"vec4({x})",
        (TypeKind.VEC2, TypeKind.VEC3): lambda x: f"vec3({x}, 0.0)",
        (TypeKind.VEC2, TypeKind.VEC4): lambda x: f"vec4({x}, 0.0, 1.0)",
        (TypeKind.VEC3, TypeKind.VEC4): lambda x: f"vec4({x}, 1.0)",
        # Matrix conversions
        (TypeKind.FLOAT, TypeKind.MAT2): lambda x: f"mat2({x})",
        (TypeKind.FLOAT, TypeKind.MAT3): lambda x: f"mat3({x})",
        (TypeKind.FLOAT, TypeKind.MAT4): lambda x: f"mat4({x})",
    }

    @classmethod
    def can_convert(cls, source: GLSLType, target: GLSLType) -> bool:
        """Check if conversion from source to target is possible"""
        return (source.kind, target.kind) in cls.CONVERSION_RULES

    @classmethod
    def convert(cls, source: GLSLType, target: GLSLType, expr: str) -> str:
        """
        Convert expression from source type to target type
        Returns converted expression or raises TypeConversionError
        """
        if source == target:
            return expr

        try:
            conversion_fn = cls.CONVERSION_RULES[(source.kind, target.kind)]
            return conversion_fn(expr)
        except KeyError:
            raise TypeConversionError(
                f"Cannot convert from {source.kind.name} to {target.kind.name}"
            )

    @classmethod
    def validate_conversion(
        cls, source: GLSLType, target: GLSLType, expr: Optional[str] = None
    ) -> bool:
        """
        Validate if conversion from source to target is valid
        Returns True if valid, False otherwise
        """
        # Same type is always valid
        if source == target:
            return True

        # Check if explicit conversion exists
        if cls.can_convert(source, target):
            return True

        # Special case: scalar to vector assignment
        if target.is_vector and source.is_scalar and source.is_numeric:
            return True

        return False

    @classmethod
    def get_conversion_code(cls, source: GLSLType, target: GLSLType, expr: str) -> str:
        """
        Get GLSL code for type conversion
        Returns conversion code or raises TypeConversionError
        """
        if source == target:
            return expr

        if not cls.validate_conversion(source, target):
            raise TypeConversionError(
                f"No valid conversion from {source.kind.name} to {target.kind.name}"
            )

        return cls.convert(source, target, expr)


__all__ = ["TypeConverter"]
