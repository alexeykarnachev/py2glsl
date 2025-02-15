"""GLSL Type System - Type Promotion Rules"""

from typing import Dict, Optional, Tuple

from .base import FLOAT, INT, VEC2, VEC3, VEC4, GLSLType, TypeKind
from .errors import TypePromotionError


class TypePromotion:
    """Handles type promotion rules for GLSL operations"""

    # Promotion rules for binary operations
    PROMOTION_RULES: Dict[Tuple[TypeKind, TypeKind], GLSLType] = {
        # Scalar promotions
        (TypeKind.INT, TypeKind.FLOAT): FLOAT,
        (TypeKind.FLOAT, TypeKind.INT): FLOAT,
        # Vector promotions
        (TypeKind.VEC2, TypeKind.FLOAT): VEC2,
        (TypeKind.FLOAT, TypeKind.VEC2): VEC2,
        (TypeKind.VEC3, TypeKind.FLOAT): VEC3,
        (TypeKind.FLOAT, TypeKind.VEC3): VEC3,
        (TypeKind.VEC4, TypeKind.FLOAT): VEC4,
        (TypeKind.FLOAT, TypeKind.VEC4): VEC4,
        # Vector-vector promotions (must be same size)
        (TypeKind.VEC2, TypeKind.VEC2): VEC2,
        (TypeKind.VEC3, TypeKind.VEC3): VEC3,
        (TypeKind.VEC4, TypeKind.VEC4): VEC4,
    }

    @classmethod
    def can_promote(cls, type1: GLSLType, type2: GLSLType) -> bool:
        """Check if types can be promoted"""
        return (type1.kind, type2.kind) in cls.PROMOTION_RULES

    @classmethod
    def promote(cls, type1: GLSLType, type2: GLSLType) -> GLSLType:
        """
        Promote two types to a common type
        Returns the promoted type or raises TypePromotionError
        """
        key = (type1.kind, type2.kind)

        # Direct promotion
        if promoted := cls.PROMOTION_RULES.get(key):
            return promoted

        # Reverse order promotion
        if promoted := cls.PROMOTION_RULES.get((type2.kind, type1.kind)):
            return promoted

        raise TypePromotionError(
            f"Cannot promote types {type1.kind.name} and {type2.kind.name}"
        )

    @classmethod
    def validate_binary_operation(
        cls, op: str, left: GLSLType, right: GLSLType
    ) -> GLSLType:
        """
        Validate binary operation and return result type
        Raises TypePromotionError if operation is invalid
        """
        try:
            # Get promoted type
            result_type = cls.promote(left, right)

            # Additional validation for specific operations
            if op in {"*", "/", "%"}:
                if not result_type.is_numeric:
                    raise TypePromotionError(
                        f"Operation '{op}' requires numeric types, got {result_type}"
                    )

            return result_type

        except TypePromotionError as e:
            raise TypePromotionError(
                f"Invalid operation '{op}' between {left.kind.name} and {right.kind.name}: {e}"
            )

    @classmethod
    def validate_assignment(cls, target: GLSLType, value: GLSLType) -> bool:
        """
        Validate if value can be assigned to target
        Returns True if valid, False otherwise
        """
        # Exact match
        if target == value:
            return True

        # Promotion possible
        if cls.can_promote(target, value):
            return True

        # Special case: scalar to vector assignment
        if target.is_vector and value.is_scalar and value.is_numeric:
            return True

        return False


__all__ = ["TypePromotion"]
