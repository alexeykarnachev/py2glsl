"""GLSL Type System - Type Validation"""

from typing import List, Optional, Union

from .base import BOOL, GLSLType
from .conversion import TypeConverter
from .errors import TypeValidationError
from .promotion import TypePromotion


class TypeValidator:
    """Central type validation system for GLSL operations"""

    # Supported binary operators and their requirements
    BINARY_OPERATORS = {
        # Arithmetic operators
        "+": {"numeric": True},
        "-": {"numeric": True},
        "*": {"numeric": True},
        "/": {"numeric": True},
        "%": {"numeric": True},
        # Comparison operators
        "==": {},
        "!=": {},
        "<": {"numeric": True},
        ">": {"numeric": True},
        "<=": {"numeric": True},
        ">=": {"numeric": True},
        # Logical operators
        "&&": {"boolean": True},
        "||": {"boolean": True},
    }

    @classmethod
    def validate_binary_operation(
        cls, op: str, left: GLSLType, right: GLSLType
    ) -> GLSLType:
        """
        Validate binary operation and return result type
        Raises TypeValidationError if operation is invalid
        """
        if op not in cls.BINARY_OPERATORS:
            raise TypeValidationError(f"Unsupported binary operator: {op}")

        # Get operator requirements
        requirements = cls.BINARY_OPERATORS[op]

        try:
            # Promote types if possible
            result_type = TypePromotion.promote(left, right)

            # Check numeric requirement
            if requirements.get("numeric") and not result_type.is_numeric:
                raise TypeValidationError(
                    f"Operator '{op}' requires numeric types, got {result_type}"
                )

            # Check boolean requirement
            if requirements.get("boolean") and not result_type.is_boolean:
                raise TypeValidationError(
                    f"Operator '{op}' requires boolean types, got {result_type}"
                )

            # Comparison operators always return bool
            if op in {"==", "!=", "<", ">", "<=", ">="}:
                return BOOL

            return result_type

        except TypeValidationError as e:
            raise TypeValidationError(
                f"Invalid operation '{op}' between {left.kind.name} and {right.kind.name}: {e}"
            )

    @classmethod
    def validate_assignment(cls, target: GLSLType, value: GLSLType) -> bool:
        """
        Validate if value can be assigned to target
        Returns True if valid, False otherwise
        """
        # Direct assignment
        if target == value:
            return True

        # Promotion possible
        if TypePromotion.can_promote(target, value):
            return True

        # Conversion possible
        if TypeConverter.can_convert(value, target):
            return True

        # Special case: scalar to vector assignment
        if target.is_vector and value.is_scalar and value.is_numeric:
            return True

        return False

    @classmethod
    def validate_function_call(
        cls, func_type: GLSLType, arg_types: List[GLSLType]
    ) -> bool:
        """
        Validate function call arguments against function type
        Returns True if valid, False otherwise
        """
        # TODO: Implement proper function signature validation
        # For now, just check if all arguments can be converted
        return all(
            TypeConverter.validate_conversion(arg, param)
            for arg, param in zip(arg_types, func_type.parameters)
        )

    @classmethod
    def validate_return(cls, return_type: GLSLType, value_type: GLSLType) -> bool:
        """
        Validate return statement
        Returns True if valid, False otherwise
        """
        # Void return
        if return_type.is_void:
            return value_type.is_void

        # Direct match
        if return_type == value_type:
            return True

        # Promotion possible
        if TypePromotion.can_promote(return_type, value_type):
            return True

        # Conversion possible
        if TypeConverter.can_convert(value_type, return_type):
            return True

        return False


__all__ = ["TypeValidator"]
