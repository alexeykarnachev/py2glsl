"""GLSL Type System - Type Errors"""


class GLSLTypeError(Exception):
    """Base class for all type-related errors"""

    def __init__(self, message: str, context=None):
        super().__init__(message)
        self.context = context
        self.message = message

    def __str__(self):
        if self.context:
            return f"{self.message}\nContext: {self.context}"
        return self.message


class TypeInferenceError(GLSLTypeError):
    """Error during type inference"""

    def __init__(self, message: str, expression=None):
        super().__init__(f"Type inference error: {message}")
        self.expression = expression


class TypePromotionError(GLSLTypeError):
    """Error during type promotion"""

    def __init__(self, type1, type2, operation=None):
        message = f"Cannot promote {type1} and {type2}"
        if operation:
            message += f" for operation '{operation}'"
        super().__init__(message)


class TypeConversionError(GLSLTypeError):
    """Error during type conversion"""

    def __init__(self, source_type, target_type, context=None):
        message = f"Cannot convert from {source_type} to {target_type}"
        super().__init__(message, context)


class TypeValidationError(GLSLTypeError):
    """Error during type validation"""

    def __init__(self, message: str, location=None):
        super().__init__(f"Type validation error: {message}")
        self.location = location


class TypeContextError(GLSLTypeError):
    """Error during type context operations"""

    def __init__(self, message: str, scope=None):
        super().__init__(f"Type context error: {message}")
        self.scope = scope


class TypeDeclarationError(GLSLTypeError):
    """Error during type declaration"""

    def __init__(self, message: str, name=None, declared_type=None):
        super().__init__(f"Type declaration error: {message}")
        self.name = name
        self.declared_type = declared_type


class TypeAssignmentError(GLSLTypeError):
    """Error during variable assignment"""

    def __init__(self, target_type, value_type, context=None):
        message = f"Cannot assign {value_type} to {target_type}"
        super().__init__(message, context)


class TypeFunctionError(GLSLTypeError):
    """Error during function type operations"""

    def __init__(self, message: str, function_name=None, signature=None):
        super().__init__(f"Function type error: {message}")
        self.function_name = function_name
        self.signature = signature


class TypeReturnError(GLSLTypeError):
    """Error during return type validation"""

    def __init__(self, expected_type, actual_type, context=None):
        message = f"Expected return type {expected_type}, got {actual_type}"
        super().__init__(message, context)


__all__ = [
    "GLSLTypeError",
    "TypeInferenceError",
    "TypePromotionError",
    "TypeConversionError",
    "TypeValidationError",
    "TypeContextError",
    "TypeDeclarationError",
    "TypeAssignmentError",
    "TypeFunctionError",
    "TypeReturnError",
]
