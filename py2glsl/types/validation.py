"""GLSL type validation and operation rules."""

from typing import Literal

from loguru import logger

from py2glsl.types import BOOL, FLOAT, GLSLType, TypeKind

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


def validate_operation(
    left: GLSLType, op: GLSLOperator, right: GLSLType
) -> GLSLType | None:
    """Validate operation between types and return result type."""
    logger.debug(f"Validating operation: {left} {op} {right}")

    # Handle arrays
    if left.array_size is not None or right.array_size is not None:
        # Array-array operations
        if left.array_size is not None and right.array_size is not None:
            if left.array_size != right.array_size or left.kind != right.kind:
                return None
            return left
        # Array-scalar operations
        if left.array_size is not None and right.kind in (TypeKind.FLOAT, TypeKind.INT):
            return left
        if right.array_size is not None and left.kind in (TypeKind.FLOAT, TypeKind.INT):
            return right
        return None

    # Handle void type
    if left.kind == TypeKind.VOID or right.kind == TypeKind.VOID:
        logger.debug("Operation involves void type - invalid")
        return None

    # Boolean operations
    if op in ("&&", "||"):
        if left.kind == TypeKind.BOOL and right.kind == TypeKind.BOOL:
            return BOOL
        if left.is_bool_vector and right.is_bool_vector and left == right:
            return left
        logger.debug("Invalid boolean operation")
        return None

    # Boolean vectors don't support arithmetic
    if (left.is_bool_vector or right.is_bool_vector) and op in (
        "+",
        "-",
        "*",
        "/",
        "%",
    ):
        logger.debug("Boolean vectors don't support arithmetic")
        return None

    # Matrix-vector multiplication
    if op == "*":
        if left.is_matrix and right.is_vector:
            if left.matrix_size() == right.vector_size():
                logger.debug(f"Valid matrix-vector multiplication: {right}")
                return right
        if right.is_matrix and left.is_vector:
            if right.matrix_size() == left.vector_size():
                logger.debug(f"Valid vector-matrix multiplication: {left}")
                return left

    # Vector operations
    if left.is_vector or right.is_vector:
        if not is_compatible_with(left, right):
            logger.debug("Incompatible vector operation")
            return None
        if op in ("==", "!="):
            return BOOL
        if op in ("<", ">", "<=", ">="):
            if left.is_bool_vector or right.is_bool_vector:
                return None
            return BOOL if left == right else None
        if op in ("+", "-", "*", "/"):
            return left if left.is_vector else right

    # Matrix operations
    if left.is_matrix or right.is_matrix:
        if op in ("==", "!="):
            return BOOL if left == right else None
        if not is_compatible_with(left, right):
            logger.debug("Incompatible matrix operation")
            return None
        return left if left.is_matrix else right

    # Numeric comparisons
    if left.is_numeric and right.is_numeric:
        if op in ("==", "!=", "<", ">", "<=", ">="):
            return BOOL
        return common_type(left, right)

    logger.debug("No valid operation found")
    return None


def is_compatible_with(left: GLSLType, right: GLSLType) -> bool:
    """Check if types can be used together in operations."""
    logger.debug(f"Checking compatibility between {left} and {right}")

    # Void type is never compatible
    if left.kind == TypeKind.VOID or right.kind == TypeKind.VOID:
        logger.debug("Void type is not compatible with any type")
        return False

    # Same types are always compatible
    if left == right:
        return True

    # Array compatibility - must match exactly in type and size
    if left.array_size is not None or right.array_size is not None:
        return left.kind == right.kind and left.array_size == right.array_size

    # Boolean vectors are only compatible with themselves
    if left.is_bool_vector or right.is_bool_vector:
        return left == right

    # Vector compatibility
    if left.is_vector and right.is_vector:
        if left.vector_size() != right.vector_size():
            logger.debug("Different vector sizes")
            return False
        # Different vector types aren't compatible for operations
        if left.kind != right.kind:
            logger.debug("Different vector types")
            return False
        return True

    # Vector-scalar compatibility (only for numeric vectors)
    if left.is_vector and not left.is_bool_vector:
        return right.kind in (TypeKind.FLOAT, TypeKind.INT)
    if right.is_vector and not right.is_bool_vector:
        return left.kind in (TypeKind.FLOAT, TypeKind.INT)

    # Matrix compatibility
    if left.is_matrix or right.is_matrix:
        if left.is_matrix and right.is_matrix:
            return left.matrix_size() == right.matrix_size()
        # Matrix-scalar compatibility
        return right.kind in (TypeKind.FLOAT, TypeKind.INT) or left.kind in (
            TypeKind.FLOAT,
            TypeKind.INT,
        )

    # Numeric type compatibility
    if left.is_numeric and right.is_numeric:
        return True

    logger.debug("No compatibility rule matched")
    return False


def can_convert_to(source: GLSLType, target: GLSLType) -> bool:
    """Check if type can be converted to another."""
    logger.debug(f"Checking conversion from {source} to {target}")

    if source == target:
        return True

    # Handle uniform source types
    if source.is_uniform:
        # Allow uniform float to int/float
        if source.kind == TypeKind.FLOAT:
            return target.kind in (TypeKind.INT, TypeKind.FLOAT)
        # Other uniform conversions follow normal rules
        base_source = GLSLType(source.kind, is_uniform=False)
        return can_convert_to(base_source, target)

    if target.kind == TypeKind.VOID or source.kind == TypeKind.VOID:
        logger.debug("Cannot convert to/from void type")
        return False

    if source.array_size is not None or target.array_size is not None:
        logger.debug("Cannot convert array types")
        return False

    if source.kind == TypeKind.INT and target.kind == TypeKind.FLOAT:
        return True

    if source.kind == TypeKind.FLOAT and target.kind == TypeKind.INT:
        return True

    # Allow vector conversions of same size (except bool vectors)
    if source.is_vector and target.is_vector:
        if source.is_bool_vector or target.is_bool_vector:
            return False
        return source.vector_size() == target.vector_size()

    logger.debug("No valid conversion found")
    return False


def common_type(left: GLSLType, right: GLSLType) -> GLSLType | None:
    """Find common type for operation result."""
    logger.debug(f"Finding common type between {left} and {right}")

    if left == right:
        return left

    if not is_compatible_with(left, right):
        logger.debug("Types are not compatible")
        return None

    # Handle arrays
    if left.array_size is not None or right.array_size is not None:
        # Array-array operations
        if left.array_size is not None and right.array_size is not None:
            if left.array_size != right.array_size:
                return None
            return left
        # Array-scalar operations
        if left.array_size is not None and right.kind in (TypeKind.FLOAT, TypeKind.INT):
            return left
        if right.array_size is not None and left.kind in (TypeKind.FLOAT, TypeKind.INT):
            return right
        return None

    # Handle vectors
    if left.is_vector or right.is_vector:
        if left.is_vector and right.is_vector:
            if left.vector_size() != right.vector_size():
                logger.debug("Different vector sizes")
                return None
            if left.is_bool_vector or right.is_bool_vector:
                return left if left == right else None
            return left if left.kind == TypeKind.FLOAT else right
        return left if left.is_vector else right

    # Handle matrices
    if left.is_matrix or right.is_matrix:
        if left.is_matrix and right.is_matrix:
            return left if left.matrix_size() == right.matrix_size() else None
        return left if left.is_matrix else right

    # Handle numeric types
    if left.kind == TypeKind.FLOAT or right.kind == TypeKind.FLOAT:
        return FLOAT

    logger.debug(f"Common type result: {left}")
    return left
