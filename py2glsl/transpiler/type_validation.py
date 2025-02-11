"""GLSL type validation and operation rules."""

from typing import Literal, Optional

from py2glsl.transpiler.type_system import (
    BOOL,
    FLOAT,
    GLSLSwizzleError,
    GLSLType,
    TypeKind,
)

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


def validate_swizzle(type_: GLSLType, components: str) -> Optional[GLSLType]:
    """Validate swizzle operation and return resulting type."""
    if not type_.is_vector:
        raise GLSLSwizzleError(f"Cannot swizzle non-vector type {type_.name}")

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
    if max_component_idx >= type_.vector_size():
        raise GLSLSwizzleError(
            f"Component index {max_component_idx} out of range for {type_.name}"
        )

    from py2glsl.transpiler.type_system import FLOAT, VEC2, VEC3, VEC4

    return {1: FLOAT, 2: VEC2, 3: VEC3, 4: VEC4}[size]


def validate_operation(
    left: GLSLType, op: GLSLOperator, right: GLSLType
) -> Optional[GLSLType]:
    """Validate operation between types and return result type."""
    # Handle void type
    if left.kind == TypeKind.VOID or right.kind == TypeKind.VOID:
        return None

    # Boolean operations
    if op in ("&&", "||"):
        if left.kind == TypeKind.BOOL and right.kind == TypeKind.BOOL:
            return BOOL
        if left.is_bool_vector and right.is_bool_vector and left == right:
            return left
        return None

    # Matrix-vector multiplication
    if op == "*":
        if left.is_matrix and right.is_vector:
            if left.matrix_size() == right.vector_size():
                return right
        if right.is_matrix and left.is_vector:
            if right.matrix_size() == left.vector_size():
                return left

    # Vector operations
    if left.is_vector or right.is_vector:
        if not is_compatible_with(left, right):
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
            return None
        return left if left.is_matrix else right

    # Numeric comparisons
    if left.is_numeric and right.is_numeric:
        if op in ("==", "!=", "<", ">", "<=", ">="):
            return BOOL
        return common_type(left, right)

    return None


def is_compatible_with(left: GLSLType, right: GLSLType) -> bool:
    """Check if types can be used together."""
    if left == right:
        return True

    if left.kind == TypeKind.VOID or right.kind == TypeKind.VOID:
        return False

    # Boolean vectors are only compatible with themselves
    if left.is_bool_vector or right.is_bool_vector:
        return left == right

    # Vector compatibility
    if left.is_vector and right.is_vector:
        if left.vector_size() != right.vector_size():
            return False
        # Different vector types aren't compatible for operations
        if left.kind != right.kind:
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
        return right.kind in (TypeKind.FLOAT, TypeKind.INT) or left.kind in (
            TypeKind.FLOAT,
            TypeKind.INT,
        )

    # Numeric type compatibility
    return left.is_numeric and right.is_numeric


def can_convert_to(source: GLSLType, target: GLSLType) -> bool:
    """Check if type can be converted to another."""
    if source == target:
        return True

    if target.kind == TypeKind.VOID or source.kind == TypeKind.VOID:
        return False

    if source.array_size is not None or target.array_size is not None:
        return False

    # Allow int to float conversion
    if source.kind == TypeKind.INT and target.kind == TypeKind.FLOAT:
        return True

    # Allow vector conversions of same size
    if source.is_vector and target.is_vector:
        return source.vector_size() == target.vector_size()

    return False


def common_type(left: GLSLType, right: GLSLType) -> Optional[GLSLType]:
    """Find common type for operation result."""
    if left == right:
        return left

    if not is_compatible_with(left, right):
        return None

    # Handle arrays
    if left.array_size is not None or right.array_size is not None:
        if left.array_size == right.array_size and left.kind == right.kind:
            return left
        return None

    # Handle vectors
    if left.is_vector or right.is_vector:
        if left.is_vector and right.is_vector:
            if left.vector_size() != right.vector_size():
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
    return left
