"""Type utility functions for IR building.

Provides type inference and manipulation functions used by the IR builder.
"""

import re

from py2glsl.transpiler.constants import (
    MATRIX_TO_VECTOR,
    MAX_SWIZZLE_LENGTH,
    SWIZZLE_CHARS,
    VECTOR_ELEMENT_TYPE,
)
from py2glsl.transpiler.ir import IRType


def parse_type_string(type_str: str) -> IRType:
    """Parse type string like 'float[3]' or 'float[]' into IRType."""
    # Match 'type[size]' with explicit size
    match = re.match(r"(\w+)\[(\d+)\]", type_str)
    if match:
        base = match.group(1)
        size = int(match.group(2))
        return IRType(base=base, array_size=size)
    # Match 'type[]' without size (size to be inferred from literal)
    match = re.match(r"(\w+)\[\]", type_str)
    if match:
        base = match.group(1)
        # array_size=-1 means "infer from literal"
        return IRType(base=base, array_size=-1)
    return IRType(base=type_str)


def infer_literal_type(value: object) -> IRType:
    """Infer IR type from literal value."""
    if isinstance(value, bool):
        return IRType("bool")
    if isinstance(value, int):
        return IRType("int")
    if isinstance(value, float):
        return IRType("float")
    return IRType("float")


def is_swizzle(attr: str) -> bool:
    """Check if attribute access is a vector swizzle."""
    if not attr or len(attr) > MAX_SWIZZLE_LENGTH:
        return False
    return all(c in SWIZZLE_CHARS for c in attr)


def swizzle_result_type(components: str) -> IRType:
    """Get result type of swizzle operation."""
    n = len(components)
    if n == 1:
        return IRType("float")
    return IRType(f"vec{n}")


def subscript_result_type(base_type: IRType) -> IRType:
    """Get result type of subscript operation."""
    # Array subscript returns the element type
    if base_type.array_size is not None:
        return IRType(base_type.base)

    base = base_type.base
    # Check vector types (vec, ivec, uvec, bvec)
    for prefix, element_type in VECTOR_ELEMENT_TYPE.items():
        if base.startswith(prefix):
            return IRType(element_type)
    # Matrix subscript returns corresponding vector
    if base in MATRIX_TO_VECTOR:
        return IRType(MATRIX_TO_VECTOR[base])
    return IRType("float")


def get_indexable_size(ir_type: IRType) -> int | None:
    """Get the size of an indexable type (vector or array)."""
    # Check for array types
    if ir_type.array_size is not None:
        return ir_type.array_size
    # Check for vector types (vec2, vec3, vec4, ivec2, etc.)
    base = ir_type.base
    for prefix in ("vec", "ivec", "uvec", "bvec"):
        if base.startswith(prefix) and len(base) == len(prefix) + 1:
            try:
                return int(base[-1])
            except ValueError:
                pass
    return None
