"""GLSL type mappings and built-in functions."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Optional, Set, Tuple

from py2glsl.types import (
    BOOL,
    BVEC2,
    BVEC3,
    BVEC4,
    FLOAT,
    INT,
    IVEC2,
    IVEC3,
    IVEC4,
    VEC2,
    VEC3,
    VEC4,
    GLSLType,
)


class ArgumentBehavior(Enum):
    """How function handles its arguments."""

    EXACT = auto()  # Arguments must match exactly
    PRESERVE = auto()  # Returns same type as first argument
    PROMOTE = auto()  # Can promote types (e.g. float -> vec4)
    FLEXIBLE = auto()  # Special handling needed


@dataclass
class GLSLFunction:
    """Definition of a GLSL built-in function."""

    return_type: Optional[GLSLType]  # None means preserve input type
    min_args: int
    max_args: int
    arg_behavior: ArgumentBehavior
    description: str = ""


# Vector type mappings
VECTOR_TYPES: Dict[str, Tuple[GLSLType, int]] = {
    "vec2": (VEC2, 2),
    "vec3": (VEC3, 3),
    "vec4": (VEC4, 4),
    "ivec2": (IVEC2, 2),
    "ivec3": (IVEC3, 3),
    "ivec4": (IVEC4, 4),
    "bvec2": (BVEC2, 2),
    "bvec3": (BVEC3, 3),
    "bvec4": (BVEC4, 4),
}

# Vector constructor sizes
VECTOR_CONSTRUCTORS: Dict[str, int] = {
    "vec2": 2,
    "vec3": 3,
    "vec4": 4,
    "ivec2": 2,
    "ivec3": 3,
    "ivec4": 4,
    "bvec2": 2,
    "bvec3": 3,
    "bvec4": 4,
}

# Valid vector constructor combinations
VALID_VECTOR_COMBINATIONS = {
    4: [  # Valid vec4 combinations
        (4,),  # vec4
        (3, 1),  # vec3 + float
        (1, 3),  # float + vec3
        (2, 1, 1),  # vec2 + float + float
        (1, 2, 1),  # float + vec2 + float
        (1, 1, 2),  # float + float + vec2
        (1, 1, 1, 1),  # four scalars
    ],
    3: [  # Valid vec3 combinations
        (3,),  # vec3
        (2, 1),  # vec2 + float
        (1, 2),  # float + vec2
        (1, 1, 1),  # three scalars
    ],
    2: [  # Valid vec2 combinations
        (2,),  # vec2
        (1, 1),  # two scalars
    ],
}

# Basic type constructors
TYPE_CONSTRUCTORS: Dict[str, GLSLType] = {
    "float": FLOAT,
    "int": INT,
    "bool": BOOL,
}

# Matrix constructors
MATRIX_CONSTRUCTORS: Dict[str, int] = {
    "mat2": 4,
    "mat3": 9,
    "mat4": 16,
}

# Built-in function definitions
_BUILTIN_DEFS: Dict[str, GLSLFunction] = {
    # Trigonometric functions
    "sin": GLSLFunction(FLOAT, 1, 1, ArgumentBehavior.PRESERVE),
    "cos": GLSLFunction(FLOAT, 1, 1, ArgumentBehavior.PRESERVE),
    "tan": GLSLFunction(FLOAT, 1, 1, ArgumentBehavior.PRESERVE),
    "asin": GLSLFunction(FLOAT, 1, 1, ArgumentBehavior.PRESERVE),
    "acos": GLSLFunction(FLOAT, 1, 1, ArgumentBehavior.PRESERVE),
    "atan": GLSLFunction(FLOAT, 1, 1, ArgumentBehavior.PRESERVE),
    # Common math functions
    "abs": GLSLFunction(None, 1, 1, ArgumentBehavior.PRESERVE),
    "sign": GLSLFunction(None, 1, 1, ArgumentBehavior.PRESERVE),
    "floor": GLSLFunction(None, 1, 1, ArgumentBehavior.PRESERVE),
    "ceil": GLSLFunction(None, 1, 1, ArgumentBehavior.PRESERVE),
    "fract": GLSLFunction(None, 1, 1, ArgumentBehavior.PRESERVE),
    "mod": GLSLFunction(None, 2, 2, ArgumentBehavior.PRESERVE),
    # Vector functions
    "length": GLSLFunction(FLOAT, 1, 1, ArgumentBehavior.EXACT),
    "distance": GLSLFunction(FLOAT, 2, 2, ArgumentBehavior.EXACT),
    "dot": GLSLFunction(FLOAT, 2, 2, ArgumentBehavior.EXACT),
    "cross": GLSLFunction(VEC3, 2, 2, ArgumentBehavior.EXACT),
    "normalize": GLSLFunction(None, 1, 1, ArgumentBehavior.PRESERVE),
    # Special functions
    "mix": GLSLFunction(None, 3, 3, ArgumentBehavior.FLEXIBLE),
    "step": GLSLFunction(None, 2, 2, ArgumentBehavior.PRESERVE),
    "smoothstep": GLSLFunction(None, 3, 3, ArgumentBehavior.PRESERVE),
    "min": GLSLFunction(None, 2, 2, ArgumentBehavior.PRESERVE),
    "max": GLSLFunction(None, 2, 2, ArgumentBehavior.PRESERVE),
    "clamp": GLSLFunction(None, 3, 3, ArgumentBehavior.PRESERVE),
}

# Generate the old-style mappings from the definitions
BUILTIN_FUNCTIONS: Set[str] = set(_BUILTIN_DEFS.keys())
BUILTIN_TYPES: Dict[str, Optional[GLSLType]] = {
    name: func.return_type for name, func in _BUILTIN_DEFS.items()
}
BUILTIN_FUNCTIONS_ARGS: Dict[str, int] = {
    name: func.min_args
    for name, func in _BUILTIN_DEFS.items()
    if func.min_args == func.max_args
}


def get_builtin_info(func_name: str) -> Optional[GLSLFunction]:
    """Get built-in function information."""
    return _BUILTIN_DEFS.get(func_name)
