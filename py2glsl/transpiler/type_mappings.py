"""GLSL type mappings and built-in functions."""

from typing import Dict, Optional, Tuple

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

# Built-in function return types
BUILTIN_TYPES: Dict[str, Optional[GLSLType]] = {
    "length": FLOAT,
    "distance": FLOAT,
    "dot": FLOAT,
    "cross": VEC3,
    "normalize": None,  # Returns same as input
    "faceforward": None,  # Returns same as input
    "reflect": None,  # Returns same as input
    "refract": None,  # Returns same as input
    "pow": FLOAT,
    "exp": FLOAT,
    "log": FLOAT,
    "exp2": FLOAT,
    "log2": FLOAT,
    "sqrt": FLOAT,
    "inversesqrt": FLOAT,
    "round": None,  # Returns same type as input
    "abs": None,  # Returns same as input
    "sign": None,  # Returns same as input
    "floor": None,  # Returns same as input
    "ceil": None,  # Returns same as input
    "fract": None,  # Returns same as input
    "mod": None,  # Returns same as input
    "min": None,  # Returns same as input
    "max": None,  # Returns same as input
    "clamp": None,  # Returns same as input
    "mix": None,  # Returns same as input
    "step": None,  # Returns same as input
    "smoothstep": None,  # Returns same as input
    "sin": FLOAT,
    "cos": FLOAT,
    "tan": FLOAT,
    "asin": FLOAT,
    "acos": FLOAT,
    "atan": FLOAT,
}

# Matrix constructors
MATRIX_CONSTRUCTORS: Dict[str, int] = {
    "mat2": 4,
    "mat3": 9,
    "mat4": 16,
}

# Built-in functions with argument count validation
BUILTIN_FUNCTIONS_ARGS = {
    "mix": 3,
    "clamp": 3,
    "smoothstep": 3,
    "cross": 2,
    "dot": 2,
    "distance": 2,
    "reflect": 2,
    "length": 1,
    "normalize": 1,
    "abs": 1,
    "sign": 1,
    "round": 1,
}

# All built-in functions
BUILTIN_FUNCTIONS = {
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "pow",
    "exp",
    "log",
    "exp2",
    "log2",
    "sqrt",
    "inversesqrt",
    "round",
    "abs",
    "sign",
    "floor",
    "ceil",
    "fract",
    "mod",
    "min",
    "max",
    "clamp",
    "mix",
    "step",
    "smoothstep",
    "length",
    "distance",
    "dot",
    "cross",
    "normalize",
    "faceforward",
    "reflect",
    "refract",
    "matrixCompMult",
    "transpose",
    "determinant",
    "inverse",
    "lessThan",
    "greaterThan",
    "lessThanEqual",
    "greaterThanEqual",
    "equal",
    "notEqual",
    "any",
    "all",
}
