"""GLSL Type System - Core Built-in Definitions"""

from typing import Callable, Dict, List, Optional, Tuple, Union

from .base import GLSLType, TypeKind
from .types import FunctionType

# ======================================================================================
#                                       SCALAR TYPES
# ======================================================================================

FLOAT = GLSLType(TypeKind.FLOAT)
INT = GLSLType(TypeKind.INT)
BOOL = GLSLType(TypeKind.BOOL)

# ======================================================================================
#                                     VECTOR/MATRIX TYPES
# ======================================================================================


class VectorType(GLSLType):
    def __init__(self, size: int, component_type: GLSLType):
        super().__init__(TypeKind.VECTOR)
        self.size = size
        self.component_type = component_type

    def __eq__(self, other):
        return (
            isinstance(other, VectorType)
            and self.size == other.size
            and self.component_type == other.component_type
        )


class MatrixType(GLSLType):
    def __init__(self, rows: int, cols: int):
        super().__init__(TypeKind.MATRIX)
        self.rows = rows
        self.cols = cols

    def __eq__(self, other):
        return (
            isinstance(other, MatrixType)
            and self.rows == other.rows
            and self.cols == other.cols
        )


VEC2 = VectorType(2, FLOAT)
VEC3 = VectorType(3, FLOAT)
VEC4 = VectorType(4, FLOAT)
MAT2 = MatrixType(2, 2)
MAT3 = MatrixType(3, 3)
MAT4 = MatrixType(4, 4)

# ======================================================================================
#                                     BUILT-IN FUNCTIONS
# ======================================================================================

BUILTIN_FUNCTIONS: Dict[str, List[FunctionType]] = {
    # Math functions
    "abs": [
        FunctionType([FLOAT], FLOAT),
        FunctionType([VEC2], VEC2),
        FunctionType([VEC3], VEC3),
        FunctionType([VEC4], VEC4),
    ],
    "sin": [FunctionType([FLOAT], FLOAT)],
    "cos": [FunctionType([FLOAT], FLOAT)],
    "tan": [FunctionType([FLOAT], FLOAT)],
    "asin": [FunctionType([FLOAT], FLOAT)],
    "acos": [FunctionType([FLOAT], FLOAT)],
    "atan": [
        FunctionType([FLOAT], FLOAT),
        FunctionType([FLOAT, FLOAT], FLOAT),  # atan(y, x)
    ],
    "pow": [FunctionType([FLOAT, FLOAT], FLOAT)],
    "exp": [FunctionType([FLOAT], FLOAT)],
    "log": [FunctionType([FLOAT], FLOAT)],
    "exp2": [FunctionType([FLOAT], FLOAT)],
    "log2": [FunctionType([FLOAT], FLOAT)],
    "sqrt": [FunctionType([FLOAT], FLOAT)],
    "inversesqrt": [FunctionType([FLOAT], FLOAT)],
    # Common functions
    "floor": [FunctionType([FLOAT], FLOAT)],
    "ceil": [FunctionType([FLOAT], FLOAT)],
    "round": [FunctionType([FLOAT], FLOAT)],
    "fract": [FunctionType([FLOAT], FLOAT)],
    "mod": [FunctionType([FLOAT, FLOAT], FLOAT), FunctionType([VEC2, FLOAT], VEC2)],
    "clamp": [
        FunctionType([FLOAT, FLOAT, FLOAT], FLOAT),
        FunctionType([VEC2, VEC2, VEC2], VEC2),
    ],
    "mix": [
        FunctionType([FLOAT, FLOAT, FLOAT], FLOAT),
        FunctionType([VEC2, VEC2, FLOAT], VEC2),
        FunctionType([VEC3, VEC3, FLOAT], VEC3),
        FunctionType([VEC4, VEC4, FLOAT], VEC4),
    ],
    "step": [FunctionType([FLOAT, FLOAT], FLOAT), FunctionType([VEC2, VEC2], VEC2)],
    "smoothstep": [
        FunctionType([FLOAT, FLOAT, FLOAT], FLOAT),
        FunctionType([VEC2, VEC2, VEC2], VEC2),
    ],
    # Geometric functions
    "length": [FunctionType([VectorType], FLOAT)],
    "distance": [FunctionType([VectorType, VectorType], FLOAT)],
    "dot": [FunctionType([VectorType, VectorType], FLOAT)],
    "cross": [FunctionType([VEC3, VEC3], VEC3)],
    "normalize": [FunctionType([VectorType], VectorType)],
    "faceforward": [FunctionType([VectorType, VectorType, VectorType], VectorType)],
    "reflect": [FunctionType([VectorType, VectorType], VectorType)],
    "refract": [FunctionType([VectorType, VectorType, FLOAT], VectorType)],
    # Matrix operations
    "transpose": [
        FunctionType([MAT2], MAT2),
        FunctionType([MAT3], MAT3),
        FunctionType([MAT4], MAT4),
    ],
    "determinant": [
        FunctionType([MAT2], FLOAT),
        FunctionType([MAT3], FLOAT),
        FunctionType([MAT4], FLOAT),
    ],
    "inverse": [
        FunctionType([MAT2], MAT2),
        FunctionType([MAT3], MAT3),
        FunctionType([MAT4], MAT4),
    ],
}

# ======================================================================================
#                                     TYPE CONSTRUCTORS
# ======================================================================================

TYPE_CONSTRUCTORS = {
    "vec2": lambda *args: VEC2,
    "vec3": lambda *args: VEC3,
    "vec4": lambda *args: VEC4,
    "mat2": lambda *args: MAT2,
    "mat3": lambda *args: MAT3,
    "mat4": lambda *args: MAT4,
    "float": lambda *args: FLOAT,
    "int": lambda *args: INT,
    "bool": lambda *args: BOOL,
}

# ======================================================================================
#                                       CONSTANTS
# ======================================================================================

BUILTIN_CONSTANTS = {"pi": FLOAT, "EPSILON": FLOAT}

# ======================================================================================
#                                       UTILITIES
# ======================================================================================


def is_glsl_builtin(name: str) -> bool:
    return name in BUILTIN_FUNCTIONS or name in BUILTIN_CONSTANTS


def get_function_overloads(name: str) -> List[FunctionType]:
    return BUILTIN_FUNCTIONS.get(name, [])


def get_constructor(type_name: str) -> Optional[Callable]:
    return TYPE_CONSTRUCTORS.get(type_name)


__all__ = [
    "FLOAT",
    "INT",
    "BOOL",
    "VEC2",
    "VEC3",
    "VEC4",
    "MAT2",
    "MAT3",
    "MAT4",
    "BUILTIN_FUNCTIONS",
    "BUILTIN_CONSTANTS",
    "TYPE_CONSTRUCTORS",
    "is_glsl_builtin",
    "get_function_overloads",
    "get_constructor",
]
