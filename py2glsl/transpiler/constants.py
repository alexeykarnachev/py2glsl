"""Constants for the GLSL shader transpiler."""

from typing import Any

# GLSL type constructors that can be called as functions
TYPE_CONSTRUCTORS = frozenset(
    {
        "vec2",
        "vec3",
        "vec4",
        "ivec2",
        "ivec3",
        "ivec4",
        "uvec2",
        "uvec3",
        "uvec4",
        "mat2",
        "mat3",
        "mat4",
        "float",
        "int",
        "bool",
    }
)

# Functions that always return a scalar float
SCALAR_RESULT_FUNCTIONS = frozenset({"length", "distance", "dot", "determinant"})

# Functions that always return bool
BOOL_RESULT_FUNCTIONS = frozenset({"any", "all", "not"})

# Functions that preserve the type of their first argument
PRESERVE_TYPE_FUNCTIONS = frozenset(
    {
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
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "tanh",
        "exp",
        "log",
        "exp2",
        "log2",
        "sqrt",
        "inversesqrt",
        "pow",
        "normalize",
        "reflect",
        "refract",
    }
)

# Matrix to vector type mapping
MATRIX_TO_VECTOR = {"mat2": "vec2", "mat3": "vec3", "mat4": "vec4"}

# Vector prefix to scalar element type mapping
VECTOR_ELEMENT_TYPE = {"vec": "float", "ivec": "int", "uvec": "uint", "bvec": "bool"}

# Maximum swizzle length (xyzw or rgba)
MAX_SWIZZLE_LENGTH = 4

# Valid swizzle characters
SWIZZLE_CHARS = frozenset("xyzwrgba")

# =============================================================================
# Builtin Function Signatures
# =============================================================================

# Vector types for signature generation
_VEC_TYPES = ["float", "vec2", "vec3", "vec4"]


def _unary_vec_sigs() -> list[tuple[str, list[str]]]:
    """Generate signatures for unary functions: T func(T) for all vec types."""
    return [(t, [t]) for t in _VEC_TYPES]


def _binary_vec_sigs() -> list[tuple[str, list[str]]]:
    """Generate signatures for binary functions: T func(T, T) for all vec types."""
    return [(t, [t, t]) for t in _VEC_TYPES]


def _ternary_vec_sigs() -> list[tuple[str, list[str]]]:
    """Generate signatures for ternary functions: T func(T, T, T) for all vec types."""
    return [(t, [t, t, t]) for t in _VEC_TYPES]


def _scalar_unary_vec_sigs() -> list[tuple[str, list[str]]]:
    """Generate signatures for functions returning float: float func(vecN)."""
    return [("float", [t]) for t in _VEC_TYPES[1:]]  # Skip float


def _scalar_binary_vec_sigs() -> list[tuple[str, list[str]]]:
    """Generate signatures for functions: float func(vecN, vecN)."""
    return [("float", [t, t]) for t in _VEC_TYPES[1:]]  # Skip float


BUILTIN_FUNCTIONS: dict[str, Any] = {
    # Trigonometric (scalar only)
    "sin": ("float", ["float"]),
    "cos": ("float", ["float"]),
    "tan": ("float", ["float"]),
    "asin": ("float", ["float"]),
    "acos": ("float", ["float"]),
    "atan": ("float", ["float"]),
    "radians": ("float", ["float"]),
    # Unary math functions: T func(T)
    "abs": _unary_vec_sigs(),
    "floor": _unary_vec_sigs(),
    "ceil": _unary_vec_sigs(),
    "fract": _unary_vec_sigs(),
    "sqrt": _unary_vec_sigs(),
    "normalize": [(t, [t]) for t in _VEC_TYPES[1:]],  # vec only
    # Binary math functions: T func(T, T)
    "min": _binary_vec_sigs(),
    "max": _binary_vec_sigs(),
    "step": _binary_vec_sigs(),
    # Binary with scalar second arg: T func(T, float)
    "mod": [(t, [t, "float" if t != "float" else t]) for t in _VEC_TYPES],
    # Ternary functions: T func(T, T, T)
    "clamp": _ternary_vec_sigs(),
    "smoothstep": _ternary_vec_sigs(),
    # mix: T func(T, T, float) for vec types, float func(float, float, float)
    "mix": [("float", ["float", "float", "float"])]
    + [(t, [t, t, "float"]) for t in _VEC_TYPES[1:]],
    # Scalar-only math
    "pow": ("float", ["float", "float"]),
    "exp": ("float", ["float"]),
    "log": ("float", ["float"]),
    "exp2": ("float", ["float"]),
    "log2": ("float", ["float"]),
    "round": ("float", ["float"]),
    # Geometric functions returning scalar
    "length": _scalar_unary_vec_sigs(),
    "distance": _scalar_binary_vec_sigs(),
    "dot": _scalar_binary_vec_sigs(),
    "cross": ("vec3", ["vec3", "vec3"]),
    # Reflection/refraction
    "reflect": [(t, [t, t]) for t in _VEC_TYPES[1:]],
    "refract": [(t, [t, t, "float"]) for t in _VEC_TYPES[1:]],
    # Type conversion
    "float": ("float", ["int"]),
    "int": ("int", ["float"]),
    # Vector constructors
    "vec2": [("vec2", ["float", "float"]), ("vec2", ["float"])],
    "vec3": [
        ("vec3", ["float", "float", "float"]),
        ("vec3", ["vec2", "float"]),
        ("vec3", ["float"]),
    ],
    "vec4": [
        ("vec4", ["float", "float", "float", "float"]),
        ("vec4", ["vec3", "float"]),
        ("vec4", ["vec2", "vec2"]),
        ("vec4", ["float"]),
    ],
    # Matrix constructors
    "mat2": [("mat2", ["float"] * 4), ("mat2", ["float"])],
    "mat3": [("mat3", ["float"] * 9), ("mat3", ["float"])],
    "mat4": [("mat4", ["float"] * 16), ("mat4", ["float"])],
}

# AST operator type names to string operators
AST_BINOP_MAP: dict[str, str] = {
    "Add": "+",
    "Sub": "-",
    "Mult": "*",
    "Div": "/",
    "FloorDiv": "//",  # Floor division - handled specially in IR builder
    "Mod": "%",
    "Pow": "**",
    "BitAnd": "&",
    "BitOr": "|",
    "BitXor": "^",
    "LShift": "<<",
    "RShift": ">>",
}

AST_UNARYOP_MAP: dict[str, str] = {
    "USub": "-",
    "UAdd": "+",
    "Not": "not",
    "Invert": "~",
}

AST_CMPOP_MAP: dict[str, str] = {
    "Eq": "==",
    "NotEq": "!=",
    "Lt": "<",
    "LtE": "<=",
    "Gt": ">",
    "GtE": ">=",
}

OPERATOR_PRECEDENCE: dict[str, int] = {
    "=": 1,
    "||": 2,
    "&&": 3,
    "==": 4,
    "!=": 4,
    "<": 5,
    ">": 5,
    "<=": 5,
    ">=": 5,
    "+": 6,
    "-": 6,
    "*": 7,
    "/": 7,
    "%": 7,
    "unary": 8,
    "call": 9,
    "member": 10,
    "?": 14,
}


# =============================================================================
# AST Operator Conversion Functions
# =============================================================================


def binop_to_str(op_type_name: str) -> str:
    """Convert AST binary operator type name to string operator."""
    return AST_BINOP_MAP.get(op_type_name, "+")


def unaryop_to_str(op_type_name: str) -> str:
    """Convert AST unary operator type name to string operator."""
    return AST_UNARYOP_MAP.get(op_type_name, "-")


def cmpop_to_str(op_type_name: str) -> str:
    """Convert AST comparison operator type name to string operator."""
    return AST_CMPOP_MAP.get(op_type_name, "==")
