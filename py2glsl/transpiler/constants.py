"""Constants for the GLSL shader transpiler."""

from typing import Any

BUILTIN_FUNCTIONS: dict[str, Any] = {
    # Trigonometric
    "sin": ("float", ["float"]),
    "cos": ("float", ["float"]),
    "tan": ("float", ["float"]),
    "asin": ("float", ["float"]),
    "acos": ("float", ["float"]),
    "atan": ("float", ["float"]),
    "radians": ("float", ["float"]),
    # Math
    "abs": [
        ("float", ["float"]),
        ("vec2", ["vec2"]),
        ("vec3", ["vec3"]),
        ("vec4", ["vec4"]),
    ],
    "floor": [
        ("float", ["float"]),
        ("vec2", ["vec2"]),
        ("vec3", ["vec3"]),
        ("vec4", ["vec4"]),
    ],
    "ceil": [
        ("float", ["float"]),
        ("vec2", ["vec2"]),
        ("vec3", ["vec3"]),
        ("vec4", ["vec4"]),
    ],
    "fract": [
        ("float", ["float"]),
        ("vec2", ["vec2"]),
        ("vec3", ["vec3"]),
        ("vec4", ["vec4"]),
    ],
    "sqrt": [
        ("float", ["float"]),
        ("vec2", ["vec2"]),
        ("vec3", ["vec3"]),
        ("vec4", ["vec4"]),
    ],
    "mod": [
        ("float", ["float", "float"]),
        ("vec2", ["vec2", "float"]),
        ("vec3", ["vec3", "float"]),
        ("vec4", ["vec4", "float"]),
    ],
    "min": [
        ("float", ["float", "float"]),
        ("vec2", ["vec2", "vec2"]),
        ("vec3", ["vec3", "vec3"]),
        ("vec4", ["vec4", "vec4"]),
    ],
    "max": [
        ("float", ["float", "float"]),
        ("vec2", ["vec2", "vec2"]),
        ("vec3", ["vec3", "vec3"]),
        ("vec4", ["vec4", "vec4"]),
    ],
    "step": [
        ("float", ["float", "float"]),
        ("vec2", ["vec2", "vec2"]),
        ("vec3", ["vec3", "vec3"]),
        ("vec4", ["vec4", "vec4"]),
    ],
    "clamp": [
        ("float", ["float", "float", "float"]),
        ("vec2", ["vec2", "vec2", "vec2"]),
        ("vec3", ["vec3", "vec3", "vec3"]),
        ("vec4", ["vec4", "vec4", "vec4"]),
    ],
    "mix": [
        ("float", ["float", "float", "float"]),
        ("vec2", ["vec2", "vec2", "float"]),
        ("vec3", ["vec3", "vec3", "float"]),
        ("vec4", ["vec4", "vec4", "float"]),
    ],
    "smoothstep": [
        ("float", ["float", "float", "float"]),
        ("vec2", ["vec2", "vec2", "vec2"]),
        ("vec3", ["vec3", "vec3", "vec3"]),
        ("vec4", ["vec4", "vec4", "vec4"]),
    ],
    "pow": ("float", ["float", "float"]),
    "exp": ("float", ["float"]),
    "log": ("float", ["float"]),
    "exp2": ("float", ["float"]),
    "log2": ("float", ["float"]),
    "round": ("float", ["float"]),
    # Geometric
    "length": [("float", ["vec2"]), ("float", ["vec3"]), ("float", ["vec4"])],
    "distance": [
        ("float", ["vec2", "vec2"]),
        ("float", ["vec3", "vec3"]),
        ("float", ["vec4", "vec4"]),
    ],
    "dot": [
        ("float", ["vec2", "vec2"]),
        ("float", ["vec3", "vec3"]),
        ("float", ["vec4", "vec4"]),
    ],
    "cross": ("vec3", ["vec3", "vec3"]),
    "normalize": [("vec2", ["vec2"]), ("vec3", ["vec3"]), ("vec4", ["vec4"])],
    "reflect": [
        ("vec2", ["vec2", "vec2"]),
        ("vec3", ["vec3", "vec3"]),
        ("vec4", ["vec4", "vec4"]),
    ],
    "refract": [
        ("vec2", ["vec2", "vec2", "float"]),
        ("vec3", ["vec3", "vec3", "float"]),
        ("vec4", ["vec4", "vec4", "float"]),
    ],
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
