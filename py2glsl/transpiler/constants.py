"""
Constants and predefined values for the GLSL shader transpiler.

This module contains dictionaries and definitions used throughout the transpiler,
including GLSL built-in functions and operator precedence mappings.
"""

from typing import Any

# Type alias for function signature
FunctionSignature = tuple[str, list[str]] | list[tuple[str, list[str]]]

# Dictionary of built-in GLSL functions with return types and parameter types
# Each function can have multiple overloads defined as a list of signatures
BUILTIN_FUNCTIONS: dict[str, Any] = {
    # Trigonometric functions
    "sin": ("float", ["float"]),
    "cos": ("float", ["float"]),
    "tan": ("float", ["float"]),
    "asin": ("float", ["float"]),
    "acos": ("float", ["float"]),
    "atan": ("float", ["float"]),
    "radians": ("float", ["float"]),
    # Mathematical functions
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
    "step": [
        ("float", ["float", "float"]),
        ("vec2", ["vec2", "vec2"]),
        ("vec3", ["vec3", "vec3"]),
        ("vec4", ["vec4", "vec4"]),
    ],
    "smoothstep": [
        ("float", ["float", "float", "float"]),
        ("vec2", ["vec2", "vec2", "vec2"]),
        ("vec3", ["vec3", "vec3", "vec3"]),
        ("vec4", ["vec4", "vec4", "vec4"]),
    ],
    "sqrt": [
        ("float", ["float"]),
        ("vec2", ["vec2"]),
        ("vec3", ["vec3"]),
        ("vec4", ["vec4"]),
    ],
    "pow": ("float", ["float", "float"]),
    "exp": ("float", ["float"]),
    "log": ("float", ["float"]),
    "exp2": ("float", ["float"]),
    "log2": ("float", ["float"]),
    "round": ("float", ["float"]),
    # Geometric functions
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
    # Type conversion functions
    "float": ("float", ["int"]),
    "int": ("int", ["float"]),
    # Vector constructors
    "vec2": [
        ("vec2", ["float", "float"]),
        ("vec2", ["float"]),  # Same value for all components
    ],
    "vec3": [
        ("vec3", ["float", "float", "float"]),
        ("vec3", ["vec2", "float"]),
        ("vec3", ["float"]),  # Same value for all components
    ],
    "vec4": [
        ("vec4", ["float", "float", "float", "float"]),
        ("vec4", ["vec3", "float"]),
        ("vec4", ["vec2", "vec2"]),
        ("vec4", ["float"]),  # Same value for all components
    ],
}


# Operator precedence for generating correct expressions
OPERATOR_PRECEDENCE: dict[str, int] = {
    # Assignment has lowest precedence
    "=": 1,
    # Logical operators
    "||": 2,  # Logical OR
    "&&": 3,  # Logical AND
    # Equality operators
    "==": 4,  # Equal
    "!=": 4,  # Not equal
    # Relational operators
    "<": 5,  # Less than
    ">": 5,  # Greater than
    "<=": 5,  # Less than or equal
    ">=": 5,  # Greater than or equal
    # Additive operators
    "+": 6,  # Addition
    "-": 6,  # Subtraction
    # Multiplicative operators
    "*": 7,  # Multiplication
    "/": 7,  # Division
    "%": 7,  # Modulo
    # Unary operators
    "unary": 8,
    # Function calls and member access
    "call": 9,
    "member": 10,
    # Ternary operator
    "?": 14,  # Ternary conditional
}
