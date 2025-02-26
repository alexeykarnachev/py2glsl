"""
Constants and predefined values for the GLSL shader transpiler.

This module contains dictionaries and definitions used throughout the transpiler,
including GLSL built-in functions and operator precedence mappings.
"""

from typing import Dict, List, Tuple

# Dictionary of built-in GLSL functions with return types and parameter types
BUILTIN_FUNCTIONS: Dict[str, Tuple[str, List[str]]] = {
    # Trigonometric functions
    "sin": ("float", ["float"]),
    "cos": ("float", ["float"]),
    "tan": ("float", ["float"]),
    "asin": ("float", ["float"]),
    "acos": ("float", ["float"]),
    "atan": ("float", ["float"]),
    # Mathematical functions
    "abs": ("float", ["float"]),
    "floor": ("float", ["float"]),
    "ceil": ("float", ["float"]),
    "fract": ("float", ["float"]),
    "mod": ("float", ["float", "float"]),
    "min": ("float", ["float", "float"]),
    "max": ("float", ["float", "float"]),
    "clamp": ("float", ["float", "float", "float"]),
    "mix": ("float", ["float", "float", "float"]),
    "step": ("float", ["float", "float"]),
    "smoothstep": ("float", ["float", "float", "float"]),
    "sqrt": ("float", ["float"]),
    "pow": ("float", ["float", "float"]),
    "exp": ("float", ["float"]),
    "log": ("float", ["float"]),
    "exp2": ("float", ["float"]),
    "log2": ("float", ["float"]),
    "round": ("float", ["float"]),
    # Geometric functions
    "length": ("float", ["vec2"]),
    "distance": ("float", ["vec2", "vec2"]),
    "dot": ("float", ["vec2", "vec2"]),
    "cross": ("vec3", ["vec3", "vec3"]),
    "normalize": ("vec3", ["vec3"]),
    "reflect": ("vec3", ["vec3", "vec3"]),
    "refract": ("vec3", ["vec3", "vec3", "float"]),
    "faceforward": ("vec3", ["vec3", "vec3", "vec3"]),
    # Vector functions with varying return types
    "mix": ("vec3", ["vec3", "vec3", "float"]),
    # Type conversion functions
    "float": ("float", ["int"]),
    "int": ("int", ["float"]),
    # Vector constructors
    "vec2": ("vec2", ["float", "float"]),
    "vec3": ("vec3", ["float", "float", "float"]),
    "vec4": ("vec4", ["float", "float", "float", "float"]),
    # Alternative vector constructors (from smaller vectors)
    "vec3": ("vec3", ["vec2", "float"]),
    "vec4": ("vec4", ["vec3", "float"]),
    "vec4": ("vec4", ["vec2", "vec2"]),
}


# Operator precedence for generating correct expressions
OPERATOR_PRECEDENCE: Dict[str, int] = {
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
