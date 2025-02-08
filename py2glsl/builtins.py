from typing import Union

from py2glsl.types import vec2, vec3, vec4

Number = Union[float, vec2, vec3, vec4]


def abs(x: Number) -> Number:
    return x  # Stub for now


def atan(y: float, x: float = 0.0) -> float:
    return 0.0  # Stub for now


def clamp(x: Number, min_val: Number, max_val: Number) -> Number:
    return x  # Stub for now


def cos(x: float) -> float:
    return 0.0  # Stub for now


def dot(x: Number, y: Number) -> float:
    return 0.0  # Stub for now


def length(x: Number) -> float:
    return 0.0  # Stub for now


def max(x: Number, y: Number) -> Number:
    return x  # Stub for now


def min(x: Number, y: Number) -> Number:
    return x  # Stub for now


def mix(x: Number, y: Number, a: Number) -> Number:
    return x  # Stub for now


def normalize(x: Number) -> Number:
    return x  # Stub for now


def sin(x: float) -> float:
    return 0.0  # Stub for now


def smoothstep(edge0: Number, edge1: Number, x: Number) -> Number:
    return x  # Stub for now
