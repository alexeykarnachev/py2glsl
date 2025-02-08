import math
from typing import Union

from py2glsl.types import Number, vec2, vec3, vec4

Number = Union[float, vec2, vec3, vec4]


def abs(x: Number) -> Number:
    return x  # Stub for now


def atan(y: float, x: float = 0.0) -> float:
    return 0.0  # Stub for now


def clamp(x: Number, min_val: Number, max_val: Number) -> Number:
    return min(max(x, min_val), max_val)


def cos(x: float) -> float:
    return 0.0  # Stub for now


def dot(x: Number, y: Number) -> float:
    if isinstance(x, vec2) and isinstance(y, vec2):
        return x.x * y.x + x.y * y.y
    elif isinstance(x, vec3) and isinstance(y, vec3):
        return x.x * y.x + x.y * y.y + x.z * y.z
    elif isinstance(x, vec4) and isinstance(y, vec4):
        return x.x * y.x + x.y * y.y + x.z * y.z + x.w * y.w
    raise TypeError(f"Cannot calculate dot product of {type(x)} and {type(y)}")


def length(x: Number) -> float:
    if isinstance(x, (int, float)):
        return abs(float(x))
    elif isinstance(x, vec2):
        return math.sqrt(x.x * x.x + x.y * x.y)
    elif isinstance(x, vec3):
        return math.sqrt(x.x * x.x + x.y * x.y + x.z * x.z)
    elif isinstance(x, vec4):
        return math.sqrt(x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w)
    raise TypeError(f"Cannot calculate length of {type(x)}")


def max(x: Number, y: Number) -> Number:
    return x  # Stub for now


def min(x: Number, y: Number) -> Number:
    return x  # Stub for now


def mix(x: Number, y: Number, a: Number) -> Number:
    return x * (1.0 - a) + y * a


def normalize(x: Number) -> Number:
    l = length(x)
    if l == 0:
        return x
    return x / l


def sin(x: float) -> float:
    return 0.0  # Stub for now


def smoothstep(edge0: Number, edge1: Number, x: Number) -> Number:
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
