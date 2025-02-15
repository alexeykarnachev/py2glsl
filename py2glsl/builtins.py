"""GLSL built-in function implementations using NumPy."""

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

Number: TypeAlias = float | NDArray[np.float32]
Vector: TypeAlias = NDArray[np.float32]

# Add pi constant
pi = np.pi


def round(x: Number) -> Number:
    """Round to nearest integer."""
    return np.round(x)


def abs(x: Number) -> Number:
    """Absolute value."""
    return np.abs(x)


def acos(x: Number) -> Number:
    """Arc cosine."""
    return np.arccos(x)


def all(x: Vector) -> bool:
    """Return true if all components are true."""
    return bool(np.all(x))


def any(x: Vector) -> bool:
    """Return true if any component is true."""
    return bool(np.any(x))


def asin(x: Number) -> Number:
    """Arc sine."""
    return np.arcsin(x)


def atan(y: float, x: float = 0.0) -> float:
    """Arc tangent. Returns atan(y/x) if x is provided."""
    return float(np.arctan2(y, x) if x != 0.0 else np.arctan(y))


def ceil(x: Number) -> Number:
    """Ceiling function."""
    return np.ceil(x)


def clamp(x: Number, minVal: Number, maxVal: Number) -> Number:
    """Constrain a value to lie between two further values."""
    return np.clip(x, minVal, maxVal)


def cos(x: Number) -> Number:
    """Cosine."""
    return np.cos(x)


def cross(x: Vector, y: Vector) -> Vector:
    """Vector cross product."""
    return np.cross(x, y)


def degrees(radians: Number) -> Number:
    """Convert radians to degrees."""
    return np.degrees(radians)


def determinant(m: Vector) -> float:
    """Matrix determinant."""
    return float(np.linalg.det(m))


def distance(p0: Vector, p1: Vector) -> float:
    """Distance between two points."""
    return float(np.linalg.norm(p1 - p0))


def dot(x: Vector, y: Vector) -> float:
    """Vector dot product."""
    return float(np.dot(x, y))


def equal(x: Vector, y: Vector) -> Vector:
    """Component-wise equal comparison."""
    return np.equal(x, y)


def exp(x: Number) -> Number:
    """Exponential."""
    return np.exp(x)


def exp2(x: Number) -> Number:
    """Base-2 exponential."""
    return np.exp2(x)


def faceforward(N: Vector, I: Vector, Nref: Vector) -> Vector:
    """Return N if dot(Nref, I) < 0, else -N."""
    return N if np.dot(Nref, I) < 0 else -N


def floor(x: Number) -> Number:
    """Floor function."""
    return np.floor(x)


def fract(x: Number) -> Number:
    """Return fractional part."""
    return x - np.floor(x)


def greaterThan(x: Vector, y: Vector) -> Vector:
    """Component-wise greater than comparison."""
    return np.greater(x, y)


def greaterThanEqual(x: Vector, y: Vector) -> Vector:
    """Component-wise greater than or equal comparison."""
    return np.greater_equal(x, y)


def inverse(m: Vector) -> Vector:
    """Matrix inverse."""
    return np.linalg.inv(m)


def inversesqrt(x: Number) -> Number:
    """Inverse square root."""
    return 1.0 / np.sqrt(x)


def length(x: Vector) -> float:
    """Vector length."""
    return float(np.linalg.norm(x))


def lessThan(x: Vector, y: Vector) -> Vector:
    """Component-wise less than comparison."""
    return np.less(x, y)


def lessThanEqual(x: Vector, y: Vector) -> Vector:
    """Component-wise less than or equal comparison."""
    return np.less_equal(x, y)


def log(x: Number) -> Number:
    """Natural logarithm."""
    return np.log(x)


def log2(x: Number) -> Number:
    """Base-2 logarithm."""
    return np.log2(x)


def max(x: Number, y: Number) -> Number:
    """Return maximum of two values."""
    return np.maximum(x, y)


def min(x: Number, y: Number) -> Number:
    """Return minimum of two values."""
    return np.minimum(x, y)


def mix(x: Number, y: Number, a: Number) -> Number:
    """Linear interpolation."""
    return x * (1.0 - a) + y * a


def mod(x: Number, y: Number) -> Number:
    """Modulo operation."""
    return np.mod(x, y)


def normalize(x: Vector) -> Vector:
    """Normalize vector."""
    return x / np.linalg.norm(x)


def notEqual(x: Vector, y: Vector) -> Vector:
    """Component-wise not equal comparison."""
    return np.not_equal(x, y)


def pow(x: Number, y: Number) -> Number:
    """Power function."""
    return np.power(x, y)


def radians(degrees: Number) -> Number:
    """Convert degrees to radians."""
    return np.radians(degrees)


def reflect(I: Vector, N: Vector) -> Vector:
    """Reflection vector."""
    return I - 2.0 * np.dot(N, I) * N


def refract(I: Vector, N: Vector, eta: float) -> Vector:
    """Refraction vector."""
    dot_ni = np.dot(N, I)
    k = 1.0 - eta * eta * (1.0 - dot_ni * dot_ni)
    if k < 0.0:
        return np.zeros_like(I)
    return eta * I - (eta * dot_ni + np.sqrt(k)) * N


def sign(x: Number) -> Number:
    """Return sign of x."""
    return np.sign(x)


def sin(x: Number) -> Number:
    """Sine."""
    return np.sin(x)


def smoothstep(edge0: Number, edge1: Number, x: Number) -> Number:
    """Smooth Hermite interpolation."""
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def sqrt(x: Number) -> Number:
    """Square root."""
    return np.sqrt(x)


def step(edge: Number, x: Number) -> Number:
    """Step function."""
    return np.where(x < edge, 0.0, 1.0)


def tan(x: Number) -> Number:
    """Tangent."""
    return np.tan(x)


def transpose(m: Vector) -> Vector:
    """Matrix transpose."""
    return np.transpose(m)
