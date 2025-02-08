from typing import TypeAlias, Union

import numpy as np

Number: TypeAlias = Union[float, np.ndarray]


def abs(x: Number) -> Number:
    """Working NumPy implementation for testing"""
    return np.abs(x)


def atan(y: float, x: float = 0.0) -> float:
    """Working NumPy implementation for testing"""
    return float(np.arctan2(y, x) if x != 0.0 else np.arctan(y))


def clamp(x: Number, minVal: Number, maxVal: Number) -> Number:
    """Working NumPy implementation for testing"""
    return np.clip(x, minVal, maxVal)


def cos(x: float) -> float:
    """Working NumPy implementation for testing"""
    return float(np.cos(x))


def dot(x: np.ndarray, y: np.ndarray) -> float:
    """Working NumPy implementation for testing"""
    return float(np.dot(x, y))


def length(x: np.ndarray) -> float:
    """Working NumPy implementation for testing"""
    return float(np.linalg.norm(x))


def max(x: Number, y: Number) -> Number:
    """Working NumPy implementation for testing"""
    return np.maximum(x, y)


def min(x: Number, y: Number) -> Number:
    """Working NumPy implementation for testing"""
    return np.minimum(x, y)


def mix(x: Number, y: Number, a: Number) -> Number:
    """Working NumPy implementation for testing"""
    return x * (1.0 - a) + y * a


def normalize(x: np.ndarray) -> np.ndarray:
    """Working NumPy implementation for testing"""
    return x / np.linalg.norm(x)


def sin(x: float) -> float:
    """Working NumPy implementation for testing"""
    return float(np.sin(x))


def smoothstep(edge0: Number, edge1: Number, x: Number) -> Number:
    """Working NumPy implementation for testing"""
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
