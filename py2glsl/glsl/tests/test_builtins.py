import numpy as np
import pytest

from py2glsl.glsl.builtins import (
    abs,
    acos,
    asin,
    atan,
    atan2,
    cos,
    cross,
    degrees,
    exp,
    floor,
    length,
    mix,
    normalize,
    pow,
    radians,
    reflect,
    sin,
    smoothstep,
    sqrt,
    tan,
    transpose,
)
from py2glsl.glsl.types import mat4, vec2, vec3


def test_template_metadata():
    assert sin.__glsl_template__ == "sin({x})"
    assert mix.__glsl_template__ == "mix({x}, {y}, {a})"
    assert cross.__glsl_template__ == "cross({x}, {y})"


@pytest.mark.parametrize(
    "func, args, expected",
    [
        (radians, (90.0,), np.pi / 2),
        (degrees, (np.pi / 2,), 90.0),
        (sin, (np.pi / 2,), 1.0),
        (cos, (0.0,), 1.0),
        (tan, (0.0,), 0.0),
        (asin, (1.0,), np.pi / 2),
        (acos, (1.0,), 0.0),
        (atan, (1.0,), np.pi / 4),
        (atan2, (1.0, 1.0), np.pi / 4),
        (pow, (2.0, 3.0), 8.0),
        (exp, (1.0,), np.e),
        (sqrt, (4.0,), 2.0),
        (abs, (-1.5,), 1.5),
        (floor, (1.9,), 1.0),
        (mix, (0.0, 1.0, 0.5), 0.5),
        (smoothstep, (0.0, 1.0, 0.5), 0.5),
    ],
)
def test_scalar_functions(func, args, expected):
    result = func(*args)
    assert np.isclose(result, expected, rtol=1e-5)


def test_vector_operations():
    v2 = vec2(3.0, 4.0)
    assert np.isclose(length(v2), 5.0)

    v3 = vec3(1.0, 0.0, 0.0)
    assert np.allclose(normalize(v3).data, [1.0, 0.0, 0.0])

    a = vec2(0.0, 0.0)
    b = vec2(1.0, 2.0)
    assert np.allclose(mix(a, b, 0.5).data, [0.5, 1.0])


def test_matrix_operations():
    m = mat4(np.eye(4))
    transposed = transpose(m)
    assert np.allclose(transposed.data, np.eye(4))


def test_geometric_functions():
    a = vec3(1.0, 0.0, 0.0)
    b = vec3(0.0, 1.0, 0.0)
    assert np.allclose(cross(a, b).data, [0.0, 0.0, 1.0])

    i = vec3(1.0, -1.0, 0.0)
    n = vec3(0.0, 1.0, 0.0)
    assert np.allclose(reflect(i, n).data, [1.0, 1.0, 0.0])
