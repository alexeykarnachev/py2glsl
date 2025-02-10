import asyncio

import glfw
import imageio.v3 as iio
import numpy as np
import pytest
from PIL import Image

from py2glsl import (
    animate,
    py2glsl,
    render_array,
    render_gif,
    render_image,
    render_video,
    vec2,
    vec4,
)
from py2glsl.builtins import length, normalize, sin, smoothstep
from py2glsl.types import Vec2, Vec3, Vec4, vec2, vec3, vec4


def test_minimal_valid_shader() -> None:
    def minimal_shader(vs_uv: vec2) -> vec4:
        return vec4(1.0, 0.0, 0.0, 1.0)

    result = py2glsl(minimal_shader)
    assert "vec4 shader(vec2 vs_uv)" in result.fragment_source
    assert "return vec4(1.0, 0.0, 0.0, 1.0)" in result.fragment_source


def test_simple_uniform():
    """Test shader with single uniform"""

    def uniform_shader(vs_uv: vec2, *, color: float) -> vec4:
        return vec4(color, 0.0, 0.0, 1.0)

    result = py2glsl(uniform_shader)
    assert "uniform float color;" in result.fragment_source


def test_simple_variable():
    """Test local variable declaration"""

    def var_shader(vs_uv: vec2) -> vec4:
        x = 1.0
        return vec4(x, 0.0, 0.0, 1.0)

    result = py2glsl(var_shader)
    assert "float x;" in result.fragment_source
    assert "x = 1.0;" in result.fragment_source


def test_simple_arithmetic():
    """Test basic arithmetic operations"""

    def math_shader(vs_uv: vec2) -> vec4:
        x = 1.0 + 2.0
        y = 3.0 * 4.0
        return vec4(x, y, 0.0, 1.0)

    result = py2glsl(math_shader)
    assert "float x;" in result.fragment_source
    assert "x = 1.0 + 2.0;" in result.fragment_source
    assert "float y;" in result.fragment_source
    assert "y = 3.0 * 4.0;" in result.fragment_source


def test_simple_vector_ops():
    """Test basic vector operations"""

    def vec_shader(vs_uv: vec2) -> vec4:
        v = vs_uv * 2.0
        return vec4(v, 0.0, 1.0)

    result = py2glsl(vec_shader)
    assert "vec2 v;" in result.fragment_source
    assert "v = vs_uv * 2.0;" in result.fragment_source


def test_simple_if():
    """Test simple if statement"""

    def if_shader(vs_uv: vec2, *, threshold: float) -> vec4:
        if threshold > 0.5:
            return vec4(1.0, 0.0, 0.0, 1.0)
        return vec4(0.0, 1.0, 0.0, 1.0)

    result = py2glsl(if_shader)
    assert "if (threshold > 0.5)" in result.fragment_source


def test_simple_function():
    """Test simple nested function"""

    def func_shader(vs_uv: vec2) -> vec4:
        def double(x: float) -> float:
            return x * 2.0

        val = double(0.5)
        return vec4(val, 0.0, 0.0, 1.0)

    result = py2glsl(func_shader)
    assert "float double(float x)" in result.fragment_source
    assert "return x * 2.0;" in result.fragment_source


def test_type_inference_simple():
    """Test basic type inference"""

    def type_shader(vs_uv: vec2) -> vec4:
        a = 1.0  # float
        b = vs_uv  # vec2
        c = vec4(1.0, 2.0, 3.0, 4.0)  # vec4
        return c

    result = py2glsl(type_shader)
    assert "float a;" in result.fragment_source
    assert "a = 1.0;" in result.fragment_source
    assert "vec2 b;" in result.fragment_source
    assert "b = vs_uv;" in result.fragment_source
    assert "vec4 c;" in result.fragment_source
    assert "c = vec4(1.0, 2.0, 3.0, 4.0);" in result.fragment_source


def test_swizzle_simple():
    """Test basic swizzling operations"""

    def swizzle_shader(vs_uv: vec2) -> vec4:
        xy = vs_uv.xy
        yx = vs_uv.yx
        return vec4(xy.x, xy.y, yx.x, yx.y)

    result = py2glsl(swizzle_shader)
    assert "vec2 xy;" in result.fragment_source
    assert "xy = vs_uv.xy;" in result.fragment_source
    assert "vec2 yx;" in result.fragment_source
    assert "yx = vs_uv.yx;" in result.fragment_source


def test_builtin_simple():
    """Test basic built-in function usage"""

    def builtin_shader(vs_uv: vec2) -> vec4:
        l = length(vs_uv)
        n = normalize(vs_uv)
        return vec4(n, l, 1.0)

    result = py2glsl(builtin_shader)
    assert "float l;" in result.fragment_source
    assert "l = length(vs_uv);" in result.fragment_source
    assert "vec2 n;" in result.fragment_source
    assert "n = normalize(vs_uv);" in result.fragment_source


def test_arithmetic():
    """Test arithmetic operations"""

    def math_shader(vs_uv: vec2) -> vec4:
        x = 1.0 + 2.0
        y = 3.0 * 4.0
        return vec4(x, y, 0.0, 1.0)

    result = py2glsl(math_shader)
    assert "float x;" in result.fragment_source
    assert "x = 1.0 + 2.0;" in result.fragment_source
    assert "float y;" in result.fragment_source
    assert "y = 3.0 * 4.0;" in result.fragment_source
