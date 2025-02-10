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


def test_nested_functions() -> None:
    def func_shader(vs_uv: vec2) -> vec4:
        def double(x: float) -> float:
            return x * 2.0

        val = double(0.5)
        return vec4(val, 0.0, 0.0, 1.0)

    result = py2glsl(func_shader)
    assert "float double(float x)" in result.fragment_source
    assert "return x * 2.0;" in result.fragment_source


def test_multiple_nested_functions():
    """Test multiple nested function definitions"""

    def shader(vs_uv: vec2) -> vec4:
        def f1(x: float) -> float:
            return x * 2.0

        def f2(x: float) -> float:
            return f1(x) + 1.0

        val = f2(0.5)
        return vec4(val, 0.0, 0.0, 1.0)

    result = py2glsl(shader)
    assert "float f1(float x)" in result.fragment_source
    assert "float f2(float x)" in result.fragment_source
    assert "return f1(x) + 1.0;" in result.fragment_source


def test_function_return_type():
    """Test function return type preservation"""

    def shader(vs_uv: vec2) -> vec4:
        def get_normal(p: vec2) -> vec2:
            return normalize(p)

        n = get_normal(vs_uv)
        return vec4(n, 0.0, 1.0)

    result = py2glsl(shader)
    assert "vec2 get_normal(vec2 p)" in result.fragment_source
    assert "vec2 n;" in result.fragment_source
    assert "n = get_normal(vs_uv);" in result.fragment_source


def test_code_formatting():
    """Test consistent code formatting"""

    def shader(vs_uv: vec2) -> vec4:
        if length(vs_uv) > 1.0:
            return vec4(1.0)
        return vec4(0.0)

    result = py2glsl(shader)
    # Check for consistent bracing style
    assert "{\n" in result.fragment_source
    assert "if (length(vs_uv) > 1.0)\n" in result.fragment_source


def test_function_calls_chain():
    """Test chained function calls"""

    def shader(vs_uv: vec2) -> vec4:
        return vec4(normalize(abs(vs_uv * 2.0 - 1.0)), 0.0, 1.0)

    result = py2glsl(shader)
    # Accept both forms of parentheses
    assert any(
        expr in result.fragment_source
        for expr in [
            "normalize(abs(vs_uv * 2.0 - 1.0))",
            "normalize(abs((vs_uv * 2.0) - 1.0))",
        ]
    )


def test_builtin_functions():
    def builtin_shader(vs_uv: vec2) -> vec4:
        l = length(vs_uv)
        n = normalize(vs_uv)
        return vec4(n, l, 1.0)

    result = py2glsl(builtin_shader)
    assert "float l;" in result.fragment_source
    assert "l = length(vs_uv);" in result.fragment_source
    assert "vec2 n;" in result.fragment_source
    assert "n = normalize(vs_uv);" in result.fragment_source


def test_builtin_functions_chain():
    """Test chained built-in function calls"""

    def shader(vs_uv: vec2) -> vec4:
        v = normalize(abs(sin(vs_uv * 6.28318530718)))
        l = length(clamp(v, 0.0, 1.0))
        return vec4(v, l, 1.0)

    result = py2glsl(shader)
    assert "normalize(abs(sin(vs_uv * 6.28318530718)))" in result.fragment_source
    assert "length(clamp(v, 0.0, 1.0))" in result.fragment_source
