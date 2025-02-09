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


def test_type_inference() -> None:
    def type_shader(vs_uv: vec2) -> vec4:
        a = 1.0
        b = vs_uv
        c = vec4(1.0, 2.0, 3.0, 4.0)
        return c

    result = py2glsl(type_shader)
    assert "float a = 1.0;" in result.fragment_source
    assert "vec2 b = vs_uv;" in result.fragment_source
    assert "vec4 c = vec4(1.0, 2.0, 3.0, 4.0);" in result.fragment_source


def test_vector_operations_types():
    """Test vector operation type preservation"""

    def shader(vs_uv: vec2) -> vec4:
        v2 = vec2(1.0, 2.0)
        v3 = vec3(v2, 3.0)
        v4 = vec4(v3, 1.0)
        scaled = v2 * 2.0  # Should stay vec2
        added = v2 + v2  # Should stay vec2
        return v4

    result = py2glsl(shader)
    assert "vec2 scaled = v2 * 2.0;" in result.fragment_source
    assert "vec2 added = v2 + v2;" in result.fragment_source


def test_math_function_types():
    """Test math function type preservation"""

    def shader(vs_uv: vec2) -> vec4:
        angle = atan(vs_uv.y, vs_uv.x)  # float
        s = sin(angle)  # float
        c = cos(angle)  # float
        return vec4(s, c, 0.0, 1.0)

    result = py2glsl(shader)
    assert "float angle = atan(vs_uv.y, vs_uv.x);" in result.fragment_source
    assert "float s = sin(angle);" in result.fragment_source
    assert "float c = cos(angle);" in result.fragment_source


def test_nested_function_types():
    """Test nested function type preservation"""

    def shader(vs_uv: vec2) -> vec4:
        def circle_sdf(p: vec2, r: float) -> float:
            return length(p) - r

        def smooth_min(a: float, b: float, k: float) -> float:
            h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
            return mix(b, a, h) - k * h * (1.0 - h)

        d = circle_sdf(vs_uv, 0.5)
        s = smooth_min(d, 0.0, 0.1)
        return vec4(s)

    result = py2glsl(shader)
    assert "float circle_sdf(vec2 p, float r)" in result.fragment_source
    assert "float smooth_min(float a, float b, float k)" in result.fragment_source
    assert "float d = circle_sdf(vs_uv, 0.5);" in result.fragment_source


def test_type_inference_complex():
    """Test complex type inference scenarios"""

    def shader(vs_uv: vec2, *, scale: float) -> vec4:
        # Type should be inferred from operations
        a = vs_uv * scale  # vec2
        b = vs_uv.x * scale  # float
        c = normalize(a)  # vec2
        d = length(a)  # float
        return vec4(c, b, d)

    result = py2glsl(shader)
    assert "vec2 a = vs_uv * scale;" in result.fragment_source
    assert "float b = vs_uv.x * scale;" in result.fragment_source
    assert "vec2 c = normalize(a);" in result.fragment_source
    assert "float d = length(a);" in result.fragment_source


def test_type_inference_consistency():
    """Test consistent type inference"""

    def shader(vs_uv: vec2, *, u_dir: vec2) -> vec4:
        n = normalize(u_dir)  # Should stay vec2
        d = dot(n, vs_uv)  # Should be float
        return vec4(d)

    result = py2glsl(shader)

    assert "vec2 n = normalize(u_dir)" in result.fragment_source
    assert "float d = dot(n, vs_uv)" in result.fragment_source


def test_builtin_function_types():
    """Test built-in function type inference"""

    def shader(vs_uv: vec2) -> vec4:
        d = length(vs_uv)  # float
        n = normalize(vs_uv)  # vec2
        m = mix(vec3(1), vec3(0), 0.5)  # vec3
        return vec4(d, n, 1.0)

    result = py2glsl(shader)
    assert "float d = length(vs_uv);" in result.fragment_source
    assert "vec2 n = normalize(vs_uv);" in result.fragment_source
    assert "vec3 m = mix(vec3(1.0), vec3(0.0), 0.5);" in result.fragment_source
