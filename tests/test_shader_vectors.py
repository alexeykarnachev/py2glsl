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


def test_vector_operations() -> None:
    def vec_shader(vs_uv: vec2) -> vec4:
        v = vs_uv * 2.0
        return vec4(v, 0.0, 1.0)

    result = py2glsl(vec_shader)
    assert "vec2 v = vs_uv * 2.0;" in result.fragment_source


def test_vector_constructors():
    """Test different vector construction patterns"""

    def shader(vs_uv: vec2) -> vec4:
        v2a = vec2(1.0, 2.0)
        v2b = vec2(v2a.x)  # same value
        v3a = vec3(v2a, 3.0)
        v3b = vec3(1.0)  # all same
        v4a = vec4(v3a, 1.0)
        v4b = vec4(v2a, v2b)
        return v4a + v4b

    result = py2glsl(shader)
    assert "vec2 v2a = vec2(1.0, 2.0);" in result.fragment_source
    assert "vec2 v2b = vec2(v2a.x);" in result.fragment_source
    assert "vec3 v3a = vec3(v2a, 3.0);" in result.fragment_source
    assert "vec3 v3b = vec3(1.0);" in result.fragment_source
    assert "vec4 v4a = vec4(v3a, 1.0);" in result.fragment_source
    assert "vec4 v4b = vec4(v2a, v2b);" in result.fragment_source


def test_complex_swizzling():
    """Test complex swizzling operations"""

    def shader(vs_uv: vec2) -> vec4:
        v4 = vec4(vs_uv, 0.0, 1.0)
        xyz = v4.xyz
        zyx = v4.zyx
        return vec4(xyz.xy, zyx.xy)

    result = py2glsl(shader)
    assert "vec4 v4 = vec4(vs_uv, 0.0, 1.0);" in result.fragment_source
    assert "vec3 xyz = v4.xyz;" in result.fragment_source
    assert "vec3 zyx = v4.zyx;" in result.fragment_source


def test_swizzling() -> None:
    def swizzle_shader(vs_uv: vec2) -> vec4:
        xy = vs_uv.xy
        yx = vs_uv.yx
        return vec4(xy.x, xy.y, yx.x, yx.y)

    result = py2glsl(swizzle_shader)
    assert "vec2 xy = vs_uv.xy;" in result.fragment_source
    assert "vec2 yx = vs_uv.yx;" in result.fragment_source


def test_complex_vector_operations() -> None:
    def shader(vs_uv: vec2, *, u_params: vec4) -> vec4:
        def polar(p: vec2) -> vec2:
            return vec2(length(p), atan(p.y, p.x))

        def cartesian(p: vec2) -> vec2:
            return vec2(p.x * cos(p.y), p.x * sin(p.y))

        p = vs_uv * 2.0 - 1.0
        pol = polar(p)
        pol.y += u_params.z * pol.x + u_params.w
        p = cartesian(pol)

        return vec4(normalize(p), length(p), 1.0)

    result = py2glsl(shader)
    assert "vec2 polar(vec2 p)" in result.fragment_source
    assert "vec2 cartesian(vec2 p)" in result.fragment_source
    assert "pol.y += (u_params.z * pol.x) + u_params.w;" in result.fragment_source


def test_vector_swizzle_formatting():
    """Test vector swizzle formatting"""

    def shader(vs_uv: vec2) -> vec4:
        v = vec4(vs_uv.xy, 0.0, 1.0)
        return v.rgba

    result = py2glsl(shader)

    assert "vs_uv.xy" in result.fragment_source
    assert "v.rgba" in result.fragment_source


def test_vector_swizzle_type():
    """Test vector swizzling with correct type inference"""

    def shader(vs_uv: vec2) -> vec4:
        v4 = vec4(1.0, 2.0, 3.0, 4.0)
        rgb = v4.rgb  # Should be vec3
        xy = v4.xy  # Should be vec2
        x = v4.x  # Should be float
        return vec4(x, xy, 1.0)

    result = py2glsl(shader)
    assert "vec3 rgb = v4.rgb;" in result.fragment_source
    assert "vec2 xy = v4.xy;" in result.fragment_source
    assert "float x = v4.x;" in result.fragment_source


def test_vector_type_hoisting():
    """Test hoisting of vector type variables"""

    def shader(vs_uv: vec2) -> vec4:
        if vs_uv.x > 0.5:
            pos = vec2(1.0, 2.0)
        else:
            pos = vec2(0.0, 0.0)
        return vec4(pos, 0.0, 1.0)

    result = py2glsl(shader)
    assert "vec2 pos;" in result.fragment_source
    assert "pos = vec2(1.0, 2.0);" in result.fragment_source
    assert "pos = vec2(0.0, 0.0);" in result.fragment_source
