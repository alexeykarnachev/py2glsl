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


def test_vertex_shader_function_params():
    """Test vs_uv usage in function parameters"""

    def shader(vs_uv: vec2) -> vec4:
        def transform(p: vec2) -> vec2:
            return p * 2.0 - 1.0

        pos = transform(vs_uv)
        return vec4(pos, 0.0, 1.0)

    result = py2glsl(shader)
    assert "vec2 transform(vec2 p)" in result.fragment_source
    assert "return (p * 2.0) - 1.0;" in result.fragment_source  # Fixed parentheses


def test_vertex_shader_input_attributes():
    """Test vertex shader input attribute declarations"""

    def shader(vs_uv: vec2) -> vec4:
        return vec4(1.0)

    result = py2glsl(shader)
    # Verify vertex shader has proper input declarations
    assert "layout(location = 0) in vec2 in_pos;" in result.vertex_source
    assert "layout(location = 1) in vec2 in_uv;" in result.vertex_source
    assert "out vec2 vs_uv;" in result.vertex_source


def test_vertex_shader_interface():
    """Test vertex shader interface generation"""

    def shader(vs_uv: vec2) -> vec4:
        return vec4(1.0, 0.0, 0.0, 1.0)

    result = py2glsl(shader)
    assert "in vec2 vs_uv;" in result.fragment_source
    assert "out vec4 fs_color;" in result.fragment_source


def test_vertex_shader_uv_usage():
    """Test proper vs_uv usage in shader"""

    def shader(vs_uv: vec2) -> vec4:
        return vec4(vs_uv.x, vs_uv.y, 0.0, 1.0)

    result = py2glsl(shader)
    assert "vs_uv.x" in result.fragment_source
    assert "vs_uv.y" in result.fragment_source


def test_vertex_shader_swizzle():
    """Test vs_uv swizzling operations"""

    def shader(vs_uv: vec2) -> vec4:
        xy = vs_uv.xy
        yx = vs_uv.yx
        return vec4(xy.x, xy.y, yx.x, yx.y)

    result = py2glsl(shader)
    assert "vec2 xy;" in result.fragment_source
    assert "vec2 yx;" in result.fragment_source
    assert "xy = vs_uv.xy;" in result.fragment_source
    assert "yx = vs_uv.yx;" in result.fragment_source


def test_vertex_shader_precision():
    """Test precision handling with vs_uv coordinates"""

    def shader(vs_uv: vec2) -> vec4:
        uv = vs_uv * 2.0 - 1.0  # Convert [0,1] to [-1,1]
        return vec4(uv, 0.0, 1.0)

    result = py2glsl(shader)
    assert "vec2 uv;" in result.fragment_source
    assert "uv = (vs_uv * 2.0) - 1.0;" in result.fragment_source


def test_vertex_shader_with_uniforms():
    """Test vs_uv interaction with uniforms"""

    def shader(vs_uv: vec2, *, u_scale: float) -> vec4:
        pos = vs_uv * u_scale
        return vec4(pos, 0.0, 1.0)

    result = py2glsl(shader)
    assert "uniform float u_scale;" in result.fragment_source
    assert "vec2 pos;" in result.fragment_source
    assert "pos = vs_uv * u_scale;" in result.fragment_source


def test_vertex_shader_complex_usage():
    """Test complex vs_uv manipulations"""

    def shader(vs_uv: vec2) -> vec4:
        def polar(p: vec2) -> vec2:
            return vec2(length(p), atan(p.y, p.x))

        center = vs_uv * 2.0 - 1.0
        polar_coords = polar(center)
        return vec4(polar_coords, 0.0, 1.0)

    result = py2glsl(shader)
    assert "vec2 center;" in result.fragment_source
    assert "vec2 polar_coords;" in result.fragment_source
    assert "center = (vs_uv * 2.0) - 1.0;" in result.fragment_source


def test_vertex_shader_resolution():
    """Test vs_uv with resolution uniform"""

    def shader(vs_uv: vec2, *, u_resolution: vec2) -> vec4:
        aspect = u_resolution.x / u_resolution.y
        pos = vs_uv
        pos.x *= aspect
        return vec4(pos, 0.0, 1.0)

    result = py2glsl(shader)
    assert "uniform vec2 u_resolution;" in result.fragment_source
    assert "float aspect;" in result.fragment_source
    assert "aspect = u_resolution.x / u_resolution.y;" in result.fragment_source


def test_vertex_shader_time():
    """Test vs_uv with time-based animation"""

    def shader(vs_uv: vec2, *, u_time: float) -> vec4:
        pos = vs_uv * 2.0 - 1.0
        angle = u_time
        x = pos.x * cos(angle) - pos.y * sin(angle)
        y = pos.x * sin(angle) + pos.y * cos(angle)
        return vec4(x, y, 0.0, 1.0)

    result = py2glsl(shader)
    assert "uniform float u_time;" in result.fragment_source
    assert "vec2 pos;" in result.fragment_source
    assert "pos = (vs_uv * 2.0) - 1.0;" in result.fragment_source


def test_vertex_shader_mouse():
    """Test vs_uv with mouse interaction"""

    def shader(vs_uv: vec2, *, u_mouse: vec2) -> vec4:
        dist = length(vs_uv - u_mouse)
        glow = 0.1 / (dist + 0.1)
        return vec4(glow, glow, glow, 1.0)

    result = py2glsl(shader)
    assert "uniform vec2 u_mouse;" in result.fragment_source
    assert "float dist;" in result.fragment_source
    assert "dist = length(vs_uv - u_mouse);" in result.fragment_source


def test_vertex_shader_coordinate_mapping():
    """Test vertex shader coordinate mapping"""

    def shader(vs_uv: vec2) -> vec4:
        # Map [0,1] to [-1,1]
        pos = vs_uv * 2.0 - 1.0
        return vec4(pos, 0.0, 1.0)

    result = py2glsl(shader)
    # Verify coordinate transformation
    assert "vec2 pos;" in result.fragment_source
    assert "pos = (vs_uv * 2.0) - 1.0;" in result.fragment_source
