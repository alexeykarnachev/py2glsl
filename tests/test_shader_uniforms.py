import asyncio
from typing import Any

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
    vec3,
    vec4,
)
from py2glsl.builtins import length, normalize, sin, smoothstep
from py2glsl.types import Vec2, Vec3, Vec4, vec2, vec3, vec4


def test_uniforms() -> None:
    def uniform_shader(vs_uv: vec2, *, color: float) -> vec4:
        return vec4(color, 0.0, 0.0, 1.0)

    result = py2glsl(uniform_shader)
    assert "uniform float color;" in result.fragment_source


def test_multiple_uniforms():
    """Test multiple uniforms with different types"""

    def shader(vs_uv: vec2, *, scale: float, offset: vec2, color: vec4) -> vec4:
        return color * vec4(vs_uv * scale + offset, 0.0, 1.0)

    result = py2glsl(shader)
    assert "uniform float scale;" in result.fragment_source
    assert "uniform vec2 offset;" in result.fragment_source
    assert "uniform vec4 color;" in result.fragment_source


def test_complex_uniforms() -> None:
    def shader(
        vs_uv: vec2,
        *,
        u_time: float,
        u_resolution: vec2,
        u_mouse: vec2,
        u_color1: vec4,
        u_color2: vec4,
        u_params: vec4,
    ) -> vec4:
        aspect = u_resolution.x / u_resolution.y
        p = vs_uv * 2.0 - 1.0
        p.x *= aspect

        mouse = u_mouse * 2.0 - 1.0
        mouse.x *= aspect

        d = length(p - mouse)
        glow = 0.02 / d

        color = mix(
            u_color1,
            u_color2,
            smoothstep(
                u_params.x, u_params.y, sin(d * u_params.z + u_time * u_params.w)
            ),
        )

        return color * glow

    result = py2glsl(shader)
    assert "uniform float u_time;" in result.fragment_source
    assert "uniform vec2 u_resolution;" in result.fragment_source
    assert "uniform vec2 u_mouse;" in result.fragment_source
    assert "uniform vec4 u_color1;" in result.fragment_source
    assert "uniform vec4 u_color2;" in result.fragment_source
    assert "uniform vec4 u_params;" in result.fragment_source
    assert "float aspect = u_resolution.x / u_resolution.y;" in result.fragment_source


def test_uniform_declaration_and_usage():
    """Test uniform declaration and usage in shaders"""

    def shader(vs_uv: vec2, *, u_time: float, u_resolution: vec2) -> vec4:
        return vec4(u_time, u_resolution.x, u_resolution.y, 1.0)

    result = py2glsl(shader)
    # Verify uniform declarations
    assert "uniform float u_time;" in result.fragment_source
    assert "uniform vec2 u_resolution;" in result.fragment_source


def test_uniform_type_validation():
    """Test uniform type validation"""

    def shader(vs_uv: vec2, *, u_float: float, u_vec2: vec2, u_vec4: vec4) -> vec4:
        return u_vec4

    result = py2glsl(shader)
    # Verify uniform type declarations
    assert "uniform float u_float;" in result.fragment_source
    assert "uniform vec2 u_vec2;" in result.fragment_source
    assert "uniform vec4 u_vec4;" in result.fragment_source


def test_integer_uniform_handling():
    """Test integer uniform type preservation"""

    def shader(vs_uv: vec2, *, frame: int) -> vec4:
        return vec4(frame, 0.0, 0.0, 1.0)

    result = py2glsl(shader)
    assert "uniform int frame;" in result.fragment_source
    assert "vec4(frame, 0.0, 0.0, 1.0)" in result.fragment_source


def test_mixed_uniform_types():
    """Test handling of mixed integer and float uniforms"""

    def shader(vs_uv: vec2, *, frame: int, time: float) -> vec4:
        return vec4(frame, time, 0.0, 1.0)

    result = py2glsl(shader)
    assert "uniform int frame;" in result.fragment_source
    assert "uniform float time;" in result.fragment_source


def test_integer_uniform_array():
    """Test integer uniform arrays"""

    def shader(vs_uv: vec2, *, frames: vec3) -> vec4:
        return vec4(frames.x, frames.y, frames.z, 1.0)

    result = py2glsl(shader)
    assert "uniform vec3 frames;" in result.fragment_source
