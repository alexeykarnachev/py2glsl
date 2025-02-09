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


def test_error_first_argument_name():
    """Test error when first argument is not named vs_uv"""
    with pytest.raises(TypeError, match="First argument must be vs_uv"):

        def shader(pos: vec2, *, u_time: float) -> vec4:
            return vec4(1.0, 0.0, 0.0, 1.0)

        py2glsl(shader)


def test_error_messages():
    """Test specific error messages"""

    def shader1(pos: float) -> vec4:
        return vec4(1.0)

    def shader2(vs_uv: vec2) -> float:
        return 1.0

    def shader3(vs_uv: vec2, time: float) -> vec4:
        return vec4(1.0)

    with pytest.raises(TypeError, match="First argument must be vs_uv"):
        py2glsl(shader1)

    with pytest.raises(TypeError, match="Shader must return vec4"):
        py2glsl(shader2)

    with pytest.raises(TypeError, match="All arguments except vs_uv must be uniforms"):
        py2glsl(shader3)


def test_bool_conversion():
    """Test basic boolean literal conversion"""

    def shader(vs_uv: vec2) -> vec4:
        x = True
        y = False
        return vec4(1.0)

    result = py2glsl(shader)
    assert "bool x = true;" in result.fragment_source
    assert "bool y = false;" in result.fragment_source


def test_integer_arithmetic():
    """Test integer arithmetic operations"""

    def shader(vs_uv: vec2, *, frame: int) -> vec4:
        x = frame + 1
        y = frame * 2
        z = frame / 2  # Should convert to float
        return vec4(x, y, z, 1.0)

    result = py2glsl(shader)
    # In GLSL, operations with integers automatically promote to float when used with vec4
    assert "float x = frame + 1.0;" in result.fragment_source
    assert "float y = frame * 2.0;" in result.fragment_source
    assert "float z = frame / 2.0;" in result.fragment_source


def test_integer_vector_construction():
    """Test using integers in vector construction"""

    def shader(vs_uv: vec2, *, frame: int) -> vec4:
        return vec4(frame, frame + 1, frame * 2, 1)

    result = py2glsl(shader)
    # GLSL automatically converts integers to float when constructing vectors
    assert "vec4(frame, frame + 1.0, frame * 2.0, 1.0)" in result.fragment_source


def test_integer_comparison():
    """Test integer comparisons"""

    def shader(vs_uv: vec2, *, frame: int) -> vec4:
        if frame > 5:
            return vec4(1.0)
        return vec4(0.0)

    result = py2glsl(shader)
    # Integer comparisons with literals should keep the literal as int
    assert "if (frame > 5.0)" in result.fragment_source


def test_integer_function_params():
    """Test integer function parameters"""

    def shader(vs_uv: vec2, *, frame: int) -> vec4:
        def step(n: int) -> float:
            return float(n) / 10.0

        return vec4(step(frame))

    result = py2glsl(shader)
    assert "float step(int n)" in result.fragment_source
    assert "return float(n) / 10.0;" in result.fragment_source


def test_integer_type_conversion():
    """Test integer to float conversion"""

    def shader(vs_uv: vec2, *, frame: int) -> vec4:
        f = float(frame)  # Explicit conversion
        return vec4(f / 10.0)

    result = py2glsl(shader)
    assert "float f = float(frame);" in result.fragment_source


def test_error_handling() -> None:
    with pytest.raises(TypeError, match="First argument must be vs_uv"):

        def invalid_shader1(pos: float, *, u_time: float) -> vec4:
            return vec4(1.0, 0.0, 0.0, 1.0)

        py2glsl(invalid_shader1)

    with pytest.raises(TypeError, match="Shader must return vec4"):

        def invalid_shader2(vs_uv: vec2) -> float:
            return 1.0

        py2glsl(invalid_shader2)

    with pytest.raises(TypeError, match="All arguments except vs_uv must be uniforms"):

        def invalid_shader3(vs_uv: vec2, time: float) -> vec4:
            return vec4(1.0)

        py2glsl(invalid_shader3)
