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


def test_control_flow() -> None:
    def if_shader(vs_uv: vec2, *, threshold: float) -> vec4:
        if threshold > 0.5:
            return vec4(1.0, 0.0, 0.0, 1.0)
        return vec4(0.0, 1.0, 0.0, 1.0)

    result = py2glsl(if_shader)
    assert "if (threshold > 0.5)" in result.fragment_source


def test_nested_control_flow() -> None:
    def shader(vs_uv: vec2, *, u_params: vec4) -> vec4:
        result = vec4(0.0)
        p = vs_uv * 2.0 - 1.0

        if length(p) < u_params.x:
            if p.x > 0.0:
                if p.y > 0.0:
                    result = vec4(1.0, 0.0, 0.0, 1.0)
                else:
                    result = vec4(0.0, 1.0, 0.0, 1.0)
            else:
                result = vec4(0.0, 0.0, 1.0, 1.0)
        else:
            result = mix(vec4(0.5), vec4(u_params.y, u_params.z, u_params.w, 1.0), 0.5)

        return result

    result = py2glsl(shader)
    # Check variable declarations
    assert "vec4 result;" in result.fragment_source
    assert "vec2 p;" in result.fragment_source
    # Check control flow
    assert "if (length(p) < u_params.x)" in result.fragment_source
    assert "if (p.x > 0.0)" in result.fragment_source
    assert "if (p.y > 0.0)" in result.fragment_source
    assert (
        "mix(vec4(0.5), vec4(u_params.y, u_params.z, u_params.w, 1.0), 0.5)"
        in result.fragment_source
    )


def test_nested_if_else():
    """Test nested if-else structures"""

    def shader(vs_uv: vec2, *, x: float, y: float) -> vec4:
        if x > 0.0:
            if y > 0.0:
                return vec4(1.0, 0.0, 0.0, 1.0)
            else:
                return vec4(0.0, 1.0, 0.0, 1.0)
        return vec4(0.0, 0.0, 1.0, 1.0)

    result = py2glsl(shader)
    shader_body = result.fragment_source[result.fragment_source.find("shader(") :]
    assert shader_body.count("if") == 2
    assert shader_body.count("else") == 1


def test_simple_for_loop():
    """Test basic for loop with integer bounds"""

    def shader(vs_uv: vec2) -> vec4:
        x = 0.0
        for i in range(5):
            x += 1.0
        return vec4(x)

    result = py2glsl(shader)
    # Check declarations
    assert "float x;" in result.fragment_source
    assert "int i;" in result.fragment_source
    # Check loop
    assert "for (int i = 0; i < 5; i++)" in result.fragment_source


def test_nested_for_loops():
    """Test nested for loops with integer bounds"""

    def shader(vs_uv: vec2) -> vec4:
        x = 0.0
        for i in range(3):
            for j in range(2):
                x += 1.0
        return vec4(x)

    result = py2glsl(shader)
    # Check declarations
    assert "float x;" in result.fragment_source
    assert "int i;" in result.fragment_source
    assert "int j;" in result.fragment_source
    # Check loops
    assert "for (int i = 0; i < 3; i++)" in result.fragment_source
    assert "for (int j = 0; j < 2; j++)" in result.fragment_source


def test_for_loop_with_range_start():
    """Test for loop with start and end range"""

    def shader(vs_uv: vec2) -> vec4:
        x = 0.0
        for i in range(1, 4):
            x += 1.0
        return vec4(x)

    result = py2glsl(shader)
    # Check declarations
    assert "float x;" in result.fragment_source
    assert "int i;" in result.fragment_source
    # Check loop
    assert "for (int i = 1; i < 4; i++)" in result.fragment_source


def test_loop_bounds_integer():
    """Test that loop bounds remain integers"""

    def shader(vs_uv: vec2) -> vec4:
        x = 0.0
        for i in range(5):  # Simple integer bound
            x += 1.0
        return vec4(x)

    result = py2glsl(shader)
    # Check declarations
    assert "float x;" in result.fragment_source
    assert "int i;" in result.fragment_source
    # Check loop
    assert "for (int i = 0; i < 5; i++)" in result.fragment_source


def test_loop_bounds_float_error():
    """Test that float loop bounds raise error"""

    def shader(vs_uv: vec2) -> vec4:
        x = 0.0
        for i in range(5.0):  # Should raise error
            x += 1.0
        return vec4(x)

    with pytest.raises(
        TypeError, match="Invalid shader function: Loop bounds must be integers"
    ):
        py2glsl(shader)


def test_loop_bounds_expression():
    """Test loop bounds with expressions"""

    def shader(vs_uv: vec2) -> vec4:
        x = 0.0
        count = 3
        for i in range(count + 2):  # Expression that evaluates to integer
            x += 1.0
        return vec4(x)

    result = py2glsl(shader)
    # Check declarations
    assert "float x;" in result.fragment_source
    assert "int count;" in result.fragment_source
    assert "int i;" in result.fragment_source
    # Check loop
    assert "for (int i = 0; i < count + 2; i++)" in result.fragment_source


def test_integer_loop_counter():
    """Test integer loop counters"""

    def shader(vs_uv: vec2, *, count: int) -> vec4:
        x = 0.0
        for i in range(count):
            x += 1.0
        return vec4(x)

    result = py2glsl(shader)
    # Check declarations
    assert "float x;" in result.fragment_source
    assert "int i;" in result.fragment_source
    # Check loop
    assert "for (int i = 0; i < count; i++)" in result.fragment_source
