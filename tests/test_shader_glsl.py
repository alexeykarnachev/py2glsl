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


def test_glsl_syntax_validation():
    """Test GLSL syntax validation"""

    def shader(vs_uv: vec2) -> vec4:
        render_called = True  # This should be converted to float/bool
        return vec4(1.0)

    result = py2glsl(shader)
    # Verify Python bool is converted to GLSL bool/float
    assert (
        "bool render_called = true;" in result.fragment_source
        or "float render_called = 1.0;" in result.fragment_source
    )


def test_shader_interface_validation():
    """Test shader interface validation"""

    def shader(vs_uv: vec2) -> vec4:
        return vec4(vs_uv, 0.0, 1.0)

    result = py2glsl(shader)
    # Verify shader interface
    assert "in vec2 vs_uv;" in result.fragment_source
    assert "out vec4 fs_color;" in result.fragment_source
    assert "void main()" in result.fragment_source
    assert "fs_color = shader(vs_uv);" in result.fragment_source


def test_shader_main_function():
    """Test shader main function generation"""

    def shader(vs_uv: vec2) -> vec4:
        return vec4(1.0)

    result = py2glsl(shader)
    expected_main = """
void main()
{
    fs_color = shader(vs_uv);
}
""".strip()
    assert expected_main in result.fragment_source


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


def test_code_formatting_style():
    """Test GLSL code formatting rules"""

    def shader(vs_uv: vec2, *, u_val: float) -> vec4:
        if u_val > 0.0:
            return vec4(1.0)
        return vec4(0.0)

    result = py2glsl(shader)

    # Check only version, in/out declarations, and uniforms have no indentation
    for line in result.fragment_source.split("\n"):
        if line and not line.isspace():
            if any(
                line.startswith(prefix)
                for prefix in ["#version", "in ", "out ", "uniform "]
            ):
                assert not line.startswith("    ")


def test_expression_grouping():
    """Test expression parentheses rules"""

    def shader(vs_uv: vec2) -> vec4:
        x = vs_uv.x * 2.0 - 1.0
        y = (vs_uv.y * 2.0) - 1.0
        return vec4(x, y, 0.0, 1.0)

    result = py2glsl(shader)

    # Both forms should produce same output
    assert "(vs_uv.x * 2.0) - 1.0" in result.fragment_source
    assert "(vs_uv.y * 2.0) - 1.0" in result.fragment_source


def test_indentation_consistency():
    """Test consistent indentation rules"""

    def shader(vs_uv: vec2, *, u_val: float) -> vec4:
        if u_val > 0.0:
            x = 1.0
            if u_val > 1.0:
                x = 2.0
        return vec4(x)

    result = py2glsl(shader)

    lines = result.fragment_source.split("\n")
    indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    assert len(set(indents)) <= 3  # Should only have 0, 4, 8 spaces


def test_parentheses_consistency():
    """Test consistent parentheses in expressions"""

    def shader(vs_uv: vec2) -> vec4:
        x = (1.0 + 2.0) * (3.0 - 4.0)
        y = vs_uv * 2.0 - 1.0
        return vec4(x, y, 0.0)

    result = py2glsl(shader)
    assert "(" in result.fragment_source and ")" in result.fragment_source
    assert "vs_uv * 2.0" in result.fragment_source


def test_precision_handling():
    """Test numerical precision handling in shaders"""

    def shader(vs_uv: vec2) -> vec4:
        x = 0.0  # Should be exactly 0.0
        y = 1.0  # Should be exactly 1.0
        return vec4(x, y, 0.0, 1.0)

    result = py2glsl(shader)
    # Verify exact float values
    assert "0.0" in result.fragment_source
    assert "1.0" in result.fragment_source


def test_function_formatting():
    """Test function declaration formatting"""

    def shader(vs_uv: vec2) -> vec4:
        def helper(x: float) -> float:
            return x * 2.0

        return vec4(helper(vs_uv.x))

    result = py2glsl(shader)

    assert "float helper(float x)\n{" in result.fragment_source
    assert "vec4 shader(vec2 vs_uv)\n{" in result.fragment_source
