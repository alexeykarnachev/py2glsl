"""Tests for rendering with different targets."""

import os

# Set environment variables before importing moderngl
os.environ["MODERNGL_FORCE_STANDALONE"] = "1"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import pytest

from py2glsl import ShaderContext
from py2glsl.builtins import cos, sin, vec4
from py2glsl.render import render_array
from py2glsl.transpiler import TargetType

# Skip all tests if NO_GPU=1
HAS_GPU = os.environ.get("NO_GPU", "0") != "1"
pytestmark = pytest.mark.gpu

TEST_SIZE = (320, 240)


def gradient_shader(ctx: ShaderContext) -> vec4:
    """A simple gradient shader."""
    return vec4(ctx.vs_uv.x, ctx.vs_uv.y, 0.5, 1.0)


def animated_shader(ctx: ShaderContext) -> vec4:
    """A time-dependent shader."""
    r = sin(ctx.vs_uv.x * 10.0 + ctx.u_time) * 0.5 + 0.5
    g = cos(ctx.vs_uv.y * 10.0 + ctx.u_time) * 0.5 + 0.5
    b = sin(ctx.u_time) * 0.5 + 0.5
    return vec4(r, g, b, 1.0)


@pytest.mark.gpu
def test_render_opengl46():
    """Test rendering with OpenGL 4.6 target."""
    if not HAS_GPU:
        pytest.skip("GPU not available")

    result = render_array(
        gradient_shader, size=TEST_SIZE, time=0.0, target=TargetType.OPENGL46
    )

    assert result.shape == (TEST_SIZE[1], TEST_SIZE[0], 4)
    assert result.dtype == np.uint8


@pytest.mark.gpu
def test_render_opengl33():
    """Test rendering with OpenGL 3.3 target."""
    if not HAS_GPU:
        pytest.skip("GPU not available")

    result = render_array(
        gradient_shader, size=TEST_SIZE, time=0.0, target=TargetType.OPENGL33
    )

    assert result.shape == (TEST_SIZE[1], TEST_SIZE[0], 4)


@pytest.mark.gpu
def test_render_animated():
    """Test rendering an animated shader at different times."""
    if not HAS_GPU:
        pytest.skip("GPU not available")

    result1 = render_array(animated_shader, size=TEST_SIZE, time=0.0)
    result2 = render_array(animated_shader, size=TEST_SIZE, time=1.0)

    # Results should be different at different times
    assert not np.array_equal(result1, result2)


@pytest.mark.gpu
def test_render_consistency():
    """Test that same input produces same output."""
    if not HAS_GPU:
        pytest.skip("GPU not available")

    result1 = render_array(gradient_shader, size=TEST_SIZE, time=0.5)
    result2 = render_array(gradient_shader, size=TEST_SIZE, time=0.5)

    # Same inputs should produce same outputs
    np.testing.assert_array_equal(result1, result2)
