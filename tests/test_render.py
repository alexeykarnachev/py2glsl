"""Tests for shader rendering functions."""

import os
import tempfile
from pathlib import Path

# Set environment variables before importing moderngl
os.environ["MODERNGL_FORCE_STANDALONE"] = "1"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import pytest
from PIL import Image

from py2glsl import ShaderContext
from py2glsl.builtins import cos, sin, vec4
from py2glsl.render import render_array, render_gif, render_image, render_video
from py2glsl.transpiler import TargetType, transpile

# Skip all tests if NO_GPU=1
HAS_GPU = os.environ.get("NO_GPU", "0") != "1"
pytestmark = pytest.mark.gpu

# Test configuration
TEST_SIZE = (208, 160)
TEST_DURATION = 0.5
TEST_FPS = 8


def simple_shader(ctx: ShaderContext) -> vec4:
    """A simple static gradient shader."""
    return vec4(ctx.vs_uv.x, ctx.vs_uv.y, 0.5, 1.0)


def time_shader(ctx: ShaderContext) -> vec4:
    """A time-dependent shader with a moving pattern."""
    r = sin(ctx.vs_uv.x * 10.0 + ctx.u_time) * 0.5 + 0.5
    g = cos(ctx.vs_uv.y * 10.0 + ctx.u_time) * 0.5 + 0.5
    b = sin(ctx.u_time) * 0.5 + 0.5
    return vec4(r, g, b, 1.0)


@pytest.mark.gpu
def test_render_array():
    """Test render_array produces correct output."""
    if not HAS_GPU:
        pytest.skip("GPU not available")

    result = render_array(simple_shader, size=TEST_SIZE, time=0.0)

    assert result.shape == (TEST_SIZE[1], TEST_SIZE[0], 4)
    assert result.dtype == np.uint8

    # Check that corners have expected gradient colors
    # Top-left (0, 0) should be dark (low r, low g)
    # Bottom-right should be bright (high r, high g)
    assert result[0, 0, 0] < 50  # low red at top-left
    assert result[-1, -1, 0] > 200  # high red at bottom-right


@pytest.mark.gpu
def test_render_image():
    """Test render_image produces a PIL Image."""
    if not HAS_GPU:
        pytest.skip("GPU not available")

    result = render_image(simple_shader, size=TEST_SIZE, time=0.0)

    assert isinstance(result, Image.Image)
    assert result.size == TEST_SIZE
    assert result.mode == "RGBA"


@pytest.mark.gpu
def test_render_image_save():
    """Test render_image can save to file."""
    if not HAS_GPU:
        pytest.skip("GPU not available")

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "test.png"
        result = render_image(
            simple_shader, size=TEST_SIZE, time=0.0, output_path=str(output_path)
        )

        assert output_path.exists()
        assert isinstance(result, Image.Image)


@pytest.mark.gpu
def test_render_gif():
    """Test render_gif produces frames."""
    if not HAS_GPU:
        pytest.skip("GPU not available")

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "test.gif"
        first_frame, frames = render_gif(
            time_shader,
            size=TEST_SIZE,
            duration=TEST_DURATION,
            fps=TEST_FPS,
            output_path=str(output_path),
        )

        expected_frames = int(TEST_DURATION * TEST_FPS)
        assert len(frames) == expected_frames
        assert output_path.exists()
        assert isinstance(first_frame, Image.Image)


@pytest.mark.gpu
def test_render_video():
    """Test render_video produces frames and file."""
    if not HAS_GPU:
        pytest.skip("GPU not available")

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "test.mp4"
        result_path, frames = render_video(
            time_shader,
            size=TEST_SIZE,
            duration=TEST_DURATION,
            fps=TEST_FPS,
            output_path=str(output_path),
        )

        expected_frames = int(TEST_DURATION * TEST_FPS)
        assert len(frames) == expected_frames
        assert Path(result_path).exists()


@pytest.mark.gpu
def test_render_with_glsl_string():
    """Test rendering with pre-transpiled GLSL string."""
    if not HAS_GPU:
        pytest.skip("GPU not available")

    glsl_code, _ = transpile(simple_shader)
    result = render_array(glsl_code, size=TEST_SIZE, time=0.0)

    assert result.shape == (TEST_SIZE[1], TEST_SIZE[0], 4)


@pytest.mark.gpu
def test_render_with_target():
    """Test rendering with explicit target."""
    if not HAS_GPU:
        pytest.skip("GPU not available")

    result = render_array(
        simple_shader, size=TEST_SIZE, time=0.0, target=TargetType.OPENGL46
    )

    assert result.shape == (TEST_SIZE[1], TEST_SIZE[0], 4)
