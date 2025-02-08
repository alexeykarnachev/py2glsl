import os
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest
from PIL import Image

from py2glsl import (
    animate,
    render_array,
    render_gif,
    render_image,
    render_video,
    vec2,
    vec4,
)


@pytest.fixture
def temp_dir(tmp_path):
    """Provide temporary directory for test outputs"""
    return tmp_path


def test_solid_color_render():
    """Test rendering solid color"""

    def red_shader(vs_uv: vec2) -> vec4:
        return vec4(1.0, 0.0, 0.0, 1.0)

    # Test array output
    arr = render_array(red_shader, size=(64, 64))
    assert arr.shape == (64, 64, 4)
    assert arr.dtype == np.float32
    np.testing.assert_array_almost_equal(arr[0, 0], np.array([1.0, 0.0, 0.0, 1.0]))


def test_gradient_render():
    """Test rendering horizontal gradient"""

    def gradient_shader(vs_uv: vec2) -> vec4:
        return vec4(vs_uv.x, 0.0, 0.0, 1.0)

    arr = render_array(gradient_shader, size=(64, 64))

    # Check left edge is black
    np.testing.assert_array_almost_equal(arr[32, 0], np.array([0.0, 0.0, 0.0, 1.0]))

    # Check right edge is red
    np.testing.assert_array_almost_equal(arr[32, -1], np.array([1.0, 0.0, 0.0, 1.0]))

    # Check middle is half red
    np.testing.assert_array_almost_equal(
        arr[32, 32], np.array([0.5, 0.0, 0.0, 1.0]), decimal=2
    )


def test_uniform_values():
    """Test uniform parameter passing"""

    def color_shader(vs_uv: vec2, *, u_color: vec4) -> vec4:
        return u_color

    arr = render_array(color_shader, size=(64, 64), u_color=(0.0, 1.0, 0.0, 1.0))

    np.testing.assert_array_almost_equal(arr[0, 0], np.array([0.0, 1.0, 0.0, 1.0]))


def test_image_output():
    """Test PIL Image output"""

    def blue_shader(vs_uv: vec2) -> vec4:
        return vec4(0.0, 0.0, 1.0, 1.0)

    img = render_image(blue_shader, size=(64, 64))
    assert isinstance(img, Image.Image)
    assert img.size == (64, 64)
    assert img.mode == "RGBA"

    # Check pixel color
    pixel = img.getpixel((0, 0))
    assert pixel == (0, 0, 255, 255)


def test_gif_animation(temp_dir):
    """Test GIF animation rendering"""

    def time_shader(vs_uv: vec2, *, u_time: float = 0.0) -> vec4:
        return vec4(sin(u_time) * 0.5 + 0.5, 0.0, 0.0, 1.0)

    output_path = temp_dir / "test.gif"
    render_gif(time_shader, str(output_path), duration=1.0, fps=10, size=(32, 32))

    # Verify file exists and is valid GIF
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    # Read and verify frames
    frames = list(iio.imiter(str(output_path)))
    assert len(frames) == 10  # 1 second at 10 fps

    # Check first and middle frames are different
    assert not np.array_equal(frames[0], frames[5])


def test_video_output(temp_dir):
    """Test video rendering"""

    def time_shader(vs_uv: vec2, *, u_time: float = 0.0) -> vec4:
        return vec4(0.0, sin(u_time) * 0.5 + 0.5, 0.0, 1.0)

    output_path = temp_dir / "test.mp4"
    render_video(time_shader, str(output_path), duration=1.0, fps=10, size=(32, 32))

    # Verify file exists and is valid video
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    # Read and verify frames
    frames = list(iio.imiter(str(output_path)))
    assert len(frames) == 10  # 1 second at 10 fps

    # Check first and middle frames are different
    assert not np.array_equal(frames[0], frames[5])


def test_resolution_uniform():
    """Test u_resolution uniform"""

    def resolution_shader(vs_uv: vec2) -> vec4:
        return vec4(u_resolution.x / 100.0, u_resolution.y / 100.0, 0.0, 1.0)

    arr = render_array(resolution_shader, size=(50, 25))
    np.testing.assert_array_almost_equal(arr[0, 0], np.array([0.5, 0.25, 0.0, 1.0]))


def test_alpha_blending():
    """Test alpha channel handling"""

    def alpha_shader(vs_uv: vec2) -> vec4:
        return vec4(1.0, 1.0, 1.0, vs_uv.x)

    arr = render_array(alpha_shader, size=(64, 64))

    # Check left edge is transparent
    np.testing.assert_array_almost_equal(arr[32, 0, 3], 0.0)

    # Check right edge is opaque
    np.testing.assert_array_almost_equal(arr[32, -1, 3], 1.0)


def test_error_handling():
    """Test error cases"""

    # Test invalid shader function
    def invalid_shader(vs_uv: vec2) -> vec3:  # Wrong return type
        return vec3(1.0, 0.0, 0.0)

    with pytest.raises(TypeError):
        render_array(invalid_shader)

    # Test invalid uniform type
    def uniform_shader(vs_uv: vec2, *, u_val: str) -> vec4:  # Invalid uniform type
        return vec4(1.0, 0.0, 0.0, 1.0)

    with pytest.raises(TypeError):
        render_array(uniform_shader, u_val="invalid")


def test_animation_frame_values():
    """Test animation frame counting and timing"""
    frames = []
    times = []

    def capture_shader(vs_uv: vec2, *, u_time: float = 0.0, u_frame: int = 0) -> vec4:
        frames.append(u_frame)
        times.append(u_time)
        return vec4(1.0)

    render_gif(capture_shader, "test.gif", duration=1.0, fps=10, size=(32, 32))

    # Check frame numbers
    assert frames == list(range(10))

    # Check frame times
    expected_times = [i / 10 for i in range(10)]
    np.testing.assert_array_almost_equal(times, expected_times)


@pytest.mark.asyncio
async def test_animate_window():
    """Test animate window creation and basic rendering"""
    window_created = False
    render_called = False

    def test_shader(vs_uv: vec2) -> vec4:
        nonlocal render_called
        render_called = True
        return vec4(1.0)

    # Run animation for a brief moment
    import asyncio

    async def run_animation():
        nonlocal window_created
        window_created = True
        animate(test_shader, size=(64, 64))

    try:
        # Run animation with timeout
        await asyncio.wait_for(run_animation(), timeout=0.5)
    except asyncio.TimeoutError:
        pass  # Expected timeout

    assert window_created
    assert render_called


def test_complex_shader():
    """Test more complex shader with multiple features"""

    def complex_shader(vs_uv: vec2, *, u_time: float = 0.0) -> vec4:
        # Center UV
        uv = vs_uv * 2.0 - 1.0

        # Create animated circle
        d = length(uv)
        radius = 0.3 + sin(u_time * 2.0) * 0.1
        circle = smoothstep(radius, radius - 0.01, d)

        # Animate color
        r = 0.5 + 0.5 * sin(u_time)
        g = 0.5 + 0.5 * sin(u_time + 2.094)
        b = 0.5 + 0.5 * sin(u_time + 4.189)

        return vec4(circle * r, circle * g, circle * b, circle)

    # Test static render
    arr = render_array(complex_shader, size=(64, 64))
    assert not np.any(np.isnan(arr))
    assert np.all(arr >= 0.0) and np.all(arr <= 1.0)

    # Test animation
    frames = []
    for i in range(5):
        frame = render_array(complex_shader, size=(64, 64), u_time=i * 0.1)
        frames.append(frame)

    # Verify frames are different
    assert not np.array_equal(frames[0], frames[-1])
