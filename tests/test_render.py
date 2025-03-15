import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from numpy.typing import NDArray
from PIL import Image

from py2glsl.builtins import cos, sin, vec2, vec4
from py2glsl.render import (
    _cleanup,
    _compile_program,
    _init_context,
    _render_frame,
    _setup_primitives,
    render_array,
    render_gif,
    render_image,
    render_video,
)
from py2glsl.transpiler import transpile

# Directory setup
TEST_DIR = Path(__file__).parent
REFERENCE_DIR = TEST_DIR / "reference"
REFERENCE_DIR.mkdir(exist_ok=True)

TEST_SIZE = (208, 160)
TEST_DURATION = 1.0
TEST_FPS = 8


# Test Shaders
def simple_shader(vs_uv: vec2, u_time: float, u_aspect: float) -> vec4:
    """A simple static gradient shader."""
    return vec4(vs_uv.x, vs_uv.y, 0.5, 1.0)


def time_shader(vs_uv: vec2, u_time: float, u_aspect: float) -> vec4:
    """A time-dependent shader with a moving pattern."""
    r = sin(vs_uv.x * 10.0 + u_time) * 0.5 + 0.5
    g = cos(vs_uv.y * 10.0 + u_time) * 0.5 + 0.5
    b = sin(u_time) * 0.5 + 0.5
    return vec4(r, g, b, 1.0)


# Transpile shaders
SIMPLE_GLSL, _ = transpile(simple_shader, main_func="simple_shader")
TIME_GLSL, _ = transpile(time_shader, main_func="time_shader")


def compare_images(
    array1: NDArray[np.uint8],
    array2: NDArray[np.uint8],
    tolerance: int = 0
) -> None:
    """Compare two RGBA arrays with a tolerance for minor rendering differences."""
    if array1.shape[-1] == 3:
        array1 = np.pad(array1, ((0, 0), (0, 0), (0, 1)), constant_values=255)
    if array2.shape[-1] == 3:
        array2 = np.pad(array2, ((0, 0), (0, 0), (0, 1)), constant_values=255)
    diff = np.abs(array1.astype(int) - array2.astype(int))
    assert np.all(diff <= tolerance), f"Images differ beyond tolerance {tolerance}"


def load_or_generate_reference(
    filename: str,
    generate_func: Callable[..., object],
    *args: object,
    **kwargs: object,
) -> NDArray[np.uint8]:
    """Load reference image or generate it if missing."""
    ref_path = REFERENCE_DIR / filename
    if not ref_path.exists():
        result = generate_func(*args, **kwargs)
        # Save the reference image regardless of result type
        if isinstance(result, Image.Image):
            result.save(ref_path)
        elif isinstance(result, np.ndarray):
            Image.fromarray(result).save(ref_path)
        elif isinstance(result, tuple) and len(result) >= 2:
            # First element is either an Image or path string, second is frames
            if hasattr(result[0], 'save'):
                result[0].save(ref_path)
            else:
                # For render_video that returns (path, frames)
                Image.fromarray(result[1][0]).save(ref_path)
        else:
            raise ValueError(f"Unsupported result type: {type(result)}")
        pytest.fail(
            f"Generated new reference image at {ref_path}. Please verify and commit."
        )
    return np.array(Image.open(ref_path).convert("RGBA"))


def generate_expected_frames(
    glsl_code: str, size: tuple[int, int], num_frames: int, fps: int
) -> list[NDArray[np.uint8]]:
    """Generate expected frames using a single OpenGL context."""
    ctx, _ = _init_context(size, windowed=False)
    program = _compile_program(ctx, glsl_code)
    vbo, vao = _setup_primitives(ctx, program)
    fbo = ctx.simple_framebuffer(size)

    frames: list[NDArray[np.uint8]] = []
    for i in range(num_frames):
        time = i / fps
        frame = _render_frame(ctx, program, vao, fbo, size, time, uniforms=None)
        assert frame is not None, "Failed to render frame"
        frames.append(frame)

    _cleanup(ctx, program, vbo, vao, fbo)
    return frames


def test_render_array() -> None:
    """Test render_array produces expected output for a static shader."""
    ref_array = load_or_generate_reference(
        "test_simple_shader.png",
        render_array,
        SIMPLE_GLSL,
        TEST_SIZE,
        time=0.0,
    )
    result_array = render_array(SIMPLE_GLSL, TEST_SIZE, time=0.0)
    assert_array_equal(result_array.shape, (TEST_SIZE[1], TEST_SIZE[0], 4))
    compare_images(result_array, ref_array)


def test_render_image() -> None:
    """Test render_image produces expected output for a static shader."""
    ref_array = load_or_generate_reference(
        "test_simple_shader.png",
        render_image,
        SIMPLE_GLSL,
        TEST_SIZE,
        time=0.0,
    )
    result_image = render_image(SIMPLE_GLSL, TEST_SIZE, time=0.0)
    result_array = np.array(result_image)
    assert result_image.size == TEST_SIZE
    compare_images(result_array, ref_array)


def test_render_gif() -> None:
    """Test render_gif produces expected frames for a time-dependent shader."""
    ref_path = REFERENCE_DIR / "test_time_shader.gif"
    num_frames = int(TEST_DURATION * TEST_FPS)

    if not ref_path.exists():
        _, _ = render_gif(
            TIME_GLSL,
            TEST_SIZE,
            duration=TEST_DURATION,
            fps=TEST_FPS,
            output_path=str(ref_path),
        )
        pytest.fail(
            f"Generated new reference GIF at {ref_path}. Please verify and commit."
        )

    _, result_frames = render_gif(
        TIME_GLSL,
        TEST_SIZE,
        duration=TEST_DURATION,
        fps=TEST_FPS,
    )
    expected_frames = generate_expected_frames(
        TIME_GLSL, TEST_SIZE, num_frames, TEST_FPS
    )

    assert (
        len(result_frames) == num_frames
    ), f"Expected {num_frames} frames, got {len(result_frames)}"
    for _i, (result_frame, expected_frame) in enumerate(
        zip(result_frames, expected_frames, strict=False)
    ):
        compare_images(result_frame, expected_frame)


def test_render_video() -> None:
    """Test render_video produces expected output for a time-dependent shader."""
    ref_path = REFERENCE_DIR / "test_time_shader.mp4"
    temp_path = TEST_DIR / "test_output.mp4"
    num_frames = int(TEST_DURATION * TEST_FPS)

    if not ref_path.exists():
        _, _ = render_video(
            TIME_GLSL,
            size=TEST_SIZE,
            duration=TEST_DURATION,
            fps=TEST_FPS,
            output_path=str(ref_path),
            codec="h264",
            quality=8,
            pixel_format="yuv420p",
        )
        pytest.fail(
            f"Generated new reference video at {ref_path}. Please verify and commit."
        )

    _, result_frames = render_video(
        TIME_GLSL,
        size=TEST_SIZE,
        duration=TEST_DURATION,
        fps=TEST_FPS,
        output_path=str(temp_path),
        codec="h264",
        quality=8,
        pixel_format="yuv420p",
    )
    expected_frames = generate_expected_frames(
        TIME_GLSL, TEST_SIZE, num_frames, TEST_FPS
    )

    assert (
        len(result_frames) == num_frames
    ), f"Expected {num_frames} frames, got {len(result_frames)}"
    for _i, (result_frame, expected_frame) in enumerate(
        zip(result_frames, expected_frames, strict=False)
    ):
        compare_images(result_frame, expected_frame)

    os.remove(temp_path)
