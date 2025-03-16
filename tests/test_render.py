import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import moderngl
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from numpy.typing import NDArray
from PIL import Image

# Import the modules we need
from py2glsl.builtins import cos, sin, vec2, vec4
from py2glsl.render import (
    render_array,
    render_gif,
    render_image,
    render_video,
)
from py2glsl.transpiler import transpile

# Skip the entire module if no GPU is available
pytestmark = pytest.mark.gpu

# Check if we should skip GPU tests
# Set NO_GPU=1 to skip GPU tests
HAS_GPU = os.environ.get("NO_GPU", "0") != "1"

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


# Context management fixture for ModernGL
@pytest.fixture(scope="module")
def gl_context() -> moderngl.Context:
    """Create a ModernGL context for all tests in this module."""
    # Create a context
    try:
        ctx = moderngl.create_context(standalone=True, require=460)
        yield ctx
    except Exception as e:
        pytest.skip(f"Failed to create ModernGL context: {e}")
    finally:
        # Release the context if it was created
        if 'ctx' in locals():
            ctx.release()


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


# Create a fixture that provides a temporary directory for each test
@pytest.fixture
def isolated_test_dir() -> Path:
    """Create a temporary directory for each test to avoid file conflicts."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


def load_or_generate_reference(
    filename: str,
    generate_func: Callable[..., object],
    *args: object,
    **kwargs: object,
) -> NDArray[np.uint8]:
    """Load reference image or generate it if missing."""
    ref_path = REFERENCE_DIR / filename
    if not ref_path.exists():
        try:
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
        except Exception as e:
            pytest.skip(f"Failed to generate reference: {e}")

        pytest.fail(
            f"Generated new reference image at {ref_path}. Please verify and commit."
        )
    return np.array(Image.open(ref_path).convert("RGBA"))


def generate_expected_frames(
    gl_context: moderngl.Context,
    glsl_code: str,
    size: tuple[int, int],
    num_frames: int,
    fps: int
) -> list[NDArray[np.uint8]]:
    """Generate expected frames one by one with individual contexts."""
    frames: list[NDArray[np.uint8]] = []

    # We'll create a new context for each frame to avoid resource conflicts
    for i in range(num_frames):
        time_value = i / fps
        try:
            # Use the render_array function directly which handles its own context
            frame = render_array(glsl_code, size, time=time_value)
            frames.append(frame)
        except Exception as e:
            pytest.skip(f"Failed to render frame {i}: {e}")

    return frames


@pytest.mark.gpu
def test_render_array(gl_context: moderngl.Context, isolated_test_dir: Path) -> None:
    """Test render_array produces expected output for a static shader."""
    if not HAS_GPU:
        pytest.skip("Skipping GPU test: HAS_GPU=0")

    ref_array = load_or_generate_reference(
        "test_simple_shader.png",
        render_array,
        SIMPLE_GLSL,
        TEST_SIZE,
        time=0.0,
    )

    # Add retry logic for more stability
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result_array = render_array(SIMPLE_GLSL, TEST_SIZE, time=0.0)
            assert_array_equal(result_array.shape, (TEST_SIZE[1], TEST_SIZE[0], 4))
            compare_images(result_array, ref_array)
            break
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(0.5)  # Short delay between retries


@pytest.mark.gpu
def test_render_image(gl_context: moderngl.Context, isolated_test_dir: Path) -> None:
    """Test render_image produces expected output for a static shader."""
    if not HAS_GPU:
        pytest.skip("Skipping GPU test: HAS_GPU=0")

    ref_array = load_or_generate_reference(
        "test_simple_shader.png",
        render_image,
        SIMPLE_GLSL,
        TEST_SIZE,
        time=0.0,
    )

    # Add retry logic for more stability
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result_image = render_image(SIMPLE_GLSL, TEST_SIZE, time=0.0)
            result_array = np.array(result_image)
            assert result_image.size == TEST_SIZE
            compare_images(result_array, ref_array)
            break
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(0.5)  # Short delay between retries


@pytest.mark.gpu
def test_render_gif(gl_context: moderngl.Context, isolated_test_dir: Path) -> None:
    """Test render_gif produces expected frames for a time-dependent shader."""
    if not HAS_GPU:
        pytest.skip("Skipping GPU test: HAS_GPU=0")

    ref_path = REFERENCE_DIR / "test_time_shader.gif"
    num_frames = int(TEST_DURATION * TEST_FPS)
    output_path = isolated_test_dir / "test_time_shader.gif"

    if not ref_path.exists():
        try:
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
        except Exception as e:
            pytest.skip(f"Failed to create reference: {e}")

    # Add retry logic for more stability
    max_retries = 3
    for attempt in range(max_retries):
        try:
            _, result_frames = render_gif(
                TIME_GLSL,
                TEST_SIZE,
                duration=TEST_DURATION,
                fps=TEST_FPS,
                output_path=str(output_path),
            )
            expected_frames = generate_expected_frames(
                gl_context, TIME_GLSL, TEST_SIZE, num_frames, TEST_FPS
            )

            assert (
                len(result_frames) == num_frames
            ), f"Expected {num_frames} frames, got {len(result_frames)}"
            for _i, (result_frame, expected_frame) in enumerate(
                zip(result_frames, expected_frames, strict=False)
            ):
                compare_images(result_frame, expected_frame)
            break
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(0.5)  # Short delay between retries


@pytest.mark.gpu
def test_render_video(gl_context: moderngl.Context, isolated_test_dir: Path) -> None:
    """Test render_video produces expected output for a time-dependent shader."""
    if not HAS_GPU:
        pytest.skip("Skipping GPU test: HAS_GPU=0")

    ref_path = REFERENCE_DIR / "test_time_shader.mp4"
    temp_path = isolated_test_dir / "test_output.mp4"
    num_frames = int(TEST_DURATION * TEST_FPS)

    if not ref_path.exists():
        try:
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
                f"Generated new reference video at {ref_path}. "
                f"Please verify and commit."
            )
        except Exception as e:
            pytest.skip(f"Failed to create reference: {e}")

    # Add retry logic for more stability
    max_retries = 3
    for attempt in range(max_retries):
        try:
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
                gl_context, TIME_GLSL, TEST_SIZE, num_frames, TEST_FPS
            )

            assert (
                len(result_frames) == num_frames
            ), f"Expected {num_frames} frames, got {len(result_frames)}"
            for _i, (result_frame, expected_frame) in enumerate(
                zip(result_frames, expected_frames, strict=False)
            ):
                compare_images(result_frame, expected_frame)

            # Skip cleanup if test passes
            break
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(0.5)  # Short delay between retries
