"""Tests for backend rendering, ensuring consistent visual results across backends.

These tests verify that shaders produce consistent visual output regardless of
which backend (standard GLSL or Shadertoy) is used. We use a single set of reference
images that both backends should match within tolerance.

The test works as follows:
1. Define test shaders with different characteristics
2. Generate reference images using any backend (if they don't exist)
3. Test each backend by comparing its output to the reference image
4. Assert that differences are within a reasonable tolerance

The goal is to ensure that users can write code once using the py2glsl API
and have it produce the expected visual result regardless of which backend
is used for rendering.
"""

import os
import subprocess
import tempfile
import time
from collections.abc import Callable, Generator
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from py2glsl.builtins import cos, sin, vec2, vec4
from py2glsl.render import render_array
from py2glsl.transpiler.backends import BackendType

# Directory setup
TEST_DIR = Path(__file__).parent
REFERENCE_DIR = TEST_DIR / "reference"
# Make sure the directory exists
REFERENCE_DIR.mkdir(exist_ok=True)

TEST_SIZE = (320, 240)
TOLERANCE = 30  # Higher tolerance for cross-backend visual similarity

# Set environment variables to help ModernGL handle contexts
os.environ["MODERNGL_FORCE_STANDALONE"] = "1"
os.environ["PYOPENGL_PLATFORM"] = (
    "egl"  # Use EGL platform to avoid display requirements
)

# Skip these tests when running in CI or without proper GPU
# Check if we should skip GPU tests
# Set NO_GPU=1 to skip GPU tests
HAS_GPU = os.environ.get("NO_GPU", "0") != "1"

# Mark the entire module for special handling
pytestmark = [
    pytest.mark.gpu,  # Mark all tests as requiring a GPU
    pytest.mark.backend,  # Mark all tests as backend tests
]


def _run_isolated_test(shader_name: str, backend_name: str) -> bool:
    """Run a single test in isolation using subprocess."""
    # Run the test in a separate process to ensure full isolation
    cmd = [
        "python",
        "-m",
        "pytest",
        # Use the full working directory path with the test file
        f"{Path.cwd()}/tests/test_backend_render.py::test_shader_backend[{backend_name}-{shader_name}-{shader_name}_test_shader]",
        "-v",
    ]

    # Run the command
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    # If the test failed, print the output
    if result.returncode != 0:
        print(f"Isolated test failed: {' '.join(cmd)}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")

    return result.returncode == 0


@pytest.fixture(scope="module", autouse=True)
def isolation() -> Generator[None, None, None]:
    """Ensure tests in this module run in isolation."""
    # Save the original current directory
    orig_dir = os.getcwd()

    # Create a temporary directory for isolated execution
    with tempfile.TemporaryDirectory() as tmpdirname:
        os.chdir(tmpdirname)
        try:
            yield
        finally:
            # Return to the original directory
            os.chdir(orig_dir)
            # Sleep after all tests in this module to ensure cleanup
            time.sleep(0.5)


# Define available backends for testing
BACKENDS = [BackendType.STANDARD, BackendType.SHADERTOY]


# Test Shaders
def simple_test_shader(vs_uv: vec2, u_time: float, u_aspect: float) -> vec4:
    """A simple shader with a consistent appearance across backends."""
    r = vs_uv.x
    g = vs_uv.y
    b = 0.5
    a = 1.0
    return vec4(r, g, b, a)


def animated_test_shader(vs_uv: vec2, u_time: float, u_aspect: float) -> vec4:
    """A time-dependent shader that should look the same across backends."""
    r = sin(vs_uv.x * 10.0 + u_time) * 0.5 + 0.5
    g = cos(vs_uv.y * 10.0 + u_time) * 0.5 + 0.5
    b = sin(u_time) * 0.5 + 0.5
    a = 1.0
    return vec4(r, g, b, a)


def complex_test_shader(vs_uv: vec2, u_time: float, u_aspect: float) -> vec4:
    """A more complex shader with multiple operations."""
    # Create a circular pattern
    uv = vec2(vs_uv.x - 0.5, vs_uv.y - 0.5) * 2.0
    uv.x *= u_aspect

    # Distance from center
    d = uv.x * uv.x + uv.y * uv.y

    # Animated ring
    ring = sin(d * 8.0 - u_time * 2.0) * 0.5 + 0.5

    # Color based on position and time
    r = sin(uv.x * 3.0 + u_time) * 0.5 + 0.5
    g = cos(uv.y * 5.0 - u_time) * 0.5 + 0.5
    b = sin(d * 4.0 + u_time * 0.5) * 0.5 + 0.5

    # Mix with ring pattern
    return vec4(r * ring, g * ring, b * ring, 1.0)


# Dictionary of test shaders for parameterized testing
TEST_SHADERS = {
    "simple": simple_test_shader,
    "animated": animated_test_shader,
    "complex": complex_test_shader,
}


def get_reference_path(name: str) -> Path:
    """Get the path to a backend-agnostic shader reference image."""
    filename = f"shader_{name}.png"
    return REFERENCE_DIR / filename


def render_safe(
    shader_func: Callable[[vec2, float, float], vec4],
    backend_type: BackendType,
    attempts: int = 3,
    fixed_time: float = 0.5,  # Use consistent time for tests
) -> np.ndarray:
    """Safely render a shader with error handling and retries.

    Since ModernGL context handling can be flaky in tests, this function
    provides retry logic and better error handling.
    """
    # Skip if no GPU is available
    if not HAS_GPU:
        pytest.skip("GPU not available - skipping rendering test")

    # For mypy
    result = None

    for attempt in range(attempts):
        try:
            # Add a small delay between attempts to let previous contexts be cleaned up
            if attempt > 0:
                time.sleep(0.5)

            # Ensure we use the same exact time for both backends
            # to get consistent results for time-dependent shaders
            result = render_array(
                shader_func, size=TEST_SIZE, time=fixed_time, backend_type=backend_type
            )
            return result
        except Exception as e:
            if attempt == attempts - 1:  # Last attempt
                pytest.fail(f"Failed to render after {attempts} attempts: {e!s}")
            print(f"Render attempt {attempt + 1} failed: {e!s}, retrying...")

    # This should never happen but makes mypy happy
    assert result is not None
    return result


def _test_backend_render(
    shader_func: Callable[[vec2, float, float], vec4],
    name: str,
    backend_type: BackendType,
) -> None:
    """Test that a shader renders visually consistent regardless of backend."""
    try:
        # Skip if no GPU is available
        if not HAS_GPU:
            pytest.skip("GPU not available - skipping rendering test")

        # Use backend-agnostic reference images
        ref_path = get_reference_path(name)

        # Generate reference image if it doesn't exist
        # Always use the standard backend for reference generation
        if not ref_path.exists():
            # We use the standard backend for reference images as the baseline
            reference_backend = BackendType.STANDARD
            current_array = render_safe(shader_func, reference_backend)
            Image.fromarray(current_array).save(ref_path)
            print(f"Generated reference image at {ref_path}")
            # If we just created a reference with this backend, no need to test
            if backend_type == reference_backend:
                return

        # Load reference image
        ref_array = np.array(Image.open(ref_path).convert("RGBA"))

        # Render current image with the specified backend
        current_array = render_safe(shader_func, backend_type)

        # Compare current render to saved reference
        diff = np.abs(current_array.astype(int) - ref_array.astype(int))
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        print(
            f"Backend {backend_type.name}: max diff: {max_diff}, mean: {mean_diff:.2f}"
        )

        error_msg = (
            f"{backend_type.name} backend render visually differs from reference "
            f"beyond tolerance {TOLERANCE}"
        )
        assert np.all(diff <= TOLERANCE), error_msg
    except Exception as e:
        print(f"Error in test_backend_render for {backend_type.name}: {e!s}")
        raise


# Parameterized test for all shader and backend combinations
@pytest.mark.parametrize("shader_name,shader_func", list(TEST_SHADERS.items()))
@pytest.mark.parametrize("backend_type", BACKENDS)
def test_shader_backend(
    shader_name: str,
    shader_func: Callable[[vec2, float, float], vec4],
    backend_type: BackendType,
) -> None:
    """Test that a shader renders consistently across backends."""
    _test_backend_render(shader_func, shader_name, backend_type)


# Direct testing approach instead of subprocess isolation
def test_all_backends_direct() -> None:
    """Test all shader and backend combinations directly."""
    # This is a simpler approach that just runs all the tests directly
    if not HAS_GPU:
        pytest.skip("GPU not available - skipping backend tests")

    for shader_name, shader_func in TEST_SHADERS.items():
        for backend_type in BACKENDS:
            # Run the test directly
            _test_backend_render(shader_func, shader_name, backend_type)
