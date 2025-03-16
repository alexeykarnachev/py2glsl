"""Tests for backend rendering, ensuring consistent results within each backend.

These tests verify that each backend (standard GLSL and Shadertoy) produces
consistent visual output over time. Each backend has its own reference images
since OpenGL and OpenGL ES may produce visually different results due to
implementation details and precision differences.

The test works as follows:
1. Define test shaders with different characteristics
2. Generate reference images for each backend separately
3. Test each backend by comparing its output to its own reference image
4. Assert that differences are within tolerance

If the reference images don't exist, they'll be generated on the first run.
When backend code is modified, these tests will catch any visual regressions
within each backend type.
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
from py2glsl.transpiler.backends.models import BackendType

# Directory setup
TEST_DIR = Path(__file__).parent
BACKEND_REFERENCE_DIR = TEST_DIR / "backend_reference"
# Make sure the directory exists
BACKEND_REFERENCE_DIR.mkdir(exist_ok=True)

TEST_SIZE = (320, 240)
TOLERANCE = 10  # Allow for precision differences between OpenGL and OpenGL ES

# Set environment variables to help ModernGL handle contexts
os.environ["MODERNGL_FORCE_STANDALONE"] = "1"
os.environ["PYOPENGL_PLATFORM"] = (
    "egl"  # Use EGL platform to avoid display requirements
)

# Skip these tests when running in CI or without proper GPU
HAS_GPU = os.environ.get("HAS_GPU", "1") == "1"

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
        f"tests/test_backend_render.py::test_shader_backend[{backend_name}-{shader_name}-{shader_name}_test_shader]",
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


def get_backend_reference_path(name: str, backend_type: BackendType) -> Path:
    """Get the path to a backend-specific reference image."""
    filename = f"{name}_{backend_type.name.lower()}.png"
    return BACKEND_REFERENCE_DIR / filename


def render_safe(
    shader_func: Callable[[vec2, float, float], vec4],
    backend_type: BackendType,
    attempts: int = 3,
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

            # Attempt to render
            result = render_array(
                shader_func, size=TEST_SIZE, time=0.5, backend_type=backend_type
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
    """Test that a shader renders consistently within a specific backend."""
    try:
        # Skip if no GPU is available
        if not HAS_GPU:
            pytest.skip("GPU not available - skipping rendering test")

        # Use backend-specific reference images
        ref_path = get_backend_reference_path(name, backend_type)

        # Generate reference image if it doesn't exist
        if not ref_path.exists():
            # Render with the specified backend
            current_array = render_safe(shader_func, backend_type)
            Image.fromarray(current_array).save(ref_path)
            print(f"Generated reference image at {ref_path}")
            # No need to test further as we just created the reference
            return

        # Load reference image
        ref_array = np.array(Image.open(ref_path).convert("RGBA"))

        # Render current image with the specified backend
        current_array = render_safe(shader_func, backend_type)

        # Compare current render to saved reference
        diff = np.abs(current_array.astype(int) - ref_array.astype(int))
        max_diff = np.max(diff)
        print(f"Maximum difference for {backend_type.name}: {max_diff}")

        error_msg = (
            f"{backend_type.name} backend render differs from reference "
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
    """Test that a shader renders consistently with a specific backend."""
    _test_backend_render(shader_func, shader_name, backend_type)


# Special test that runs each test in isolation using subprocesses
# to ensure complete isolation of OpenGL contexts
@pytest.mark.skip(reason="Subprocess tests are only for diagnostic use")
def test_all_backends_in_isolation() -> None:
    """Run all shader and backend combinations in isolation."""
    # This test only runs tests individually to verify each one passes
    # It helps diagnose issues with OpenGL context that may occur when
    # tests are run together.
    if not HAS_GPU:
        pytest.skip("GPU not available - skipping isolation tests")

    failures = 0
    for shader_name in TEST_SHADERS:
        for backend in BACKENDS:
            backend_name = f"BackendType.{backend.name}"
            success = _run_isolated_test(shader_name, backend_name)
            if not success:
                failures += 1

    assert failures == 0, f"{failures} tests failed when run in isolation"
