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

from collections.abc import Callable
from pathlib import Path

import numpy as np
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


def get_backend_reference_path(name: str, backend_type: BackendType) -> Path:
    """Get the path to a backend-specific reference image."""
    filename = f"{name}_{backend_type.name.lower()}.png"
    return BACKEND_REFERENCE_DIR / filename


def _test_backend_render(
    shader_func: Callable[[vec2, float, float], vec4],
    name: str,
    backend_type: BackendType,
) -> None:
    """Test that a shader renders consistently within a specific backend."""
    try:
        # Use backend-specific reference images
        ref_path = get_backend_reference_path(name, backend_type)

        # Generate reference image if it doesn't exist
        if not ref_path.exists():
            # Render with the specified backend
            current_array = render_array(
                shader_func, size=TEST_SIZE, time=0.5, backend_type=backend_type
            )
            Image.fromarray(current_array).save(ref_path)
            print(f"Generated reference image at {ref_path}")
            # No need to test further as we just created the reference
            return

        # Load reference image
        ref_array = np.array(Image.open(ref_path).convert("RGBA"))

        # Render current image with the specified backend
        current_array = render_array(
            shader_func, size=TEST_SIZE, time=0.5, backend_type=backend_type
        )

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


def test_simple_shader_standard() -> None:
    """Test that simple shader renders consistently with standard backend."""
    _test_backend_render(simple_test_shader, "simple_shader", BackendType.STANDARD)


def test_simple_shader_shadertoy() -> None:
    """Test that simple shader renders consistently with shadertoy backend."""
    _test_backend_render(simple_test_shader, "simple_shader", BackendType.SHADERTOY)


def test_animated_shader_standard() -> None:
    """Test that animated shader renders consistently with standard backend."""
    _test_backend_render(animated_test_shader, "animated_shader", BackendType.STANDARD)


def test_animated_shader_shadertoy() -> None:
    """Test that animated shader renders consistently with shadertoy backend."""
    _test_backend_render(animated_test_shader, "animated_shader", BackendType.SHADERTOY)


def test_complex_shader_standard() -> None:
    """Test that complex shader renders consistently with standard backend."""
    _test_backend_render(complex_test_shader, "complex_shader", BackendType.STANDARD)


def test_complex_shader_shadertoy() -> None:
    """Test that complex shader renders consistently with shadertoy backend."""
    _test_backend_render(complex_test_shader, "complex_shader", BackendType.SHADERTOY)
