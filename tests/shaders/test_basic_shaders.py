"""Test suite for basic shader functionality."""

import pytest

from py2glsl import py2glsl, vec2, vec4
from py2glsl.builtins import (
    abs,
    atan,
    clamp,
    cos,
    distance,
    dot,
    floor,
    fract,
    length,
    min,
    mix,
    mod,
    normalize,
    sin,
    smoothstep,
    sqrt,
)

from .utils import verify_shader_output


def test_solid_color(tmp_path):
    """Test basic solid color shader."""

    def solid_color(vs_uv: vec2, *, u_color: vec4) -> vec4:
        return u_color

    verify_shader_output(
        shader_func=solid_color,
        test_name="solid_color",
        tmp_path=tmp_path,
        uniforms={"u_color": (1.0, 0.0, 0.0, 1.0)},
    )


def test_uv_visualization(tmp_path):
    """Test UV coordinate visualization."""

    def uv_vis(vs_uv: vec2) -> vec4:
        return vec4(vs_uv.x, vs_uv.y, 0.0, 1.0)

    verify_shader_output(
        shader_func=uv_vis,
        test_name="uv_visualization",
        tmp_path=tmp_path,
    )


def test_checkerboard(tmp_path):
    """Test checkerboard pattern."""

    def checkerboard(vs_uv: vec2, *, u_scale: float) -> vec4:
        uv = vs_uv * u_scale
        checker = mod(floor(uv.x) + floor(uv.y), 2.0)
        return vec4(checker, checker, checker, 1.0)

    verify_shader_output(
        shader_func=checkerboard,
        test_name="checkerboard",
        tmp_path=tmp_path,
        uniforms={"u_scale": 8.0},
    )


def test_circular_gradient(tmp_path):
    """Test circular gradient pattern."""

    def circular_gradient(vs_uv: vec2) -> vec4:
        uv = vs_uv * 2.0 - 1.0
        d = length(uv)
        color = smoothstep(0.0, 1.0, 1.0 - d)
        return vec4(color, color, color, 1.0)

    verify_shader_output(
        shader_func=circular_gradient,
        test_name="circular_gradient",
        tmp_path=tmp_path,
    )


def test_polar_coordinates(tmp_path):
    """Test polar coordinate pattern."""

    def polar_pattern(vs_uv: vec2) -> vec4:
        uv = vs_uv * 2.0 - 1.0
        # Use atan() instead of atan2()
        angle = atan(uv.y, uv.x) / 6.28318530718 + 0.5  # 2π
        radius = length(uv)
        return vec4(angle, radius, 1.0 - radius, 1.0)

    verify_shader_output(
        shader_func=polar_pattern,
        test_name="polar_pattern",
        tmp_path=tmp_path,
    )


def test_sine_wave_pattern(tmp_path):
    """Test sine wave pattern."""

    def sine_pattern(vs_uv: vec2, *, u_frequency: float) -> vec4:
        uv = vs_uv * 2.0 - 1.0
        wave = sin(uv.x * u_frequency) * 0.5 + 0.5
        return vec4(wave, wave, wave, 1.0)

    verify_shader_output(
        shader_func=sine_pattern,
        test_name="sine_pattern",
        tmp_path=tmp_path,
        uniforms={"u_frequency": 10.0},
    )


def test_voronoi_cells(tmp_path):
    """Test basic Voronoi cell pattern."""

    def voronoi(vs_uv: vec2, *, u_scale: float) -> vec4:
        uv = vs_uv * u_scale
        cell_uv = fract(uv)
        min_dist = 1.0

        for x in range(-1, 2):
            for y in range(-1, 2):
                cell_point = vec2(x, y)
                point_pos = cell_point + fract(
                    sin(
                        vec2(
                            dot(floor(uv) + cell_point, vec2(127.1, 311.7)),
                            dot(floor(uv) + cell_point, vec2(269.5, 183.3)),
                        )
                    )
                    * 43758.5453123
                )
                min_dist = min(min_dist, length(cell_point + point_pos - cell_uv))

        return vec4(min_dist, min_dist, min_dist, 1.0)

    verify_shader_output(
        shader_func=voronoi,
        test_name="voronoi",
        tmp_path=tmp_path,
        uniforms={"u_scale": 5.0},
    )


def test_rainbow_gradient(tmp_path):
    """Test rainbow color gradient."""

    def rainbow(vs_uv: vec2) -> vec4:
        hue = vs_uv.x
        h = mod(hue * 6.0, 6.0)
        c = vec4(
            abs(h - 3.0) - 1.0,
            2.0 - abs(h - 2.0),
            2.0 - abs(h - 4.0),
            1.0,
        )
        return clamp(c, 0.0, 1.0)

    verify_shader_output(
        shader_func=rainbow,
        test_name="rainbow",
        tmp_path=tmp_path,
    )


def test_mandelbrot(tmp_path):
    """Test Mandelbrot set visualization."""

    def mandelbrot(vs_uv: vec2, *, u_zoom: float) -> vec4:
        uv = (vs_uv * 4.0 - 2.0) / u_zoom
        c = vec2(uv.x, uv.y)
        z = vec2(0.0, 0.0)
        iter_count = 0.0  # Use float instead of int
        max_iter = 100.0

        for i in range(100):  # Fixed iteration count
            if length(z) > 2.0:
                break
            z = vec2(z.x * z.x - z.y * z.y + c.x, 2.0 * z.x * z.y + c.y)
            iter_count += 1.0

        color = iter_count / max_iter
        return vec4(color, color * 0.5, color * 0.25, 1.0)

    verify_shader_output(
        shader_func=mandelbrot,
        test_name="mandelbrot",
        tmp_path=tmp_path,
        uniforms={"u_zoom": 1.0},
    )


def test_plasma_effect(tmp_path):
    """Test plasma effect shader."""

    def plasma(vs_uv: vec2, *, u_time: float) -> vec4:
        uv = vs_uv * 2.0 - 1.0
        v1 = sin(uv.x * 10.0 + u_time)
        v2 = sin(10.0 * (uv.x * sin(u_time / 2.0) + uv.y * cos(u_time / 3.0)))
        v3 = sin(sqrt(100.0 * (uv.x * uv.x + uv.y * uv.y) + 1.0 + u_time))
        color = (v1 + v2 + v3) / 3.0
        return vec4(
            sin(color * 3.14159) * 0.5 + 0.5,
            sin(color * 3.14159 + 2.09440) * 0.5 + 0.5,  # 2π/3
            sin(color * 3.14159 + 4.18879) * 0.5 + 0.5,  # 4π/3
            1.0,
        )

    verify_shader_output(
        shader_func=plasma,
        test_name="plasma",
        tmp_path=tmp_path,
        uniforms={"u_time": 1.234},
    )


def test_truchet_tiles(tmp_path):
    """Test Truchet tiles pattern."""

    def truchet(vs_uv: vec2, *, u_scale: float) -> vec4:
        uv = vs_uv * u_scale
        cell = floor(uv)
        cell_uv = fract(uv)

        # Random value for cell
        rand = fract(sin(dot(cell, vec2(12.9898, 78.233))) * 43758.5453)

        # Choose pattern based on random value
        d = 0.0
        if rand < 0.5:
            d = min(
                distance(cell_uv, vec2(0.0, 0.0)),
                distance(cell_uv, vec2(1.0, 1.0)),
            )
        else:
            d = min(
                distance(cell_uv, vec2(1.0, 0.0)),
                distance(cell_uv, vec2(0.0, 1.0)),
            )

        pattern = smoothstep(0.05, 0.1, d)
        return vec4(pattern, pattern, pattern, 1.0)

    verify_shader_output(
        shader_func=truchet,
        test_name="truchet",
        tmp_path=tmp_path,
        uniforms={"u_scale": 4.0},
    )


@pytest.mark.parametrize(
    "test_name,color1,color2",
    [
        ("gradient_red_blue", (1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)),
        ("gradient_yellow_purple", (1.0, 1.0, 0.0, 1.0), (0.5, 0.0, 0.5, 1.0)),
        ("gradient_green_orange", (0.0, 1.0, 0.0, 1.0), (1.0, 0.5, 0.0, 1.0)),
    ],
)
def test_linear_gradients(
    tmp_path, test_name: str, color1: tuple[float, ...], color2: tuple[float, ...]
):
    """Test parametrized linear gradients."""

    def linear_gradient(vs_uv: vec2, *, u_color1: vec4, u_color2: vec4) -> vec4:
        return mix(u_color1, u_color2, vs_uv.x)

    verify_shader_output(
        shader_func=linear_gradient,
        test_name=test_name,
        tmp_path=tmp_path,
        uniforms={"u_color1": color1, "u_color2": color2},
    )


def test_comprehensive_shader(tmp_path):
    """Test shader with multiple features but manageable complexity."""

    def comprehensive(
        vs_uv: vec2, *, u_color1: vec4, u_color2: vec4, u_time: float, u_scale: vec2
    ) -> vec4:
        # UV manipulation
        uv = vs_uv * 2.0 - 1.0
        scaled_uv = uv * u_scale

        # Trig-based pattern
        angle = atan(uv.y, uv.x)
        radius = length(uv)
        spiral = sin(angle * 3.0 + u_time + radius * 5.0)

        # Grid pattern
        grid_uv = fract(scaled_uv) - 0.5
        grid = smoothstep(0.1, 0.2, abs(grid_uv.x) * abs(grid_uv.y))

        # Combine patterns with iteration
        result = vec4(0.0, 0.0, 0.0, 0.0)

        # Iterate and build layers
        for i in range(3):
            # Calculate offset based on time and layer
            t = float(i)  # Store converted index
            offset = vec2(sin(u_time + t), cos(u_time - t))

            # Calculate UV with offset
            layer_uv = uv + offset * 0.2

            # Generate noise pattern
            noise_factor = dot(layer_uv, vec2(12.9898, 78.233))
            time_factor = 43758.5453 + t
            layer_val = sin(noise_factor * time_factor) * 0.5 + 0.5

            # Mix colors based on patterns
            blend_factor = layer_val * spiral * grid
            mixed_color: vec4 = mix(
                u_color1, u_color2, blend_factor
            )  # Changed variable name and added type annotation

            # Add layer with falloff
            falloff = 1.0 / (t + 1.0)
            result += mixed_color * falloff

        # Post-processing
        result = clamp(result, 0.0, 1.0)
        vignette = 1.0 - smoothstep(0.5, 1.5, radius)
        result *= vignette

        return result

    verify_shader_output(
        shader_func=comprehensive,
        test_name="comprehensive",
        tmp_path=tmp_path,
        uniforms={
            "u_color1": (1.0, 0.2, 0.1, 1.0),
            "u_color2": (0.1, 0.4, 1.0, 1.0),
            "u_time": 1.234,
            "u_scale": (3.0, 3.0),
        },
    )


# def test_complex_nested_shader(tmp_path):
#     """Test shader with complex nested structures and features."""
#
#     def complex_shader(
#         vs_uv: vec2,
#         *,
#         u_color1: vec4,
#         u_color2: vec4,
#         u_time: float,
#         u_scale: float,
#     ) -> vec4:
#         def create_rotation_matrix(angle: float) -> mat2:
#             c = cos(angle)
#             s = sin(angle)
#             return mat2(c, -s, s, c)
#
#         def create_fractal(p: vec2, iterations: int) -> float:
#             z = vec2(0.0, 0.0)
#             c = p
#             value = 0.0
#
#             for i in range(iterations):
#                 # Early break optimization
#                 if length(z) > 2.0:
#                     break
#
#                 # Complex number multiplication
#                 z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c
#
#                 value += exp(-length(z))
#
#             return value / float(iterations)
#
#         def create_pattern(uv: vec2, time: float) -> vec4:
#             # Rotate UV space
#             rot = create_rotation_matrix(time * 0.5)
#             rotated_uv = rot * (uv * 2.0 - 1.0)
#
#             # Create multiple layers
#             color = vec4(0.0, 0.0, 0.0, 0.0)
#             scale = 1.0
#
#             for i in range(3):  # Layer loop
#                 offset = vec2(
#                     sin(time + float(i) * 0.5), cos(time * 0.7 + float(i) * 0.5)
#                 )
#
#                 scaled_uv = rotated_uv * scale + offset
#
#                 # Create grid pattern
#                 for x in range(-1, 2):  # Grid X
#                     for y in range(-1, 2):  # Grid Y
#                         cell_uv = scaled_uv + vec2(x - 0.0, y - 0.0)
#
#                         # Calculate fractal value
#                         fractal = create_fractal(cell_uv * 0.5, 8)
#
#                         # Create cell color
#                         cell_color = mix(u_color1, u_color2, fractal)
#
#                         # Apply distance falloff
#                         dist = length(vec2(x - 0.0, y - 0.0))
#                         falloff = smoothstep(2.0, 0.0, dist)
#
#                         color += cell_color * falloff * 0.3
#
#                 scale *= 1.5
#
#             return color
#
#         # Main shader logic
#         uv = vs_uv * u_scale
#
#         # Create base pattern
#         pattern = create_pattern(uv, u_time)
#
#         # Apply post-processing effects
#         final_color = pattern
#
#         # Vignette effect
#         center_dist = length(vs_uv * 2.0 - 1.0)
#         vignette = 1.0 - smoothstep(0.5, 1.5, center_dist)
#         final_color *= vignette
#
#         # Ensure valid alpha
#         final_color.a = clamp(final_color.a, 0.0, 1.0)
#
#         return final_color
#
#     # Test the shader with specific uniforms
#     verify_shader_output(
#         shader_func=complex_shader,
#         test_name="complex_nested",
#         tmp_path=tmp_path,
#         uniforms={
#             "u_color1": (1.0, 0.2, 0.1, 1.0),
#             "u_color2": (0.1, 0.4, 1.0, 1.0),
#             "u_time": 1.234,
#             "u_scale": 3.0,
#         },
#     )
