"""Test suite for basic shader functionality."""

import pytest

from py2glsl import vec2, vec3, vec4
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
    sin,
    smoothstep,
    sqrt,
)
from py2glsl.types.constructors import mat2

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


"""Test shaders for type conversion behavior."""


def test_loop_integer_context(tmp_path):
    """Test integer context in loop variables."""

    def loop_shader(vs_uv: vec2) -> vec4:
        color = vec4(0.0)
        # Loop variable should be int despite float literals
        for i in range(4):  # i should be int
            color += vec4(float(i) / 3.0)  # Need explicit float conversion
        return color

    verify_shader_output(
        shader_func=loop_shader,
        test_name="loop_integer",
        tmp_path=tmp_path,
    )


def test_float_to_int_conversion(tmp_path):
    """Test float to int conversion in different contexts."""

    def conversion_shader(vs_uv: vec2) -> vec4:
        # Float to int conversion in mod context
        grid_x = mod(floor(vs_uv.x * 8.0), 2.0)  # floor returns float
        grid_y = mod(floor(vs_uv.y * 8.0), 2.0)

        # Loop using float value converted to int
        color = vec4(0.0)
        cell_count = floor(grid_x * 4.0)  # This is float
        for i in range(int(cell_count)):  # Need explicit int conversion for range
            color += vec4(0.25)

        return color + vec4(grid_x * grid_y)

    verify_shader_output(
        shader_func=conversion_shader,
        test_name="float_to_int",
        tmp_path=tmp_path,
    )


def test_mixed_numeric_operations(tmp_path):
    """Test mixed float/int operations."""

    def mixed_shader(vs_uv: vec2, *, u_steps: float) -> vec4:
        # u_steps is float but needs to be int in range
        steps = int(u_steps)  # Explicit conversion for range
        color = vec4(0.0)

        for i in range(steps):
            # i is int, but used in float context
            t = i / float(steps - 1)  # Need explicit float conversion for division
            color += vec4(t)  # t is float, auto-converts in vec4 constructor

        return color

    verify_shader_output(
        shader_func=mixed_shader,
        test_name="mixed_numeric",
        tmp_path=tmp_path,
        uniforms={"u_steps": 4.0},
    )


@pytest.mark.parametrize(
    "grid_size",
    [2.0, 4.0, 8.0],  # Float values that need int conversion
)
def test_grid_pattern_conversion(tmp_path, grid_size: float):
    """Test grid pattern with float to int conversion."""

    def grid_shader(vs_uv: vec2, *, u_grid: float) -> vec4:
        # Convert float grid size to int for range
        cells = int(u_grid)
        color = vec4(0.0)

        # Nested loops with integer context
        for x in range(cells):
            for y in range(cells):
                # Convert back to float for position calculation
                cell_x = float(x) / u_grid
                cell_y = float(y) / u_grid

                if (
                    vs_uv.x >= cell_x
                    and vs_uv.x < cell_x + 1.0 / u_grid
                    and vs_uv.y >= cell_y
                    and vs_uv.y < cell_y + 1.0 / u_grid
                ):
                    # Alternate cell colors
                    if mod(float(x + y), 2.0) < 1.0:
                        color = vec4(1.0)

        return color

    verify_shader_output(
        shader_func=grid_shader,
        test_name=f"grid_pattern_{int(grid_size)}",
        tmp_path=tmp_path,
        uniforms={"u_grid": grid_size},
    )


def test_complex_type_conversion(tmp_path):
    """Test complex scenarios with type conversion."""

    def complex_shader(vs_uv: vec2, *, u_iterations: float) -> vec4:
        # Start with float uniform but need int for iteration
        max_iter = int(u_iterations)
        color = vec4(0.0)

        # Initialize with integer coordinates
        pos_x = int(vs_uv.x * 10.0)  # Explicit conversion to int
        pos_y = int(vs_uv.y * 10.0)

        # Accumulate in float context
        value = 0.0
        for i in range(max_iter):
            # Mix int and float operations
            offset = float(i + pos_x + pos_y) / float(max_iter)
            value += mod(offset, 1.0)

            # Integer-based condition
            if i % 2 == 0:  # Modulo requires int
                value *= 0.5  # Float operation

        return vec4(value)

    verify_shader_output(
        shader_func=complex_shader,
        test_name="complex_conversion",
        tmp_path=tmp_path,
        uniforms={"u_iterations": 5.0},
    )


def test_nested_and_global_functions_shader(tmp_path):
    """Test shader with nested and global functions."""

    def get_color_out(t: float) -> vec3:
        return vec3(0.5 * (sin(t) + 1.0), 0.0, 0.0)

    def main_shader(vs_uv: vec2, *, u_time: float) -> vec4:
        def get_color_in(p: vec2) -> vec3:
            return vec3(0.0, 1.0, 0.0)

        color = get_color_in(vs_uv) + get_color_out(u_time)
        return vec4(color, 1.0)

    verify_shader_output(
        shader_func=main_shader,
        test_name="nested_and_global_functions",
        tmp_path=tmp_path,
        uniforms={"u_time": 1.0},
    )


def test_vector_component_shader(tmp_path):
    """Test shader with vector component modification."""

    def component_shader(vs_uv: vec2, *, u_aspect: float) -> vec4:
        p = vs_uv
        p.x *= u_aspect  # Direct component modification
        return vec4(p, 0.0, 1.0)

    verify_shader_output(
        shader_func=component_shader,
        test_name="vector_component",
        tmp_path=tmp_path,
        uniforms={"u_aspect": 1.5},
    )
