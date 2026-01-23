"""Python Syntax Features Showcase

This example demonstrates all Python syntax features supported by py2glsl,
including the newer Pythonic features like tuple unpacking, default parameters,
negative indexing, power optimization, and more.

Features demonstrated:
1. Global constants (const float, const int, const bool)
2. Constant expressions evaluated at compile time (TAU = PI * 2.0)
3. Mutable globals
4. Helper functions with various parameter/return types
5. Default function parameters
6. Tuple returns (multiple return values)
7. Tuple unpacking from vectors
8. Negative indexing (v[-1], arr[-2])
9. Integer/floor division (//)
10. Power optimization (x**2, x**3, x**0.5)
11. min/max with multiple arguments
12. Arrays with typed annotations
13. Vector swizzling and component access
14. Control flow (if/elif/else, for, while, break, continue)
15. Augmented assignments (+=, -=, *=, /=)
16. Comparison chaining (a < b < c)
17. Boolean operators (and, or, not)
18. Ternary expressions
19. Struct definitions with dataclasses
20. Matrix operations

Example Usage:
    # Interactive preview
    py2glsl show examples/syntax_features.py

    # Render image
    py2glsl image examples/syntax_features.py syntax.png

    # Export to GLSL
    py2glsl export examples/syntax_features.py -o syntax.glsl
"""

from dataclasses import dataclass

from py2glsl import ShaderContext
from py2glsl.builtins import (
    abs,
    array,
    atan,
    clamp,
    cos,
    length,
    max,
    min,
    mix,
    sin,
    smoothstep,
    vec2,
    vec3,
    vec4,
)

# =============================================================================
# GLOBAL CONSTANTS - Evaluated at compile time
# =============================================================================

# Simple constants (type inferred)
PI = 3.14159
MAX_STEPS = 64
DEBUG_MODE = False

# Annotated constants
EPSILON: float = 0.001
LAYER_COUNT: int = 4

# Constant expressions - computed at transpile time!
TAU = PI * 2.0
HALF_PI = PI / 2.0
QUARTER_PI = PI / 4.0

# Mutable global (will be non-const in GLSL)
frame_accumulator = 0.0


# =============================================================================
# STRUCTS - Using dataclasses
# =============================================================================


@dataclass
class ColorPalette:
    """A color palette with primary and accent colors."""

    primary: vec3
    accent: vec3
    background: vec3
    intensity: float


@dataclass
class ShapeResult:
    """Result of a shape distance calculation."""

    distance: float
    color: vec3
    glow: float


# =============================================================================
# HELPER FUNCTIONS - Various signatures
# =============================================================================


def rotate2d(p: vec2, angle: float) -> vec2:
    """Rotate a 2D point around origin."""
    c = cos(angle)
    s = sin(angle)
    return vec2(p.x * c - p.y * s, p.x * s + p.y * c)


# Default parameters - filled in at call site when omitted
def blend_colors(a: vec3, b: vec3, t: float = 0.5) -> vec3:
    """Blend two colors with optional mix factor."""
    return mix(a, b, t)


def adjust_brightness(
    color: vec3, brightness: float = 1.0, contrast: float = 1.0
) -> vec3:
    """Adjust color brightness and contrast with defaults."""
    adjusted = (color - 0.5) * contrast + 0.5
    return adjusted * brightness


# Tuple return - returns vec2 in GLSL
def to_polar(p: vec2) -> tuple[float, float]:
    """Convert cartesian to polar coordinates."""
    return length(p), atan(p.y, p.x)


def get_bounds(v: vec3) -> tuple[float, float, float]:
    """Get min, max, and range of vector components."""
    # min/max with multiple arguments - chains into nested calls
    lo = min(v.x, v.y, v.z)
    hi = max(v.x, v.y, v.z)
    return lo, hi, hi - lo


# =============================================================================
# DISTANCE FUNCTIONS - Using various features
# =============================================================================


def sd_circle(p: vec2, radius: float) -> float:
    """Signed distance to circle."""
    return length(p) - radius


def sd_box(p: vec2, size: vec2) -> float:
    """Signed distance to box."""
    d = abs(p) - size
    outside = length(max(d, vec2(0.0, 0.0)))
    # Negative indexing - d[-1] is d[1] (d.y)
    inside = min(max(d.x, d[-1]), 0.0)
    return outside + inside


def sd_rounded_box(p: vec2, size: vec2, radius: float = 0.1) -> float:
    """Signed distance to rounded box with default corner radius."""
    return sd_box(p, size - vec2(radius, radius)) - radius


def sd_ring(p: vec2, outer: float, thickness: float = 0.1) -> float:
    """Signed distance to ring with default thickness."""
    return abs(length(p) - outer) - thickness


# =============================================================================
# MAIN SHADER
# =============================================================================


def shader(ctx: ShaderContext) -> vec4:
    """Main shader demonstrating all Python syntax features."""

    # Tuple unpacking from vec2
    u, v = ctx.vs_uv

    # Center and aspect-correct coordinates
    uv = vec2(u - 0.5, v - 0.5) * 2.0
    uv.x *= ctx.u_aspect

    # Use mutable global
    frame_accumulator = ctx.u_time * 0.1

    # ---------------------------------------------------------------------
    # POWER OPTIMIZATION: x**2 -> x*x, x**0.5 -> sqrt(x)
    # ---------------------------------------------------------------------

    # These get optimized at transpile time:
    dist_squared = uv.x**2 + uv.y**2  # -> uv.x * uv.x + uv.y * uv.y
    dist = dist_squared**0.5  # -> sqrt(dist_squared)

    # x**3 and x**4 also optimized
    cubic_falloff = (1.0 - dist) ** 3  # -> (1-dist)*(1-dist)*(1-dist)

    # ---------------------------------------------------------------------
    # TUPLE UNPACKING AND POLAR COORDINATES
    # ---------------------------------------------------------------------

    # Tuple return + unpacking
    radius, angle = to_polar(uv)

    # Get bounds using multi-arg min/max
    test_vec = vec3(sin(ctx.u_time), cos(ctx.u_time * 1.3), sin(ctx.u_time * 0.7))
    lo, hi, range_val = get_bounds(test_vec)

    # ---------------------------------------------------------------------
    # INTEGER/FLOOR DIVISION
    # ---------------------------------------------------------------------

    # Floor division on floats -> floor(a / b)
    grid_x = uv.x // 0.25
    grid_y = uv.y // 0.25

    # ---------------------------------------------------------------------
    # ARRAYS AND NEGATIVE INDEXING
    # ---------------------------------------------------------------------

    # Array with type annotation
    colors: array[vec3, 4] = [  # type: ignore[valid-type]
        vec3(1.0, 0.2, 0.3),  # Red
        vec3(0.2, 1.0, 0.3),  # Green
        vec3(0.2, 0.3, 1.0),  # Blue
        vec3(1.0, 0.8, 0.2),  # Yellow
    ]

    # Negative indexing on array
    last_color = colors[-1]  # colors[3] - Yellow
    second_last = colors[-2]  # colors[2] - Blue

    # Negative indexing on vec4
    test_v = vec4(1.0, 2.0, 3.0, 4.0)
    last_component = test_v[-1]  # test_v[3] = 4.0

    # Tuple unpacking from vec3
    r, g, b = last_color

    # Tuple unpacking from swizzle
    x_comp, y_comp = uv.xy

    # Use grid coordinates for pattern
    grid_pattern = sin(grid_x * 3.14159) * sin(grid_y * 3.14159)

    # Use bounds values for normalization
    normalized_val = (test_vec.x - lo) / max(range_val, 0.001)

    # Use unpacked components
    unpacked_color = vec3(r * normalized_val, g * grid_pattern, b * cubic_falloff)

    # Use second_last, last_component, x_comp, y_comp
    detail = second_last * last_component * 0.1
    coord_factor = (x_comp + y_comp) * 0.5 + hi * 0.01

    # ---------------------------------------------------------------------
    # CONTROL FLOW
    # ---------------------------------------------------------------------

    color = vec3(0.0, 0.0, 0.0)

    # For loop with constant bound
    for i in range(LAYER_COUNT):
        layer_angle = angle + float(i) * QUARTER_PI + ctx.u_time * 0.5
        layer_radius = 0.3 + float(i) * 0.15

        # Rotate point for this layer
        rotated = rotate2d(uv, layer_angle * 0.5)

        # Distance to shape (using default parameter)
        d = sd_rounded_box(rotated, vec2(layer_radius, layer_radius * 0.5))

        # If/elif/else
        if i == 0:
            shape_color = colors[0]
        elif i == 1:
            shape_color = colors[1]
        elif i == 2:
            shape_color = colors[2]
        else:
            shape_color = colors[3]

        # Smooth shape edge
        shape_mask = smoothstep(0.02, 0.0, d)

        # Blend using default parameter (t=0.5 omitted)
        blended = blend_colors(color, shape_color)
        color = mix(color, blended, shape_mask * 0.8)

    # While loop with break
    iterations = 0
    glow = 0.0
    while iterations < 8:
        ring_dist = sd_ring(uv, 0.2 + float(iterations) * 0.1)
        glow += 0.02 / (abs(ring_dist) + 0.01)
        iterations += 1
        if glow > 1.0:
            break

    # Add glow
    glow_color = vec3(0.5, 0.7, 1.0)
    color += glow_color * min(glow, 1.0) * 0.3

    # ---------------------------------------------------------------------
    # COMPARISON CHAINING AND BOOLEAN OPS
    # ---------------------------------------------------------------------

    # Comparison chaining: a < b < c -> (a < b) && (b < c)
    in_center = -0.5 < uv.x < 0.5 and -0.5 < uv.y < 0.5

    # Boolean operations
    on_edge = (abs(uv.x) > 0.9 or abs(uv.y) > 0.9) and not in_center

    if on_edge:
        color = vec3(0.1, 0.1, 0.1)

    # ---------------------------------------------------------------------
    # TERNARY EXPRESSION
    # ---------------------------------------------------------------------

    # Conditional expression
    final_alpha = 1.0 if radius < 1.5 else 0.5

    # ---------------------------------------------------------------------
    # CONSTANT EXPRESSIONS (computed at compile time)
    # ---------------------------------------------------------------------

    # TAU, HALF_PI, QUARTER_PI are pre-computed constants
    # frame_accumulator is a mutable global that gets updated
    wave = sin(angle * 4.0 + frame_accumulator * TAU) * 0.5 + 0.5
    color = mix(color, color * 1.2, wave * 0.3)

    # ---------------------------------------------------------------------
    # STRUCT USAGE
    # ---------------------------------------------------------------------

    palette = ColorPalette(
        primary=vec3(0.9, 0.3, 0.4),
        accent=vec3(0.3, 0.8, 0.9),
        background=vec3(0.05, 0.05, 0.1),
        intensity=1.2,
    )

    # Apply palette
    color = mix(palette.background, color, smoothstep(1.5, 0.0, radius))
    color = color * palette.intensity

    # ---------------------------------------------------------------------
    # AUGMENTED ASSIGNMENTS
    # ---------------------------------------------------------------------

    color *= 1.1  # Exposure boost
    color += vec3(0.02, 0.01, 0.03)  # Color lift

    # Mix in unpacked color, detail, and coord_factor
    color = mix(color, unpacked_color, 0.05)
    color += detail * 0.01
    color *= 1.0 + coord_factor * 0.01

    # Clamp final color
    color = clamp(color, vec3(0.0, 0.0, 0.0), vec3(1.0, 1.0, 1.0))

    # ---------------------------------------------------------------------
    # BRIGHTNESS ADJUSTMENT WITH MULTIPLE DEFAULTS
    # ---------------------------------------------------------------------

    # Call with no defaults -> uses brightness=1.0, contrast=1.0
    color = adjust_brightness(color)

    # Call with one default -> contrast=1.0 filled in
    # color = adjust_brightness(color, 1.1)

    return vec4(color, final_alpha)
