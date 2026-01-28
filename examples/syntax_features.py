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
12. Arrays with typed annotations (list[T] with size inference)
13. Vector swizzling and component access
14. Control flow (if/elif/else, for, while, break, continue)
15. Augmented assignments (+=, -=, *=, /=)
16. Comparison chaining (a < b < c)
17. Boolean operators (and, or, not)
18. Ternary expressions
19. Struct definitions with dataclasses
20. Matrix operations
21. List comprehensions (unrolled at compile time)
22. for item in array (direct array iteration)
23. enumerate(array) (index + element iteration)
24. Filtered list comprehensions ([x for x in range(n) if cond])
25. Nested list comprehensions ([f(i,j) for i in range(n) for j in range(m)])
26. sum() builtin (array sum, generator sum)
27. len() builtin (compile-time array/vector length)
28. Walrus operator (:= inline assignment)
29. Regular classes with __init__ (auto-converted to structs)
30. Instance methods (converted to functions with struct as first param)

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

# Constants for filtered comprehensions
NUM_SAMPLES = 8

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
# CLASSES - Regular Python classes (converted to structs with methods)
# =============================================================================


class Circle:
    """A circle defined by center and radius, with distance method."""

    def __init__(self, center: vec2, radius: float):
        self.center = center
        self.radius = radius

    def signed_distance(self, point: vec2) -> float:
        """Return signed distance from point to circle edge."""
        return length(point - self.center) - self.radius

    def contains(self, point: vec2) -> float:
        """Return 1.0 if point is inside, 0.0 otherwise."""
        d = self.signed_distance(point)
        return 1.0 if d < 0.0 else 0.0


class Transform2D:
    """A 2D transformation with position and scale."""

    def __init__(self, position: vec2, scale: float):
        self.position = position
        self.scale = scale

    def apply(self, point: vec2) -> vec2:
        """Apply the transformation to a point."""
        return (point - self.position) / self.scale

    def apply_inverse(self, point: vec2) -> vec2:
        """Apply the inverse transformation."""
        return point * self.scale + self.position


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

    # Array with type annotation (size inferred from literal)
    colors: list[vec3] = [
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

    # ---------------------------------------------------------------------
    # LIST COMPREHENSIONS - Unrolled at compile time
    # ---------------------------------------------------------------------

    # Simple list comprehension - generates array initialization
    weights: list[float] = [1.0 / float(i + 1) for i in range(4)]
    # Transpiles to: float[4](1.0, 0.5, 0.333..., 0.25)

    # List comprehension with expression
    offsets: list[float] = [float(i) * 0.1 - 0.2 for i in range(5)]
    # Transpiles to: float[5](-0.2, -0.1, 0.0, 0.1, 0.2)

    # List comprehension with step
    even_values: list[float] = [float(i) for i in range(0, 8, 2)]
    # Transpiles to: float[4](0.0, 2.0, 4.0, 6.0)

    # List comprehension for vec3 array
    gradient_colors: list[vec3] = [
        vec3(float(i) * 0.5, 1.0 - float(i) * 0.3, 0.5) for i in range(3)
    ]

    # List comprehension with structs
    sample_palettes: list[ColorPalette] = [
        ColorPalette(
            primary=vec3(float(i) * 0.3, 0.2, 0.1),
            accent=vec3(0.1, float(i) * 0.3, 0.2),
            background=vec3(0.05),
            intensity=1.0 + float(i) * 0.1,
        )
        for i in range(3)
    ]

    # Use the list comprehension results
    weighted_sum = (
        weights[0] * offsets[0]
        + weights[1] * offsets[1]
        + weights[2] * offsets[2]
        + weights[3] * offsets[3]
    )

    # Use even_values and gradient_colors
    pattern_val = sin(even_values[1] * uv.x + even_values[2] * uv.y + ctx.u_time)
    blend_t = pattern_val * 0.5 + 0.5
    gradient_sample = mix(gradient_colors[0], gradient_colors[2], blend_t)

    # ---------------------------------------------------------------------
    # FILTERED LIST COMPREHENSIONS - With compile-time if conditions
    # ---------------------------------------------------------------------

    # Only include even indices: [0, 2, 4, 6]
    even_indices: list[int] = [i for i in range(NUM_SAMPLES) if i % 2 == 0]
    # Transpiles to: int[4](0, 2, 4, 6)

    # Only include values where condition holds
    small_offsets: list[float] = [
        float(i) * 0.1 for i in range(NUM_SAMPLES) if i % 3 != 0
    ]

    # ---------------------------------------------------------------------
    # NESTED LIST COMPREHENSIONS - Multiple for clauses
    # ---------------------------------------------------------------------

    # 2D grid: generates 3x3 = 9 elements
    grid_values: list[float] = [float(i + j * 3) for i in range(3) for j in range(3)]
    # Transpiles to: float[9](0, 3, 6, 1, 4, 7, 2, 5, 8)

    # Use grid_values and filtered results
    grid_sum_val = grid_values[0] + grid_values[4] + grid_values[8]

    # ---------------------------------------------------------------------
    # sum() BUILTIN - Unrolled addition
    # ---------------------------------------------------------------------

    # sum() over an array - unrolls to chained additions
    total_weight = sum(weights)
    # Transpiles to: weights[0] + weights[1] + weights[2] + weights[3]

    # sum() over a generator expression
    gen_total = sum(float(i) * 0.1 for i in range(4))
    # Transpiles to: float(0) * 0.1 + float(1) * 0.1 + ...

    # ---------------------------------------------------------------------
    # len() BUILTIN - Compile-time constant
    # ---------------------------------------------------------------------

    # len() on array - replaced with compile-time constant
    num_colors = len(colors)
    # Transpiles to: int num_colors = 4;

    # Use len() in expression
    color_scale = 1.0 / float(num_colors)

    # ---------------------------------------------------------------------
    # WALRUS OPERATOR (:=) - Inline assignment
    # ---------------------------------------------------------------------

    # Walrus operator hoists the assignment before the if statement
    # Python: if (d := length(p)) < threshold:
    # GLSL:   float d = length(p); if (d < threshold) { ... }
    walrus_color = vec3(0.0)
    center_p = uv * 0.5
    if (center_d := length(center_p)) < 0.4:
        walrus_color = vec3(1.0 - center_d * 2.0, center_d, 0.5)

    # ---------------------------------------------------------------------
    # FOR ITEM IN ARRAY - Direct array iteration
    # ---------------------------------------------------------------------

    # Iterate directly over array elements (no index needed)
    color_sum = vec3(0.0)
    for c in colors:
        color_sum = color_sum + c
    # Transpiles to: for (int _i0 = 0; _i0 < 4; _i0++) {
    #     vec3 c = colors[_i0]; color_sum = color_sum + c; }

    avg_color = color_sum * color_scale

    # ---------------------------------------------------------------------
    # ENUMERATE - Index + element iteration
    # ---------------------------------------------------------------------

    # Iterate with both index and element
    weighted_color = vec3(0.0)
    for idx, col in enumerate(colors):
        weighted_color = weighted_color + col * float(idx)
    # Transpiles to: for (int idx = 0; idx < 4; idx++) {
    #     vec3 col = colors[idx]; weighted_color += col * float(idx); }

    # ---------------------------------------------------------------------
    # CLASSES WITH METHODS - Regular Python classes
    # ---------------------------------------------------------------------

    # Create a circle using regular class with __init__
    circle = Circle(vec2(0.5, 0.5), 0.25)
    # Transpiles to: Circle circle = Circle(vec2(0.5, 0.5), 0.25);

    # Call instance method - transpiles to function call with struct as first arg
    circle_dist = circle.signed_distance(ctx.vs_uv)
    # Transpiles to: float circle_dist = Circle_signed_distance(circle, vs_uv);

    # Use method result
    circle_mask = circle.contains(ctx.vs_uv)
    # Transpiles to: float circle_mask = Circle_contains(circle, vs_uv);

    # Another class with methods
    transform = Transform2D(vec2(0.5, 0.5), 2.0)
    transformed_uv = transform.apply(ctx.vs_uv)
    # Transpiles to: vec2 transformed_uv = Transform2D_apply(transform, vs_uv);

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

    # Mix in list comprehension results
    color = mix(color, gradient_sample, 0.1)
    color += vec3(weighted_sum * 0.01)

    # Use struct list comprehension result
    color = mix(color, sample_palettes[1].primary, 0.05)

    # Mix in new feature results
    color = mix(color, avg_color, 0.05)
    color = mix(color, walrus_color, 0.1)
    color += weighted_color * 0.01
    color += vec3(total_weight * 0.01, gen_total * 0.01, grid_sum_val * 0.001)
    color += vec3(float(even_indices[0]) * 0.001)
    color += vec3(small_offsets[0] * 0.001)
    color += vec3(grid_values[0] * 0.001)

    # Mix in class/method results
    color = mix(color, vec3(1.0, 0.5, 0.0), circle_mask * 0.1)
    color += vec3(circle_dist * 0.01)
    color += vec3(transformed_uv.x * 0.01, transformed_uv.y * 0.01, 0.0)

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
