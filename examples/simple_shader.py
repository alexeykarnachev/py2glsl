"""Animated Plasma Shader Example

This example demonstrates a classic "plasma" effect - a colorful, animated
pattern created using sine waves and distance functions. It's an excellent
starting point for beginners to shader programming with py2glsl.

The shader creates a hypnotic, pulsating pattern by:
1. Normalizing UV coordinates to center the effect
2. Calculating distance from center points
3. Using sine waves with time to create smooth animation
4. Mapping the results to RGB color components

Key concepts demonstrated:
1. Using built-in GLSL functions (sin, length)
2. Utilizing uniform variables (time, aspect ratio)
3. Proper UV coordinate handling with aspect ratio correction
4. Creating smooth animations with trigonometric functions

Perfect for:
- Learning shader basics
- Creating background effects
- Understanding animation principles

To run this example:
    # Interactive preview
    py2glsl show run examples/simple_shader.py
    # Render image
    py2glsl image render examples/simple_shader.py plasma.png
    # Create animation
    py2glsl gif render examples/simple_shader.py plasma.gif --duration 3
    # Export code for external use
    py2glsl code export examples/simple_shader.py plasma.glsl
    # Export for Shadertoy
    py2glsl code export examples/simple_shader.py shadertoy.glsl --target shadertoy \
        --shadertoy-compatible --main shader
"""

from py2glsl import ShaderContext
from py2glsl.builtins import length, sin, vec2, vec4

# Global constant with type annotation
PI: float = 3.14159


def shader(ctx: ShaderContext) -> vec4:
    """A simple animated plasma shader."""
    # Center UV coordinates
    uv = ctx.vs_uv * 2.0 - vec2(1.0, 1.0)
    uv.x *= ctx.u_aspect

    # Calculate distance from center
    d = length(uv)

    # Create animated color based on distance and time
    color = sin(d * 10.0 - ctx.u_time * 2.0) * 0.5 + 0.5

    # Return RGBA color
    return vec4(color, color * 0.5, 1.0 - color, 1.0)
