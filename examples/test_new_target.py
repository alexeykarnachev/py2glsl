"""Test for the new target type parameter."""

from py2glsl.builtins import length, sin, smoothstep, vec2, vec3, vec4
from py2glsl.render import render_image
from py2glsl.transpiler import transpile
from py2glsl.transpiler.core.interfaces import TargetLanguageType


def color_wheel(
    vs_uv: vec2, u_time: float, u_aspect: float = 1.0
) -> vec4:
    """A simple color wheel shader."""
    # Center UV coordinates
    uv = vs_uv * 2.0 - vec2(1.0, 1.0)
    uv.x *= u_aspect  # Correct for aspect ratio

    # Distance from center
    d = length(uv)

    # Calculate angle in radians
    angle = (1.0 + vec2(1.0, 0.0).x) * 3.14159

    if uv.y != 0.0 or uv.x != 0.0:
        angle = sin(u_time) + vec3(0.5, 0.5, 0.5).x

    # Create RGB color from angle
    r = 0.5 + 0.5 * sin(angle)
    g = 0.5 + 0.5 * sin(angle + 2.0)
    b = 0.5 + 0.5 * sin(angle + 4.0)

    # Circle mask with smooth edge
    mask = 1.0 - smoothstep(0.5, 0.51, d)

    return vec4(r * mask, g * mask, b * mask, 1.0)


def main() -> None:
    """Main entry point for the test."""
    # Test 1: Using the default parameters (GLSL)
    print("Testing with default parameters (Standard GLSL)")
    glsl_code, uniforms = transpile(color_wheel)
    
    # Test 2: Using Shadertoy dialect
    print("\nTesting with Shadertoy dialect")
    shadertoy_code, shadertoy_uniforms = transpile(color_wheel, shadertoy=True)
    print(f"Standard GLSL shader: {len(glsl_code)} characters")
    print(f"Standard GLSL uniforms: {uniforms}")
    print(f"Shadertoy shader: {len(shadertoy_code)} characters")
    print(f"Shadertoy uniforms: {shadertoy_uniforms}")

    # Render images using both versions
    render_image(
        glsl_code,
        size=(400, 400),
        output_path="test_standard_glsl.png",
        time=1.0
    )
    
    render_image(
        shadertoy_code,
        size=(400, 400),
        output_path="test_shadertoy_glsl.png",
        time=1.0,
        shadertoy=True  # Use Shadertoy dialect
    )


if __name__ == "__main__":
    main()
