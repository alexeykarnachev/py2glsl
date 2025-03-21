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
    # Test 1: Using the new target_type parameter
    print("Testing with new target_type parameter (GLSL)")
    glsl_code, uniforms = transpile(
        color_wheel, target_type=TargetLanguageType.GLSL
    )
    print(f"Generated shader with {len(glsl_code)} characters")
    print(f"Uniforms: {uniforms}")

    # Render an image
    render_image(
        glsl_code,
        size=(400, 400),
        output_path="test_glsl_target.png",
        time=1.0
    )


if __name__ == "__main__":
    main()
