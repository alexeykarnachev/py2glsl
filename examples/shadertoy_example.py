"""Example demonstrating the Shadertoy backend for py2glsl."""

from py2glsl.builtins import length, mix, sin, smoothstep, vec2, vec3, vec4
from py2glsl.render import render_image
from py2glsl.transpiler import transpile
from py2glsl.transpiler.backends.models import BackendType


def shadertoy_shader(vs_uv: vec2, u_time: float) -> vec4:
    """A simple Shadertoy-compatible shader.

    Args:
        vs_uv: UV coordinates (0-1)
        u_time: Current time in seconds

    Returns:
        Final color (RGBA)
    """
    # Create a simple color gradient based on UV coordinates
    col = vec3(vs_uv.x, vs_uv.y, sin(u_time) * 0.5 + 0.5)

    # Add a pulsing circle
    center = vec2(0.5, 0.5)
    radius = 0.25 + sin(u_time * 2.0) * 0.05
    dist = length(vs_uv - center)
    circle = smoothstep(radius + 0.01, radius, dist)

    # Mix the circle with the background
    final_color = mix(col, vec3(1.0, 1.0, 1.0), circle)

    # Create vec4 with alpha=1.0
    return vec4(final_color.x, final_color.y, final_color.z, 1.0)


if __name__ == "__main__":
    # Transpile to Shadertoy GLSL
    glsl_code, used_uniforms = transpile(
        shadertoy_shader, backend_type=BackendType.SHADERTOY
    )

    print("\nGenerated Shadertoy GLSL code:")
    print("-" * 40)
    print(glsl_code)
    print("-" * 40)
    print(f"Used uniforms: {used_uniforms}")

    from py2glsl.render import render_image

    # Render the shader with Shadertoy backend
    try:
        # Render a still image
        image = render_image(
            shadertoy_shader,
            size=(800, 600),
            time=1.0,  # Set a specific time value
            backend_type=BackendType.SHADERTOY,
            output_path="shadertoy_output.png",
        )
        print("Image saved to shadertoy_output.png")
    except Exception as e:
        print(f"Error rendering image: {e}")
