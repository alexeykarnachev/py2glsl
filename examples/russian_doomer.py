from py2glsl.builtins import sin, vec2, vec4


def shader(vs_uv: vec2, u_time: float, u_aspect: float) -> vec4:
    """Shader example demonstrating flexible vector construction."""
    # Using flexible vector construction:
    blue = 0.5 * (sin(u_time) + 1.0)

    # Vector construction with vec2 + scalar components
    color = vec4(vs_uv, blue, 1.0)

    # Usage of scalar constructor
    # Use the color directly instead of creating a separate variable
    return color
