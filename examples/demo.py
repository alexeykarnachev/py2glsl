from py2glsl import animate, vec2, vec4
from py2glsl.glsl.builtins import abs, cos, fract, length, mix, sin, smoothstep, sqrt


def hex_grid(uv: vec2, scale: float) -> float:
    """Generate hexagonal grid pattern"""
    uv *= scale * vec2(2.0, sqrt(3.0))
    hex_coord = fract(uv) - 0.5
    return abs(hex_coord.x + hex_coord.y * 0.5)


def dynamic_colors(intensity: float, phase: float) -> vec4:
    """Create dynamic color palette with phase offset"""
    r = sin(intensity * 6.0 + phase) * 0.5 + 0.5
    g = cos(intensity * 4.0 + phase * 1.2) * 0.5 + 0.5
    b = sin(intensity * 3.0 + phase * 0.8) * 0.5 + 0.5
    return vec4(r, g, b, 1.0)


def mouse_interaction(uv: vec2, mouse_uv: vec2, time: float) -> float:
    """Create radial waves from mouse position"""
    mouse_dist = length(uv - mouse_uv)
    return sin(mouse_dist * 15.0 - time * 4.0) * 0.5 + 0.5


def post_processing(color: vec4, uv: vec2, time: float) -> vec4:
    """Apply final screen effects"""
    # Vignette
    vignette = 1.0 - length(uv) * 0.5
    color.rgb *= vignette**2

    # Scanlines
    scanline = sin(uv.y * 1000.0 + time * 5.0) * 0.1 + 0.9
    color.rgb *= scanline

    return color


def main_shader(
    vs_uv: vec2,
    /,
    u_time: float,
    u_resolution: vec2,
    u_speed: float = 1.0,
    u_scale: float = 5.0,
    u_color_speed: float = 0.5,
    u_mouse_uv: vec2 = vec2(0.5),
) -> vec4:
    # Normalize coordinates
    uv = vs_uv * vec2(u_resolution.x / u_resolution.y, 1.0)
    t = u_time * u_speed

    # Pattern layers
    hex_pattern = hex_grid(uv, u_scale)
    plasma_wave = sin(length(uv * 2.0) * 10.0 - t * 2.0)
    mouse_wave = mouse_interaction(uv, u_mouse_uv, t)

    # Combine effects
    intensity = (plasma_wave * 0.6 + hex_pattern * 0.4 + mouse_wave * 0.3) / 2.0

    # Color processing
    color = dynamic_colors(intensity, t * u_color_speed)
    return post_processing(color, vs_uv, t)


if __name__ == "__main__":
    animate(main_shader, size=(1200, 800))
