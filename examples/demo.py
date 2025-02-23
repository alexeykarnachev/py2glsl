from py2glsl import animate, vec2, vec4
from py2glsl.glsl.builtins import abs, cos, fract, length, mix, sin, smoothstep, sqrt


def psychedelic_plasma(
    vs_uv: vec2,
    /,
    u_time: float,
    u_resolution: vec2,
    u_speed: float = 1.0,
    u_scale: float = 5.0,
    u_color_speed: float = 0.5,
    u_mouse_uv: vec2 = vec2(0.5),
) -> vec4:
    """
    Enhanced plasma effect with multiple features:
    - Dynamic color cycling
    - Multiple pattern layers
    - Post-processing effects
    - Mouse interaction
    """
    # Normalize coordinates
    uv = vs_uv
    uv.x *= u_resolution.x / u_resolution.y

    # Time variations
    t = u_time * u_speed
    moving_uv = uv * u_scale + vec2(sin(t * 0.5), cos(t * 0.3)) * 0.2

    # Gradient background
    bg = mix(vec4(0.1, 0.1, 0.3, 1.0), vec4(0.0, 0.0, 0.1, 1.0), length(uv))

    # Plasma layer 1 - Swirling colors
    d1 = length(uv + vec2(sin(t * 0.7), cos(t * 0.6)) * 0.3)
    layer1 = sin(d1 * 8.0 + t * 3.0) * 0.5 + 0.5

    # Plasma layer 2 - Hexagonal pattern
    hex_uv = vec2(moving_uv.x + moving_uv.y * 0.5, moving_uv.y * sqrt(3.0) / 2.0)
    hex_grid = abs(fract(hex_uv) - 0.5)
    hex_pattern = smoothstep(0.4, 0.5, max(hex_grid.x, hex_grid.y))
    layer2 = sin(hex_pattern * 10.0 - t * 2.0) * 0.5 + 0.5

    # Layer 3 - Mouse interaction
    mouse_dist = length(uv - u_mouse_uv)
    mouse_wave = sin(mouse_dist * 15.0 - t * 4.0) * 0.5 + 0.5

    # Combine layers
    intensity = (layer1 * 0.6 + layer2 * 0.4 + mouse_wave * 0.3) / 2.0

    # Dynamic color palette
    color_phase = t * u_color_speed
    r = sin(intensity * 6.0 + color_phase) * 0.5 + 0.5
    g = cos(intensity * 4.0 + color_phase * 1.2) * 0.5 + 0.5
    b = sin(intensity * 3.0 + color_phase * 0.8) * 0.5 + 0.5

    # Final color with post-processing
    color = vec4(r, g, b, 1.0)

    # Vignette effect
    vignette = 1.0 - length(uv) * 0.5
    color.rgb *= vignette**2

    # Scanlines
    scanline = sin(vs_uv.y * u_resolution.y * 0.7 + t * 5.0) * 0.1 + 0.9
    color.rgb *= scanline

    return color


if __name__ == "__main__":
    # Run interactive viewer with default parameters
    animate(psychedelic_plasma, size=(1200, 800))
