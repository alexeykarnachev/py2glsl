from py2glsl import animate, vec2, vec4
from py2glsl.glsl.builtins import atan, cos, length, sin


def plasma(vs_uv: vec2, /, *, u_time: float, u_resolution: vec2) -> vec4:
    # Normalize and center UV
    uv = vs_uv * 2.0 - 1.0
    uv.x *= u_resolution.x / u_resolution.y

    # Animated parameters
    time = u_time * 0.5
    speed = 0.5

    # Base pattern
    d = length(uv)
    angle = atan(uv.y, uv.x)

    # Layer 1: Rotating circles
    layer1 = sin(d * 12.0 + time * 3.0) * 0.5 + 0.5

    # Layer 2: Angular waves
    layer2 = cos(angle * 6.0 + time * 2.0) * 0.3 + 0.7

    # Layer 3: Pulsing grid
    layer3 = sin(uv.x * 15.0 + time * 1.4) * sin(uv.y * 10.0 + time * 1.1) * 0.8 + 0.2

    # Combine layers
    intensity = (layer1 + layer2 + layer3) / 3.0

    # Color gradient
    r = sin(intensity * 6.0 + time) * 0.5 + 0.5
    g = cos(intensity * 4.0 + time * 0.7) * 0.5 + 0.5
    b = sin(intensity * 3.0 + time * 1.3) * 0.5 + 0.5

    return vec4(r, g, b, 1.0)


if __name__ == "__main__":
    animate(plasma, size=(800, 600))
