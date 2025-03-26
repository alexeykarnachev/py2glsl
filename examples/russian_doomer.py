from py2glsl.builtins import sin, vec2, vec4


def shader(vs_uv: vec2, u_time: float, u_aspect: float) -> vec4:
    blue = 0.5 * (sin(u_time) + 1.0)
    color = vec4(vs_uv, blue, 1.0)
    return color
