from py2glsl import vec2, vec4
from py2glsl.renderer import animate


def shader(vs_uv: vec2, /, u_time: float) -> vec4:
    return vec4(vs_uv, u_time, 1.0)


if __name__ == "__main__":
    animate(shader)
