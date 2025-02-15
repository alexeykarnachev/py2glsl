from py2glsl import py2glsl, sin, vec2, vec3, vec4


def shader(vs_uv: vec2, *, u_aspect: float) -> vec4:
    p = vs_uv
    p.x *= u_aspect
    return vec4(p, 0.0, 1.0)


result = py2glsl(shader)
print(result.fragment_source)

from py2glsl import animate

animate(shader)
