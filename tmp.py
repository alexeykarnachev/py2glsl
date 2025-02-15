from dataclasses import dataclass

from py2glsl import (
    abs,
    acos,
    atan,
    cos,
    cross,
    length,
    max,
    min,
    normalize,
    pi,
    py2glsl,
    radians,
    round,
    sin,
    tan,
    vec2,
    vec3,
    vec4,
)

# Constants
RM_MAX_DIST = 10000.0
RM_MAX_N_STEPS = 64
RM_EPS = 0.0001
NORMAL_DERIVATIVE_STEP = 0.015


@dataclass
class RayMarchResult:
    i: int
    p: vec3
    ro: vec3
    rd: vec3
    dist: float
    sd_last: float
    sd_min: float
    sd_min_shape: float
    normal: vec3
    has_normal: bool


def get_sd_shape(p: vec3) -> float:
    return length(p) - 1.0


def quantize_normal(v: vec3, q: float) -> vec3:
    # Convert to spherical coordinates
    t = atan(v.y, v.x)
    p = acos(v.z)

    # Quantize angles
    qt = round(t / q) * q
    qp = round(p / q) * q

    # Back to Cartesian
    sp = sin(qp)
    return vec3(sp * cos(qt), sp * sin(qt), cos(qp))


def attenuate(d: float, coeffs: vec3) -> float:
    return 1.0 / (coeffs.x + coeffs.y * d + coeffs.z * d * d)


def march(ro: vec3, rd: vec3) -> RayMarchResult:
    rm = RayMarchResult(
        i=0,
        p=ro,
        ro=ro,
        rd=rd,
        dist=0.0,
        sd_last=0.0,
        sd_min=RM_MAX_DIST,
        sd_min_shape=RM_MAX_DIST,
        normal=vec3(0.0),
        has_normal=False,
    )

    for i in range(RM_MAX_N_STEPS):
        rm.p = rm.p + rm.rd * rm.sd_last
        sd_step_shape = get_sd_shape(rm.p)

        rm.sd_last = sd_step_shape
        rm.sd_min_shape = min(rm.sd_min_shape, sd_step_shape)
        rm.sd_min = min(rm.sd_min, sd_step_shape)
        rm.dist += length(rm.p - rm.ro)

        if rm.sd_last < RM_EPS or rm.dist > RM_MAX_DIST:
            if rm.sd_last < RM_EPS:
                rm.normal = vec3(1.0)
            break

    # Normals
    if rm.sd_last < RM_EPS:
        if rm.sd_last == rm.sd_min_shape:
            e = vec2(NORMAL_DERIVATIVE_STEP, 0.0)
            rm.normal = normalize(
                vec3(
                    get_sd_shape(rm.p + vec3(e.x, e.y, e.y))
                    - get_sd_shape(rm.p - vec3(e.x, e.y, e.y)),
                    get_sd_shape(rm.p + vec3(e.y, e.x, e.y))
                    - get_sd_shape(rm.p - vec3(e.y, e.x, e.y)),
                    get_sd_shape(rm.p + vec3(e.y, e.y, e.x))
                    - get_sd_shape(rm.p - vec3(e.y, e.y, e.x)),
                )
            )
            rm.has_normal = True

    return rm


def main_shader(vs_uv: vec2, *, u_time: float, u_aspect: float) -> vec4:
    # Screen position
    screen_pos = vs_uv * 2.0 - 1.0
    screen_pos.x *= u_aspect

    # Camera setup
    fov = radians(70.0)
    screen_dist = 1.0 / tan(0.5 * fov)
    cam_pos = vec3(5.0, 5.0, 5.0)
    look_at = vec3(0.0, 0.0, 0.0)

    # Camera basis vectors
    forward = normalize(look_at - cam_pos)
    world_up = vec3(0.0, 1.0, 0.0)
    right = normalize(cross(forward, world_up))
    up = normalize(cross(right, forward))

    # Ray setup
    screen_center = cam_pos + forward * screen_dist
    sp = screen_center + right * screen_pos.x + up * screen_pos.y

    ro0 = cam_pos
    rd0 = normalize(sp - cam_pos)

    # Orthographic
    ro1 = sp * 4.0
    rd1 = normalize(look_at - cam_pos)

    # Mix perspective and orthographic
    ro = mix(ro0, ro1, 1.0)
    rd = mix(rd0, rd1, 1.0)

    # Ray march
    rm = march(ro, rd)

    # Color
    d = abs(max(0.0, rm.sd_min_shape))
    a = attenuate(d, vec3(0.01, 8.0, 8.0))

    color = vec3(0.0)
    if rm.has_normal:
        normal = quantize_normal(rm.normal, pi / 8)
        color = abs(normal)

    return vec4(color, 1.0)


# Generate GLSL code
result = py2glsl(main_shader)
print(result.fragment_source)

# Or display in real-time
from py2glsl import animate

animate(main_shader)
