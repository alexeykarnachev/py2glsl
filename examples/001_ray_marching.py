from py2glsl import animate, transpile
from py2glsl.builtins import (
    RayMarchResult,
    abs,
    cos,
    cross,
    length,
    max,
    min,
    mix,
    normalize,
    radians,
    sin,
    tan,
    vec2,
    vec3,
    vec4,
)

# Global constants
PI: float = 3.141592
RM_MAX_DIST: float = 10000.0
RM_MAX_N_STEPS: int = 64
RM_EPS: float = 0.0001
NORMAL_DERIVATIVE_STEP: float = 0.015


def get_sd_shape(p: vec3) -> float:
    d = length(max(abs(p) - 1.0, 0.0)) - 0.2
    return d


def march(ro: vec3, rd: vec3) -> RayMarchResult:
    rm: RayMarchResult = RayMarchResult(
        0, ro, vec3(0.0), ro, rd, 0.0, 0.0, RM_MAX_DIST, RM_MAX_DIST
    )
    for i in range(RM_MAX_N_STEPS):
        rm.p = rm.p + rm.rd * rm.sd_last
        sd_step_shape = get_sd_shape(rm.p)
        rm.sd_last = sd_step_shape
        rm.sd_min_shape = min(rm.sd_min_shape, sd_step_shape)
        rm.sd_min = min(rm.sd_min, sd_step_shape)
        rm.dist = rm.dist + length(rm.p - rm.ro)
        if rm.sd_last < RM_EPS or rm.dist > RM_MAX_DIST:
            if rm.sd_last < RM_EPS:
                rm.n = vec3(1.0)
            break

    if rm.sd_last < RM_EPS:
        h = RM_EPS
        eps = vec3(h, 0.0, 0.0)
        rm.n = vec3(0.0)
        if rm.sd_last == rm.sd_min_shape:
            e = vec2(NORMAL_DERIVATIVE_STEP, 0.0)
            rm.n = normalize(
                vec3(
                    get_sd_shape(rm.p + e.xyy) - get_sd_shape(rm.p - e.xyy),
                    get_sd_shape(rm.p + e.yxy) - get_sd_shape(rm.p - e.yxy),
                    get_sd_shape(rm.p + e.yyx) - get_sd_shape(rm.p - e.yyx),
                )
            )
    return rm


def sin01(x: float, a: float, f: float, phase: float) -> float:
    return a * 0.5 * (sin(PI * f * (x + phase)) + 1.0)


def attenuate(d: float, coeffs: vec3) -> float:
    return 1.0 / (coeffs.x + coeffs.y * d + coeffs.z * d * d)


def main_shader(vs_uv: vec2, u_time: float, u_aspect: float) -> vec4:
    screen_pos = vs_uv * 2.0 - 1.0
    screen_pos.x *= u_aspect

    fov = radians(70.0)
    screen_dist = 1.0 / tan(0.5 * fov)
    cam_pos = vec3(5.0 * sin(u_time), 5.0, 5.0 * cos(u_time))
    look_at = vec3(0.0, 0.0, 0.0)

    forward = normalize(look_at - cam_pos)
    world_up = vec3(0.0, 1.0, 0.0)
    right = normalize(cross(forward, world_up))
    up = normalize(cross(right, forward))

    screen_center = cam_pos + forward * screen_dist
    sp = screen_center + right * screen_pos.x + up * screen_pos.y

    ro0 = cam_pos
    rd0 = normalize(sp - cam_pos)
    ro1 = sp * 4.0
    rd1 = normalize(look_at - cam_pos)

    ro = mix(ro0, ro1, 1.0)
    rd = mix(rd0, rd1, 1.0)
    rm = march(ro, rd)

    color = vec3(0.0)
    d = abs(max(0.0, rm.sd_min_shape))
    a = attenuate(d, vec3(0.01, 8.0, 8.0))
    color = 1.0 * abs(rm.n)

    return vec4(color, 1.0)


if __name__ == "__main__":
    glsl_code, used_uniforms = transpile(main_shader)
    animate(glsl_code, used_uniforms)
