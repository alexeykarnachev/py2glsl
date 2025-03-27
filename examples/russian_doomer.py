from dataclasses import dataclass

from py2glsl.builtins import (
    abs,
    cos,
    cross,
    length,
    max,
    min,
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
RM_MAX_STEPS: int = 64
RM_EPS: float = 0.0001
NORMAL_DERIVATIVE_STEP: float = 0.015


@dataclass
class RayMarchResult:
    steps: int
    p: vec3
    normal: vec3
    ro: vec3
    rd: vec3
    dist: float
    sd_last: float
    sd_min: float
    sd_min_shape: float
    has_normal: bool


def get_sd_shape(p: vec3) -> float:
    # Rounded box SDF
    d = length(max(abs(p) - vec3(1.0, 1.0, 1.0), vec3(0.0, 0.0, 0.0))) - 0.2
    return d


def march(ro: vec3, rd: vec3) -> RayMarchResult:
    # Initialize result
    rm = RayMarchResult(
        steps=0,
        p=ro,
        normal=vec3(0.0, 0.0, 0.0),
        ro=ro,
        rd=rd,
        dist=0.0,
        sd_last=0.0,
        sd_min=RM_MAX_DIST,
        sd_min_shape=RM_MAX_DIST,
        has_normal=False,
    )

    # March the ray
    for i in range(RM_MAX_STEPS):
        rm.steps = i
        rm.p = rm.p + rm.rd * rm.sd_last
        sd_step_shape = get_sd_shape(rm.p)

        rm.sd_last = sd_step_shape
        rm.sd_min_shape = min(rm.sd_min_shape, sd_step_shape)
        rm.sd_min = min(rm.sd_min, sd_step_shape)
        rm.dist = rm.dist + length(rm.p - rm.ro)

        # Check termination conditions
        if rm.sd_last < RM_EPS or rm.dist > RM_MAX_DIST:
            break

    # Calculate normal if we hit something
    if rm.sd_last < RM_EPS:
        # Use central differences for better normal quality
        e = vec2(NORMAL_DERIVATIVE_STEP, 0.0)
        rm.normal = normalize(
            vec3(
                get_sd_shape(rm.p + vec3(e.x, 0.0, 0.0))
                - get_sd_shape(rm.p - vec3(e.x, 0.0, 0.0)),
                get_sd_shape(rm.p + vec3(0.0, e.x, 0.0))
                - get_sd_shape(rm.p - vec3(0.0, e.x, 0.0)),
                get_sd_shape(rm.p + vec3(0.0, 0.0, e.x))
                - get_sd_shape(rm.p - vec3(0.0, 0.0, e.x)),
            )
        )
        rm.has_normal = True

    return rm


def shader(vs_uv: vec2, u_time: float, u_aspect: float) -> vec4:
    # Screen position
    sp = vs_uv * 2.0 - vec2(1.0, 1.0)
    sp.x *= u_aspect

    # Camera setup
    fov = radians(70.0)
    screen_dist = 1.0 / tan(0.5 * fov)

    # Animation parameters
    animation_speed = 0.3
    camera_height = 5.0
    camera_distance = 5.0

    # Normalized time for animation cycle
    angle = u_time * animation_speed

    # Calculate camera position
    cam_pos = vec3(
        camera_distance * sin(angle),  # X position
        camera_height,  # Fixed height
        camera_distance * cos(angle),  # Z position
    )
    look_at = vec3(0.0, 0.0, 0.0)

    # Camera basis vectors
    forward = normalize(look_at - cam_pos)
    world_up = vec3(0.0, 1.0, 0.0)
    right = normalize(cross(forward, world_up))
    up = normalize(cross(right, forward))

    # Ray setup
    screen_center = cam_pos + forward * screen_dist
    ray_target = screen_center + right * sp.x + up * sp.y

    # Perspective ray
    ro = cam_pos
    rd = normalize(ray_target - cam_pos)

    # Ray march (currently unused but would be used for shading)
    _rm = march(ro, rd)

    # Fixed color for now
    color = vec3(1.0, 1.0, 1.0)

    return vec4(color.x, color.y, color.z, 1.0)
