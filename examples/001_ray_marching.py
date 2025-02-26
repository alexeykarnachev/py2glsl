from dataclasses import dataclass

from py2glsl.builtins import (
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
from py2glsl.render import animate
from py2glsl.transpiler import transpile  # Import transpile explicitly

# Global constants - these need to be properly annotated and passed to the transpiler
PI: float = 3.141592
RM_MAX_DIST: float = 10000.0
RM_MAX_STEPS: int = 64
RM_EPS: float = 0.0001
NORMAL_DERIVATIVE_STEP: float = 0.015


@dataclass
class RayMarchResult:
    """Result of a ray marching operation.

    Attributes:
        steps: Number of steps taken during marching
        p: Final position reached
        normal: Surface normal at hit point
        ro: Ray origin
        rd: Ray direction
        dist: Total distance traveled
        sd_last: Last signed distance value
        sd_min: Minimum signed distance encountered
        sd_min_shape: Minimum signed distance to the shape
        has_normal: Whether a valid normal was calculated
    """

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
    """Calculate signed distance to a rounded box shape.

    Args:
        p: Position to evaluate

    Returns:
        Signed distance to the shape
    """
    # Rounded box SDF
    d = length(max(abs(p) - 1.0, 0.0)) - 0.2
    return d


def march(ro: vec3, rd: vec3) -> RayMarchResult:
    """Perform ray marching from origin along direction.

    Args:
        ro: Ray origin
        rd: Ray direction (normalized)

    Returns:
        Ray marching result with hit information
    """
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


def attenuate(d: float, coeffs: vec3) -> float:
    """Apply attenuation based on distance.

    Args:
        d: Distance value
        coeffs: Attenuation coefficients (constant, linear, quadratic)

    Returns:
        Attenuated value
    """
    return 1.0 / (coeffs.x + coeffs.y * d + coeffs.z * d * d)


def main_shader(vs_uv: vec2, u_time: float, u_aspect: float) -> vec4:
    """Main shader function.

    Args:
        vs_uv: UV coordinates (0-1)
        u_time: Current time in seconds
        u_aspect: Aspect ratio (width/height)

    Returns:
        Final color (RGBA)
    """
    # Screen position
    screen_pos = vs_uv * 2.0 - 1.0
    screen_pos.x *= u_aspect

    # Camera setup
    fov = radians(70.0)
    screen_dist = 1.0 / tan(0.5 * fov)
    cam_pos = vec3(5.0 * sin(u_time), 5.0, 5.0 * cos(u_time))
    look_at = vec3(0.0, 0.0, 0.0)

    # Camera basis vectors
    forward = normalize(look_at - cam_pos)
    world_up = vec3(0.0, 1.0, 0.0)
    right = normalize(cross(forward, world_up))
    up = normalize(cross(right, forward))

    # Ray setup
    screen_center = cam_pos + forward * screen_dist
    sp = screen_center + right * screen_pos.x + up * screen_pos.y

    # Perspective ray
    ro0 = cam_pos
    rd0 = normalize(sp - cam_pos)

    # Orthographic ray
    ro1 = sp * 4.0
    rd1 = normalize(look_at - cam_pos)

    # Mix perspective and orthographic
    ro = mix(ro0, ro1, 0.0)  # 0.0 = perspective, 1.0 = orthographic
    rd = mix(rd0, rd1, 0.0)

    # Ray march
    rm = march(ro, rd)

    # Color calculation
    color = vec3(0.0, 0.0, 0.0)

    if rm.has_normal:
        # Use normal as color
        color = abs(rm.normal)

        # Apply distance attenuation
        d = abs(max(0.0, rm.sd_min_shape))
        a = attenuate(d, vec3(0.01, 8.0, 8.0))
        color = color * a
    else:
        # Background color - simple gradient
        color = vec3(0.1, 0.2, 0.3) * (1.0 - length(screen_pos) * 0.5)

    return vec4(color, 1.0)


if __name__ == "__main__":
    # Pass all functions, the struct AND GLOBAL CONSTANTS explicitly to transpile
    glsl_code, used_uniforms = transpile(
        # Functions
        get_sd_shape,
        march,
        attenuate,
        main_shader,
        # Struct
        RayMarchResult,
        # Global constants
        PI=PI,
        RM_MAX_DIST=RM_MAX_DIST,
        RM_MAX_STEPS=RM_MAX_STEPS,
        RM_EPS=RM_EPS,
        NORMAL_DERIVATIVE_STEP=NORMAL_DERIVATIVE_STEP,
        # Main function
        main_func="main_shader",
    )
    animate(glsl_code, used_uniforms=used_uniforms)

    # Optional: For debugging
    print("Generated GLSL code:")
    print(glsl_code)
