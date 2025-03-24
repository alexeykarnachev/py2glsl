"""3D Ray Marching Example

This example demonstrates a ray marching technique to render 3D scenes with py2glsl.
Ray marching is an advanced technique that shoots rays from a camera and "marches"
along them in small steps until hitting a surface, making it possible to render
complex scenes with signed distance functions (SDFs).

Key concepts demonstrated:
1. Dataclass-based struct definition with RayMarchResult
2. Complex function dependencies (march -> get_sd_shape -> attenuate)
3. Global constants for configuration
4. Proper ray casting with camera matrices
5. Surface normal calculation using central differences
6. Distance-based attenuation for lighting effects

Technical highlights:
- Animated camera path that orbits the scene
- Proper perspective projection
- Rounded box SDF implementation
- Normal-based coloring with distance attenuation

To run this example:
    # Interactive preview
    py2glsl show run examples/raymarching.py
    # Render image
    py2glsl image render examples/raymarching.py output.png
    # Create animation
    py2glsl gif render examples/raymarching.py output.gif --duration 5 --fps 30
    # Export code for use in external renderers
    py2glsl code export examples/raymarching.py output.glsl
    # Export code for Shadertoy
    py2glsl code export examples/raymarching.py shadertoy.glsl --target shadertoy
"""

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
    """Result of a ray marching operation."""

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
    """Calculate signed distance to a rounded box shape."""
    # Rounded box SDF
    d = length(max(abs(p) - vec3(1.0, 1.0, 1.0), vec3(0.0, 0.0, 0.0))) - 0.2
    return d


def march(ro: vec3, rd: vec3) -> RayMarchResult:
    """Perform ray marching from origin along direction."""
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
    """Apply attenuation based on distance."""
    return 1.0 / (coeffs.x + coeffs.y * d + coeffs.z * d * d)


def simple_shader(vs_uv: vec2, u_time: float, u_aspect: float) -> vec4:
    """Main shader function."""
    # Screen position
    screen_pos = vs_uv * 2.0 - vec2(1.0, 1.0)
    screen_pos.x *= u_aspect

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
        camera_height,                 # Fixed height
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
    sp = screen_center + right * screen_pos.x + up * screen_pos.y

    # Perspective ray
    ro = cam_pos
    rd = normalize(sp - cam_pos)

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

    return vec4(color.x, color.y, color.z, 1.0)
