"""CLI-Compatible Raymarching Example

This is a simplified version of the raymarching example that works well
with the py2glsl CLI tool. It has the same visual output but avoids using
custom struct types which require special handling during transpilation.

Key features:
1. Uses individual variables instead of a custom RayMarchResult struct
2. Direct function calls for march, get_normal, etc.
3. Compatible with automatic CLI transpilation

You can use this example with the CLI commands:

# Export to GLSL shader
py2glsl code export examples/raymarching_simple.py raymarching.glsl

# Export for Shadertoy
py2glsl code export examples/raymarching_simple.py shadertoy.glsl --target shadertoy

# Export for direct Shadertoy paste (no built-in uniforms)
py2glsl code export examples/raymarching_simple.py shadertoy.glsl --target shadertoy --shadertoy-compatible

# Interactive preview
py2glsl show run examples/raymarching_simple.py

# Render image
py2glsl image render examples/raymarching_simple.py output.png

# Create animation
py2glsl gif render examples/raymarching_simple.py output.gif --duration 5 --fps 30
"""

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


def get_sd_shape(p: vec3) -> float:
    """Calculate signed distance to a rounded box shape."""
    # Rounded box SDF
    d = length(max(abs(p) - vec3(1.0, 1.0, 1.0), vec3(0.0, 0.0, 0.0))) - 0.2
    return d


def get_normal(p: vec3) -> vec3:
    """Calculate normal using central differences."""
    e = vec2(NORMAL_DERIVATIVE_STEP, 0.0)
    normal = normalize(
        vec3(
            get_sd_shape(p + vec3(e.x, 0.0, 0.0)) - get_sd_shape(p - vec3(e.x, 0.0, 0.0)),
            get_sd_shape(p + vec3(0.0, e.x, 0.0)) - get_sd_shape(p - vec3(0.0, e.x, 0.0)),
            get_sd_shape(p + vec3(0.0, 0.0, e.x)) - get_sd_shape(p - vec3(0.0, 0.0, e.x)),
        )
    )
    return normal


def march(ro: vec3, rd: vec3) -> vec3:
    """Perform ray marching from origin along direction, returns hit position."""
    p = ro
    sd_last = 0.0
    dist = 0.0
    
    # March the ray
    for i in range(RM_MAX_STEPS):
        p = p + rd * sd_last
        sd_step_shape = get_sd_shape(p)
        sd_last = sd_step_shape
        dist = dist + length(p - ro)

        # Check termination conditions
        if sd_last < RM_EPS or dist > RM_MAX_DIST:
            break

    return p


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

    # Ray march to get hit position
    hit_pos = march(ro, rd)
    
    # Check if we hit something
    hit_dist = length(hit_pos - ro)
    has_hit = hit_dist < RM_MAX_DIST

    # Color calculation
    color = vec3(0.0, 0.0, 0.0)

    if has_hit:
        # Calculate normal
        normal = get_normal(hit_pos)
        
        # Use normal as color
        color = abs(normal)

        # Apply distance attenuation
        d = length(hit_pos - ro)
        a = attenuate(d, vec3(0.01, 8.0, 8.0))
        color = color * a
    else:
        # Background color - simple gradient
        color = vec3(0.1, 0.2, 0.3) * (1.0 - length(screen_pos) * 0.5)

    return vec4(color.x, color.y, color.z, 1.0)