"""Comprehensive Feature Showcase

This example demonstrates virtually all py2glsl features in a single shader.
It creates a complex, animated scene that tests:

1. Multiple dataclass-based structs with various field types
2. Extensive use of builtin functions (sin, cos, abs, min, max, normalize, etc.)
3. All control flow: if/elif/else, for loops, while loops, break statements
4. Vector swizzling (xyy, yxy, etc.)
5. Matrix operations (mat2, mat3, mat4)
6. Nested function calls and dependencies
7. Augmented assignments (+=, -=, *=, /=)
8. Comparison and boolean operators
9. Ternary expressions (if/else inline)
10. Global constants
11. Distance functions and mix/smoothstep for smooth blending

The shader creates a psychedelic animated scene with rotating shapes, distance fields,
and smooth color transitions.

Example Usage:
    # Interactive preview
    py2glsl show run examples/feature_showcase.py --main shader

    # Render image
    py2glsl image render examples/feature_showcase.py showcase.png --main shader

    # Create animation
    py2glsl gif render examples/feature_showcase.py showcase.gif --duration 5 --fps 30

    # Export to GLSL
    py2glsl code export examples/feature_showcase.py showcase.glsl

    # Export for Shadertoy
    py2glsl code export examples/feature_showcase.py shadertoy.glsl --target shadertoy \
        --shadertoy-compatible
"""

from dataclasses import dataclass

from py2glsl import ShaderContext
from py2glsl.builtins import (
    abs,
    cos,
    cross,
    dot,
    fract,
    length,
    mat2,
    mat3,
    max,
    min,
    mix,
    normalize,
    radians,
    sin,
    smoothstep,
    tan,
    vec2,
    vec3,
    vec4,
)

# Global constants
PI: float = 3.141592653589793
TAU: float = 6.283185307179586
EPSILON: float = 0.001
MAX_ITERATIONS: int = 100
SHAPE_COUNT: int = 5


@dataclass
class Material:
    """Material properties for shading."""

    albedo: vec3
    roughness: float
    metallic: float
    emission: float


@dataclass
class SceneObject:
    """Object in the scene with position and material."""

    position: vec3
    scale: float
    rotation: float
    material: Material
    distance: float


@dataclass
class RayHit:
    """Ray intersection result."""

    hit: bool
    distance: float
    position: vec3
    normal: vec3
    material: Material
    iterations: int


def rotate_2d(p: vec2, angle: float) -> vec2:
    """Rotate a 2D point by angle using mat2."""
    c = cos(angle)
    s = sin(angle)
    # Column-major matrix
    m = mat2(c, -s, s, c)
    # Matrix multiplication using subscript notation
    result = vec2(
        m[0][0] * p.x + m[1][0] * p.y,
        m[0][1] * p.x + m[1][1] * p.y,
    )
    return result


def sdf_sphere(p: vec3, radius: float) -> float:
    """Signed distance function for a sphere."""
    return length(p) - radius


def sdf_box(p: vec3, size: vec3) -> float:
    """Signed distance function for a box."""
    q = abs(p) - size
    outside = length(max(q, vec3(0.0, 0.0, 0.0)))
    inside = min(max(q.x, max(q.y, q.z)), 0.0)
    return outside + inside


def sdf_torus(p: vec3, major_radius: float, minor_radius: float) -> float:
    """Signed distance function for a torus."""
    q = vec2(length(vec2(p.x, p.z)) - major_radius, p.y)
    return length(q) - minor_radius


def smooth_min(a: float, b: float, k: float) -> float:
    """Smooth minimum for blending distances."""
    h = max(k - abs(a - b), 0.0) / k
    return min(a, b) - h * h * k * 0.25


def calculate_normal(p: vec3, scene_dist: float) -> vec3:
    """Calculate surface normal using central differences."""
    eps = EPSILON
    e = vec2(eps, 0.0)

    # Sample distances around the point
    dx = scene_distance(p + vec3(e.x, 0.0, 0.0)) - scene_distance(
        p - vec3(e.x, 0.0, 0.0)
    )
    dy = scene_distance(p + vec3(0.0, e.x, 0.0)) - scene_distance(
        p - vec3(0.0, e.x, 0.0)
    )
    dz = scene_distance(p + vec3(0.0, 0.0, e.x)) - scene_distance(
        p - vec3(0.0, 0.0, e.x)
    )

    return normalize(vec3(dx, dy, dz))


def scene_distance(p: vec3) -> float:
    """Calculate minimum distance to scene geometry."""
    # Test all control flow and operations
    total_dist = 1000.0

    # For loop with range
    for i in range(SHAPE_COUNT):
        offset = vec3(
            cos(float(i) * TAU / float(SHAPE_COUNT)) * 2.0,
            sin(float(i) * 1.5) * 0.5,
            sin(float(i) * TAU / float(SHAPE_COUNT)) * 2.0,
        )

        # Use augmented assignment
        local_p = p - offset

        # Test if/elif/else
        if i == 0:
            # Sphere
            d = sdf_sphere(local_p, 0.5)
        elif i == 1:
            # Box
            d = sdf_box(local_p, vec3(0.4, 0.4, 0.4))
        elif i == 2:
            # Torus
            d = sdf_torus(local_p, 0.5, 0.2)
        else:
            # Smooth blend
            d1 = sdf_sphere(local_p, 0.4)
            d2 = sdf_box(local_p, vec3(0.3, 0.3, 0.3))
            d = smooth_min(d1, d2, 0.3)

        # Use smooth minimum to blend
        total_dist = smooth_min(total_dist, d, 0.1)

    # Add a ground plane using max
    ground = p.y + 1.0
    total_dist = min(total_dist, ground)

    return total_dist


def get_material(p: vec3) -> Material:
    """Get material properties at a position."""
    # Use vector swizzling
    color_base = fract(p.xyy * 0.5)

    # Use mix for interpolation
    albedo = mix(vec3(0.8, 0.2, 0.3), vec3(0.2, 0.5, 0.9), color_base.x)

    # Calculate roughness with smoothstep
    roughness = smoothstep(0.0, 1.0, color_base.y)

    # Metallic based on position
    metallic = abs(sin(p.x * 2.0 + p.z * 2.0)) * 0.5 + 0.5

    # Emission
    emission = max(0.0, sin(length(p) * 5.0)) * 0.5

    return Material(
        albedo=albedo, roughness=roughness, metallic=metallic, emission=emission
    )


def trace_ray(origin: vec3, direction: vec3) -> RayHit:
    """Trace a ray through the scene."""
    t = 0.0
    iterations = 0
    hit = False
    final_pos = origin
    final_normal = vec3(0.0, 1.0, 0.0)
    final_material = Material(
        albedo=vec3(0.0, 0.0, 0.0), roughness=0.5, metallic=0.0, emission=0.0
    )

    # While loop with break
    while iterations < MAX_ITERATIONS:
        iterations += 1
        pos = origin + direction * t
        d = scene_distance(pos)

        # Augmented assignments
        t += d

        # Check hit condition
        if d < EPSILON:
            hit = True
            final_pos = pos
            final_normal = calculate_normal(pos, d)
            final_material = get_material(pos)
            break

        # Check miss condition
        if t > 100.0:
            break

    return RayHit(
        hit=hit,
        distance=t,
        position=final_pos,
        normal=final_normal,
        material=final_material,
        iterations=iterations,
    )


def apply_lighting(
    hit: RayHit, ray_dir: vec3, light_pos: vec3, light_color: vec3
) -> vec3:
    """Apply lighting calculations."""
    # Check if we hit anything
    if not hit.hit:
        # Background gradient
        t = ray_dir.y * 0.5 + 0.5
        return mix(vec3(0.1, 0.1, 0.2), vec3(0.5, 0.7, 1.0), t)

    # Light direction
    light_dir = normalize(light_pos - hit.position)

    # Diffuse lighting (Lambertian)
    diffuse_strength = max(dot(hit.normal, light_dir), 0.0)
    diffuse = hit.material.albedo * diffuse_strength * light_color

    # Specular lighting (Blinn-Phong)
    view_dir = normalize(-ray_dir)
    half_dir = normalize(light_dir + view_dir)
    spec_strength = max(dot(hit.normal, half_dir), 0.0)

    # Roughness affects specular power
    specular_power = mix(128.0, 8.0, hit.material.roughness)
    specular = vec3(1.0, 1.0, 1.0) * (spec_strength**specular_power)

    # Metallic affects specular color
    specular = mix(specular, hit.material.albedo * specular, hit.material.metallic)

    # Ambient occlusion approximation
    ao = 1.0 - float(hit.iterations) / float(MAX_ITERATIONS)
    ambient = hit.material.albedo * 0.1 * ao

    # Emission
    emission = hit.material.albedo * hit.material.emission

    # Combine all lighting
    color: vec3 = ambient + diffuse + specular + emission

    return color


def shader(ctx: ShaderContext) -> vec4:
    """Main shader function showcasing all features."""
    # Screen-space coordinates
    uv = ctx.vs_uv * 2.0 - vec2(1.0, 1.0)
    uv.x *= ctx.u_aspect

    # Camera setup with matrix operations
    camera_angle = ctx.u_time * 0.3
    camera_distance = 6.0
    camera_height = 2.0

    # Camera position using trigonometry
    cam_pos = vec3(
        cos(camera_angle) * camera_distance,
        camera_height + sin(ctx.u_time * 0.5) * 0.5,
        sin(camera_angle) * camera_distance,
    )

    # Look-at target
    target = vec3(0.0, 0.0, 0.0)

    # Camera basis vectors
    forward = normalize(target - cam_pos)
    right = normalize(cross(vec3(0.0, 1.0, 0.0), forward))
    up = cross(forward, right)

    # Use mat3 for camera matrix (demonstration)
    # Column-major: right, up, forward
    cam_matrix = mat3(
        right.x,
        right.y,
        right.z,
        up.x,
        up.y,
        up.z,
        forward.x,
        forward.y,
        forward.z,
    )

    # Ray direction
    fov = radians(60.0)
    focal_length = 1.0 / tan(fov * 0.5)

    # Using matrix multiplication for ray direction
    ray_dir_local = vec3(uv.x, uv.y, focal_length)
    ray_dir = normalize(
        vec3(
            dot(
                vec3(cam_matrix[0][0], cam_matrix[1][0], cam_matrix[2][0]),
                ray_dir_local,
            ),
            dot(
                vec3(cam_matrix[0][1], cam_matrix[1][1], cam_matrix[2][1]),
                ray_dir_local,
            ),
            dot(
                vec3(cam_matrix[0][2], cam_matrix[1][2], cam_matrix[2][2]),
                ray_dir_local,
            ),
        )
    )

    # Trace the ray
    hit = trace_ray(cam_pos, ray_dir)

    # Animated light position
    light_angle = ctx.u_time * 0.7
    light_pos = vec3(cos(light_angle) * 4.0, 3.0, sin(light_angle) * 4.0)
    light_color = vec3(1.0, 0.95, 0.9)

    # Apply lighting
    color = apply_lighting(hit, ray_dir, light_pos, light_color)

    # Test ternary expression via if/else inline behavior
    # Add distance-based fog
    fog_amount = 1.0 - min(hit.distance / 50.0, 1.0) if hit.hit else 0.0
    fog_color = vec3(0.5, 0.6, 0.7)
    color = mix(color, fog_color, fog_amount)

    # Vignette effect
    vignette = smoothstep(0.8, 0.2, length(uv))
    color = color * vignette

    # Test swizzling (xyy, yxy) without changing result
    color = color + (color.xyy + color.yxy) * 0.0

    # Color grading (test augmented assignment operators)
    color *= 1.2  # Exposure
    color -= vec3(0.02, 0.02, 0.02)  # Lift
    color = max(color, vec3(0.0, 0.0, 0.0))  # Clamp

    # Gamma correction
    gamma = 2.2
    color = vec3(
        color.x ** (1.0 / gamma), color.y ** (1.0 / gamma), color.z ** (1.0 / gamma)
    )

    return vec4(color.x, color.y, color.z, 1.0)
