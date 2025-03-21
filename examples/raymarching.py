"""Enhanced ray marching example demonstrating py2glsl's modular architecture.

This example shows a 3D ray marching scene that can be rendered with
different target languages and backends. It demonstrates the new architecture
where backends are treated as first-class target languages.

Run with:
    python examples/raymarching.py                                # Standard GLSL target
    python examples/raymarching.py --target shadertoy             # Shadertoy target
    python examples/raymarching.py --save-image output.png        # Save still image
    python examples/raymarching.py --save-video output.mp4        # Save video
    python examples/raymarching.py --save-gif output.gif          # Save animated GIF

Animation control:
    python examples/raymarching.py --fps 60                       # Higher frame rate
    python examples/raymarching.py --animation-speed 0.6          # Faster animation
    python examples/raymarching.py --camera-height 3.0            # Lower camera
    python examples/raymarching.py --camera-distance 8.0          # Camera further away
    python examples/raymarching.py --time-offset 2.0              # Set start point

Future targets will be added as they become available:
    python examples/raymarching.py --target hlsl                  # HLSL target (future)
    python examples/raymarching.py --target wgsl                  # WGSL target (future)

This example demonstrates all major features of py2glsl:
1. Struct transpilation with RayMarchResult
2. Function dependencies (march -> get_sd_shape -> attenuate)
3. Global constants (PI, RM_MAX_DIST, etc.)
4. Multiple rendering methods (interactive, image, video, GIF)
5. Animation control via command-line parameters
6. Cross-target compatibility with identical output
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, cast

import typer

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
from py2glsl.render import animate, render_gif, render_image, render_video
from py2glsl.transpiler import transpile
from py2glsl.transpiler.core.interfaces import TargetLanguageType

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
    # Rounded box SDF - Fixed type compatibility
    d = length(max(abs(p) - vec3(1.0, 1.0, 1.0), vec3(0.0, 0.0, 0.0))) - 0.2
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


# Default mouse position
DEFAULT_MOUSE: vec2 = vec2(0.5, 0.5)


def main_shader(
    vs_uv: vec2,
    u_time: float,
    u_aspect: float,
    u_mouse: vec2 = DEFAULT_MOUSE,
    animation_speed: float = 0.5,  # Animation speed control
    animation_loop: float = 6.28,  # Full rotation period in seconds (2π)
    camera_height: float = 5.0,  # Camera height above origin
    camera_distance: float = 5.0,  # Camera distance from center
) -> vec4:
    """Main shader function.

    Args:
        vs_uv: UV coordinates (0-1)
        u_time: Current time in seconds
        u_aspect: Aspect ratio (width/height)
        u_mouse: Mouse position normalized (0-1), optional
        animation_speed: Controls how fast the animation plays, default 0.5
        animation_loop: Period for one full rotation in seconds, default 2π
        camera_height: Height of camera above origin, default 5.0
        camera_distance: Distance of camera from center axis, default 5.0

    Returns:
        Final color (RGBA)
    """
    # Screen position
    screen_pos = vs_uv * 2.0 - vec2(1.0, 1.0)
    screen_pos.x *= u_aspect

    # Camera setup
    fov = radians(70.0)
    screen_dist = 1.0 / tan(0.5 * fov)

    # CRITICAL FIX FOR ANIMATION CONSISTENCY:
    # 1. First normalize time to a cyclic period (forces identical cycle length)
    # 2. Then apply animation speed to control how fast we go through the cycle
    # This ensures the SAME animation pattern regardless of backend or render mode
    normalized_time = u_time * animation_speed

    # We take sin/cos of this angle to create circular camera motion
    angle = normalized_time

    # Calculate camera position with precise control
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

    return vec4(color.x, color.y, color.z, 1.0)


def main(
    target: Annotated[
        str,
        typer.Option(
            "--target",
            "-t",
            help="Target language/backend to use (glsl, shadertoy, hlsl, wgsl)",
        ),
    ] = "glsl",
    save_image: Annotated[
        Path | None,
        typer.Option(
            "--save-image", "-i", help="Save a still image to the specified path"
        ),
    ] = None,
    save_video: Annotated[
        Path | None,
        typer.Option("--save-video", "-v", help="Save a video to the specified path"),
    ] = None,
    save_gif: Annotated[
        Path | None,
        typer.Option(
            "--save-gif", "-g", help="Save an animated GIF to the specified path"
        ),
    ] = None,
    width: Annotated[
        int, typer.Option("--width", "-w", help="Width of the output window/image")
    ] = 800,
    height: Annotated[
        int, typer.Option("--height", "-h", help="Height of the output window/image")
    ] = 600,
    duration: Annotated[
        float,
        typer.Option("--duration", "-d", help="Duration of the video/GIF in seconds"),
    ] = 10.0,
    fps: Annotated[
        int, typer.Option("--fps", help="Frames per second for video/GIF")
    ] = 30,
    animation_speed: Annotated[
        float,
        typer.Option("--animation-speed", help="Speed factor for animation"),
    ] = 0.3,
    camera_height: Annotated[
        float, typer.Option("--camera-height", help="Camera height above origin")
    ] = 5.0,
    camera_distance: Annotated[
        float, typer.Option("--camera-distance", help="Camera distance from center")
    ] = 5.0,
    animation_loop: Annotated[
        float, typer.Option("--animation-loop", help="Full rotation period in seconds")
    ] = 6.28,
    time_offset: Annotated[
        float,
        typer.Option("--time-offset", help="Starting time for animation sequence"),
    ] = 0.0,
) -> None:
    """Run the ray marching example with the specified target language and options.

    Args:
        target: Target language/backend to use ("glsl", "shadertoy", "hlsl", "wgsl")
        save_image: Save a still image to the specified path
        save_video: Save a video to the specified path
        save_gif: Save an animated GIF to the specified path
        width: Width of the output window/image
        height: Height of the output window/image
        duration: Duration of the video/GIF in seconds
        fps: Frames per second for video/GIF
        animation_speed: Speed factor for animation (higher = faster)
        camera_height: Camera height above origin
        camera_distance: Camera distance from center
        animation_loop: Full rotation period in seconds (2π by default)
        time_offset: Starting time for animation sequence
    """
    # Map the target string to the corresponding TargetLanguageType
    target_type = None

    # Convert command-line target argument to TargetLanguageType
    target = target.lower()
    if target in ("glsl", "standard"):
        target_type = TargetLanguageType.GLSL
    elif target == "shadertoy":
        target_type = TargetLanguageType.SHADERTOY
    elif target == "hlsl":
        print("HLSL target is not yet implemented. Using GLSL instead.")
        target_type = TargetLanguageType.GLSL
    elif target == "wgsl":
        print("WGSL target is not yet implemented. Using GLSL instead.")
        target_type = TargetLanguageType.GLSL
    else:
        print(f"Unknown target: {target}. Using GLSL as default.")
        target_type = TargetLanguageType.GLSL

    # Get corresponding backend type for rendering
    from py2glsl.transpiler.backends.models import BackendType

    if target_type == TargetLanguageType.SHADERTOY:
        backend_type = BackendType.SHADERTOY
    else:
        backend_type = BackendType.STANDARD

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
        # Target language specification - using the new architecture
        target_type=target_type,
    )

    print(f"Using {target_type.name} target language")
    print(f"Used uniforms: {used_uniforms}")

    # Handle different output modes
    # All animation parameters are now passed directly from command line arguments
    # This eliminates hardcoded values and allows full control via CLI parameters

    # Define animation parameters as shader uniforms
    # Using cast to handle type compatibility with render functions
    animation_params = {
        "animation_speed": animation_speed,
        "animation_loop": animation_loop,
        "camera_height": camera_height,
        "camera_distance": camera_distance,
    }

    # This will be passed to all rendering functions
    render_uniforms = cast(dict[str, float | tuple[float, ...]], animation_params)

    if save_image:
        # Render a still image at a specific time in the cycle
        print(f"Rendering still image to {save_image}...")
        render_image(
            shader_input=glsl_code,  # Use the pre-transpiled code
            size=(width, height),
            time=time_offset,  # Use the specified time offset
            backend_type=backend_type,  # Use the mapped backend_type enum
            output_path=str(save_image),
            uniforms=render_uniforms,
        )
        print(f"Image saved to {save_image}")
    elif save_video:
        # Render a video
        print(f"Rendering {duration}s video at {fps}fps to {save_video}...")
        render_video(
            shader_input=glsl_code,  # Use the pre-transpiled code
            size=(width, height),
            duration=duration,
            fps=fps,
            backend_type=backend_type,  # Use the mapped backend_type enum
            output_path=str(save_video),
            time_offset=time_offset,  # Use consistent time offset
            uniforms=render_uniforms,
        )
        print(f"Video saved to {save_video}")
    elif save_gif:
        # Render an animated GIF
        print(f"Rendering {duration}s GIF at {fps}fps to {save_gif}...")
        render_gif(
            shader_input=glsl_code,  # Use the pre-transpiled code
            size=(width, height),
            duration=duration,
            fps=fps,
            backend_type=backend_type,  # Use the mapped backend_type enum
            output_path=str(save_gif),
            time_offset=time_offset,  # Use consistent time offset
            uniforms=render_uniforms,
        )
        print(f"GIF saved to {save_gif}")
    else:
        # Run interactive animation - glfw.get_time() is used here
        print(f"Running interactive animation at {fps}fps (press ESC to exit)...")
        # Pass animation speed for consistent behavior with GIF/video
        animate(
            shader_input=glsl_code,
            backend_type=backend_type,  # Use the mapped backend_type enum
            size=(width, height),
            uniforms=render_uniforms,
            fps=fps,  # Pass the fps parameter to control frame rate
        )


if __name__ == "__main__":
    typer.run(main)
