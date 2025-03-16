"""Enhanced ray marching example demonstrating py2glsl backends.

This example shows a 3D ray marching scene that can be rendered with
different backends (standard OpenGL or Shadertoy).

Run with:
    python examples/raymarching.py
    python examples/raymarching.py --backend shadertoy
    python examples/raymarching.py --save-image output.png
    python examples/raymarching.py --save-video output.mp4
    python examples/raymarching.py --save-gif output.gif
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

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
from py2glsl.transpiler.backends.models import BackendType

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
    vs_uv: vec2, u_time: float, u_aspect: float, u_mouse: vec2 = DEFAULT_MOUSE
) -> vec4:
    """Main shader function.

    Args:
        vs_uv: UV coordinates (0-1)
        u_time: Current time in seconds
        u_aspect: Aspect ratio (width/height)
        u_mouse: Mouse position normalized (0-1), optional

    Returns:
        Final color (RGBA)
    """
    # Screen position
    screen_pos = vs_uv * 2.0 - vec2(1.0, 1.0)
    screen_pos.x *= u_aspect

    # Camera setup
    fov = radians(70.0)
    screen_dist = 1.0 / tan(0.5 * fov)

    # Normalize camera movement - loop every 2π seconds
    # Using sin/cos with raw time automatically loops
    t = u_time
    cam_pos = vec3(5.0 * sin(t), 5.0, 5.0 * cos(t))
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


def app() -> None:
    """CLI application entry point."""
    typer.run(main)


def main(
    backend: Annotated[
        str,
        typer.Option("--backend", "-b", help="Backend to use (standard or shadertoy)"),
    ] = "standard",
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
) -> None:
    """Run the ray marching example with the specified backend and options.

    Args:
        backend: Backend to use ("standard" or "shadertoy")
        save_image: Save a still image to the specified path
        save_video: Save a video to the specified path
        save_gif: Save an animated GIF to the specified path
        width: Width of the output window/image
        height: Height of the output window/image
        duration: Duration of the video/GIF in seconds
        fps: Frames per second for video/GIF
    """
    # Determine the backend type
    backend_type = BackendType.STANDARD
    if backend.lower() == "shadertoy":
        backend_type = BackendType.SHADERTOY

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
        # Backend specification
        backend_type=backend_type,
    )

    print(f"Using {backend} backend (BackendType.{backend_type.name})")
    print(f"Used uniforms: {used_uniforms}")

    # Handle different output modes
    # Set a consistent time offset to ensure animations are consistent
    # This matches the starting point for all rendering modes
    time_offset = 0.0

    if save_image:
        # Render a still image
        print(f"Rendering still image to {save_image}...")
        render_image(
            shader_input=glsl_code,  # Use the pre-transpiled GLSL code
            size=(width, height),
            time=time_offset + 2.0,  # Set specific time value for the still image
            backend_type=backend_type,
            output_path=str(save_image),
        )
        print(f"Image saved to {save_image}")
    elif save_video:
        # Render a video
        print(f"Rendering {duration}s video at {fps}fps to {save_video}...")
        render_video(
            shader_input=glsl_code,  # Use the pre-transpiled GLSL code
            size=(width, height),
            duration=duration,
            fps=fps,
            backend_type=backend_type,
            output_path=str(save_video),
            time_offset=time_offset,  # Use consistent time offset
        )
        print(f"Video saved to {save_video}")
    elif save_gif:
        # Render an animated GIF
        print(f"Rendering {duration}s GIF at {fps}fps to {save_gif}...")
        render_gif(
            shader_input=glsl_code,  # Use the pre-transpiled GLSL code
            size=(width, height),
            duration=duration,
            fps=fps,
            backend_type=backend_type,
            output_path=str(save_gif),
            time_offset=time_offset,  # Use consistent time offset
        )
        print(f"GIF saved to {save_gif}")
    else:
        # Run interactive animation - glfw.get_time() is used here
        print("Running interactive animation (press ESC to exit)...")
        # No custom uniforms needed, time comes from glfw.get_time()
        animate(glsl_code, backend_type=backend_type, size=(width, height))


if __name__ == "__main__":
    app()
