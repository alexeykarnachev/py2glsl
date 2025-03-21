# py2glsl 🎨

Transform Python functions into GLSL shaders with zero boilerplate.
Write complex shaders in pure Python with type hinting,
including custom structs and global constants, then render them as real-time animations,
images, GIFs, or videos—all with proper IDE support and no GLSL knowledge required
(though it helps!).

## Quick Start

Install using uv:

```bash
uv pip install git+https://github.com/akarnachev/py2glsl.git
```

Create a simple animated shader:

```python
from py2glsl.builtins import length, sin, vec2, vec4
from py2glsl.render import animate


def plasma(vs_uv: vec2, u_time: float, u_aspect: float) -> vec4:
    """A simple animated plasma shader."""
    uv = vs_uv * 2.0 - 1.0  # Center UV coordinates
    d = length(uv)
    color = sin(d * 10.0 - u_time * 2.0) * 0.5 + 0.5
    return vec4(color, color * 0.5, 1.0 - color, 1.0)


# Run real-time animation at 30fps
animate(plasma, fps=30)
```

## Features

- Python-to-GLSL Transpilation: Write shaders in Python with full type hinting,
custom structs, and global constants—automatically converted to GLSL.
- Built-in GLSL Functions: Use familiar functions like sin, cos, length, normalize,
and more directly in Python.
- Flexible Rendering:
  - Real-time animations with animate (with framerate control)
  - Static images with render_image
  - Animated GIFs with render_gif
  - Videos with render_video
- Multiple Target Languages:
  - Standard GLSL
  - Shadertoy
  - More coming soon (HLSL, WGSL)
- Debugging Support: Access raw frames or generated GLSL code for inspection.
- IDE-Friendly: Leverages Python’s type system for autocompletion and error checking.
- No GLSL Boilerplate: Focus on shader logic without writing vertex/fragment wrappers.

## Installation

For users:

```bash
uv pip install git+https://github.com/akarnachev/py2glsl.git
```

For development:

```bash
git clone https://github.com/akarnachev/py2glsl.git
cd py2glsl
uv venv
source .venv/bin/activate  # or .venv/Scripts/activate on Windows
uv sync
```

Install pre-commit hooks:

```bash
uv pip install pre-commit
pre-commit install
```

## Usage

### Basic Shader

```python
from py2glsl.builtins import length, smoothstep, vec2, vec4
from py2glsl.render import render_image


def circle(vs_uv: vec2, u_time: float, u_aspect: float) -> vec4:
    """A static circle shader."""
    d = length(vs_uv * 2.0 - 1.0)
    color = 1.0 - smoothstep(0.0, 0.01, d - 0.5)
    return vec4(color, color, color, 1.0)


# Save as PNG
render_image(circle).save("circle.png")
```

### Animated Shader (GIF)
```python
from py2glsl.builtins import length, sin, vec2, vec4
from py2glsl.render import render_gif
from py2glsl.transpiler.backends.models import BackendType


def ripple(vs_uv: vec2, u_time: float, u_aspect: float) -> vec4:
    """An animated ripple effect."""
    uv = vs_uv * 2.0 - 1.0
    d = length(uv)
    wave = sin(d * 10.0 - u_time * 2.0) * 0.5 + 0.5
    return vec4(wave, wave * 0.5, 1.0 - wave, 1.0)


# Create animated GIF
_, frames = render_gif(ripple, duration=2.0, fps=30, output_path="ripple.gif")

# For Shadertoy compatibility:
_, frames = render_gif(ripple, duration=2.0, fps=30, output_path="ripple.gif", 
                      backend_type=BackendType.SHADERTOY)
```

### Advanced Example: Ray Marching

Here’s a more complex example using ray marching with structs and global constants:

```python
from dataclasses import dataclass

from py2glsl.builtins import length, sin, vec2, vec3, vec4
from py2glsl.render import animate
from py2glsl.transpiler import transpile
from py2glsl.transpiler.core.interfaces import TargetLanguageType

# Global constants
PI: float = 3.141592
RM_MAX_DIST: float = 10000.0
RM_MAX_STEPS: int = 64
RM_EPS: float = 0.0001


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
    """Signed distance to a sphere."""
    return length(p) - 1.0


def march(ro: vec3, rd: vec3) -> RayMarchResult:
    """Ray marching function."""
    rm = RayMarchResult(
        steps=0,
        p=ro,
        normal=vec3(0.0),
        ro=ro,
        rd=rd,
        dist=0.0,
        sd_last=0.0,
        sd_min=RM_MAX_DIST,
        sd_min_shape=RM_MAX_DIST,
        has_normal=False,
    )
    for i in range(RM_MAX_STEPS):
        rm.steps = i
        rm.p = rm.p + rm.rd * rm.sd_last
        rm.sd_last = get_sd_shape(rm.p)
        rm.dist = rm.dist + length(rm.p - rm.ro)
        if rm.sd_last < RM_EPS or rm.dist > RM_MAX_DIST:
            break
    return rm


def shader(vs_uv: vec2, u_time: float, u_aspect: float) -> vec4:
    """Ray-marched sphere with animation."""
    ro = vec3(0.0, 0.0, 5.0 + sin(u_time))
    rd = normalize(vec3(vs_uv * 2.0 - 1.0, -1.0))
    rm = march(ro, rd)
    color = vec3(0.1, 0.2, 0.3)  # Background
    if rm.sd_last < RM_EPS:
        color = vec3(1.0, 0.5, 0.2)  # Hit color
    return vec4(color, 1.0)


# Transpile with constants and structs
glsl_code, _ = transpile(
    march,
    get_sd_shape,
    shader,
    RayMarchResult,
    PI=PI,
    RM_MAX_DIST=RM_MAX_DIST,
    RM_MAX_STEPS=RM_MAX_STEPS,
    RM_EPS=RM_EPS,
    main_func="shader",
    # Optional: specify target language
    target_type=TargetLanguageType.GLSL  # or TargetLanguageType.SHADERTOY
)
# Run animation with 30fps limit
animate(glsl_code, fps=30)
```

### Debugging GLSL Output

```python
from py2glsl.builtins import vec2, vec4
from py2glsl.transpiler import transpile
from py2glsl.transpiler.core.interfaces import TargetLanguageType


def simple(vs_uv: vec2, u_time: float, u_aspect: float) -> vec4:
    return vec4(vs_uv, 0.0, 1.0)


glsl_code, uniforms = transpile(simple, main_func="simple")
print("Fragment Shader:")
print(glsl_code)

# Transpile to Shadertoy format
shadertoy_code, shadertoy_uniforms = transpile(
    simple, 
    main_func="simple",
    target_type=TargetLanguageType.SHADERTOY
)
print("Shadertoy Fragment Shader:")
print(shadertoy_code)
```

### Advanced Rendering Options

```python
from py2glsl.builtins import vec2, vec4, sin, length
from py2glsl.render import animate, render_video, render_gif
from py2glsl.transpiler.backends.models import BackendType

def my_shader(vs_uv: vec2, u_time: float, u_aspect: float) -> vec4:
    """Simple animated color shader."""
    d = length(vs_uv * 2.0 - 1.0)
    c = sin(d * 10.0 - u_time * 2.0) * 0.5 + 0.5
    return vec4(c, c * 0.5, 1.0 - c, 1.0)

# Real-time animation with frame rate control
animate(my_shader, fps=30)  # Cap at 30fps
animate(my_shader, fps=0)   # Unlimited frame rate (default)

# Interactive animation with Shadertoy compatibility
animate(my_shader, backend_type=BackendType.SHADERTOY, fps=60)

# Video rendering with different settings
render_video(
    my_shader,
    size=(1920, 1080),  # Full HD
    duration=5.0,       # 5 seconds
    fps=60,             # 60 frames per second
    output_path="shader.mp4",
    codec="h264",       # Video codec
    quality=8,          # Quality level (0-10)
    backend_type=BackendType.STANDARD,  # Use GLSL standard format
)

# GIF with custom parameters
render_gif(
    my_shader,
    size=(600, 600),    # Square dimensions
    duration=3.0,       # 3 seconds loop
    fps=24,             # 24 frames per second
    output_path="shader.gif",
    time_offset=1.0,    # Start animation from time=1.0
)
```

## License

MIT License - see [LICENSE](./LICENSE) for details.
