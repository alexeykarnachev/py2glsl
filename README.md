# py2glsl 🎨

<p align="center">
  <img src="thumbnail.gif" alt="py2glsl" width="600"/>
</p>

Transform Python functions into GLSL shaders with zero boilerplate.
Write complex shaders in pure Python with type hinting,
including custom structs and global constants, then render them as real-time animations,
images, GIFs, or videos—all with proper IDE support and no GLSL knowledge required
(though it helps!).

## Quick Start

Install using uv to get both the library and command-line tool:

```bash
uv pip install py2glsl
```

Create a simple animated shader file `plasma.py`:

```python
from py2glsl.builtins import length, sin, vec2, vec4

def shader(vs_uv: vec2, u_time: float, u_aspect: float) -> vec4:
    """A simple animated plasma shader."""
    uv = vs_uv * 2.0 - 1.0  # Center UV coordinates
    d = length(uv)
    color = sin(d * 10.0 - u_time * 2.0) * 0.5 + 0.5
    # Example of flexible vector construction
    return vec4(color, color * 0.5, 1.0 - color, 1.0)
```

Run it using the command-line interface:

```bash
# Interactive animation preview
py2glsl animate plasma.py

# Live animation with auto-reload on file changes
py2glsl animate plasma.py --watch

# Run in background (detached) with auto-reload
py2glsl animate plasma.py --watch --detach

# Run animation for a specific duration (e.g., 5 seconds)
py2glsl animate plasma.py --max-runtime 5.0

# Save as image
py2glsl render-image plasma.py output.png

# Create animated GIF
py2glsl render-gif plasma.py animation.gif --duration 5.0

# Create video 
py2glsl render-video plasma.py animation.mp4 --duration 5.0

# Specify a particular function to use (default auto-detects shader function)
py2glsl animate plasma.py --main my_custom_shader

# Export code for Shadertoy
py2glsl export-code plasma.py shadertoy.glsl --target shadertoy --format wrapped

# Export Shadertoy-compatible code (removes built-in uniforms)
py2glsl export-code plasma.py shadertoy_ready.glsl --target shadertoy --format wrapped --shadertoy-compatible
```

Or use the library directly in your code:

```python
from py2glsl.render import animate
from plasma import plasma  # Import your shader function

# Run real-time animation at 30fps
animate(plasma, fps=30)
```

## Features

- **Python-to-GLSL Transpilation**: Write shaders in Python with full type hinting,
  custom structs, and global constants—automatically converted to GLSL.
- **Built-in GLSL Functions**: Use familiar functions like sin, cos, length, normalize,
  and more directly in Python.
- **Command-Line Interface**:
  - Interactive animation with `py2glsl animate`
  - Live animation with auto-reload using `py2glsl animate --watch` 
  - Detached mode operation with `--detach` flag for headless usage
  - Static image rendering with `py2glsl render-image`
  - Video rendering with `py2glsl render-video`
  - GIF creation with `py2glsl render-gif`
  - Code export with `py2glsl export-code` (includes Shadertoy-compatibility mode)
- **Flexible Rendering API**:
  - Real-time animations with `animate()` (with framerate control)
  - Static images with `render_image()`
  - Animated GIFs with `render_gif()`
  - Videos with `render_video()`
- **Multiple Target Languages**:
  - Standard GLSL
  - Shadertoy
  - More coming soon (HLSL, WGSL)
- **IDE-Friendly**: Leverages Python's type system for autocompletion and error checking.
- **No GLSL Boilerplate**: Focus on shader logic without writing vertex/fragment wrappers.
- **Convenient Vectors**: Flexible vector initialization, like `vec4(vs_uv, 0.0, 1.0)` or `vec3(1.0)`.
- **Relaxed Type Annotations**: Optional return type hints in shader functions.

## Installation

For users:

```bash
# Using uv (recommended)
uv pip install py2glsl

# From source with uv
uv pip install git+https://github.com/alexeykarnachev/py2glsl.git

# Using pipx (for command-line usage only)
pipx install py2glsl
```

For development:

```bash
git clone https://github.com/alexeykarnachev/py2glsl.git
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

## Command-Line Interface

py2glsl provides a comprehensive command-line interface for working with shaders.

### Interactive Animation

```bash
py2glsl animate shader_file.py [OPTIONS]

Options:
  -t, --target TEXT      Target language (glsl, shadertoy)  [default: glsl]
  -m, --main TEXT        Specific shader function to use
  -w, --width INTEGER    Window width  [default: 800]
  -h, --height INTEGER   Window height  [default: 600]
  --fps INTEGER          Target framerate (0 for unlimited)  [default: 30]
  -d, --detach           Run in background (immediately returns control to shell)
  --watch                Watch shader file and auto-reload on changes
  --max-runtime FLOAT    Maximum runtime in seconds (stops animation after this time)
```

### Render to Image

```bash
py2glsl render-image shader_file.py output.png [OPTIONS]

Options:
  -t, --target TEXT      Target language (glsl, shadertoy)  [default: glsl]
  -m, --main TEXT        Specific shader function to use
  -w, --width INTEGER    Image width  [default: 800]
  -h, --height INTEGER   Image height  [default: 600]
  --time FLOAT           Time value for the image  [default: 0.0]
  -d, --detach           Run in detached mode (no output/logging)
```

### Render to Video

```bash
py2glsl render-video shader_file.py output.mp4 [OPTIONS]

Options:
  -t, --target TEXT      Target language (glsl, shadertoy)  [default: glsl]
  -m, --main TEXT        Specific shader function to use
  -w, --width INTEGER    Video width  [default: 800]
  -h, --height INTEGER   Video height  [default: 600]
  --fps INTEGER          Frames per second  [default: 30]
  -d, --duration FLOAT   Duration in seconds  [default: 5.0]
  --time-offset FLOAT    Starting time for animation  [default: 0.0]
  --codec TEXT           Video codec (h264, vp9, etc.)  [default: h264]
  -q, --quality INTEGER  Video quality (0-10)  [default: 8]
```

### Render to GIF

```bash
py2glsl render-gif shader_file.py output.gif [OPTIONS]

Options:
  -t, --target TEXT      Target language (glsl, shadertoy)  [default: glsl]
  -m, --main TEXT        Specific shader function to use
  -w, --width INTEGER    GIF width  [default: 800]
  -h, --height INTEGER   GIF height  [default: 600]
  --fps INTEGER          Frames per second  [default: 30]
  -d, --duration FLOAT   Duration in seconds  [default: 5.0]
  --time-offset FLOAT    Starting time for animation  [default: 0.0]
```

### Export Shader Code

```bash
py2glsl export-code shader_file.py output.glsl [OPTIONS]

Options:
  -t, --target TEXT           Target language (glsl, shadertoy)  [default: glsl]
  -m, --main TEXT             Specific shader function to use
  -f, --format TEXT           Code format (plain, commented, wrapped)  [default: plain]
  -s, --shadertoy-compatible  Process code for direct Shadertoy paste (removes version and uniforms)
```

## Library Usage

### Basic Shader

```python
from py2glsl.builtins import length, smoothstep, vec2, vec4
from py2glsl.render import render_image


def main(vs_uv: vec2, u_time: float, u_aspect: float) -> vec4:
    """A static circle shader."""
    d = length(vs_uv * 2.0 - 1.0)
    color = 1.0 - smoothstep(0.0, 0.01, d - 0.5)
    return vec4(color, color, color, 1.0)


# Save as PNG
render_image(main).save("circle.png")
```

### Animated Shader (GIF)
```python
from py2glsl.builtins import length, sin, vec2, vec4
from py2glsl.render import render_gif
from py2glsl.transpiler.backends.models import BackendType


def main(vs_uv: vec2, u_time: float, u_aspect: float) -> vec4:
    """An animated ripple effect."""
    uv = vs_uv * 2.0 - 1.0
    d = length(uv)
    wave = sin(d * 10.0 - u_time * 2.0) * 0.5 + 0.5
    return vec4(wave, wave * 0.5, 1.0 - wave, 1.0)


# Create animated GIF
_, frames = render_gif(main, duration=2.0, fps=30, output_path="ripple.gif")

# For Shadertoy compatibility:
_, frames = render_gif(main, duration=2.0, fps=30, output_path="ripple.gif", 
                      backend_type=BackendType.SHADERTOY)
```

### Advanced Example: Ray Marching

Here's a more complex example using ray marching with structs and global constants:

```python
from dataclasses import dataclass

from py2glsl.builtins import length, sin, vec2, vec3, vec4, normalize
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


def main(vs_uv: vec2, u_time: float, u_aspect: float):  # Return type hint is optional
    """Ray-marched sphere with animation."""
    ro = vec3(0.0, 0.0, 5.0 + sin(u_time))
    # Convenient vector construction with vs_uv directly in vec3
    rd = normalize(vec3(vs_uv * 2.0 - 1.0, -1.0))
    rm = march(ro, rd)
    
    # Create background color with scalar constructor
    bg_color = vec3(0.1)  # Same as vec3(0.1, 0.1, 0.1)
    
    # Determine final color
    color = bg_color
    if rm.sd_last < RM_EPS:
        # Hit color with standard construction
        color = vec3(1.0, 0.5, 0.2)
    
    # Add alpha channel to create vec4
    return vec4(color, 1.0)  # vec4(vec3, float) constructor


# Transpile with constants and structs
glsl_code, _ = transpile(
    march,
    get_sd_shape,
    main,
    RayMarchResult,
    PI=PI,
    RM_MAX_DIST=RM_MAX_DIST,
    RM_MAX_STEPS=RM_MAX_STEPS,
    RM_EPS=RM_EPS,
    # Optional: specify main function - defaults to "main" 
    # main_func="main",
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


def main(vs_uv: vec2, u_time: float, u_aspect: float) -> vec4:
    return vec4(vs_uv, 0.0, 1.0)


glsl_code, uniforms = transpile(main)
print("Fragment Shader:")
print(glsl_code)

# Transpile to Shadertoy format
shadertoy_code, shadertoy_uniforms = transpile(
    main, 
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

def main(vs_uv: vec2, u_time: float, u_aspect: float) -> vec4:
    """Simple animated color shader."""
    d = length(vs_uv * 2.0 - 1.0)
    c = sin(d * 10.0 - u_time * 2.0) * 0.5 + 0.5
    return vec4(c, c * 0.5, 1.0 - c, 1.0)

# Real-time animation with frame rate control
animate(main, fps=30)                   # Cap at 30fps
animate(main, fps=0)                    # Unlimited frame rate (default)
animate(main, fps=30, detached=True)    # Run without logging output (silent mode)
animate(main, fps=30, max_runtime=5.0)  # Run for 5 seconds then stop

# Interactive animation with Shadertoy compatibility
animate(main, backend_type=BackendType.SHADERTOY, fps=60)

# Video rendering with different settings
render_video(
    main,
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
    main,
    size=(600, 600),    # Square dimensions
    duration=3.0,       # 3 seconds loop
    fps=24,             # 24 frames per second
    output_path="shader.gif",
    time_offset=1.0,    # Start animation from time=1.0
)
```

## License

MIT License - see [LICENSE](./LICENSE) for details.