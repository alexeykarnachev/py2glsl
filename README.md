# py2glsl 🎨

Transform Python functions into GLSL shaders with zero boilerplate. Write shaders in pure Python with proper type checking, then render them to images, GIFs, or videos, or display them in real-time.

## Quick Start

Install using `uv`:
```bash
uv pip install git+https://github.com/yourusername/py2glsl.git
```

Create a simple animated shader:
```python
from py2glsl import animate, vec2, vec4, py2glsl
from py2glsl.builtins import sin, length

def plasma(vs_uv: vec2, *, u_time: float) -> vec4:
    # Center and scale UV coordinates
    uv = vs_uv * 2.0 - 1.0
    
    # Create animated plasma effect
    d = length(uv)
    color = sin(d * 10.0 - u_time * 2.0) * 0.5 + 0.5
    
    return vec4(color, color * 0.5, 1.0 - color, 1.0)

# Print generated GLSL code
result = py2glsl(plasma)
print("Fragment Shader:")
print(result.fragment_source)

# Display animation in real-time window
animate(plasma)
```

## Features

- Write GLSL shaders in Python with type hinting
- Automatic variable declaration and type inference
- Support for uniforms and built-in GLSL functions
- Multiple rendering options:
  - Real-time animation window
  - Static image output
  - GIF animation
  - Video rendering
- Clean Python syntax with proper IDE support
- No standalone GLSL knowledge required (but understanding basic fragment shaders helps)

## Installation

For development:
```bash
git clone https://github.com/yourusername/py2glsl.git
cd py2glsl
uv venv
source .venv/bin/activate  # or .venv/Scripts/activate on Windows
uv sync
```

## Usage

### Basic Shader

```python
from py2glsl import render_image, vec2, vec4
from py2glsl.builtins import length, smoothstep

def circle(vs_uv: vec2) -> vec4:
    d = length(vs_uv * 2.0 - 1.0)
    color = 1.0 - smoothstep(0.0, 0.01, d - 0.5)
    return vec4(color, color, color, 1.0)

# Save as PNG
render_image(circle).save("circle.png")
```

### Render Animation (gif or video)
```python
from py2glsl import render_gif, vec2, vec4
from py2glsl.builtins import sin, length

def ripple(vs_uv: vec2, *, u_time: float) -> vec4:
    uv = vs_uv * 2.0 - 1.0
    d = length(uv)
    wave = sin(d * 10.0 - u_time * 2.0) * 0.5 + 0.5
    return vec4(wave, wave * 0.5, 1.0 - wave, 1.0)

# Create animated GIF
render_gif(ripple, "ripple.gif", duration=2.0, fps=30)
```

### Print Generated GLSL Code
```python
from py2glsl import py2glsl, vec2, vec4

def simple(vs_uv: vec2) -> vec4:
    return vec4(vs_uv, 0.0, 1.0)

# Get GLSL code
result = py2glsl(simple)
print("Fragment Shader:")
print(result.fragment_source)
print("\nVertex Shader:")
print(result.vertex_source)
```

## License
MIT License - see [LICENSE](LICENSE) for details.

