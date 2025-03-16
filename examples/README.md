# py2glsl Examples

This directory contains examples demonstrating how to use py2glsl to
write shaders in Python and transpile them to GLSL.

## Raymarching Example

The `raymarching.py` example demonstrates a 3D ray marching scene that can be rendered with different backends (standard OpenGL or Shadertoy).
It showcases:

- A 3D raymarching implementation with normal-based coloring
- Support for both Standard and Shadertoy backends
- CLI interface with Typer for easy configuration

### Usage

Run the example interactively:

```bash
python examples/raymarching.py
```

Render with the Shadertoy backend:

```bash
python examples/raymarching.py --backend shadertoy
```

Save a still image:

```bash
python examples/raymarching.py --save-image output.png
```

Save a video:

```bash
python examples/raymarching.py --save-video output.mp4 --duration 5 --fps 30
```

Save an animated GIF:

```bash
python examples/raymarching.py --save-gif output.gif --duration 3 --fps 15
```

Customize the output size:

```bash
python examples/raymarching.py --width 1920 --height 1080
```

### Command Line Options

| Option         | Short | Description                                     | Default   |
|----------------|-------|-------------------------------------------------|-----------|
| `--backend`    | `-b`  | Backend to use (standard or shadertoy)          | standard  |
| `--save-image` | `-i`  | Save a still image to the specified path        | None      |
| `--save-video` | `-v`  | Save a video to the specified path              | None      |
| `--save-gif`   | `-g`  | Save an animated GIF to the specified path      | None      |
| `--width`      | `-w`  | Width of the output window/image                | 800       |
| `--height`     | `-h`  | Height of the output window/image               | 600       |
| `--duration`   | `-d`  | Duration of the video/GIF in seconds            | 10.0      |
| `--fps`        |       | Frames per second for video/GIF                 | 30        |

## Backend Differences

The example demonstrates the differences between the Standard and Shadertoy backends:

1. **Standard Backend:**
   - Uses `#version 460 core` GLSL
   - Coordinate system is normalized (0-1)
   - Uses `u_time`, `u_aspect`, `u_mouse` uniforms

2. **Shadertoy Backend:**
   - Uses `#version 330 core` with precision qualifiers
   - Coordinates transformed from pixel-based to normalized
   - Uses Shadertoy-compatible uniforms (`iResolution`, `iTime`, `iMouse`, etc.)

This example can be used as a starting point for creating your own shaders with py2glsl.
