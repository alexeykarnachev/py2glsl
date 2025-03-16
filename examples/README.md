# py2glsl Examples

This directory contains examples demonstrating various features of py2glsl.

## Shadertoy Integration

The `shadertoy_example.py` demonstrates how to use the Shadertoy backend to generate Shadertoy-compatible GLSL code from Python.

Key features:
- Uses the Shadertoy backend to generate compatible GLSL
- Automatically maps Python functions to Shadertoy entry points
- Translates standard uniforms to Shadertoy built-ins (e.g., `u_time` to `iTime`)
- Adds required precision qualifiers for OpenGL ES
- Renders the shader locally using the renderer

To run the example:
```bash
python examples/shadertoy_example.py
```

### Using in actual Shadertoy website

If you want to use the generated code in the actual Shadertoy website, you can modify it slightly:
1. Keep only the `mainImage()` function, remove the `main()` function
2. Set the appropriate precision qualifiers
3. Paste into Shadertoy editor