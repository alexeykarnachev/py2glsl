"""Shader context providing access to built-in inputs and uniforms.

The ShaderContext provides all standard shader inputs and uniforms:

    from py2glsl import ShaderContext, vec4, sin

    def my_shader(ctx: ShaderContext) -> vec4:
        r = sin(ctx.u_time) * 0.5 + 0.5
        return vec4(ctx.vs_uv.x, ctx.vs_uv.y, r, 1.0)
"""

from py2glsl.builtins import vec2


class ShaderContext:
    """Context object providing shader builtins.

    Attributes:
        vs_uv: UV coordinates (0-1 range), vec2 input from vertex shader
        u_time: Time in seconds, float uniform
        u_resolution: Viewport resolution (width, height), vec2 uniform
        u_aspect: Aspect ratio (width/height), float uniform
        u_mouse_pos: Mouse position in pixels, vec2 uniform
        u_mouse_uv: Mouse position normalized (0-1 range), vec2 uniform
    """

    # Inputs (from vertex shader)
    vs_uv: vec2

    # Uniforms
    u_time: float
    u_resolution: vec2
    u_aspect: float
    u_mouse_pos: vec2
    u_mouse_uv: vec2

    def __init__(self) -> None:
        """Initialize with default values for runtime execution."""
        self.vs_uv = vec2(0.0, 0.0)
        self.u_time = 0.0
        self.u_resolution = vec2(1.0, 1.0)
        self.u_aspect = 1.0
        self.u_mouse_pos = vec2(0.0, 0.0)
        self.u_mouse_uv = vec2(0.0, 0.0)


# Type mapping for transpiler: name -> (glsl_type, storage_class)
CONTEXT_BUILTINS: dict[str, tuple[str, str]] = {
    "vs_uv": ("vec2", "input"),
    "u_time": ("float", "uniform"),
    "u_resolution": ("vec2", "uniform"),
    "u_aspect": ("float", "uniform"),
    "u_mouse_pos": ("vec2", "uniform"),
    "u_mouse_uv": ("vec2", "uniform"),
}
