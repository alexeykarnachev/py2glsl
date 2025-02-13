import math

from py2glsl import py2glsl, vec2, vec4

from .utils import verify_shader_output


def test_solid_color(tmp_path):
    """Test basic solid color shader."""

    def solid_color(vs_uv: vec2, *, u_color: vec4) -> vec4:
        return u_color

    # Add debug prints
    result = py2glsl(solid_color)
    print("\nVertex Shader:")
    print(result.vertex_source)
    print("\nFragment Shader:")
    print(result.fragment_source)

    verify_shader_output(
        shader_func=solid_color,
        test_name="solid_color",
        tmp_path=tmp_path,
        uniforms={"u_color": (1.0, 0.0, 0.0, 1.0)},  # Pure red
    )


def test_uv_visualization(tmp_path):
    """Test UV coordinate visualization."""

    def uv_vis(vs_uv: vec2) -> vec4:
        return vec4(vs_uv.x, vs_uv.y, 0.0, 1.0)

    verify_shader_output(
        shader_func=uv_vis,
        test_name="uv_visualization",
        tmp_path=tmp_path,
    )


def test_animated_pattern(tmp_path):
    """Test animated pattern with time uniform."""

    def animated(vs_uv: vec2, *, u_time: float) -> vec4:
        # Center UVs
        uv = vs_uv * 2.0 - 1.0

        # Create animated circular pattern
        d = math.sqrt(uv.x * uv.x + uv.y * uv.y)
        pattern = math.sin(d * 10.0 - u_time * 2.0) * 0.5 + 0.5

        return vec4(pattern, pattern * 0.5, 1.0 - pattern, 1.0)

    verify_shader_output(
        shader_func=animated,
        test_name="animated_pattern",
        tmp_path=tmp_path,
        uniforms={"u_time": 1.234},  # Fixed time for reproducible output
    )
