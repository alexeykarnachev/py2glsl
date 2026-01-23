"""Tests for the transpiler module."""

from dataclasses import dataclass

import pytest

from py2glsl import ShaderContext
from py2glsl.builtins import dot, max, normalize, vec2, vec3, vec4
from py2glsl.transpiler import transpile
from py2glsl.transpiler.models import TranspilerError


class TestTranspile:
    """Test cases for the transpile function."""

    def test_transpile_with_context(self):
        """Test transpiling with ShaderContext."""

        def my_shader(ctx: ShaderContext) -> vec4:
            return vec4(ctx.vs_uv.x, ctx.vs_uv.y, 0.5, 1.0)

        glsl_code, uniforms = transpile(my_shader)

        assert "vec4 my_shader()" in glsl_code
        assert "return vec4(vs_uv.x, vs_uv.y, 0.5, 1.0);" in glsl_code
        assert "in vec2 vs_uv;" in glsl_code
        assert "uniform float u_time;" in glsl_code
        assert "u_time" in uniforms

    def test_transpile_helper_and_main(self):
        """Test transpiling with helper and main functions."""

        def helper(pos: "vec2") -> "float":
            return pos.x + pos.y

        def main_shader(ctx: ShaderContext) -> vec4:
            value = helper(ctx.vs_uv)
            return vec4(value, 0.0, 0.0, 1.0)

        glsl_code, uniforms = transpile(helper, main_shader, main_func="main_shader")

        assert "float helper(vec2 pos)" in glsl_code
        assert "return pos.x + pos.y;" in glsl_code
        assert "vec4 main_shader()" in glsl_code
        assert "float value = helper(vs_uv);" in glsl_code
        assert "return vec4(value, 0.0, 0.0, 1.0);" in glsl_code
        assert "in vec2 vs_uv;" in glsl_code
        assert "u_time" in uniforms

    def test_transpile_complex_setup(self):
        """Test transpiling a complex setup with structs, helpers, and globals."""

        @dataclass
        class Light:
            position: "vec3"
            color: "vec3"
            intensity: "float" = 1.0

        def calc_diffuse(normal: "vec3", light_dir: "vec3") -> "float":
            return max(dot(normal, light_dir), 0.0)

        def apply_light(pos: "vec3", light: "Light") -> "vec3":
            normal = normalize(vec3(0.0, 0.0, 1.0))
            light_dir = normalize(light.position - pos)
            diffuse = calc_diffuse(normal, light_dir)
            return light.color * diffuse * light.intensity

        def shader(ctx: ShaderContext) -> vec4:
            pos = vec3(ctx.vs_uv.x * 2.0 - 1.0, ctx.vs_uv.y * 2.0 - 1.0, 0.0)
            light = Light(vec3(1.0, 2.0, 3.0), vec3(1.0, 1.0, 1.0), 0.8)
            color = apply_light(pos, light)
            return vec4(color.x, color.y, color.z, 1.0)

        glsl_code, uniforms = transpile(
            Light, calc_diffuse, apply_light, shader, main_func="shader", MAX_DIST=100.0
        )

        # Check struct
        assert "struct Light {" in glsl_code
        assert "vec3 position;" in glsl_code
        assert "vec3 color;" in glsl_code
        assert "float intensity;" in glsl_code

        # Check global
        assert "const float MAX_DIST = 100.0;" in glsl_code

        # Check helper function
        assert "float calc_diffuse(vec3 normal, vec3 light_dir)" in glsl_code
        assert "return max(dot(normal, light_dir), 0.0);" in glsl_code

        # Check variable declarations
        assert "in vec2 vs_uv;" in glsl_code

        # Check uniforms
        assert "u_time" in uniforms

    def test_transpile_helper_no_return_type(self):
        """Test that helper function without return type raises an error."""

        def helper(x: "float"):  # No return type annotation
            return x * 2.0

        def main_shader(ctx: ShaderContext) -> vec4:
            return vec4(helper(ctx.vs_uv.x), 0.0, 0.0, 1.0)

        with pytest.raises(
            TranspilerError,
            match="Helper function 'helper' lacks return type annotation",
        ):
            transpile(helper, main_shader, main_func="main_shader")

    def test_transpile_unsupported_item(self):
        """Test that unsupported items raise an error."""
        item = 42

        with pytest.raises(TranspilerError, match="Unsupported item type"):
            transpile(item)  # type: ignore
