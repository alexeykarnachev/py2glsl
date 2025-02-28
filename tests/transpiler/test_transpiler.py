from builtins import max as python_max  # Using Python's built-in max
from dataclasses import dataclass

import pytest

from py2glsl.builtins import dot, normalize, vec2, vec3, vec4
from py2glsl.transpiler import transpile
from py2glsl.transpiler.errors import TranspilerError


class TestTranspile:
    """Test cases for the transpile function."""

    def test_transpile_helper_and_main(self):
        """Test transpiling with helper and main functions."""

        # Arrange
        def helper(pos: "vec2") -> "float":
            return pos.x + pos.y

        def main_shader(vs_uv: "vec2", u_scale: "float") -> "vec4":
            value = helper(vs_uv) * u_scale
            return vec4(value, 0.0, 0.0, 1.0)  # type: ignore

        # Act
        glsl_code, uniforms = transpile(helper, main_shader, main_func="main_shader")

        # Assert
        assert "float helper(vec2 pos)" in glsl_code
        assert "return pos.x + pos.y;" in glsl_code
        assert "vec4 main_shader(vec2 vs_uv, float u_scale)" in glsl_code
        assert "float value = helper(vs_uv) * u_scale;" in glsl_code
        assert "return vec4(value, 0.0, 0.0, 1.0);" in glsl_code
        assert uniforms == {"u_scale"}

    def test_transpile_complex_setup(self):
        """Test transpiling a complex setup with structs, helpers, and globals."""

        # Arrange
        @dataclass
        class Light:
            position: "vec3"
            color: "vec3"
            intensity: "float" = 1.0

        def calc_diffuse(normal: "vec3", light_dir: "vec3") -> "float":
            return max(dot(normal, light_dir), 0.0)  # type: ignore

        def shader(vs_uv: "vec2", u_light: "Light") -> "vec4":
            pos = vec3(vs_uv.x * 2.0 - 1.0, vs_uv.y * 2.0 - 1.0, 0.0)  # type: ignore
            normal = normalize(vec3(0.0, 0.0, 1.0))  # type: ignore
            light_dir = normalize(u_light.position - pos)  # type: ignore
            diffuse = calc_diffuse(normal, light_dir)
            return vec4(u_light.color * diffuse * u_light.intensity, 1.0)  # type: ignore

        # Act
        glsl_code, uniforms = transpile(
            Light, calc_diffuse, shader, main_func="shader", MAX_DIST=100.0
        )

        # Assert
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

        # Check main function
        assert "vec4 shader(vec2 vs_uv, Light u_light)" in glsl_code
        assert (
            "vec3 pos = vec3(vs_uv.x * 2.0 - 1.0, vs_uv.y * 2.0 - 1.0, 0.0);"
            in glsl_code
        )
        assert "vec3 normal = normalize(vec3(0.0, 0.0, 1.0));" in glsl_code
        assert "vec3 light_dir = normalize(u_light.position - pos);" in glsl_code
        assert "float diffuse = calc_diffuse(normal, light_dir);" in glsl_code
        assert (
            "return vec4(u_light.color * diffuse * u_light.intensity, 1.0);"
            in glsl_code
        )

        # Check uniforms
        assert uniforms == {"u_light"}

    def test_transpile_helper_no_return_type(self):
        """Test that helper function without return type raises an error."""

        # Arrange
        def helper(x: "float"):  # No return type annotation
            return x * 2.0

        def main_shader(vs_uv: "vec2") -> "vec4":
            return vec4(helper(vs_uv.x), 0.0, 0.0, 1.0)  # type: ignore

        # Act & Assert
        with pytest.raises(
            TranspilerError,
            match="Helper function 'helper' lacks return type annotation",
        ):
            transpile(helper, main_shader, main_func="main_shader")

    def test_transpile_unsupported_item(self):
        """Test that unsupported items raise an error."""
        # Arrange - int is not a valid item for transpilation
        item = 42

        # Act & Assert
        with pytest.raises(TranspilerError, match="Unsupported item type"):
            transpile(item)  # type: ignore
