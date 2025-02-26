"""Tests for the main transpiler module."""

from dataclasses import dataclass

import pytest

from py2glsl.transpiler import TranspilerError, transpile


class TestTranspile:
    """Tests for the transpile function."""

    def test_transpile_string(self):
        """Test transpiling code from a string."""
        # Arrange
        shader_code = """
        def shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
            return vec4(1.0, 0.0, 0.0, 1.0)
        """

        # Act
        glsl_code, uniforms = transpile(shader_code)

        # Assert
        assert "#version 460 core" in glsl_code
        assert "uniform float u_time;" in glsl_code
        assert "vec4 shader(vec2 vs_uv, float u_time)" in glsl_code
        assert "return vec4(1.0, 0.0, 0.0, 1.0);" in glsl_code
        assert "void main() {" in glsl_code
        assert "fragColor = shader(vs_uv, u_time);" in glsl_code
        assert uniforms == {"u_time"}

    def test_transpile_function(self):
        """Test transpiling a function object."""

        # Arrange
        def simple_shader(vs_uv: "vec2", u_time: "float") -> "vec4":
            return vec4(vs_uv.x, vs_uv.y, 0.0, 1.0)  # type: ignore

        # Act
        glsl_code, uniforms = transpile(simple_shader)

        # Assert
        assert "#version 460 core" in glsl_code
        assert "uniform float u_time;" in glsl_code
        assert "vec4 simple_shader(vec2 vs_uv, float u_time)" in glsl_code
        assert "return vec4(vs_uv.x, vs_uv.y, 0.0, 1.0);" in glsl_code
        assert uniforms == {"u_time"}

    def test_transpile_with_custom_main(self):
        """Test transpiling with a custom main function name."""
        # Arrange
        shader_code = """
        def my_shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
            return vec4(1.0, 0.0, 0.0, 1.0)
        """

        # Act
        glsl_code, uniforms = transpile(shader_code, main_func="my_shader")

        # Assert
        assert "vec4 my_shader(vec2 vs_uv, float u_time)" in glsl_code
        assert "fragColor = my_shader(vs_uv, u_time);" in glsl_code

    def test_transpile_with_struct(self):
        """Test transpiling with a dataclass struct."""

        # Arrange
        @dataclass
        class Material:
            color: "vec3"
            shininess: "float" = 32.0

        def shader(vs_uv: "vec2", u_material: "Material") -> "vec4":
            return vec4(u_material.color, 1.0)  # type: ignore

        # Act
        glsl_code, uniforms = transpile(Material, shader)

        # Assert
        assert "struct Material {" in glsl_code
        assert "vec3 color;" in glsl_code
        assert "float shininess;" in glsl_code
        assert "uniform Material u_material;" in glsl_code
        assert "return vec4(u_material.color, 1.0);" in glsl_code
        assert uniforms == {"u_material"}

    def test_transpile_with_globals(self):
        """Test transpiling with global constants."""
        # Arrange
        shader_code = """
        def shader(vs_uv: 'vec2') -> 'vec4':
            return vec4(sin(PI * vs_uv.x), 0.0, 0.0, 1.0)
        """

        # Act
        glsl_code, uniforms = transpile(shader_code, PI=3.14159)

        # Assert
        assert "const float PI = 3.14159;" in glsl_code
        assert "return vec4(sin(PI * vs_uv.x), 0.0, 0.0, 1.0);" in glsl_code

    def test_transpile_helper_and_main(self):
        """Test transpiling with helper and main functions."""

        # Arrange
        def helper(pos: "vec2") -> "float":
            return pos.x + pos.y

        def main_shader(vs_uv: "vec2", u_scale: "float") -> "vec4":
            value = helper(vs_uv) * u_scale
            return vec4(value, 0.0, 0.0, 1.0)  # type: ignore

        # Act
        glsl_code, uniforms = transpile(helper, main_shader)

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
        glsl_code, uniforms = transpile(Light, calc_diffuse, shader, MAX_DIST=100.0)

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

    def test_transpile_test_prefixed_function(self):
        """Test that functions with 'test_' prefix are rejected."""

        # Arrange
        def test_shader(vs_uv: "vec2") -> "vec4":
            return vec4(1.0, 0.0, 0.0, 1.0)  # type: ignore

        # Act & Assert
        with pytest.raises(TranspilerError, match="excluded due to 'test_' prefix"):
            transpile(test_shader)

    def test_transpile_missing_main(self):
        """Test that missing main function raises an error."""
        # Arrange
        shader_code = """
        def helper(x: 'float') -> 'float':
            return x * 2.0
        """

        # Act & Assert
        with pytest.raises(TranspilerError, match="Main function 'shader' not found"):
            transpile(shader_code)

    def test_transpile_empty_code(self):
        """Test that empty code raises an error."""
        # Act & Assert
        with pytest.raises(TranspilerError, match="Empty shader code provided"):
            transpile("")

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
            transpile(helper, main_shader)

    def test_transpile_unsupported_item(self):
        """Test that unsupported items raise an error."""
        # Arrange - int is not a valid item for transpilation
        item = 42

        # Act & Assert
        with pytest.raises(TranspilerError, match="Unsupported item type"):
            transpile(item)  # type: ignore
