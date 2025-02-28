"""Tests for GLSL builtin function support and overloading."""

from py2glsl.builtins import length, mix, smoothstep, vec2, vec3, vec4
from py2glsl.transpiler import transpile


class TestBuiltinFunctions:
    """Test cases for GLSL builtin functions with overloaded signatures."""

    def test_mix_float_overload(self):
        """Test the mix() function with float parameters."""

        # Arrange
        def shader(t: "float") -> "vec4":
            a = 0.5
            b = 1.0
            c = mix(a, b, t)  # float, float, float -> float
            return vec4(c, c, c, 1.0)  # type: ignore

        # Act
        glsl_code, _ = transpile(shader)

        # Assert
        assert "float c = mix(a, b, t);" in glsl_code

    def test_mix_vec2_overload(self):
        """Test the mix() function with vec2 parameters."""

        # Arrange
        def shader(t: "float") -> "vec4":
            a = vec2(0.0, 0.5)  # type: ignore
            b = vec2(1.0, 1.0)  # type: ignore
            c = mix(a, b, t)  # vec2, vec2, float -> vec2
            return vec4(c.x, c.y, 0.0, 1.0)  # type: ignore

        # Act
        glsl_code, _ = transpile(shader)

        # Assert
        assert "vec2 c = mix(a, b, t);" in glsl_code

    def test_mix_vec3_overload(self):
        """Test the mix() function with vec3 parameters."""

        # Arrange
        def shader(t: "float") -> "vec4":
            a = vec3(0.0, 0.5, 0.3)  # type: ignore
            b = vec3(1.0, 1.0, 0.8)  # type: ignore
            c = mix(a, b, t)  # vec3, vec3, float -> vec3
            return vec4(c, 1.0)  # type: ignore

        # Act
        glsl_code, _ = transpile(shader)

        # Assert
        assert "vec3 c = mix(a, b, t);" in glsl_code

    def test_mix_vec4_overload(self):
        """Test the mix() function with vec4 parameters."""

        # Arrange
        def shader(t: "float") -> "vec4":
            a = vec4(0.0, 0.5, 0.3, 1.0)  # type: ignore
            b = vec4(1.0, 1.0, 0.8, 1.0)  # type: ignore
            c = mix(a, b, t)  # vec4, vec4, float -> vec4
            return c

        # Act
        glsl_code, _ = transpile(shader)

        # Assert
        assert "vec4 c = mix(a, b, t);" in glsl_code

    def test_smoothstep_float_overload(self):
        """Test the smoothstep() function with float parameters."""

        # Arrange
        def shader(t: "float") -> "vec4":
            edge0 = 0.1
            edge1 = 0.9
            result = smoothstep(edge0, edge1, t)  # float, float, float -> float
            return vec4(result, result, result, 1.0)  # type: ignore

        # Act
        glsl_code, _ = transpile(shader)

        # Assert
        assert "float result = smoothstep(edge0, edge1, t);" in glsl_code

    def test_smoothstep_vec2_overload(self):
        """Test the smoothstep() function with vec2 parameters."""

        # Arrange
        def shader(uv: "vec2") -> "vec4":
            edge0 = vec2(0.1, 0.2)  # type: ignore
            edge1 = vec2(0.8, 0.9)  # type: ignore
            result = smoothstep(edge0, edge1, uv)  # vec2, vec2, vec2 -> vec2
            return vec4(result, 0.0, 1.0)  # type: ignore

        # Act
        glsl_code, _ = transpile(shader)

        # Assert
        assert "vec2 result = smoothstep(edge0, edge1, uv);" in glsl_code

    def test_length_overloads(self):
        """Test the length() function with different vector types."""

        # Arrange
        def shader() -> "vec4":
            v2 = vec2(1.0, 2.0)  # type: ignore
            v3 = vec3(1.0, 2.0, 3.0)  # type: ignore
            v4 = vec4(1.0, 2.0, 3.0, 4.0)  # type: ignore

            len2 = length(v2)  # vec2 -> float
            len3 = length(v3)  # vec3 -> float
            len4 = length(v4)  # vec4 -> float

            return vec4(len2, len3, len4, 1.0)  # type: ignore

        # Act
        glsl_code, _ = transpile(shader)

        # Assert
        assert "float len2 = length(v2);" in glsl_code
        assert "float len3 = length(v3);" in glsl_code
        assert "float len4 = length(v4);" in glsl_code

    def test_min_max_overloads(self):
        """Test the min()/max() functions with various vector types."""

        # Arrange
        def shader() -> "vec4":
            # Scalar version
            a = 1.0
            b = 2.0
            min_float = min(a, b)  # float, float -> float
            max_float = max(a, b)  # float, float -> float

            # Vector versions
            v2a = vec2(1.0, 3.0)  # type: ignore
            v2b = vec2(2.0, 2.0)  # type: ignore
            min_vec2 = min(v2a, v2b)  # vec2, vec2 -> vec2
            # Use both min and max functions in result to ensure both work correctly
            max_vec2 = max(v2a, v2b)  # vec2, vec2 -> vec2

            # Test vec3 versions to cover all cases
            v3a = vec3(1.0, 3.0, 5.0)  # type: ignore
            v3b = vec3(2.0, 2.0, 4.0)  # type: ignore
            # Also test vec3 min function to verify it works too
            min_vec3 = min(v3a, v3b)  # vec3, vec3 -> vec3

            return vec4(min_float, max_float, min_vec2.x + max_vec2.y, min_vec3.z)  # type: ignore

        # Act
        glsl_code, _ = transpile(shader)

        # Assert
        assert "float min_float = min(a, b);" in glsl_code
        assert "float max_float = max(a, b);" in glsl_code
        assert "vec2 min_vec2 = min(v2a, v2b);" in glsl_code
        assert "vec2 max_vec2 = max(v2a, v2b);" in glsl_code
        assert "vec3 min_vec3 = min(v3a, v3b);" in glsl_code
