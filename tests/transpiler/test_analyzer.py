"""Tests for shader analyzer functionality."""

import ast
from textwrap import dedent

import pytest

from py2glsl.transpiler.analyzer import GLSLContext, ShaderAnalysis, ShaderAnalyzer
from py2glsl.types import (
    BOOL,
    BVEC2,
    FLOAT,
    INT,
    VEC2,
    VEC3,
    VEC4,
    GLSLType,
    GLSLTypeError,
    TypeKind,
)


class TestShaderAnalyzer:
    """Test shader analyzer functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.analyzer = ShaderAnalyzer()

    def parse_and_analyze(self, code: str) -> ShaderAnalysis:
        """Helper to parse and analyze code."""
        tree = ast.parse(dedent(code))
        return self.analyzer.analyze(tree)

    def test_basic_function_analysis(self):
        """Test basic function analysis."""
        code = """
        def simple(vs_uv: vec2) -> vec4:
            return vec4(vs_uv, 0.0, 1.0)
        """
        analysis = self.parse_and_analyze(code)

        assert analysis.main_function is not None
        assert len(analysis.functions) == 0
        assert analysis.var_types["simple"]["vs_uv"] == VEC2

    def test_uniform_analysis(self):
        """Test uniform variable analysis."""
        code = """
        def shader(vs_uv: vec2, *, u_time: float) -> vec4:
            return vec4(vs_uv, sin(u_time), 1.0)
        """
        analysis = self.parse_and_analyze(code)

        assert "u_time" in analysis.uniforms
        assert analysis.uniforms["u_time"].kind == TypeKind.FLOAT
        assert analysis.uniforms["u_time"].is_uniform

    def test_nested_function_analysis(self):
        """Test nested function analysis."""
        code = """
        def main(vs_uv: vec2) -> vec4:
            def helper(v: vec2) -> float:
                return v.x + v.y
            return vec4(vs_uv, helper(vs_uv), 1.0)
        """
        analysis = self.parse_and_analyze(code)

        assert len(analysis.functions) == 1
        helper_func = analysis.functions[0]
        assert helper_func.name == "helper"
        assert analysis.var_types["helper"]["v"] == VEC2

    def test_type_inference_constants(self):
        """Test type inference for constants."""
        code = """
        def test(vs_uv: vec2) -> vec4:
            a = 1.0
            b = 2
            c = True
            return vec4(a, b, float(c), 1.0)
        """
        analysis = self.parse_and_analyze(code)

        assert analysis.var_types["test"]["a"] == FLOAT
        assert analysis.var_types["test"]["b"] == FLOAT  # Non-loop context
        assert analysis.var_types["test"]["c"] == BOOL

    def test_type_inference_vectors(self):
        """Test type inference for vector operations."""
        code = """
        def test(vs_uv: vec2) -> vec4:
            v2 = vec2(1.0, 2.0)
            v3 = vec3(v2, 3.0)
            v4 = vec4(v3, 1.0)
            return v4
        """
        analysis = self.parse_and_analyze(code)

        assert analysis.var_types["test"]["v2"] == VEC2
        assert analysis.var_types["test"]["v3"] == VEC3
        assert analysis.var_types["test"]["v4"] == VEC4

    def test_loop_variable_types(self):
        """Test type inference in loops."""
        code = """
        def test(vs_uv: vec2) -> vec4:
            result = vec4(0.0)
            for i in range(5):
                result += vec4(float(i))
            return result
        """
        analysis = self.parse_and_analyze(code)

        assert analysis.var_types["test"]["i"] == INT
        assert analysis.var_types["test"]["result"] == VEC4

    def test_swizzle_validation(self):
        """Test vector swizzle validation."""
        code = """
        def test(vs_uv: vec2) -> vec4:
            v2 = vs_uv.xy
            v3 = vec3(vs_uv.x, vs_uv.y, 1.0)
            return vec4(v3, 1.0)
        """
        analysis = self.parse_and_analyze(code)

        assert analysis.var_types["test"]["v2"] == VEC2
        assert analysis.var_types["test"]["v3"] == VEC3

    def test_invalid_swizzle(self):
        """Test invalid swizzle detection."""
        code = """
        def test(vs_uv: vec2) -> vec4:
            v = vs_uv.xyz  # vec2 doesn't have z component
            return vec4(v, 1.0)
        """
        with pytest.raises(GLSLTypeError, match="Invalid swizzle"):
            self.parse_and_analyze(code)

    def test_type_error_detection(self):
        """Test various type error detections."""
        invalid_codes = [
            # Invalid operation
            """
            def test(vs_uv: vec2) -> vec4:
                return vec4(vs_uv * vec3(1.0))
            """,
            # Invalid assignment
            """
            def test(vs_uv: vec2) -> vec4:
                v3: vec3 = vec2(1.0)
                return vec4(v3, 1.0)
            """,
            # Invalid return type
            """
            def test(vs_uv: vec2) -> vec3:
                return vec4(1.0)
            """,
            # Invalid function argument
            """
            def test(vs_uv) -> vec4:  # Missing type annotation
                return vec4(vs_uv, 0.0, 1.0)
            """,
        ]

        for code in invalid_codes:
            with pytest.raises(GLSLTypeError):
                self.parse_and_analyze(code)

    def test_binary_operations(self):
        """Test binary operation type inference."""
        code = """
        def test(vs_uv: vec2) -> vec4:
            a = vec2(1.0) + vec2(2.0)
            b = vec3(1.0) * 2.0
            c = 3.0 / vec4(2.0)
            return vec4(a.x, b.x, c.x, 1.0)
        """
        analysis = self.parse_and_analyze(code)

        assert analysis.var_types["test"]["a"] == VEC2
        assert analysis.var_types["test"]["b"] == VEC3
        assert analysis.var_types["test"]["c"] == VEC4

    def test_comparison_operations(self):
        """Test comparison operation type inference."""
        code = """
        def test(vs_uv: vec2) -> vec4:
            a = 1.0 < 2.0
            b = vec2(1.0) == vec2(2.0)
            c = bvec2(a, b)
            return vec4(float(c.x), float(c.y), 0.0, 1.0)
        """
        analysis = self.parse_and_analyze(code)

        assert analysis.var_types["test"]["a"] == BOOL
        assert analysis.var_types["test"]["b"] == BOOL
        assert analysis.var_types["test"]["c"] == BVEC2

    def test_function_scope_isolation(self):
        """Test variable scope isolation between functions."""
        code = """
        def helper(x: float) -> float:
            y = x + 1.0
            return y

        def main(vs_uv: vec2) -> vec4:
            x = 2.0
            y = helper(x)
            return vec4(vs_uv, y, 1.0)
        """
        analysis = self.parse_and_analyze(code)

        assert "x" in analysis.var_types["helper"]
        assert "y" in analysis.var_types["helper"]
        assert "x" in analysis.var_types["main"]
        assert "y" in analysis.var_types["main"]
        assert analysis.var_types["helper"]["x"] == FLOAT
        assert analysis.var_types["main"]["x"] == FLOAT

    def test_context_tracking(self):
        """Test context tracking in different scenarios."""
        code = """
        def test(vs_uv: vec2) -> vec4:
            result = vec4(0.0)
            for i in range(5):
                if i < 3:
                    result += vec4(float(i))
                else:
                    result *= 2.0
            return result
        """
        analysis = self.parse_and_analyze(code)

        # Verify loop variable is integer
        assert analysis.var_types["test"]["i"] == INT
        # Verify result variable maintains vec4 type
        assert analysis.var_types["test"]["result"] == VEC4
