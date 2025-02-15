"""Tests for GLSL transpiler functionality."""

import ast
from textwrap import dedent

import pytest

from py2glsl.transpiler.analyzer import ShaderAnalyzer
from py2glsl.transpiler.generator import GeneratedShader, GLSLGenerator
from py2glsl.types import (
    BOOL,
    FLOAT,
    INT,
    VEC2,
    VEC3,
    VEC4,
    GLSLType,
    GLSLTypeError,
    TypeKind,
)


def analyze_shader(code: str):
    """Helper function to analyze shader code."""
    tree = ast.parse(dedent(code))
    analyzer = ShaderAnalyzer()
    return analyzer.analyze(tree)


# Basic functionality tests
def test_basic_shader_generation():
    """Test basic shader generation."""
    code = """
    def shader(vs_uv: vec2) -> vec4:
        return vec4(vs_uv, 0.0, 1.0)
    """
    analysis = analyze_shader(code)
    generator = GLSLGenerator(analysis)
    result = generator.generate()

    assert isinstance(result, GeneratedShader)
    assert "#version 460" in result.fragment_source
    assert "in vec2 vs_uv;" in result.fragment_source
    assert "out vec4 fs_color;" in result.fragment_source
    assert "vec4 shader(vec2 vs_uv)" in result.fragment_source
    assert "return vec4(vs_uv, 0.0, 1.0);" in result.fragment_source


# Type system tests
def test_type_inference_constants():
    """Test type inference for constants."""
    code = """
    def test(vs_uv: vec2) -> vec4:
        a = 1.0
        b = 2
        c = True
        return vec4(a, b, float(c), 1.0)
    """
    analysis = analyze_shader(code)
    assert analysis.var_types["test"]["a"] == FLOAT
    assert analysis.var_types["test"]["b"] == FLOAT  # Non-loop context
    assert analysis.var_types["test"]["c"] == BOOL


def test_type_inference_vectors():
    """Test type inference for vector operations."""
    code = """
    def test(vs_uv: vec2) -> vec4:
        v2 = vec2(1.0, 2.0)
        v3 = vec3(v2, 3.0)
        v4 = vec4(v3, 1.0)
        return v4
    """
    analysis = analyze_shader(code)
    assert analysis.var_types["test"]["v2"] == VEC2
    assert analysis.var_types["test"]["v3"] == VEC3
    assert analysis.var_types["test"]["v4"] == VEC4


# Vector operations tests
def test_vector_component_modification():
    """Test vector component modification."""
    code = """
    def shader(vs_uv: vec2, *, u_aspect: float) -> vec4:
        p = vs_uv
        p.x *= u_aspect
        return vec4(p, 0.0, 1.0)
    """
    analysis = analyze_shader(code)
    generator = GLSLGenerator(analysis)
    result = generator.generate()
    assert "p.x *= u_aspect;" in result.fragment_source


def test_vector_swizzle():
    """Test vector swizzle operations."""
    code = """
    def test(vs_uv: vec2) -> vec4:
        v2 = vs_uv.xy
        v3 = vec3(vs_uv.x, vs_uv.y, 1.0)
        return vec4(v3, 1.0)
    """
    analysis = analyze_shader(code)
    assert analysis.var_types["test"]["v2"] == VEC2
    assert analysis.var_types["test"]["v3"] == VEC3


# Control flow tests
def test_if_statement_validation():
    """Test if statement type validation."""
    code = """
    def shader(vs_uv: vec2) -> vec4:
        if vs_uv:  # Invalid: condition must be boolean
            return vec4(1.0)
        return vec4(0.0)
    """
    analysis = analyze_shader(code)
    generator = GLSLGenerator(analysis)
    with pytest.raises(GLSLTypeError, match="If condition must be boolean"):
        generator.generate()


def test_loop_variable_types():
    """Test type inference in loops."""
    code = """
    def test(vs_uv: vec2) -> vec4:
        result = vec4(0.0)
        for i in range(5):
            result += vec4(float(i))
        return result
    """
    analysis = analyze_shader(code)
    assert analysis.var_types["test"]["i"] == INT
    assert analysis.var_types["test"]["result"] == VEC4


# Function and scope tests
def test_nested_functions():
    """Test nested function handling."""
    code = """
    def main_shader(vs_uv: vec2, *, u_time: float) -> vec4:
        def get_color_in(p: vec2) -> vec3:
            return vec3(0.0, 1.0, 0.0)
        color = get_color_in(vs_uv)
        return vec4(color, 1.0)
    """
    analysis = analyze_shader(code)
    generator = GLSLGenerator(analysis)
    result = generator.generate()
    assert "vec3 get_color_in(vec2 p)" in result.fragment_source


def test_global_function_reference():
    """Test global function reference."""
    code = """
    def get_color_out(t: float) -> vec3:
        return vec3(0.5 * (sin(t) + 1.0), 0.0, 0.0)

    def main_shader(vs_uv: vec2, *, u_time: float) -> vec4:
        color = get_color_out(u_time)
        return vec4(color, 1.0)
    """
    analysis = analyze_shader(code)
    generator = GLSLGenerator(analysis)
    result = generator.generate()
    assert "vec3 get_color_out(float t)" in result.fragment_source


# Uniform and built-in tests
def test_uniform_generation():
    """Test uniform variable generation."""
    code = """
    def shader(vs_uv: vec2, *, u_time: float, u_aspect: float) -> vec4:
        return vec4(vs_uv * u_aspect, sin(u_time), 1.0)
    """
    analysis = analyze_shader(code)
    generator = GLSLGenerator(analysis)
    result = generator.generate()
    assert "uniform float u_time;" in result.fragment_source
    assert "uniform float u_aspect;" in result.fragment_source


# Error cases tests
def test_invalid_swizzle():
    """Test invalid swizzle detection."""
    code = """
    def test(vs_uv: vec2) -> vec4:
        v = vs_uv.xyz  # vec2 doesn't have z component
        return vec4(v, 1.0)
    """
    with pytest.raises(GLSLTypeError, match="Invalid swizzle"):
        analyze_shader(code)


def test_invalid_type_assignment():
    """Test invalid type assignment."""
    code = """
    def test(vs_uv: vec2) -> vec4:
        x: float = vs_uv  # Invalid: cannot assign vec2 to float
        return vec4(x)
    """
    analysis = analyze_shader(code)
    generator = GLSLGenerator(analysis)
    with pytest.raises(GLSLTypeError, match="Cannot assign value of type"):
        generator.generate()


def test_invalid_return_type():
    """Test invalid return type."""
    code = """
    def test(vs_uv: vec2) -> vec3:
        return vec4(1.0)  # Invalid: returning vec4 from vec3 function
    """
    analysis = analyze_shader(code)
    generator = GLSLGenerator(analysis)
    with pytest.raises(GLSLTypeError, match="Cannot return value of type"):
        generator.generate()


def test_numeric_operations():
    """Test numeric operations translation."""
    code = """
    def shader(vs_uv: vec2) -> vec4:
        # Basic arithmetic
        a = 2.0 + 3.0
        b = 5.0 - 1.0
        c = 2.0 * 3.0
        d = 6.0 / 2.0
        # Compound operations
        e = (a + b) * (c - d)
        # Mixed integer and float
        f = 2 * 3.0
        g = 5.0 / 2
        return vec4(a/10.0, b/10.0, c/10.0, 1.0)
    """
    analysis = analyze_shader(code)
    generator = GLSLGenerator(analysis)
    result = generator.generate()

    assert "a = 2.0 + 3.0;" in result.fragment_source
    assert "b = 5.0 - 1.0;" in result.fragment_source
    assert "c = 2.0 * 3.0;" in result.fragment_source
    assert "d = 6.0 / 2.0;" in result.fragment_source
    assert "e = (a + b) * (c - d);" in result.fragment_source
    assert "f = 2.0 * 3.0;" in result.fragment_source
    assert "g = 5.0 / 2.0;" in result.fragment_source


def test_vector_operations():
    """Test vector operations translation."""
    code = """
    def shader(vs_uv: vec2) -> vec4:
        # Vector construction
        v2 = vec2(1.0, 2.0)
        v3 = vec3(v2, 3.0)
        v4 = vec4(v3, 1.0)
        
        # Vector arithmetic
        scaled = v2 * 2.0
        added = v2 + vec2(1.0)
        mixed = v2 * vec2(2.0, 3.0)
        
        # Vector swizzling
        xy = v3.xy
        yx = v3.yx
        rgb = v3.rgb
        
        return v4
    """
    analysis = analyze_shader(code)
    generator = GLSLGenerator(analysis)
    result = generator.generate()

    assert "vec2 v2 = vec2(1.0, 2.0);" in result.fragment_source
    assert "vec3 v3 = vec3(v2, 3.0);" in result.fragment_source
    assert "vec4 v4 = vec4(v3, 1.0);" in result.fragment_source
    assert "scaled = v2 * 2.0;" in result.fragment_source
    assert "added = v2 + vec2(1.0);" in result.fragment_source
    assert "mixed = v2 * vec2(2.0, 3.0);" in result.fragment_source
    assert "xy = v3.xy;" in result.fragment_source
    assert "yx = v3.yx;" in result.fragment_source
    assert "rgb = v3.rgb;" in result.fragment_source


def test_control_flow_translation():
    """Test control flow statements translation."""
    code = """
    def shader(vs_uv: vec2) -> vec4:
        color = vec4(0.0)
        
        # If-else with comparison operators
        if vs_uv.x > 0.5:
            color = vec4(1.0, 0.0, 0.0, 1.0)
        elif vs_uv.x > 0.25:
            color = vec4(0.0, 1.0, 0.0, 1.0)
        else:
            color = vec4(0.0, 0.0, 1.0, 1.0)
        
        # For loop with range
        for i in range(3):
            color += vec4(0.1)
            
        # Break and continue
        for i in range(5):
            if i > 3:
                break
            if i < 2:
                continue
            color += vec4(0.05)
            
        return color
    """
    analysis = analyze_shader(code)
    generator = GLSLGenerator(analysis)
    result = generator.generate()

    assert "if (vs_uv.x > 0.5)" in result.fragment_source
    assert "else if (vs_uv.x > 0.25)" in result.fragment_source
    assert "else" in result.fragment_source
    assert "for (int i = 0; i < 3; i++)" in result.fragment_source
    assert "break;" in result.fragment_source
    assert "continue;" in result.fragment_source


def test_builtin_functions():
    """Test built-in function translation."""
    code = """
    def shader(vs_uv: vec2) -> vec4:
        # Trigonometric functions
        s = sin(vs_uv.x)
        c = cos(vs_uv.y)
        t = tan(vs_uv.x)
        
        # Common math functions
        l = length(vs_uv)
        n = normalize(vs_uv)
        d = dot(vs_uv, vec2(1.0))
        
        # Mix and step functions
        m = mix(0.0, 1.0, 0.5)
        st = step(0.5, vs_uv.x)
        sm = smoothstep(0.0, 1.0, vs_uv.x)
        
        # Other math functions
        f = floor(vs_uv.x)
        cl = ceil(vs_uv.y)
        fr = fract(vs_uv.x)
        
        return vec4(s, c, t, 1.0)
    """
    analysis = analyze_shader(code)
    generator = GLSLGenerator(analysis)
    result = generator.generate()

    # Check function calls are properly translated
    assert "sin(vs_uv.x)" in result.fragment_source
    assert "cos(vs_uv.y)" in result.fragment_source
    assert "tan(vs_uv.x)" in result.fragment_source
    assert "length(vs_uv)" in result.fragment_source
    assert "normalize(vs_uv)" in result.fragment_source
    assert "dot(vs_uv, vec2(1.0))" in result.fragment_source
    assert "mix(0.0, 1.0, 0.5)" in result.fragment_source
    assert "step(0.5, vs_uv.x)" in result.fragment_source
    assert "smoothstep(0.0, 1.0, vs_uv.x)" in result.fragment_source
    assert "floor(vs_uv.x)" in result.fragment_source
    assert "ceil(vs_uv.y)" in result.fragment_source
    assert "fract(vs_uv.x)" in result.fragment_source


def test_type_conversions():
    """Test type conversion handling."""
    code = """
    def shader(vs_uv: vec2) -> vec4:
        # Numeric conversions
        i = int(5.7)
        f = float(3)
        
        # Vector conversions
        v2f = vec2(1.0)        # Scalar to vec2
        v2i = vec2(i)          # Int to vec2
        v3f = vec3(v2f, 0.0)   # vec2 to vec3
        v4f = vec4(v3f, 1.0)   # vec3 to vec4
        
        return v4f
    """
    analysis = analyze_shader(code)
    generator = GLSLGenerator(analysis)
    result = generator.generate()

    assert "int(5.7)" in result.fragment_source
    assert "float(3)" in result.fragment_source
    assert "vec2(1.0)" in result.fragment_source
    assert "vec2(float(i))" in result.fragment_source
    assert "vec3(v2f, 0.0)" in result.fragment_source
    assert "vec4(v3f, 1.0)" in result.fragment_source


def test_compound_expressions():
    """Test complex expression translation."""
    code = """
    def shader(vs_uv: vec2) -> vec4:
        # Compound vector operations
        v = normalize(vs_uv * 2.0 - 1.0)
        
        # Nested function calls
        c = cos(sin(vs_uv.x * 3.14159))
        
        # Complex math expressions
        f = mix(
            smoothstep(0.0, 1.0, v.x),
            step(0.5, v.y),
            abs(sin(v.x * 3.14159))
        )
        
        return vec4(v, f, 1.0)
    """
    analysis = analyze_shader(code)
    generator = GLSLGenerator(analysis)
    result = generator.generate()

    assert "normalize(vs_uv * 2.0 - 1.0)" in result.fragment_source
    assert "cos(sin(vs_uv.x * 3.14159))" in result.fragment_source
    assert (
        "mix(smoothstep(0.0, 1.0, v.x), step(0.5, v.y), abs(sin(v.x * 3.14159)))"
        in result.fragment_source
    )
