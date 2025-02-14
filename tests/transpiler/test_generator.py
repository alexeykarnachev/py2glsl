"""Tests for GLSL code generator."""

import ast
from textwrap import dedent

import pytest

from py2glsl.transpiler.analyzer import ShaderAnalyzer
from py2glsl.transpiler.generator import GLSLGenerator
from py2glsl.types import GLSLTypeError


def generate_shader(code: str) -> str:
    """Helper to generate GLSL code from Python code."""
    tree = ast.parse(dedent(code))
    analyzer = ShaderAnalyzer()
    analysis = analyzer.analyze(tree)
    generator = GLSLGenerator(analysis)
    return generator.generate().fragment_source


def test_basic_shader_generation():
    """Test basic shader generation."""
    code = """
    def shader(vs_uv: vec2) -> vec4:
        return vec4(vs_uv, 0.0, 1.0)
    """
    glsl = generate_shader(code)
    assert "#version 460" in glsl
    assert "in vec2 vs_uv;" in glsl
    assert "out vec4 fs_color;" in glsl
    assert "vec4 shader(vec2 vs_uv)" in glsl
    assert "return vec4(vs_uv, 0.0, 1.0);" in glsl


def test_vector_constructor_validation():
    """Test vector constructor validation."""
    code = """
    def shader(vs_uv: vec2) -> vec4:
        return vec4(vs_uv, vs_uv)  # Invalid: too many components
    """
    with pytest.raises(GLSLTypeError, match="Invalid number of components"):
        generate_shader(code)


def test_for_loop_validation():
    """Test for loop validation."""
    code = """
    def shader(vs_uv: vec2) -> vec4:
        for i in range(1.5):  # Invalid: range argument must be integer
            pass
        return vec4(0.0)
    """
    with pytest.raises(GLSLTypeError, match="Range argument must be integer"):
        generate_shader(code)


def test_uniform_generation():
    """Test uniform variable generation."""
    code = """
    def shader(vs_uv: vec2, *, u_color: vec4) -> vec4:
        return u_color
    """
    glsl = generate_shader(code)
    assert "uniform vec4 u_color;" in glsl


def test_variable_declaration():
    """Test variable declaration and initialization."""
    code = """
    def shader(vs_uv: vec2) -> vec4:
        color: vec4 = vec4(1.0)
        return color
    """
    glsl = generate_shader(code)
    assert "vec4 color = vec4(1.0, 1.0, 1.0, 1.0);" in glsl


def test_if_statement_validation():
    """Test if statement validation."""
    code = """
    def shader(vs_uv: vec2) -> vec4:
        if vs_uv:  # Invalid: condition must be boolean
            return vec4(1.0)
        return vec4(0.0)
    """
    with pytest.raises(GLSLTypeError, match="If condition must be boolean"):
        generate_shader(code)


def test_type_mismatch_validation():
    """Test type mismatch validation."""
    code = """
    def shader(vs_uv: vec2) -> vec4:
        x: float = vs_uv  # Invalid: cannot assign vec2 to float
        return vec4(x)
    """
    with pytest.raises(GLSLTypeError, match="Cannot assign"):
        generate_shader(code)


def test_return_type_validation():
    """Test return type validation."""
    code = """
    def shader(vs_uv: vec2) -> vec4:
        return vs_uv  # Invalid: cannot return vec2 from vec4 function
    """
    with pytest.raises(GLSLTypeError, match="Cannot return"):
        generate_shader(code)


def test_complex_shader_generation():
    """Test generation of a more complex shader."""
    code = """
    def shader(vs_uv: vec2) -> vec4:
        pos: vec2 = vs_uv * 2.0 - 1.0
        d: float = length(pos)
        color: float = sin(d * 10.0)
        return vec4(color, color * 0.5, 1.0 - color, 1.0)
    """
    glsl = generate_shader(code)
    assert "vec2 pos = vs_uv * 2.0 - 1.0;" in glsl
    assert "float d = length(pos);" in glsl
    assert "float color = sin(d * 10.0);" in glsl
    assert "return vec4(color, color * 0.5, 1.0 - color, 1.0);" in glsl


def test_nested_function_generation():
    """Test generation of nested functions."""
    code = """
    def shader(vs_uv: vec2) -> vec4:
        def circle(p: vec2, r: float) -> float:
            return smoothstep(r, r - 0.01, length(p))
        
        pos = vs_uv * 2.0 - 1.0
        c = circle(pos, 0.5)
        return vec4(c, c, c, 1.0)
    """
    glsl = generate_shader(code)
    assert "float circle(vec2 p, float r)" in glsl
    assert "return smoothstep(r, r - 0.01, length(p));" in glsl
    assert "vec4 shader(vec2 vs_uv)" in glsl


def test_builtin_uniform_inclusion():
    """Test built-in uniform inclusion."""
    code = """
    def shader(vs_uv: vec2) -> vec4:
        return vec4(vs_uv, sin(u_time), 1.0)
    """
    glsl = generate_shader(code)
    assert "uniform float u_time;" in glsl
    assert "uniform float u_aspect;" in glsl
    assert "uniform int u_frame;" in glsl
    assert "uniform vec2 u_mouse;" in glsl
