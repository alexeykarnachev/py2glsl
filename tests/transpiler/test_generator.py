"""Tests for GLSL code generator."""

import ast
from textwrap import dedent

import pytest

from py2glsl.transpiler.analyzer import ShaderAnalysis
from py2glsl.transpiler.generator import GeneratedShader, GLSLGenerator
from py2glsl.types import BOOL, FLOAT, INT, VEC2, VEC3, VEC4, GLSLTypeError


@pytest.fixture
def basic_analysis() -> ShaderAnalysis:
    """Create basic shader analysis for testing."""
    analysis = ShaderAnalysis()
    analysis.var_types = {
        "global": {"shader": VEC4},
        "shader": {"vs_uv": VEC2},
    }
    analysis.hoisted_vars = {"global": set(), "shader": set()}
    return analysis


def test_generator_initialization(basic_analysis):
    """Test generator initialization."""
    generator = GLSLGenerator(basic_analysis)
    assert generator.analysis == basic_analysis
    assert generator.indent_level == 0
    assert generator.lines == []
    assert generator.current_scope == "global"
    assert generator.scope_stack == []
    assert generator.declared_vars == {"global": set()}


def test_indentation():
    """Test indentation handling."""
    generator = GLSLGenerator(ShaderAnalysis())
    assert generator.indent() == ""
    generator.indent_level = 1
    assert generator.indent() == "    "
    generator.indent_level = 2
    assert generator.indent() == "        "


def test_basic_shader_generation(basic_analysis):
    """Test basic shader generation."""
    code = dedent(
        """
    def shader(vs_uv: vec2) -> vec4:
        return vec4(vs_uv, 0.0, 1.0)
    """
    )

    tree = ast.parse(code)
    basic_analysis.main_function = tree.body[0]

    generator = GLSLGenerator(basic_analysis)
    result = generator.generate()

    assert isinstance(result, GeneratedShader)
    assert "#version 460" in result.fragment_source
    assert "in vec2 vs_uv;" in result.fragment_source
    assert "out vec4 fs_color;" in result.fragment_source
    assert "vec4 shader(vec2 vs_uv)" in result.fragment_source
    assert "return vec4(vs_uv, 0.0, 1.0);" in result.fragment_source


def test_vector_constructor_validation(basic_analysis):
    """Test vector constructor validation."""
    code = dedent(
        """
    def shader(vs_uv: vec2) -> vec4:
        return vec4(vs_uv, vs_uv)  # Invalid: too many components
    """
    )

    tree = ast.parse(code)
    basic_analysis.main_function = tree.body[0]

    generator = GLSLGenerator(basic_analysis)
    with pytest.raises(GLSLTypeError, match="Invalid number of components"):
        generator.generate()


def test_if_statement_validation(basic_analysis):
    """Test if statement type validation."""
    code = dedent(
        """
    def shader(vs_uv: vec2) -> vec4:
        if vs_uv:  # Invalid: condition must be boolean
            return vec4(1.0)
        return vec4(0.0)
    """
    )

    tree = ast.parse(code)
    basic_analysis.main_function = tree.body[0]

    generator = GLSLGenerator(basic_analysis)
    with pytest.raises(GLSLTypeError, match="If condition must be boolean"):
        generator.generate()


def test_for_loop_validation(basic_analysis):
    """Test for loop validation."""
    code = dedent(
        """
    def shader(vs_uv: vec2) -> vec4:
        for i in range(1.5):  # Invalid: range argument must be integer
            pass
        return vec4(0.0)
    """
    )

    tree = ast.parse(code)
    basic_analysis.main_function = tree.body[0]

    generator = GLSLGenerator(basic_analysis)
    with pytest.raises(GLSLTypeError, match="Range argument must be integer"):
        generator.generate()


def test_assignment_type_validation(basic_analysis):
    """Test assignment type validation."""
    code = dedent(
        """
    def shader(vs_uv: vec2) -> vec4:
        x: float = vs_uv  # Invalid: cannot assign vec2 to float
        return vec4(x)
    """
    )

    tree = ast.parse(code)
    basic_analysis.main_function = tree.body[0]
    basic_analysis.var_types["shader"]["x"] = FLOAT
    basic_analysis.hoisted_vars["shader"].add("x")

    generator = GLSLGenerator(basic_analysis)
    with pytest.raises(GLSLTypeError, match="Cannot assign value of type"):
        generator.generate()


def test_return_type_validation(basic_analysis):
    """Test return type validation."""
    code = dedent(
        """
    def shader(vs_uv: vec2) -> vec4:
        return vs_uv  # Invalid: cannot return vec2 from vec4 function
    """
    )

    tree = ast.parse(code)
    basic_analysis.main_function = tree.body[0]

    generator = GLSLGenerator(basic_analysis)
    with pytest.raises(GLSLTypeError, match="Cannot return value of type"):
        generator.generate()


def test_complex_shader_generation(basic_analysis):
    """Test generation of a more complex shader."""
    code = dedent(
        """
    def shader(vs_uv: vec2) -> vec4:
        pos: vec2 = vs_uv * 2.0 - 1.0
        d: float = length(pos)
        color: float = sin(d * 10.0)
        return vec4(color, color * 0.5, 1.0 - color, 1.0)
    """
    )

    tree = ast.parse(code)
    basic_analysis.main_function = tree.body[0]
    basic_analysis.var_types["shader"].update(
        {
            "pos": VEC2,
            "d": FLOAT,
            "color": FLOAT,
        }
    )
    basic_analysis.hoisted_vars["shader"].update({"pos", "d", "color"})

    generator = GLSLGenerator(basic_analysis)
    result = generator.generate()

    assert "vec2 pos;" in result.fragment_source
    assert "float d;" in result.fragment_source
    assert "float color;" in result.fragment_source
    assert "pos = vs_uv * 2.0 - 1.0;" in result.fragment_source
    assert "d = length(pos);" in result.fragment_source
    assert "color = sin(d * 10.0);" in result.fragment_source


def test_uniform_generation(basic_analysis):
    """Test uniform variable generation."""
    basic_analysis.uniforms = {
        "u_time": FLOAT,
        "u_resolution": VEC2,
    }

    generator = GLSLGenerator(basic_analysis)
    result = generator.generate()

    assert "uniform float u_time;" in result.fragment_source
    assert "uniform vec2 u_resolution;" in result.fragment_source


def test_vertex_shader_inclusion(basic_analysis):
    """Test that vertex shader is included in output."""
    generator = GLSLGenerator(basic_analysis)
    result = generator.generate()

    assert "layout(location = 0) in vec2 in_pos;" in result.vertex_source
    assert "layout(location = 1) in vec2 in_uv;" in result.vertex_source
    assert "out vec2 vs_uv;" in result.vertex_source
    assert "gl_Position = vec4(in_pos, 0.0, 1.0);" in result.vertex_source
