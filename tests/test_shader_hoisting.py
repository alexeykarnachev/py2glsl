import numpy as np
import pytest

from py2glsl import py2glsl, render_array, vec2, vec4
from py2glsl.builtins import length, normalize, sin, smoothstep
from py2glsl.types import Vec2, Vec3, Vec4, vec2, vec3, vec4


def test_basic_variable_hoisting():
    """Test basic variable hoisting at function scope"""

    def shader(vs_uv: vec2) -> vec4:
        x = 1.0
        return vec4(x, 0.0, 0.0, 1.0)

    result = py2glsl(shader)
    assert "float x;" in result.fragment_source
    assert "x = 1.0;" in result.fragment_source

    # Verify shader compiles and runs
    arr = render_array(shader, size=(64, 64))
    assert not np.any(np.isnan(arr))


def test_if_scope_hoisting():
    """Test variable hoisting from if statements"""

    def shader(vs_uv: vec2) -> vec4:
        if vs_uv.x > 0.5:
            x = 1.0
        else:
            x = 0.0
        return vec4(x, 0.0, 0.0, 1.0)

    result = py2glsl(shader)
    assert "float x;" in result.fragment_source
    assert "x = 1.0;" in result.fragment_source
    assert "x = 0.0;" in result.fragment_source

    # Verify shader compiles and runs
    arr = render_array(shader, size=(64, 64))
    assert not np.any(np.isnan(arr))


def test_loop_scope_hoisting():
    """Test variable hoisting from loop scope"""

    def shader(vs_uv: vec2) -> vec4:
        sum = 0.0
        for i in range(4):
            x = float(i)
            sum += x
        return vec4(sum, 0.0, 0.0, 1.0)

    result = py2glsl(shader)
    assert "float sum;" in result.fragment_source
    assert "float x;" in result.fragment_source
    assert "sum = 0.0;" in result.fragment_source
    assert "x = float(i);" in result.fragment_source

    # Verify shader compiles and runs
    arr = render_array(shader, size=(64, 64))
    assert not np.any(np.isnan(arr))


def test_nested_scope_hoisting():
    """Test hoisting with nested scopes"""

    def shader(vs_uv: vec2) -> vec4:
        if vs_uv.x > 0.5:
            x = 1.0
            if vs_uv.y > 0.5:
                y = 2.0
                x = y
        return vec4(x, 0.0, 0.0, 1.0)

    result = py2glsl(shader)
    assert "float x;" in result.fragment_source
    assert "float y;" in result.fragment_source
    assert "x = 1.0;" in result.fragment_source
    assert "y = 2.0;" in result.fragment_source
    assert "x = y;" in result.fragment_source

    # Verify shader compiles and runs
    arr = render_array(shader, size=(64, 64))
    assert not np.any(np.isnan(arr))


def test_multiple_variable_hoisting():
    """Test hoisting of multiple variables"""

    def shader(vs_uv: vec2) -> vec4:
        x = 1.0
        y = 2.0
        v2 = vec2(x, y)
        v3 = vec3(v2, 3.0)
        return vec4(v3, 1.0)

    result = py2glsl(shader)
    assert "float x;" in result.fragment_source
    assert "float y;" in result.fragment_source
    assert "vec2 v2;" in result.fragment_source
    assert "vec3 v3;" in result.fragment_source

    # Verify shader compiles and runs
    arr = render_array(shader, size=(64, 64))
    assert not np.any(np.isnan(arr))


def test_function_scope_isolation():
    """Test variable isolation between functions"""

    def shader(vs_uv: vec2) -> vec4:
        def func1() -> float:
            x = 1.0  # Should be isolated
            return x

        def func2() -> float:
            x = 2.0  # Different x
            return x

        return vec4(func1() + func2(), 0.0, 0.0, 1.0)

    result = py2glsl(shader)
    func1_part = result.fragment_source[
        result.fragment_source.find("func1") : result.fragment_source.find("func2")
    ]
    func2_part = result.fragment_source[result.fragment_source.find("func2") :]

    assert "float x;" in func1_part
    assert "float x;" in func2_part

    # Verify shader compiles and runs
    arr = render_array(shader, size=(64, 64))
    assert not np.any(np.isnan(arr))


def test_reused_variable_single_declaration():
    """Test that reused variables are only declared once"""

    def shader(vs_uv: vec2) -> vec4:
        x = 1.0
        x = 2.0  # Reuse
        x = 3.0  # Reuse again
        return vec4(x, 0.0, 0.0, 1.0)

    result = py2glsl(shader)
    assert result.fragment_source.count("float x;") == 1
    assert result.fragment_source.count("x =") == 3

    # Verify shader compiles and runs
    arr = render_array(shader, size=(64, 64))
    assert not np.any(np.isnan(arr))


def test_hoisting_with_immediate_init():
    """Test variables are not redeclared when hoisted and immediately initialized"""

    def shader(vs_uv: vec2) -> vec4:
        x = 1.0  # Should be hoisted and initialized in one statement
        return vec4(x, 0.0, 0.0, 1.0)

    result = py2glsl(shader)
    assert result.fragment_source.count("float x") == 1

    # Verify shader compiles and runs
    arr = render_array(shader, size=(64, 64))
    assert not np.any(np.isnan(arr))


def test_hoisting_multiple_variables():
    """Test multiple variables are hoisted correctly without redeclaration"""

    def shader(vs_uv: vec2) -> vec4:
        x = 1.0
        y = 2.0
        z = x + y
        return vec4(x, y, z, 1.0)

    result = py2glsl(shader)
    assert result.fragment_source.count("float x") == 1
    assert result.fragment_source.count("float y") == 1
    assert result.fragment_source.count("float z") == 1

    # Verify shader compiles and runs
    arr = render_array(shader, size=(64, 64))
    assert not np.any(np.isnan(arr))


def test_hoisting_with_vector_components():
    """Test hoisting with vector component assignments"""

    def shader(vs_uv: vec2) -> vec4:
        v = vec2(1.0, 2.0)
        x = v.x
        y = v.y
        return vec4(x, y, 0.0, 1.0)

    result = py2glsl(shader)
    assert result.fragment_source.count("vec2 v") == 1
    assert result.fragment_source.count("float x") == 1
    assert result.fragment_source.count("float y") == 1

    # Verify shader compiles and runs
    arr = render_array(shader, size=(64, 64))
    assert not np.any(np.isnan(arr))


def test_hoisting_with_complex_expressions():
    """Test hoisting with more complex expressions similar to failing shader"""

    def shader(vs_uv: vec2, *, u_time: float = 0.0) -> vec4:
        x = sin(u_time)
        y = x * 2.0
        z = smoothstep(0.0, 1.0, y)
        return vec4(x, y, z, 1.0)

    result = py2glsl(shader)
    assert result.fragment_source.count("float x") == 1
    assert result.fragment_source.count("float y") == 1
    assert result.fragment_source.count("float z") == 1

    # Verify shader compiles and runs
    arr = render_array(shader, size=(64, 64))
    assert not np.any(np.isnan(arr))
