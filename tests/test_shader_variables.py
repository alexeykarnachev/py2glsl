import asyncio

import glfw
import imageio.v3 as iio
import numpy as np
import pytest
from PIL import Image

from py2glsl import (
    animate,
    py2glsl,
    render_array,
    render_gif,
    render_image,
    render_video,
    vec2,
    vec4,
)
from py2glsl.builtins import length, normalize, sin, smoothstep
from py2glsl.types import Vec2, Vec3, Vec4, vec2, vec3, vec4


def test_variables() -> None:
    def var_shader(vs_uv: vec2) -> vec4:
        x = 1.0
        return vec4(x, 0.0, 0.0, 1.0)

    result = py2glsl(var_shader)
    assert "float x = 1.0;" in result.fragment_source


def test_chained_assignments():
    """Test chained assignments"""

    def shader(vs_uv: vec2) -> vec4:
        x = y = z = 1.0
        return vec4(x, y, z, 1.0)

    result = py2glsl(shader)
    assert "float x = 1.0;" in result.fragment_source
    assert "float y = 1.0;" in result.fragment_source
    assert "float z = 1.0;" in result.fragment_source


def test_compound_assignments() -> None:
    def shader(vs_uv: vec2) -> vec4:
        x = 1.0
        x += 2.0
        x *= 3.0
        return vec4(x, 0.0, 0.0, 1.0)

    result = py2glsl(shader)
    assert "x += 2.0;" in result.fragment_source
    assert "x *= 3.0;" in result.fragment_source


def test_variable_scoping():
    """Test variable scoping and redeclaration in different blocks"""

    def shader(vs_uv: vec2) -> vec4:
        if length(vs_uv) > 0.5:
            x = 1.0
        else:
            x = 0.0
        return vec4(x, 0.0, 0.0, 1.0)

    result = py2glsl(shader)
    assert "float x;" in result.fragment_source  # Variable should be hoisted
    assert "x = 1.0;" in result.fragment_source  # Assignment without redeclaration
    assert "x = 0.0;" in result.fragment_source  # Assignment without redeclaration


def test_variable_reuse():
    """Test reusing variable names in different blocks"""

    def shader(vs_uv: vec2) -> vec4:
        local_uv = vs_uv * 2.0 - 1.0
        if length(local_uv) > 0.5:
            local_uv = local_uv * 0.5  # Reuse variable
        return vec4(local_uv, 0.0, 1.0)

    result = py2glsl(shader)
    assert "vec2 local_uv;" in result.fragment_source  # Should be hoisted
    assert "local_uv = vs_uv * 2.0 - 1.0;" in result.fragment_source
    assert "local_uv = local_uv * 0.5;" in result.fragment_source


def test_loop_variable_scope():
    """Test loop variable scoping"""

    def shader(vs_uv: vec2) -> vec4:
        sum = 0.0
        for i in range(4):
            x = float(i) * 0.25  # New variable in loop
            sum += x
        return vec4(sum, 0.0, 0.0, 1.0)

    result = py2glsl(shader)
    assert "float sum;" in result.fragment_source  # Hoisted
    assert "float x;" in result.fragment_source  # Loop variable should be hoisted
    assert "sum = 0.0;" in result.fragment_source
    assert "x = float(i) * 0.25;" in result.fragment_source


def test_nested_scope_variables():
    """Test variables in nested scopes"""

    def shader(vs_uv: vec2) -> vec4:
        color = vec3(0.0)
        if length(vs_uv) > 0.5:
            factor = 1.0
            if vs_uv.x > 0.0:
                factor = 2.0
            color = vec3(factor)
        return vec4(color, 1.0)

    result = py2glsl(shader)
    assert "vec3 color;" in result.fragment_source
    assert "float factor;" in result.fragment_source
    assert "color = vec3(0.0);" in result.fragment_source
    assert "factor = 1.0;" in result.fragment_source
    assert "factor = 2.0;" in result.fragment_source


def test_conditional_variable_initialization():
    """Test variables that are conditionally initialized"""

    def shader(vs_uv: vec2) -> vec4:
        if vs_uv.x > 0.5:
            color = vec3(1.0, 0.0, 0.0)
        else:
            color = vec3(0.0, 1.0, 0.0)
        return vec4(color, 1.0)

    result = py2glsl(shader)
    assert "vec3 color;" in result.fragment_source  # Should be hoisted
    assert "color = vec3(1.0, 0.0, 0.0);" in result.fragment_source
    assert "color = vec3(0.0, 1.0, 0.0);" in result.fragment_source


def test_complex_scope_nesting():
    """Test complex nested scopes with multiple variables"""

    def shader(vs_uv: vec2) -> vec4:
        d = length(vs_uv)
        if d < 0.5:
            color = vec3(1.0)
            if vs_uv.x > 0.0:
                factor = 0.5
                color = color * factor
            else:
                factor = 0.25
                color = color * factor
        else:
            color = vec3(0.0)
            factor = 0.1
            color = color + factor
        return vec4(color, 1.0)

    result = py2glsl(shader)
    assert "float d;" in result.fragment_source
    assert "vec3 color;" in result.fragment_source
    assert "float factor;" in result.fragment_source


def test_loop_variable_reuse():
    """Test reusing variables across multiple loops"""

    def shader(vs_uv: vec2) -> vec4:
        sum = 0.0
        for i in range(4):
            val = float(i) * 0.25
            sum += val
        for i in range(2):
            val = float(i) * 0.5
            sum += val
        return vec4(sum, 0.0, 0.0, 1.0)

    result = py2glsl(shader)
    assert "float sum;" in result.fragment_source
    assert "float val;" in result.fragment_source
    assert "int i;" in result.fragment_source
