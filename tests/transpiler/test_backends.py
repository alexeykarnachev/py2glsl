"""Tests for the transpiler backends module."""

import ast

import pytest

from py2glsl.transpiler.backends import create_backend
from py2glsl.transpiler.backends.base import GLSLBackend
from py2glsl.transpiler.backends.models import BackendType
from py2glsl.transpiler.models import CollectedInfo, FunctionInfo


@pytest.fixture
def basic_collected_info() -> CollectedInfo:
    """Fixture providing a basic collected info for testing."""
    info = CollectedInfo()

    # Create a simple shader function
    main_func_code = """
def main(vs_uv: vec2, u_time: float) -> vec4:
    return vec4(vs_uv.x, vs_uv.y, 0.0, 1.0)
"""
    main_func_node = ast.parse(main_func_code).body[0]
    assert isinstance(main_func_node, ast.FunctionDef)  # Type check for mypy

    info.functions["main"] = FunctionInfo(
        name="main",
        node=main_func_node,
        return_type="vec4",
        param_types=["vec2", "float"],
    )

    return info


def test_create_backend() -> None:
    """Test creating backends of different types."""
    # Test creating standard backend
    backend = create_backend(BackendType.STANDARD)
    assert isinstance(backend, GLSLBackend)

    # Test creating shadertoy backend
    backend = create_backend(BackendType.SHADERTOY)
    assert isinstance(backend, GLSLBackend)

    # Test with default parameter (should be standard)
    backend = create_backend()
    assert isinstance(backend, GLSLBackend)

    # Test with invalid backend type
    with pytest.raises(ValueError):
        create_backend("invalid_type")  # type: ignore


def test_standard_backend_generation(basic_collected_info: CollectedInfo) -> None:
    """Test code generation with the standard backend."""
    # Create a standard backend
    backend = create_backend(BackendType.STANDARD)

    # Generate code
    glsl_code, uniforms = backend.generate_code(basic_collected_info, "main")

    # Verify basic structure
    assert "#version 460 core" in glsl_code
    assert "uniform float u_time;" in glsl_code
    assert "vec4 main(vec2 vs_uv, float u_time)" in glsl_code
    assert "void main()" in glsl_code
    assert "fragColor = main(vs_uv, u_time);" in glsl_code

    # Verify uniforms
    assert "u_time" in uniforms


def test_shadertoy_backend_generation(basic_collected_info: CollectedInfo) -> None:
    """Test code generation with the shadertoy backend."""
    # Create a shadertoy backend
    backend = create_backend(BackendType.SHADERTOY)

    # Generate code
    glsl_code, uniforms = backend.generate_code(basic_collected_info, "main")

    # Verify basic structure
    assert "#version 300 es" in glsl_code
    assert "precision mediump float;" in glsl_code
    assert "uniform float iTime;" in glsl_code
    assert "vec4 main(vec2 vs_uv, float u_time)" in glsl_code
    assert "mainImage(vec2 fragCoord)" in glsl_code
    assert "vec2 vs_uv = fragCoord / iResolution.xy;" in glsl_code
    assert "out vec4 fragColor;" in glsl_code

    # Verify uniforms
    assert "iTime" in uniforms
    assert "iResolution" in uniforms
