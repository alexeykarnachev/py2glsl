"""Tests for the py2glsl command-line interface."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from py2glsl.main import app

# Set up the CLI runner
runner = CliRunner()


@pytest.fixture
def sample_shader_file():
    """Create a temporary shader file for testing."""
    with NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write('''
from py2glsl.builtins import length, sin, vec2, vec4

def simple_shader(vs_uv: vec2, u_time: float, u_aspect: float) -> vec4:
    # A simple test shader
    uv = vs_uv * 2.0 - 1.0
    d = length(uv)
    color = sin(d * 10.0 - u_time * 2.0) * 0.5 + 0.5
    return vec4(color, color * 0.5, 1.0 - color, 1.0)
''')
        path = f.name

    yield path

    # Clean up the temporary file
    os.unlink(path)


def test_help():
    """Test that the CLI help command works."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Transform Python functions into GLSL shaders" in result.stdout


def test_show_help():
    """Test that the show subcommand help works."""
    result = runner.invoke(app, ["show", "--help"])
    assert result.exit_code == 0
    assert "Run interactive shader preview" in result.stdout


def test_image_help():
    """Test that the image subcommand help works."""
    result = runner.invoke(app, ["image", "--help"])
    assert result.exit_code == 0
    assert "Render shader to static image" in result.stdout

    result = runner.invoke(app, ["image", "render", "--help"])
    assert result.exit_code == 0
    assert "Python file containing shader functions" in result.stdout


def test_code_export(sample_shader_file):
    """Test code export functionality."""
    with TemporaryDirectory() as temp_dir:
        output_file = Path(temp_dir) / "output.glsl"

        # Test with different format options
        for format_opt in ["plain", "commented", "wrapped"]:
            result = runner.invoke(
                app, [
                    "code", "export",
                    sample_shader_file,
                    str(output_file),
                    "--format", format_opt
                ]
            )

            assert result.exit_code == 0
            assert os.path.exists(output_file)

            # Check file content based on format
            with open(output_file) as f:
                content = f.read()

            if format_opt == "plain":
                assert "#version" in content
                assert "// Generated by py2glsl" not in content
            elif format_opt == "commented":
                assert "// Generated by py2glsl" in content
            elif format_opt == "wrapped" and "shadertoy" in content.lower():
                assert "/*" in content
                assert "*/" in content


@pytest.mark.gpu
def test_render_image(sample_shader_file):
    """Test rendering to image."""
    with TemporaryDirectory() as temp_dir:
        output_file = Path(temp_dir) / "output.png"

        result = runner.invoke(
            app, [
                "image", "render",
                sample_shader_file,
                str(output_file),
                "--width", "200",
                "--height", "200"
            ]
        )

        assert result.exit_code == 0
        assert os.path.exists(output_file)
        # Could add image verification here


@pytest.mark.gpu
def test_render_gif(sample_shader_file):
    """Test rendering to GIF."""
    with TemporaryDirectory() as temp_dir:
        output_file = Path(temp_dir) / "output.gif"

        result = runner.invoke(
            app, [
                "gif", "render",
                sample_shader_file,
                str(output_file),
                "--width", "200",
                "--height", "200",
                "--fps", "10",
                "--duration", "0.5"  # Keep duration short for testing
            ]
        )

        assert result.exit_code == 0
        assert os.path.exists(output_file)
        # Could add GIF verification here


@pytest.mark.parametrize(
    "target_format",
    ["glsl", "shadertoy"]
)
def test_target_formats(sample_shader_file, target_format):
    """Test different target formats."""
    with TemporaryDirectory() as temp_dir:
        output_file = Path(temp_dir) / f"output_{target_format}.glsl"

        result = runner.invoke(
            app, [
                "code", "export",
                sample_shader_file,
                str(output_file),
                "--target", target_format,
                "--format", "commented"
            ]
        )

        assert result.exit_code == 0
        assert os.path.exists(output_file)

        with open(output_file) as f:
            content = f.read()

        # Check target-specific content patterns
        if target_format == "shadertoy":
            # Shadertoy should have mainImage or the simple_shader function
            assert "mainImage" in content or "simple_shader" in content
        else:
            # Standard GLSL should have main or the simple_shader function
            assert "main" in content or "simple_shader" in content


def test_shadertoy_compatibility(sample_shader_file):
    """Test the shadertoy compatibility flag."""
    with TemporaryDirectory() as temp_dir:
        # Test with flag on
        compat_file = Path(temp_dir) / "shadertoy_compatible.glsl"
        result = runner.invoke(
            app, [
                "code", "export",
                sample_shader_file,
                str(compat_file),
                "--target", "shadertoy",
                "--shadertoy-compatible"
            ]
        )

        assert result.exit_code == 0

        # Test with flag off
        regular_file = Path(temp_dir) / "shadertoy_regular.glsl"
        result = runner.invoke(
            app, [
                "code", "export",
                sample_shader_file,
                str(regular_file),
                "--target", "shadertoy"
            ]
        )

        assert result.exit_code == 0

        # Compare files
        with open(compat_file) as f:
            compat_content = f.read()

        with open(regular_file) as f:
            regular_content = f.read()

        # Compatible version should not have these uniforms
        # (at least one should be missing)
        shadertoy_uniforms_missing = (
            "uniform vec3 iResolution;" not in compat_content or
            "uniform float iTime;" not in compat_content
        )
        assert shadertoy_uniforms_missing

        # Regular version should have at least some uniforms
        assert (
            "uniform vec3 iResolution;" in regular_content or
            "uniform float iTime;" in regular_content or
            "uniform" in regular_content
        )

        # Both should still have the mainImage function
        assert "mainImage" in compat_content
        assert "mainImage" in regular_content


@patch("py2glsl.main._get_transpiled_shader")
def test_error_handling(mock_transpile, sample_shader_file):
    """Test error handling in CLI."""
    # Simulate transpilation error
    mock_transpile.side_effect = ValueError("Test error")

    result = runner.invoke(
        app, [
            "code", "export",
            sample_shader_file,
            "output.glsl"
        ]
    )

    # We only check that the exit code is non-zero, indicating an error
    assert result.exit_code != 0
