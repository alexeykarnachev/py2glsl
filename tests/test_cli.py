"""Tests for the py2glsl command-line interface."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from py2glsl.main import app

runner = CliRunner()


@pytest.fixture
def sample_shader_file():
    """Create a temporary shader file for testing."""
    with NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write("""
from py2glsl import ShaderContext
from py2glsl.builtins import length, sin, vec4

def simple_shader(ctx: ShaderContext) -> vec4:
    uv = ctx.vs_uv * 2.0 - 1.0
    d = length(uv)
    color = sin(d * 10.0 - ctx.u_time * 2.0) * 0.5 + 0.5
    return vec4(color, color * 0.5, 1.0 - color, 1.0)
""")
        path = f.name

    yield path
    os.unlink(path)


def test_help():
    """Test that the CLI help command works."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Transform Python functions into GLSL shaders" in result.stdout


def test_show_help():
    """Test that the show command help works."""
    result = runner.invoke(app, ["show", "--help"])
    assert result.exit_code == 0
    assert "interactive" in result.stdout.lower()


def test_image_help():
    """Test that the image command help works."""
    result = runner.invoke(app, ["image", "--help"])
    assert result.exit_code == 0
    assert "static image" in result.stdout.lower()


def test_export_to_stdout(sample_shader_file):
    """Test export to stdout (no output file)."""
    result = runner.invoke(app, ["export", sample_shader_file])
    assert result.exit_code == 0
    assert "#version" in result.stdout
    assert "simple_shader" in result.stdout


def test_export_to_file(sample_shader_file):
    """Test export to file."""
    with TemporaryDirectory() as temp_dir:
        output_file = Path(temp_dir) / "output.glsl"
        result = runner.invoke(app, ["export", sample_shader_file, str(output_file)])
        assert result.exit_code == 0
        assert output_file.exists()

        content = output_file.read_text()
        assert "#version" in content
        assert "simple_shader" in content


@pytest.mark.gpu
def test_render_image(sample_shader_file):
    """Test rendering to image."""
    with TemporaryDirectory() as temp_dir:
        output_file = Path(temp_dir) / "output.png"
        result = runner.invoke(
            app,
            [
                "image",
                sample_shader_file,
                str(output_file),
                "--width",
                "200",
                "--height",
                "200",
            ],
        )
        assert result.exit_code == 0
        assert output_file.exists()


@pytest.mark.gpu
def test_render_gif(sample_shader_file):
    """Test rendering to GIF."""
    with TemporaryDirectory() as temp_dir:
        output_file = Path(temp_dir) / "output.gif"
        result = runner.invoke(
            app,
            [
                "gif",
                sample_shader_file,
                str(output_file),
                "--width",
                "200",
                "--height",
                "200",
                "--fps",
                "10",
                "--duration",
                "0.5",
            ],
        )
        assert result.exit_code == 0
        assert output_file.exists()


@pytest.mark.parametrize("target", ["glsl", "shadertoy"])
def test_target_formats(sample_shader_file, target):
    """Test different target formats."""
    with TemporaryDirectory() as temp_dir:
        output_file = Path(temp_dir) / f"output_{target}.glsl"
        result = runner.invoke(
            app,
            ["export", sample_shader_file, str(output_file), "--target", target],
        )
        assert result.exit_code == 0
        assert output_file.exists()

        content = output_file.read_text()
        if target == "shadertoy":
            assert "mainImage" in content
        else:
            assert "void main()" in content


def test_shadertoy_export(sample_shader_file):
    """Test Shadertoy target exports clean paste-able code."""
    with TemporaryDirectory() as temp_dir:
        # Export to Shadertoy format
        shadertoy_file = Path(temp_dir) / "shader.glsl"
        result = runner.invoke(
            app,
            [
                "export",
                sample_shader_file,
                str(shadertoy_file),
                "--target",
                "shadertoy",
            ],
        )
        assert result.exit_code == 0

        shadertoy_code = shadertoy_file.read_text()

        # Shadertoy output should NOT have version or uniform declarations
        assert "#version" not in shadertoy_code
        assert (
            "uniform " not in shadertoy_code
        )  # trailing space to avoid matching in comments
        # Should not have top-level in/out declarations (but mainImage signature is OK)
        assert "in vec2 vs_uv" not in shadertoy_code
        assert "out vec4 fragColor;" not in shadertoy_code

        # But should have mainImage function
        assert "void mainImage(" in shadertoy_code

        # Export to regular GLSL for comparison
        glsl_file = Path(temp_dir) / "regular.glsl"
        result = runner.invoke(
            app,
            ["export", sample_shader_file, str(glsl_file), "--target", "glsl"],
        )
        assert result.exit_code == 0

        glsl_code = glsl_file.read_text()

        # Regular GLSL should have version and declarations
        assert "#version" in glsl_code
        assert "void main()" in glsl_code


@patch("py2glsl.main._get_transpiled_shader")
def test_error_handling(mock_transpile, sample_shader_file):
    """Test error handling in CLI."""
    mock_transpile.side_effect = ValueError("Test error")
    result = runner.invoke(app, ["export", sample_shader_file, "output.glsl"])
    assert result.exit_code != 0
