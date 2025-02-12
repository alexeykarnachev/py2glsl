"""Tests for OpenGL context management."""

import moderngl
import pytest

from py2glsl.render.context import (
    GLConfig,
    GLContextError,
    create_standalone_context,
    create_window_context,
    get_framebuffer_size,
    setup_context,
)


def test_standalone_context_creation():
    """Test standalone context creation."""
    with create_standalone_context() as ctx:
        assert ctx is not None
        assert ctx.version_code >= 460  # OpenGL 4.6


def test_window_context_creation():
    """Test window context creation."""
    with create_window_context(title="Test Window") as (ctx, window):
        assert ctx is not None
        assert window is not None
        assert ctx.version_code >= 460


def test_invalid_gl_version():
    """Test context creation with invalid OpenGL version."""
    config = GLConfig(major_version=9, minor_version=9)  # Invalid version
    with pytest.raises(GLContextError):
        with create_standalone_context(config=config):
            pass


def test_context_cleanup():
    """Test proper context cleanup."""
    ctx_id = None
    with create_standalone_context() as ctx:
        ctx_id = id(ctx)
        # Check if context is valid by attempting to create a buffer
        buffer = ctx.buffer(reserve=4)
        assert buffer is not None
        buffer.release()

    # Context should be invalid after cleanup
    with pytest.raises(Exception):
        ctx.buffer(reserve=4)


def test_framebuffer_size():
    """Test framebuffer size retrieval."""
    size = (640, 480)
    with create_window_context(size=size) as (_, window):
        width, height = get_framebuffer_size(window)
        assert width == size[0]
        assert height == size[1]


def test_context_setup():
    """Test context setup configuration."""
    with create_standalone_context() as ctx:
        setup_context(ctx)
        # Create a simple program to test blending
        prog = ctx.program(
            vertex_shader="""
                #version 460
                in vec2 in_pos;
                void main() {
                    gl_Position = vec4(in_pos, 0.0, 1.0);
                }
            """,
            fragment_shader="""
                #version 460
                out vec4 fragColor;
                void main() {
                    fragColor = vec4(1.0);
                }
            """,
        )
        assert prog is not None


def test_custom_config():
    """Test custom OpenGL configuration."""
    config = GLConfig(
        major_version=4,
        minor_version=6,
        samples=8,
        vsync=False,
        debug=True,
    )
    with create_standalone_context(config=config) as ctx:
        assert ctx is not None
        assert ctx.version_code >= 460


@pytest.mark.xfail(reason="Multiple contexts not supported in all environments")
def test_multiple_contexts():
    """Test creating multiple contexts."""
    with create_standalone_context() as ctx1:
        with create_standalone_context() as ctx2:
            assert ctx1 is not None
            assert ctx2 is not None
            assert id(ctx1) != id(ctx2)
