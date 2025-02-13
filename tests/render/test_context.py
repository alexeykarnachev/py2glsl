"""Tests for OpenGL context management."""

import glfw
import moderngl
import pytest

from py2glsl.render.context import (
    GLConfig,
    GLContextError,
    create_standalone_context,
    create_window_context,
    get_framebuffer_size,
    poll_events,
    setup_context,
    should_close,
    swap_buffers,
)


# Fixtures
@pytest.fixture
def default_config():
    """Default OpenGL configuration."""
    return GLConfig()


@pytest.fixture
def custom_config():
    """Custom OpenGL configuration."""
    return GLConfig(
        major_version=4,
        minor_version=6,
        samples=8,
        vsync=False,
    )


# GLConfig Tests
class TestGLConfig:
    """Test suite for GLConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = GLConfig()
        assert config.major_version == 4
        assert config.minor_version == 6
        assert config.samples == 4
        assert config.vsync is True

    def test_custom_values(self, custom_config):
        """Test custom configuration values."""
        assert custom_config.major_version == 4
        assert custom_config.minor_version == 6
        assert custom_config.samples == 8
        assert custom_config.vsync is False

    @pytest.mark.parametrize(
        "major,minor,error_msg",
        [
            (2, 0, "Unsupported OpenGL version: 2.0"),
            (5, 0, "Unsupported OpenGL version: 5.0"),
            (4, 7, "Unsupported OpenGL version: 4.7"),
        ],
    )
    def test_invalid_versions(self, major, minor, error_msg):
        """Test invalid OpenGL versions."""
        with pytest.raises(GLContextError, match=error_msg):
            GLConfig(major_version=major, minor_version=minor)


# Standalone Context Tests
class TestStandaloneContext:
    """Test suite for standalone context."""

    def test_context_creation(self):
        """Test basic context creation."""
        with create_standalone_context() as ctx:
            assert isinstance(ctx, moderngl.Context)
            assert ctx.version_code >= 460

    def test_context_with_custom_config(self, custom_config):
        """Test context creation with custom config."""
        with create_standalone_context(config=custom_config) as ctx:
            assert isinstance(ctx, moderngl.Context)
            assert ctx.version_code >= 460

    def test_context_cleanup(self):
        """Test context cleanup."""
        with create_standalone_context() as ctx:
            buffer = ctx.buffer(reserve=4)
            assert buffer is not None
            buffer.release()

        # Context should be invalid after cleanup
        with pytest.raises(Exception):
            ctx.buffer(reserve=4)


# Window Context Tests
class TestWindowContext:
    """Test suite for window context."""

    def test_window_creation(self):
        """Test window context creation."""
        with create_window_context(title="Test Window") as (ctx, window):
            assert isinstance(ctx, moderngl.Context)
            assert window is not None
            assert ctx.version_code >= 460

    def test_framebuffer_size(self):
        """Test framebuffer size retrieval."""
        size = (640, 480)
        with create_window_context(size=size) as (_, window):
            width, height = get_framebuffer_size(window)
            assert width == size[0]
            assert height == size[1]

    def test_window_events(self):
        """Test window event handling."""
        with create_window_context() as (_, window):
            poll_events()
            assert not should_close(window)
            swap_buffers(window)


# Context Setup Tests
class TestContextSetup:
    """Test suite for context setup."""

    def test_blend_setup(self):
        """Test blending setup."""
        with create_standalone_context() as ctx:
            setup_context(ctx)
            # Since we can't check blend_func directly, verify that we can:
            # 1. Enable blending
            # 2. Set blend function without errors
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

    def test_shader_compilation(self):
        """Test shader compilation in context."""
        with create_standalone_context() as ctx:
            setup_context(ctx)
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
            # ModernGL raises an exception if program compilation fails
            # So if we get here, the program is valid
            assert prog is not None
            # We can verify the program exists and has a valid GL object
            assert prog.glo > 0


# Error Handling Tests
class TestErrorHandling:
    """Test suite for error handling."""

    def test_glfw_initialization_failure(self, monkeypatch):
        """Test GLFW initialization failure."""

        def mock_init():
            return False

        monkeypatch.setattr(glfw, "init", mock_init)

        with pytest.raises(GLContextError, match="Failed to initialize GLFW"):
            with create_standalone_context():
                pass

    def test_window_creation_failure(self, monkeypatch):
        """Test window creation failure."""

        def mock_create_window(*args, **kwargs):
            return None

        monkeypatch.setattr(glfw, "create_window", mock_create_window)

        with pytest.raises(GLContextError, match="Failed to create GLFW window"):
            with create_standalone_context():
                pass
