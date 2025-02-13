"""OpenGL context management."""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import glfw
import moderngl


class GLContextError(Exception):
    """OpenGL context error."""


@dataclass
class GLConfig:
    """OpenGL context configuration."""

    major_version: int = 4
    minor_version: int = 6
    samples: int = 4
    vsync: bool = True
    debug: bool = False

    def __post_init__(self) -> None:
        """Validate OpenGL version."""
        if (
            self.major_version < 3
            or (self.major_version == 4 and self.minor_version > 6)
            or self.major_version > 4
        ):
            raise GLContextError(
                f"Unsupported OpenGL version: {self.major_version}.{self.minor_version}"
            )
        if self.samples < 0:
            raise GLContextError(f"Invalid samples value: {self.samples}")


def setup_context(ctx: moderngl.Context) -> None:
    """Setup common context settings."""
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA


@contextmanager
def create_standalone_context(
    *, config: GLConfig | None = None
) -> Iterator[moderngl.Context]:
    """Create standalone OpenGL context.

    Args:
        config: Optional GL configuration

    Yields:
        ModernGL context
    """
    if not glfw.init():
        raise GLContextError("Failed to initialize GLFW")

    window = None
    try:
        cfg = config or GLConfig()

        # Configure window hints
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, cfg.major_version)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, cfg.minor_version)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.SAMPLES, cfg.samples)
        if cfg.debug:
            glfw.window_hint(glfw.OPENGL_DEBUG_CONTEXT, True)
        glfw.window_hint(glfw.VISIBLE, False)

        # Create hidden window for context
        window = glfw.create_window(1, 1, "", None, None)
        if not window:
            raise GLContextError("Failed to create GLFW window")

        glfw.make_context_current(window)

        # Create and setup context
        ctx = moderngl.create_context()
        setup_context(ctx)
        yield ctx

    finally:
        if window:
            glfw.destroy_window(window)
        glfw.terminate()


@contextmanager
def create_window_context(
    size: tuple[int, int] = (800, 600),
    title: str = "",
    config: GLConfig | None = None,
) -> Iterator[tuple[moderngl.Context, Any]]:
    """Create window context with proper cleanup.

    Args:
        size: Window size (width, height)
        title: Window title
        config: Optional GL configuration

    Yields:
        Tuple of (ModernGL context, GLFW window)
    """
    if not glfw.init():
        raise GLContextError("Failed to initialize GLFW")

    window = None
    try:
        cfg = config or GLConfig()

        # Configure window hints
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, cfg.major_version)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, cfg.minor_version)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.SAMPLES, cfg.samples)
        if cfg.debug:
            glfw.window_hint(glfw.OPENGL_DEBUG_CONTEXT, True)

        # Create window
        window = glfw.create_window(size[0], size[1], title, None, None)
        if not window:
            raise GLContextError("Failed to create GLFW window")

        glfw.make_context_current(window)
        if cfg.vsync:
            glfw.swap_interval(1)

        # Create and setup context
        ctx = moderngl.create_context()
        setup_context(ctx)

        yield ctx, window

    finally:
        if window:
            glfw.destroy_window(window)
        glfw.terminate()


def get_framebuffer_size(window: Any) -> tuple[int, int]:
    """Get framebuffer size.

    Args:
        window: GLFW window

    Returns:
        Tuple of (width, height)
    """
    return glfw.get_framebuffer_size(window)


def poll_events() -> None:
    """Poll for window events."""
    glfw.poll_events()


def should_close(window: Any) -> bool:
    """Check if window should close.

    Args:
        window: GLFW window

    Returns:
        True if window should close
    """
    return glfw.window_should_close(window)


def swap_buffers(window: Any) -> None:
    """Swap window buffers.

    Args:
        window: GLFW window
    """
    glfw.swap_buffers(window)


__all__ = [
    "GLContextError",
    "GLConfig",
    "setup_context",
    "create_standalone_context",
    "create_window_context",
    "get_framebuffer_size",
    "poll_events",
    "should_close",
    "swap_buffers",
]
