"""OpenGL context management."""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Tuple

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

    def __post_init__(self) -> None:
        """Validate OpenGL version."""
        if (
            self.major_version < 3
            or self.major_version > 4
            or (self.major_version == 4 and self.minor_version > 6)
        ):
            raise GLContextError(
                f"Unsupported OpenGL version: {self.major_version}.{self.minor_version}"
            )


def setup_context(ctx: moderngl.Context) -> None:
    """Setup common context settings."""
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA


@contextmanager
def create_standalone_context(
    *, config: GLConfig | None = None
) -> Iterator[moderngl.Context]:
    """Create standalone OpenGL context for offscreen rendering."""
    if not glfw.init():
        raise GLContextError("Failed to initialize GLFW")

    window = None
    try:
        cfg = config or GLConfig()

        # Configure context
        glfw.window_hint(glfw.VISIBLE, False)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, cfg.major_version)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, cfg.minor_version)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.SAMPLES, cfg.samples)

        # Create hidden window
        window = glfw.create_window(1, 1, "", None, None)
        if not window:
            raise GLContextError("Failed to create GLFW window")

        glfw.make_context_current(window)

        # Setup ModernGL context
        ctx = moderngl.create_context()
        setup_context(ctx)
        yield ctx

    finally:
        if window:
            glfw.destroy_window(window)
        glfw.terminate()


@contextmanager
def create_window_context(
    size: Tuple[int, int] = (800, 600),
    title: str = "",
    config: GLConfig | None = None,
) -> Iterator[Tuple[moderngl.Context, Any]]:
    """Create window context for real-time display."""
    if not glfw.init():
        raise GLContextError("Failed to initialize GLFW")

    window = None
    try:
        cfg = config or GLConfig()

        # Configure context
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, cfg.major_version)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, cfg.minor_version)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.SAMPLES, cfg.samples)

        # Create visible window
        window = glfw.create_window(size[0], size[1], title, None, None)
        if not window:
            raise GLContextError("Failed to create GLFW window")

        glfw.make_context_current(window)
        if cfg.vsync:
            glfw.swap_interval(1)

        # Setup ModernGL context
        ctx = moderngl.create_context()
        setup_context(ctx)
        yield ctx, window

    finally:
        if window:
            glfw.destroy_window(window)
        glfw.terminate()


def get_framebuffer_size(window: Any) -> Tuple[int, int]:
    """Get window framebuffer size."""
    return glfw.get_framebuffer_size(window)


def poll_events() -> None:
    """Process pending window events."""
    glfw.poll_events()


def should_close(window: Any) -> bool:
    """Check if window should close."""
    return glfw.window_should_close(window)


def swap_buffers(window: Any) -> None:
    """Swap window buffers."""
    glfw.swap_buffers(window)


__all__ = [
    "GLContextError",
    "GLConfig",
    "create_standalone_context",
    "create_window_context",
    "get_framebuffer_size",
    "poll_events",
    "should_close",
    "swap_buffers",
]
