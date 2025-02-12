"""OpenGL context management."""

from collections.abc import Iterator
from contextlib import contextmanager

import moderngl

Size = tuple[int, int]


class GLContextError(Exception):
    """OpenGL context error."""


class GLConfig:
    """OpenGL context configuration."""

    def __init__(
        self,
        major_version: int = 4,
        minor_version: int = 6,
        samples: int = 4,
        vsync: bool = True,
    ):
        self.major_version = major_version
        self.minor_version = minor_version
        self.samples = samples
        self.vsync = vsync


@contextmanager
def create_standalone_context() -> Iterator[moderngl.Context]:
    """Create standalone OpenGL context."""
    ctx = moderngl.create_standalone_context()
    try:
        setup_context(ctx)
        yield ctx
    finally:
        ctx.release()


def setup_context(ctx: moderngl.Context) -> None:
    """Setup common context settings."""
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA


def create_window_context(size: Size, title: str = "", config: GLConfig = None):
    """Create window context."""
    raise NotImplementedError("Window context creation not implemented")


def get_framebuffer_size(window) -> Size:
    """Get framebuffer size."""
    raise NotImplementedError("Framebuffer size not implemented")


def poll_events() -> None:
    """Poll for window events."""
    raise NotImplementedError("Event polling not implemented")


def should_close(window) -> bool:
    """Check if window should close."""
    raise NotImplementedError("Window close check not implemented")


def swap_buffers(window) -> None:
    """Swap window buffers."""
    raise NotImplementedError("Buffer swapping not implemented")
