"""Rendering functionality for shaders."""

from .animation import animate
from .buffers import BufferError, BufferLayout, create_quad_buffer, create_vertex_array
from .context import (
    GLConfig,
    GLContextError,
    Size,
    create_standalone_context,
    create_window_context,
    get_framebuffer_size,
    poll_events,
    setup_context,
    should_close,
    swap_buffers,
)
from .render import render_array, render_gif, render_image, render_video

__all__ = [
    # Main rendering functions
    "render_array",
    "render_image",
    "render_gif",
    "render_video",
    "animate",
    # Buffer management
    "BufferError",
    "BufferLayout",
    "create_quad_buffer",
    "create_vertex_array",
    # Context management
    "GLConfig",
    "GLContextError",
    "Size",
    "create_standalone_context",
    "create_window_context",
    "get_framebuffer_size",
    "setup_context",
    "poll_events",
    "should_close",
    "swap_buffers",
]
