from dataclasses import dataclass
from typing import List, Optional, Tuple

import moderngl
import numpy as np
from loguru import logger

from .context import Size


@dataclass
class BufferLayout:
    """OpenGL buffer layout description."""

    name: str  # Attribute name in shader
    components: int
    data_type: str  # 'f', 'i', etc.
    normalized: bool = False
    stride: int = 0
    offset: int = 0


class BufferError(Exception):
    """Buffer creation or management error."""


def create_quad_buffer(ctx: moderngl.Context, size: Size) -> moderngl.Buffer:
    """Create a fullscreen quad buffer."""
    w, h = size
    # Map UV coordinates to pixel centers
    u_min = 0.5 / w
    u_max = (w - 0.5) / w
    v_min = 0.5 / h
    v_max = (h - 0.5) / h

    vertices = np.array(
        [
            # x,    y,     u,    v
            -1.0,
            -1.0,
            u_min,
            v_min,  # Bottom left
            1.0,
            -1.0,
            u_max,
            v_min,  # Bottom right
            -1.0,
            1.0,
            u_min,
            v_max,  # Top left
            1.0,
            1.0,
            u_max,
            v_max,  # Top right
        ],
        dtype="f4",
    )

    return ctx.buffer(vertices.tobytes())


def create_vertex_array(
    ctx: moderngl.Context,
    program: moderngl.Program,
    buffer: moderngl.Buffer,
    layout: List[BufferLayout] | None = None,
) -> moderngl.VertexArray:
    """Create a vertex array with specified layout."""
    if layout is None:
        # Default layout for fullscreen quad
        return ctx.vertex_array(
            program,
            [(buffer, "2f 2f", "in_pos", "in_uv")],
            skip_errors=True,
        )

    try:
        # Create bindings based on layout
        bindings = []
        for l in layout:
            if l.data_type not in ("f", "i", "u"):
                raise BufferError(f"Invalid data type: {l.data_type}")

            format_str = f"{l.components}{l.data_type}"
            if l.normalized:
                format_str = f"{l.components}f1"
            if l.stride:
                format_str = f"{format_str}/{l.stride}"
            if l.offset:
                format_str = f"{format_str}@{l.offset}"

            bindings.append((buffer, format_str, l.name))

        return ctx.vertex_array(program, bindings)
    except Exception as e:
        raise BufferError(f"Failed to create vertex array: {e}") from e
