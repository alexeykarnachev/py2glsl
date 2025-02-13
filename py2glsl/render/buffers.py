import moderngl
import numpy as np


class BufferError(Exception):
    """Buffer creation or management error."""


def create_quad_buffer(ctx: moderngl.Context) -> moderngl.Buffer:
    """Create a fullscreen quad buffer with UV coordinates."""
    vertices = np.array(
        [
            # x,    y,     u,    v
            -1.0,
            -1.0,
            0.0,
            0.0,  # Bottom left
            1.0,
            -1.0,
            1.0,
            0.0,  # Bottom right
            -1.0,
            1.0,
            0.0,
            1.0,  # Top left
            1.0,
            1.0,
            1.0,
            1.0,  # Top right
        ],
        dtype="f4",
    )
    return ctx.buffer(vertices.tobytes())


def create_vertex_array(
    ctx: moderngl.Context,
    program: moderngl.Program,
    buffer: moderngl.Buffer,
) -> moderngl.VertexArray:
    """Create a vertex array for shader rendering."""
    try:
        return ctx.vertex_array(
            program,
            [(buffer, "2f 2f", "in_pos", "in_uv")],
            skip_errors=True,
        )
    except Exception as e:
        raise BufferError(f"Failed to create vertex array: {e}") from e
