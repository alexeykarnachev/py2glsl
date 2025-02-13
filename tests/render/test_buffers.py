"""Tests for basic buffer and vertex array creation."""

import moderngl
import numpy as np
import pytest

from py2glsl.render.buffers import BufferError, create_quad_buffer, create_vertex_array
from py2glsl.render.context import create_standalone_context


def test_quad_buffer_creation():
    """Test basic quad buffer creation."""
    with create_standalone_context() as ctx:
        buffer = create_quad_buffer(ctx)

        # Verify buffer was created
        assert buffer is not None

        # Verify buffer data
        data = np.frombuffer(buffer.read(), dtype="f4").reshape(-1, 4)
        assert data.shape == (4, 4)  # 4 vertices, 4 components each

        # Check vertex positions (x, y)
        assert np.allclose(
            data[:, 0:2],
            [
                [-1.0, -1.0],  # Bottom left
                [1.0, -1.0],  # Bottom right
                [-1.0, 1.0],  # Top left
                [1.0, 1.0],  # Top right
            ],
        )

        # Check UV coordinates
        assert np.allclose(
            data[:, 2:4],
            [
                [0.0, 0.0],  # Bottom left
                [1.0, 0.0],  # Bottom right
                [0.0, 1.0],  # Top left
                [1.0, 1.0],  # Top right
            ],
        )


def test_vertex_array_creation():
    """Test vertex array creation with shader program."""
    with create_standalone_context() as ctx:
        # Create a simple shader program
        program = ctx.program(
            vertex_shader="""
                #version 460
                in vec2 in_pos;
                in vec2 in_uv;
                out vec2 v_uv;
                void main() {
                    gl_Position = vec4(in_pos, 0.0, 1.0);
                    v_uv = in_uv;
                }
            """,
            fragment_shader="""
                #version 460
                in vec2 v_uv;
                out vec4 f_color;
                void main() {
                    f_color = vec4(v_uv, 0.0, 1.0);
                }
            """,
        )

        # Create buffer and vertex array
        buffer = create_quad_buffer(ctx)
        vao = create_vertex_array(ctx, program, buffer)

        assert vao is not None

        # Test rendering
        fbo = ctx.framebuffer(color_attachments=[ctx.texture((64, 64), 4)])
        fbo.use()
        vao.render(moderngl.TRIANGLE_STRIP)


def test_invalid_program():
    """Test error handling with invalid shader program."""
    with create_standalone_context() as ctx:
        buffer = create_quad_buffer(ctx)

        # Create program with invalid attribute names
        program = ctx.program(
            vertex_shader="""
                #version 460
                in vec2 wrong_pos;  # Wrong attribute name
                in vec2 wrong_uv;   # Wrong attribute name
                void main() {
                    gl_Position = vec4(wrong_pos, 0.0, 1.0);
                }
            """,
            fragment_shader="""
                #version 460
                out vec4 f_color;
                void main() {
                    f_color = vec4(1.0);
                }
            """,
        )

        with pytest.raises(BufferError):
            create_vertex_array(ctx, program, buffer)
