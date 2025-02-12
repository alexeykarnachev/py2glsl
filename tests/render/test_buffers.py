"""Tests for vertex buffer and attribute management."""

import moderngl
import numpy as np
import pytest

from py2glsl.render.buffers import (
    BufferError,
    BufferLayout,
    create_quad_buffer,
    create_vertex_array,
)
from py2glsl.render.context import create_standalone_context


def test_create_quad_buffer():
    """Test quad buffer creation."""
    with create_standalone_context() as ctx:
        # Test with pixel-centered coordinates
        buffer = create_quad_buffer(ctx, size=(512, 512), pixel_centered=True)
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

        # Check UV coordinates are pixel-centered
        w, h = 512, 512
        expected_uvs = np.array(
            [
                [0.5 / w, 0.5 / h],  # Bottom left
                [(w - 0.5) / w, 0.5 / h],  # Bottom right
                [0.5 / w, (h - 0.5) / h],  # Top left
                [(w - 0.5) / w, (h - 0.5) / h],  # Top right
            ]
        )
        assert np.allclose(data[:, 2:4], expected_uvs)

        buffer.release()


def test_create_quad_buffer_non_centered():
    """Test quad buffer creation with non-centered coordinates."""
    with create_standalone_context() as ctx:
        buffer = create_quad_buffer(ctx, size=(512, 512), pixel_centered=False)
        assert buffer is not None

        # Verify buffer data
        data = np.frombuffer(buffer.read(), dtype="f4").reshape(-1, 4)

        # Check UV coordinates are not centered
        expected_uvs = np.array(
            [
                [0.0, 0.0],  # Bottom left
                [1.0, 0.0],  # Bottom right
                [0.0, 1.0],  # Top left
                [1.0, 1.0],  # Top right
            ]
        )
        assert np.allclose(data[:, 2:4], expected_uvs)

        buffer.release()


def test_create_vertex_array():
    """Test vertex array creation."""
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

        # Create buffer and layout
        buffer = create_quad_buffer(ctx, (512, 512))
        layout = [
            BufferLayout(name="in_pos", components=2, data_type="f"),  # Position
            BufferLayout(name="in_uv", components=2, data_type="f"),  # UV
        ]

        # Create vertex array
        vao = create_vertex_array(ctx, program, buffer, layout)
        assert vao is not None

        vao.release()
        buffer.release()
        program.release()


def test_invalid_buffer_layout():
    """Test vertex array creation with invalid layout."""
    with create_standalone_context() as ctx:
        program = ctx.program(
            vertex_shader="""
                #version 460
                in vec2 in_pos;
                void main() {
                    gl_Position = vec4(in_pos, 0.0, 1.0);
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

        buffer = create_quad_buffer(ctx, (512, 512))

        # Invalid data type
        invalid_layout = [BufferLayout(name="in_pos", components=2, data_type="x")]

        with pytest.raises(BufferError):
            create_vertex_array(ctx, program, buffer, invalid_layout)

        buffer.release()
        program.release()


def test_basic_layout():
    """Test basic (non-interleaved) vertex array layout."""
    with create_standalone_context() as ctx:
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

        buffer = create_quad_buffer(ctx, (512, 512))
        layout = [
            BufferLayout(name="in_pos", components=2, data_type="f"),
            BufferLayout(name="in_uv", components=2, data_type="f"),
        ]

        vao = create_vertex_array(ctx, program, buffer, layout)
        assert vao is not None

        # Test rendering
        fbo = ctx.framebuffer(color_attachments=[ctx.texture((64, 64), 4)])
        fbo.use()
        vao.render(moderngl.TRIANGLE_STRIP)

        vao.release()
        fbo.release()
        buffer.release()
        program.release()


def test_interleaved_layout():
    """Test interleaved vertex array layout."""
    with create_standalone_context() as ctx:
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

        buffer = create_quad_buffer(ctx, (512, 512))
        layout = [
            BufferLayout(
                name="in_pos",
                components=2,
                data_type="f",
                stride=16,  # 4 floats * 4 bytes
                offset=0,
            ),
            BufferLayout(
                name="in_uv",
                components=2,
                data_type="f",
                stride=16,  # 4 floats * 4 bytes
                offset=8,  # After 2 floats
            ),
        ]

        vao = create_vertex_array(ctx, program, buffer, layout)
        assert vao is not None

        # Test rendering
        fbo = ctx.framebuffer(color_attachments=[ctx.texture((64, 64), 4)])
        fbo.use()
        vao.render(moderngl.TRIANGLE_STRIP)

        vao.release()
        fbo.release()
        buffer.release()
        program.release()


def test_normalized_layout():
    """Test normalized vertex array layout."""
    with create_standalone_context() as ctx:
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

        buffer = create_quad_buffer(ctx, (512, 512))
        layout = [
            BufferLayout(
                name="in_pos",
                components=2,
                data_type="f",
                normalized=True,
            ),
            BufferLayout(
                name="in_uv",
                components=2,
                data_type="f",
                normalized=True,
            ),
        ]

        vao = create_vertex_array(ctx, program, buffer, layout)
        assert vao is not None

        # Test rendering
        fbo = ctx.framebuffer(color_attachments=[ctx.texture((64, 64), 4)])
        fbo.use()
        vao.render(moderngl.TRIANGLE_STRIP)

        vao.release()
        fbo.release()
        buffer.release()
        program.release()
