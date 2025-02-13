"""Core rendering functions for shader output."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import imageio
import moderngl
import numpy as np
from PIL import Image

from ..transpiler import py2glsl
from .context import create_standalone_context
from .program import ShaderProgram


@dataclass
class RenderConfig:
    """Render configuration settings."""

    size: tuple[int, int] = (512, 512)
    fps: float = 30.0
    duration: Optional[float] = None
    codec: str = "h264"
    optimize_gif: bool = False


class ShaderRenderer:
    """Handles shader rendering to various formats."""

    def __init__(self, shader_func: Callable, config: Optional[RenderConfig] = None):
        """Initialize renderer.

        Args:
            shader_func: Python shader function
            config: Optional render configuration
        """
        self.shader_func = shader_func
        self.config = config or RenderConfig()
        self.shader_result = py2glsl(shader_func)

    def setup_render_context(
        self, ctx: moderngl.Context
    ) -> tuple[ShaderProgram, moderngl.VertexArray, moderngl.Framebuffer]:
        """Set up rendering context with program and framebuffer.

        Args:
            ctx: ModernGL context

        Returns:
            Tuple of (program, vertex array object, framebuffer)
        """
        # Create shader program
        program = ShaderProgram(
            ctx,
            self.shader_result.vertex_source,
            self.shader_result.fragment_source,
        )

        # Create quad buffer for fullscreen rendering
        quad_vertices = np.array(
            [
                -1.0,
                -1.0,
                0.0,
                1.0,  # Bottom left (flipped v from 0.0 to 1.0)
                1.0,
                -1.0,
                1.0,
                1.0,  # Bottom right (flipped v from 0.0 to 1.0)
                -1.0,
                1.0,
                0.0,
                0.0,  # Top left (flipped v from 1.0 to 0.0)
                1.0,
                1.0,
                1.0,
                0.0,  # Top right (flipped v from 1.0 to 0.0)
            ],
            dtype="f4",
        )

        quad_buffer = ctx.buffer(quad_vertices.tobytes())

        # Create vertex array with attributes
        vao = ctx.vertex_array(
            program.program,  # Use the underlying ModernGL program
            [(quad_buffer, "2f 2f", "in_pos", "in_uv")],
        )

        # Create framebuffer for rendering
        fbo = ctx.framebuffer(color_attachments=[ctx.texture(self.config.size, 4)])

        return program, vao, fbo

    def render_frame(
        self,
        ctx,
        program: ShaderProgram,
        vao,
        fbo,
        time: float = 0.0,
        frame: int = 0,
        **uniforms,
    ) -> np.ndarray:
        """Render single frame.

        Args:
            ctx: ModernGL context
            program: Shader program wrapper
            vao: Vertex array object
            fbo: Framebuffer object
            time: Current time in seconds
            frame: Current frame number
            **uniforms: Additional uniforms

        Returns:
            Numpy array with frame data
        """
        fbo.use()

        # Update uniforms
        frame_uniforms = {**uniforms, "u_time": time, "u_frame": frame}
        program.set_uniforms(frame_uniforms)

        # Render
        ctx.clear()
        vao.render(mode=ctx.TRIANGLE_STRIP)

        # Read pixels
        data = np.frombuffer(fbo.read(components=4, alignment=1), dtype=np.uint8)
        return data.reshape(*self.config.size, 4)

    def to_array(self, **uniforms) -> np.ndarray:
        """Render shader to numpy array.

        Args:
            **uniforms: Shader uniforms

        Returns:
            Normalized RGBA numpy array
        """
        with create_standalone_context() as ctx:
            program, vao, fbo = self.setup_render_context(ctx)
            data = self.render_frame(ctx, program, vao, fbo, **uniforms)
            return data.astype(np.float32) / 255.0

    def to_image(self, **uniforms) -> Image.Image:
        """Render shader to PIL Image.

        Args:
            **uniforms: Shader uniforms

        Returns:
            PIL Image
        """
        data = self.to_array(**uniforms)
        return Image.fromarray((data * 255).astype(np.uint8))

    def to_gif(self, filename: str | Path, **uniforms) -> None:
        """Render shader animation to GIF.

        Args:
            filename: Output filename
            **uniforms: Shader uniforms
        """
        if not self.config.duration:
            raise ValueError("Duration must be set for animation rendering")

        frames = []
        frame_count = int(self.config.duration * self.config.fps)
        frame_duration = int(1000 / self.config.fps)

        with create_standalone_context() as ctx:
            program, vao, fbo = self.setup_render_context(ctx)

            for frame in range(frame_count):
                time = frame / self.config.fps
                data = self.render_frame(
                    ctx, program, vao, fbo, time, frame, **uniforms
                )
                img = Image.fromarray(data, mode="RGBA")
                frames.append(img.convert("RGB"))

        if frames:
            frames[0].save(
                filename,
                save_all=True,
                append_images=frames[1:],
                duration=frame_duration,
                loop=0,
                optimize=self.config.optimize_gif,
            )

    def to_video(self, filename: str | Path, **uniforms) -> None:
        """Render shader animation to video.

        Args:
            filename: Output filename
            **uniforms: Shader uniforms
        """
        if not self.config.duration:
            raise ValueError("Duration must be set for animation rendering")

        writer = imageio.get_writer(
            filename,
            fps=self.config.fps,
            codec=self.config.codec,
            quality=8,
            pixelformat="yuv420p",
        )

        try:
            frame_count = int(self.config.duration * self.config.fps)
            for frame in range(frame_count):
                time = frame / self.config.fps
                frame_uniforms = {**uniforms, "u_time": time, "u_frame": frame}
                frame_data = self.to_array(**frame_uniforms)
                writer.append_data((frame_data * 255).astype(np.uint8))
        finally:
            writer.close()


# Convenience functions that use ShaderRenderer internally
def render_array(shader_func, size=(512, 512), **uniforms) -> np.ndarray:
    """Render shader to numpy array."""
    config = RenderConfig(size=size)
    renderer = ShaderRenderer(shader_func, config)
    return renderer.to_array(**uniforms)


def render_image(shader_func, size=(512, 512), **uniforms) -> Image.Image:
    """Render shader to PIL Image."""
    config = RenderConfig(size=size)
    renderer = ShaderRenderer(shader_func, config)
    return renderer.to_image(**uniforms)


def render_gif(
    shader_func, filename, duration, fps=30.0, size=(512, 512), **uniforms
) -> None:
    """Render shader animation to GIF."""
    config = RenderConfig(size=size, fps=fps, duration=duration)
    renderer = ShaderRenderer(shader_func, config)
    renderer.to_gif(filename, **uniforms)


def render_video(
    shader_func, filename, duration, fps=30.0, size=(512, 512), codec="h264", **uniforms
) -> None:
    """Render shader animation to video."""
    config = RenderConfig(size=size, fps=fps, duration=duration, codec=codec)
    renderer = ShaderRenderer(shader_func, config)
    renderer.to_video(filename, **uniforms)
