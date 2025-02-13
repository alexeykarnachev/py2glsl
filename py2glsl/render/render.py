# render/render.py
"""Core rendering functions."""

import imageio
import numpy as np
from PIL import Image

from ..transpiler import py2glsl
from ..transpiler.constants import VERTEX_SHADER
from .buffers import create_quad_buffer, create_vertex_array
from .context import create_standalone_context


def render_array(shader_func, size=(512, 512), **uniforms) -> np.ndarray:
    """Render shader to numpy array."""
    with create_standalone_context() as ctx:
        # Create quad buffer
        quad = create_quad_buffer(ctx)

        # Convert shader to GLSL
        shader_result = py2glsl(shader_func)

        # Create shader program
        program = ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=shader_result.fragment_source,
        )

        # Set built-in uniforms
        if "u_aspect" in shader_result.uniforms:
            u_aspect = size[0] / size[1]
            program["u_aspect"].value = u_aspect

        # Set custom uniforms
        for name, value in uniforms.items():
            if name in shader_result.uniforms:
                try:
                    uniform_type = shader_result.uniforms[name]
                    if uniform_type == "int":
                        program[name].value = int(value)
                    elif isinstance(value, (tuple, list, np.ndarray)):
                        program[name].value = tuple(map(float, value))
                    else:
                        program[name].value = float(value)
                except KeyError:
                    continue

        # Create vertex array
        vao = create_vertex_array(ctx, program, quad)

        # Create framebuffer
        fbo = ctx.framebuffer(color_attachments=[ctx.texture(size, 4)])
        fbo.use()

        # Clear and render
        ctx.clear()
        vao.render(mode=ctx.TRIANGLE_STRIP)

        # Read pixels and normalize
        data = np.frombuffer(fbo.read(components=4, alignment=1), dtype=np.uint8)
        data = data.reshape(*size, 4)
        return data.astype(np.float32) / 255.0


def render_image(shader_func, size=(512, 512), **uniforms) -> Image.Image:
    """Render shader to PIL Image."""
    data = render_array(shader_func, size, **uniforms)
    return Image.fromarray((data * 255).astype(np.uint8))


def render_gif(
    shader_func, filename, duration, fps=30.0, size=(512, 512), **uniforms
) -> None:
    """Render shader animation to GIF."""
    frames = []
    frame_count = int(duration * fps)
    frame_duration = int(1000 / fps)

    with create_standalone_context() as ctx:
        quad = create_quad_buffer(ctx, size)
        shader_result = py2glsl(shader_func)
        program = ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=shader_result.fragment_source,
        )
        vao = create_vertex_array(ctx, program, quad)
        fbo = ctx.framebuffer(color_attachments=[ctx.texture(size, 4)])

        for frame in range(frame_count):
            time = frame / fps
            frame_uniforms = {**uniforms, "u_time": time, "u_frame": frame}

            # Set uniforms
            for name, value in frame_uniforms.items():
                if name in shader_result.uniforms:
                    try:
                        uniform_type = shader_result.uniforms[name]
                        if uniform_type == "int":
                            program[name].value = int(value)
                        elif isinstance(value, (tuple, list, np.ndarray)):
                            program[name].value = tuple(map(float, value))
                        else:
                            program[name].value = float(value)
                    except KeyError:
                        continue

            # Render frame
            fbo.use()
            ctx.clear()
            vao.render(mode=ctx.TRIANGLE_STRIP)

            # Capture frame
            data = np.frombuffer(fbo.read(components=4, alignment=1), dtype=np.uint8)
            frame_data = data.reshape(*size, 4)
            img = Image.fromarray(frame_data, mode="RGBA")
            frames.append(img.convert("RGB"))

    # Save GIF
    if frames:
        frames[0].save(
            filename,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration,
            loop=0,
            optimize=False,
        )


def render_video(
    shader_func, filename, duration, fps=30.0, size=(512, 512), codec="h264", **uniforms
) -> None:
    """Render shader animation to video file."""
    writer = imageio.get_writer(
        filename, fps=fps, codec=codec, quality=8, pixelformat="yuv420p"
    )

    try:
        frame_count = int(duration * fps)
        for frame in range(frame_count):
            time = frame / fps
            frame_uniforms = {**uniforms, "u_time": time, "u_frame": frame}
            frame_data = render_array(shader_func, size=size, **frame_uniforms)
            writer.append_data((frame_data * 255).astype(np.uint8))
    finally:
        writer.close()
