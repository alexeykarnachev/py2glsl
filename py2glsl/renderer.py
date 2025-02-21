from time import perf_counter, sleep
from typing import Callable, Tuple

import glfw
import imageio
import moderngl
import numpy as np
from PIL import Image

from py2glsl.transpiler.core import transpile


class RenderContext:
    """Base class for ModernGL rendering contexts"""

    _glfw_initialized = False

    def __init__(self, size: Tuple[int, int], visible: bool = False):
        if not RenderContext._glfw_initialized:
            if not glfw.init():
                raise RuntimeError("GLFW initialization failed")
            RenderContext._glfw_initialized = True

        self.size = size
        self.ctx: moderngl.Context
        self.window: glfw._GLFWwindow
        self.program: moderngl.Program
        self.vao: moderngl.VertexArray
        self.fbo: moderngl.Framebuffer
        self.vbo: moderngl.Buffer

        # Configure OpenGL context
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.VISIBLE, visible)

        # Create window and context
        self.window = glfw.create_window(*size, "Py2GLSL", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Window creation failed")

        glfw.make_context_current(self.window)
        self.ctx = moderngl.create_context()

        # Fullscreen quad geometry
        self._init_geometry()
        self.start_time = perf_counter()
        self.frame_count = 0

    def _init_geometry(self):
        """Initialize vertex buffer for fullscreen quad"""
        vertices = np.array(
            [
                -1.0,
                -1.0,  # Bottom-left
                1.0,
                -1.0,  # Bottom-right
                -1.0,
                1.0,  # Top-left
                1.0,
                1.0,  # Top-right
            ],
            dtype="f4",
        )

        self.vbo = self.ctx.buffer(vertices.tobytes())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def release(self):
        """Clean up all GPU resources"""
        if hasattr(self, "vao") and self.vao:
            self.vao.release()
        if hasattr(self, "vbo") and self.vbo:
            self.vbo.release()
        if hasattr(self, "program") and self.program:
            self.program.release()
        if self.window:
            glfw.destroy_window(self.window)

    def load_shader(self, shader_func: Callable):
        """Compile and load shader program"""
        transpiled = transpile(shader_func)

        try:
            self.program = self.ctx.program(
                vertex_shader=transpiled.vertex_src,
                fragment_shader=transpiled.fragment_src,
            )
            self.vao = self.ctx.vertex_array(self.program, [(self.vbo, "2f", "a_pos")])
        except moderngl.Error as e:
            raise RuntimeError(f"Shader compilation failed: {str(e)}") from e

        if self.vao is None:
            raise RuntimeError("Failed to create vertex array object")

    def _set_core_uniforms(self, time: float = 0.0, frame: int = 0):
        """Set standard uniforms if present in the shader"""
        if self.program is None:
            return

        # Explicit uniform assignments
        if "u_resolution" in self.program:
            self.program["u_resolution"].value = self.size
        if "u_aspect" in self.program:
            self.program["u_aspect"].value = self.size[0] / self.size[1]
        if "u_time" in self.program:
            self.program["u_time"].value = time
        if "u_frame" in self.program:
            self.program["u_frame"].value = frame


class ImageRenderer(RenderContext):
    """Offscreen renderer for images and videos"""

    def __init__(self, size: Tuple[int, int]):
        super().__init__(size, visible=False)
        self.fbo = self.ctx.framebuffer(color_attachments=[self.ctx.texture(size, 4)])

    def render_frame(self, time: float = 0.0, frame: int = 0) -> Image.Image:
        """Render single frame to PIL Image"""
        if self.vao is None or self.program is None:
            raise RuntimeError("Shader not loaded. Call load_shader() first")

        self.fbo.use()
        self.ctx.clear()
        self._set_core_uniforms(time, frame)
        self.vao.render(mode=moderngl.TRIANGLE_STRIP)
        return Image.frombytes("RGBA", self.size, self.fbo.read(components=4))


class AnimationWindow(RenderContext):
    """Interactive real-time rendering window"""

    def __init__(self, size: Tuple[int, int]):
        super().__init__(size, visible=True)
        self._setup_callbacks()

    def _setup_callbacks(self):
        glfw.set_window_size_callback(self.window, self._on_resize)
        glfw.set_key_callback(self.window, self._on_key)

    def _on_resize(self, window, width, height):
        self.size = (width, height)
        self.ctx.viewport = (0, 0, width, height)
        self._set_core_uniforms()

    def _on_key(self, window, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)

    def run(self, fps: int = 60):
        """Run main animation loop"""
        if self.vao is None or self.program is None:
            raise RuntimeError("Shader not loaded. Call load_shader() first")

        last_frame_time = perf_counter()
        while not glfw.window_should_close(self.window):
            glfw.poll_events()

            # Update timing
            current_time = perf_counter() - self.start_time
            self._set_core_uniforms(current_time, self.frame_count)

            # Render frame
            self.ctx.clear()
            self.vao.render(mode=moderngl.TRIANGLE_STRIP)
            glfw.swap_buffers(self.window)

            # Precise FPS control
            elapsed = perf_counter() - last_frame_time
            sleep_time = max(0.0, (1.0 / fps) - elapsed)
            sleep(sleep_time)

            self.frame_count += 1
            last_frame_time = perf_counter()


def render_image(
    shader_func: Callable,
    size: Tuple[int, int] = (512, 512),
    time: float = 0.0,
    frame: int = 0,
) -> Image.Image:
    """Render static image from shader"""
    with ImageRenderer(size) as renderer:
        renderer.load_shader(shader_func)
        return renderer.render_frame(time, frame)


def render_gif(
    shader_func: Callable,
    output_path: str = "output.gif",
    duration: float = 2.0,
    fps: int = 30,
    size: Tuple[int, int] = (256, 256),
):
    """Render animated GIF from shader"""
    with ImageRenderer(size) as renderer:
        renderer.load_shader(shader_func)
        frames = []

        total_frames = int(duration * fps)
        for frame_idx in range(total_frames):
            t = frame_idx / fps
            frames.append(renderer.render_frame(t, frame_idx))

        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=1000 / fps,
            loop=0,
        )


def render_video(
    shader_func: Callable,
    output_path: str = "output.mp4",
    duration: float = 2.0,
    fps: int = 30,
    size: Tuple[int, int] = (1280, 720),
    codec: str = "libx264",
    quality: str = "high",
):
    """Render video from shader using imageio"""
    with ImageRenderer(size) as renderer:
        renderer.load_shader(shader_func)
        frames = []

        total_frames = int(duration * fps)
        for frame_idx in range(total_frames):
            t = frame_idx / fps
            frames.append(renderer.render_frame(t, frame_idx))

        writer = imageio.get_writer(
            output_path,
            fps=fps,
            codec=codec,
            quality=9 if quality == "high" else 5,
        )
        for frame_img in frames:
            writer.append_data(np.array(frame_img))
        writer.close()


def animate(
    shader_func: Callable,
    size: Tuple[int, int] = (800, 600),
    fps: int = 60,
):
    """Start real-time animation window"""
    with AnimationWindow(size) as renderer:
        renderer.load_shader(shader_func)
        renderer.run(fps)
