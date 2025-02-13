from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import moderngl
import numpy as np

from ..types.errors import GLSLError


@dataclass
class ShaderAttribute:
    """Shader attribute configuration."""

    name: str
    location: int
    type: str


@dataclass
class ShaderVarying:
    """Shader varying configuration."""

    name: str
    type: str


class ShaderProgram:
    """Manages shader program and associated buffers."""

    # Default attributes for the quad rendering
    DEFAULT_ATTRIBUTES = [
        ShaderAttribute("in_pos", 0, "vec2"),
        ShaderAttribute("in_uv", 1, "vec2"),
    ]

    # Default varyings for fragment shader input
    DEFAULT_VARYINGS = [
        ShaderVarying("vs_uv", "vec2"),
    ]

    # Quad vertices for fullscreen rendering
    QUAD_VERTICES = np.array(
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

    def __init__(
        self,
        ctx: moderngl.Context,
        vertex_source: str,
        fragment_source: str,
        attributes: Optional[List[ShaderAttribute]] = None,
        varyings: Optional[List[ShaderVarying]] = None,
    ):
        """Initialize shader program.

        Args:
            ctx: ModernGL context
            vertex_source: Vertex shader source
            fragment_source: Fragment shader source
            attributes: Optional custom attributes
            varyings: Optional custom varyings
        """
        self.ctx = ctx
        self.attributes = attributes or self.DEFAULT_ATTRIBUTES
        self.varyings = varyings or self.DEFAULT_VARYINGS

        # Create program and buffers
        self.program = self._create_program(vertex_source, fragment_source)
        self.quad_buffer = self._create_quad_buffer()
        self.vao = self._create_vertex_array()

    def _create_program(
        self, vertex_source: str, fragment_source: str
    ) -> moderngl.Program:
        """Create shader program with proper configuration."""
        try:
            program = self.ctx.program(
                vertex_shader=vertex_source,
                fragment_shader=fragment_source,
                varyings=[v.name for v in self.varyings],
            )

            # Set attribute locations
            for attr in self.attributes:
                program[attr.name].location = attr.location

            return program
        except Exception as e:
            raise GLSLError(f"Failed to create shader program: {e}")

    def _create_quad_buffer(self) -> moderngl.Buffer:
        """Create fullscreen quad buffer."""
        try:
            return self.ctx.buffer(self.QUAD_VERTICES.tobytes())
        except Exception as e:
            raise GLSLError(f"Failed to create quad buffer: {e}")

    def _create_vertex_array(self) -> moderngl.VertexArray:
        """Create vertex array with attributes."""
        try:
            return self.ctx.vertex_array(
                self.program,
                [(self.quad_buffer, "2f 2f", "in_pos", "in_uv")],
                skip_errors=False,
            )
        except Exception as e:
            raise GLSLError(f"Failed to create vertex array: {e}")

    def set_uniform(self, name: str, value: Any) -> None:
        """Set uniform value with type checking."""
        if name not in self.program:
            return

        try:
            if isinstance(value, (tuple, list, np.ndarray)):
                self.program[name].value = tuple(map(float, value))
            elif isinstance(value, bool):
                self.program[name].value = int(value)
            elif isinstance(value, int):
                self.program[name].value = value
            else:
                self.program[name].value = float(value)
        except Exception as e:
            raise GLSLError(f"Failed to set uniform {name}: {e}")

    def set_uniforms(self, uniforms: Dict[str, Any]) -> None:
        """Set multiple uniforms at once."""
        for name, value in uniforms.items():
            self.set_uniform(name, value)

    def set_built_in_uniforms(
        self, size: Tuple[int, int], time: float = 0.0, frame: int = 0
    ) -> None:
        """Set built-in uniforms."""
        built_ins = {
            "u_time": time,
            "u_frame": frame,
            "u_aspect": float(size[0]) / float(size[1]),
            "u_resolution": size,
        }
        self.set_uniforms(built_ins)

    def render(self, mode: int = moderngl.TRIANGLE_STRIP) -> None:
        """Render the shader program."""
        self.vao.render(mode=mode)

    @property
    def attributes_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about program attributes."""
        return {
            attr.name: {
                "location": attr.location,
                "type": attr.type,
            }
            for attr in self.attributes
        }

    @property
    def uniforms_info(self) -> Dict[str, str]:
        """Get information about program uniforms."""
        return {
            name: str(uniform.type) for name, uniform in self.program._members.items()
        }
