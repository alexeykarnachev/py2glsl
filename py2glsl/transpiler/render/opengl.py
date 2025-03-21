"""OpenGL rendering backend implementation.

This module provides rendering backends for OpenGL-based shaders.
"""

from typing import Any

from py2glsl.transpiler.core.interfaces import RenderInterface


class StandardOpenGLRenderer(RenderInterface):
    """Standard OpenGL rendering backend."""

    def get_vertex_code(self) -> str:
        """Get the standard OpenGL vertex shader."""
        return """
#version 460 core
in vec2 in_position;
out vec2 vs_uv;
void main() {
    vs_uv = (in_position + 1.0) * 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

    def setup_uniforms(self, params: dict[str, Any]) -> dict[str, Any]:
        """Transform standard uniform values for OpenGL."""
        # Standard OpenGL doesn't need any transformation
        return params.copy()

    def get_render_requirements(self) -> dict[str, Any]:
        """Get OpenGL rendering requirements."""
        return {
            "version_major": 4,
            "version_minor": 6,
            "profile": "core",
        }


class ShadertoyOpenGLRenderer(RenderInterface):
    """Shadertoy OpenGL rendering backend."""

    def __init__(self) -> None:
        """Initialize the Shadertoy renderer."""
        self._frame_count = 0
        self.uniform_mapping = {
            "u_time": "iTime",
            "u_resolution": "iResolution",
            "u_mouse_pos": "iMouse",
            "u_aspect": None,  # Calculated from iResolution
        }

    def get_vertex_code(self) -> str:
        """Get the Shadertoy compatible vertex shader."""
        return """
#version 330 core
in vec2 in_position;
out vec2 vs_uv;
void main() {
    vs_uv = (in_position + 1.0) * 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

    def setup_uniforms(self, params: dict[str, Any]) -> dict[str, Any]:
        """Transform standard uniforms to Shadertoy uniforms."""
        result = {}

        # Copy all params that don't need mapping
        for name, value in params.items():
            if name not in self.uniform_mapping:
                result[name] = value

        # Transform mapped uniforms
        if "u_resolution" in params:
            width, height = params["u_resolution"]
            result["iResolution"] = (width, height, 0.0)

        if "u_time" in params:
            result["iTime"] = params["u_time"]

        if "u_mouse_pos" in params:
            mouse_x, mouse_y = params["u_mouse_pos"]
            # In Shadertoy, iMouse is (x, y, click_x, click_y)
            result["iMouse"] = (mouse_x, mouse_y, 0.0, 0.0)

        # Add other Shadertoy uniforms
        result["iTimeDelta"] = 0.016  # Assume 60 FPS for now
        result["iFrame"] = self._frame_count
        self._frame_count += 1

        # Date and time (simplified)
        from datetime import datetime

        now = datetime.now()
        secs = (now.hour * 3600 + now.minute * 60 + now.second + now.microsecond / 1000000)
        result["iDate"] = (now.year, now.month, now.day, secs)

        # Other defaults
        result["iSampleRate"] = 44100.0
        result["iChannelTime"] = [0.0, 0.0, 0.0, 0.0]
        result["iChannelResolution"] = [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ]

        return result

    def get_render_requirements(self) -> dict[str, Any]:
        """Get OpenGL rendering requirements for Shadertoy."""
        return {
            "version_major": 3,
            "version_minor": 3,
            "profile": "core",
        }
