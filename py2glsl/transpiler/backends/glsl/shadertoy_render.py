"""Shadertoy GLSL render backend implementation."""

import time
from datetime import datetime
from typing import Any

from py2glsl.transpiler.backends.render import BaseRenderBackend, UniformProvider


class ShadertoyUniformProvider(UniformProvider):
    """Shadertoy uniform provider implementation."""

    def get_uniform_type_mapping(self) -> dict[str, str]:
        """Get the mapping of uniform names to their GLSL types."""
        return {
            # Basic uniforms (always available)
            "iResolution": "vec3",  # viewport resolution (in pixels)
            "iTime": "float",  # shader playback time (in seconds)
            "iTimeDelta": "float",  # render time (in seconds)
            "iFrame": "int",  # shader playback frame
            "iMouse": "vec4",  # mouse pixel coords. xy: current, zw: click
            "iDate": "vec4",  # (year, month, day, time in seconds)
            "iSampleRate": "float",  # sound sample rate (i.e., 44100)
            # Channel uniforms (for textures, etc.)
            "iChannel0": "sampler2D",
            "iChannel1": "sampler2D",
            "iChannel2": "sampler2D",
            "iChannel3": "sampler2D",
            "iChannelTime": "float[4]",  # channel playback time (in seconds)
            "iChannelResolution": "vec3[4]",  # channel resolution (in pixels)
        }

    def get_uniform_name_mapping(self) -> dict[str, str]:
        """Get the mapping of standard uniform names to backend-specific names."""
        return {
            "u_time": "iTime",
            "u_resolution": "iResolution.xy",
            "u_mouse_pos": "iMouse.xy",
            "u_aspect": "iResolution.x / iResolution.y",
        }


class ShadertoyRenderBackend(BaseRenderBackend):
    """Shadertoy GLSL rendering backend implementation."""

    def __init__(self) -> None:
        """Initialize the Shadertoy render backend."""
        super().__init__(ShadertoyUniformProvider())
        self._frame_count = 0
        self._last_time = time.time()

    def get_vertex_shader(self) -> str:
        """Get the Shadertoy vertex shader."""
        return """
#version 330 core
in vec2 in_position;
out vec2 vs_uv;
void main() {
    vs_uv = (in_position + 1.0) * 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

    def get_opengl_version(self) -> tuple[int, int]:
        """Get the OpenGL version (3.3)."""
        return (3, 3)

    def get_opengl_profile(self) -> str:
        """Get the OpenGL profile (core)."""
        return "core"

    def setup_uniforms(self, params: dict[str, Any]) -> dict[str, Any]:
        """Transform standard uniforms to Shadertoy-specific uniforms."""
        # Get base transformations from parent class
        result = super().setup_uniforms(params)

        # Calculate frame time delta
        current_time = time.time()
        time_delta = current_time - self._last_time
        self._last_time = current_time
        self._frame_count += 1

        # Add Shadertoy-specific uniforms
        if "u_resolution" in params:
            width, height = params["u_resolution"]
            result["iResolution"] = (width, height, 0.0)

        if "u_mouse_pos" in params and "u_mouse_uv" in params:
            mouse_x, mouse_y = params["u_mouse_pos"]
            # In Shadertoy, iMouse is (x, y, click_x, click_y)
            # For simplicity, we don't track clicks in this implementation
            result["iMouse"] = (mouse_x, mouse_y, 0.0, 0.0)

        # Add other Shadertoy uniforms
        result["iTimeDelta"] = time_delta
        result["iFrame"] = self._frame_count

        # Get current date and time
        now = datetime.now()
        seconds_since_midnight = (
            now.hour * 3600 + now.minute * 60 + now.second + now.microsecond / 1000000.0
        )
        result["iDate"] = (now.year, now.month, now.day, seconds_since_midnight)

        # Add default values for channels
        result["iSampleRate"] = 44100.0
        result["iChannelTime"] = [0.0, 0.0, 0.0, 0.0]
        result["iChannelResolution"] = [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ]

        return result
