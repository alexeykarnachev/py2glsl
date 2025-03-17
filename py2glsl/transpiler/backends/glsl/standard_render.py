"""Standard GLSL render backend implementation."""


from py2glsl.transpiler.backends.render import BaseRenderBackend, UniformProvider


class StandardGLSLUniformProvider(UniformProvider):
    """Standard GLSL uniform provider implementation."""

    def get_uniform_type_mapping(self) -> dict[str, str]:
        """Get the mapping of uniform names to their GLSL types."""
        return {
            "u_resolution": "vec2",
            "u_time": "float",
            "u_aspect": "float",
            "u_mouse_pos": "vec2",
            "u_mouse_uv": "vec2",
        }

    def get_uniform_name_mapping(self) -> dict[str, str]:
        """Get the mapping of standard uniform names to backend-specific names."""
        # Standard backend uses the original names - no mapping needed
        return {}


class StandardGLSLRenderBackend(BaseRenderBackend):
    """Standard GLSL rendering backend implementation."""

    def __init__(self) -> None:
        """Initialize the standard render backend."""
        super().__init__(StandardGLSLUniformProvider())

    def get_vertex_shader(self) -> str:
        """Get the standard vertex shader."""
        return """
#version 460 core
in vec2 in_position;
out vec2 vs_uv;
void main() {
    vs_uv = (in_position + 1.0) * 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

    def get_opengl_version(self) -> tuple[int, int]:
        """Get the OpenGL version (4.6)."""
        return (4, 6)

    def get_opengl_profile(self) -> str:
        """Get the OpenGL profile (core)."""
        return "core"
