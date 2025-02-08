from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ShaderResult:
    fragment_source: str


def py2glsl(shader_func: Callable[..., Any]) -> ShaderResult:
    """Convert Python shader function to GLSL."""
    # For now, just return a minimal shader
    return ShaderResult(
        fragment_source="""
#version 460

in vec2 vs_uv;
out vec4 fs_color;

void main() {
    fs_color = vec4(1.0);
}
""".strip()
    )
