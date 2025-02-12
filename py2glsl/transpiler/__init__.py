"""GLSL shader transpiler."""

from dataclasses import dataclass
from typing import Dict, Optional

from py2glsl.transpiler.generator import GeneratedShader


@dataclass
class ShaderResult:
    """Result of shader transpilation."""

    fragment_source: str
    vertex_source: Optional[str]
    uniforms: Dict[str, str]


def py2glsl(shader_func) -> GeneratedShader:
    """Transform Python function into GLSL shader."""
    from .analyzer import ShaderAnalyzer
    from .generator import GLSLGenerator

    # Analyze shader function
    analyzer = ShaderAnalyzer()
    analysis = analyzer.analyze(shader_func)

    # Generate GLSL code
    generator = GLSLGenerator(analysis)
    return generator.generate()


__all__ = [
    "py2glsl",
    "ShaderResult",
]
