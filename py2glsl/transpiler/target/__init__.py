"""Target language management module.

This module provides factory functions for creating target language instances.
"""

from py2glsl.transpiler.core.interfaces import (
    LanguageAdapter,
    RenderInterface,
    TargetLanguage,
    TargetLanguageType,
)
from py2glsl.transpiler.render.opengl import (
    ShadertoyOpenGLRenderer,
    StandardOpenGLRenderer,
)
from py2glsl.transpiler.target.glsl import GLSLStandardDialect
from py2glsl.transpiler.target.shadertoy import ShadertoyGLSLDialect


class GLSLAdapter(LanguageAdapter):
    """Adapter for GLSL languages to OpenGL renderers."""

    def adapt(self, code: str, uniforms: set[str]) -> tuple[str, dict[str, str]]:
        """Adapt GLSL code for the specific OpenGL renderer.

        Args:
            code: Generated GLSL code
            uniforms: Set of uniform names used in the code

        Returns:
            Tuple of (adapted code, uniform mapping)
        """
        # Create mapping of uniforms to their actual names in the renderer
        uniform_mapping = {}
        for uniform in uniforms:
            # Get standard-to-backend mapping using renderer's setup_uniforms mechanism
            test_params = {uniform: None}
            mapped_params = self.renderer.setup_uniforms(test_params)

            # If the uniform was mapped to something else, add it to the mapping
            for key in mapped_params:
                if key != uniform:
                    uniform_mapping[uniform] = key
                    break

        return code, uniform_mapping


def create_target(
    target_type: TargetLanguageType = TargetLanguageType.GLSL
) -> tuple[TargetLanguage, RenderInterface, LanguageAdapter]:
    """Create a target language, renderer, and adapter based on type.

    Args:
        target_type: The type of target language to create

    Returns:
        Tuple of (target language, renderer, adapter)

    Raises:
        ValueError: If the target type is not supported
    """
    if target_type == TargetLanguageType.GLSL:
        language = GLSLStandardDialect()
        renderer = StandardOpenGLRenderer()
        adapter = GLSLAdapter(language, renderer)
        return language, renderer, adapter
    elif target_type == TargetLanguageType.HLSL:
        # Future expansion point
        raise ValueError("HLSL target not yet implemented")
    elif target_type == TargetLanguageType.WGSL:
        # Future expansion point
        raise ValueError("WGSL target not yet implemented")
    else:
        raise ValueError(f"Unsupported target type: {target_type}")


def create_glsl_target(
    shadertoy: bool = False
) -> tuple[TargetLanguage, RenderInterface, LanguageAdapter]:
    """Create a GLSL target with optional Shadertoy compatibility.

    Args:
        shadertoy: Whether to create a Shadertoy-compatible target

    Returns:
        Tuple of (target language, renderer, adapter)
    """
    if shadertoy:
        language: TargetLanguage = ShadertoyGLSLDialect()
        renderer: RenderInterface = ShadertoyOpenGLRenderer()
    else:
        language = GLSLStandardDialect()
        renderer = StandardOpenGLRenderer()

    adapter = GLSLAdapter(language, renderer)
    return language, renderer, adapter
