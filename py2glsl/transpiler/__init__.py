"""GLSL shader transpiler for Python."""

import ast
import inspect
import textwrap
from dataclasses import dataclass
from typing import Any, Dict

from loguru import logger

from .analyzer import ShaderAnalysis, ShaderAnalyzer
from .constants import VERTEX_SHADER
from .formatter import GLSLFormatter
from .generator import GLSLGenerator
from .types import GLSLType


@dataclass
class ShaderResult:
    """Result of shader transformation."""

    fragment_source: str
    uniforms: Dict[str, str]
    vertex_source: str = VERTEX_SHADER


def py2glsl(func: Any) -> ShaderResult:
    """Transform Python shader function to GLSL.

    Args:
        func: Python function to transform to GLSL shader

    Returns:
        ShaderResult containing generated GLSL code

    Raises:
        TypeError: If shader function is invalid
        ValueError: If shader code is invalid
    """
    try:
        # Get source code and clean it
        source = inspect.getsource(func)
        logger.debug(f"Original source:\n{source}")

        # Clean up source indentation
        source = textwrap.dedent(source)
        logger.debug(f"After dedent:\n{source}")

        # Get AST first
        try:
            tree = ast.parse(source)
            logger.debug("Successfully parsed AST")
        except IndentationError:
            # If initial parse fails, try more aggressive cleaning
            clean_lines = [line.lstrip() for line in source.splitlines()]
            source = "\n".join(clean_lines)
            logger.debug(f"After aggressive cleaning:\n{source}")
            tree = ast.parse(source)

        # Extract just the function definition
        if isinstance(tree.body[0], ast.FunctionDef):
            func_def = tree.body[0]
            # Create new AST with just the function
            new_tree = ast.Module(body=[func_def], type_ignores=[])
            logger.debug(f"Extracted function AST:\n{ast.dump(new_tree, indent=2)}")
        else:
            raise ValueError("Could not find function definition")

    except (TypeError, OSError) as e:
        logger.error(f"Failed to get source code: {e}")
        raise TypeError("Could not get source code for shader function") from e
    except SyntaxError as e:
        logger.error(f"Failed to parse AST: {e}\nSource:\n{source}")
        raise ValueError(f"Invalid Python syntax in shader function: {str(e)}") from e

    # Analyze shader using AST directly
    try:
        analyzer = ShaderAnalyzer()
        analysis = analyzer.analyze(new_tree)
    except (TypeError, ValueError) as e:
        logger.error(f"Shader analysis failed: {e}")
        raise TypeError(f"Invalid shader function: {str(e)}") from e

    # Generate GLSL
    try:
        generator = GLSLGenerator(analysis)
        fragment_source = generator.generate()
        logger.debug(f"Generated GLSL:\n{fragment_source}")
    except (TypeError, ValueError) as e:
        logger.error(f"GLSL generation failed: {e}")
        raise ValueError(f"Could not generate GLSL code: {str(e)}") from e

    # Extract uniform information for result
    uniforms = {
        name: str(glsl_type).split()[-1]  # Get base type without qualifiers
        for name, glsl_type in analysis.uniforms.items()
    }
    logger.debug(f"Extracted uniforms: {uniforms}")

    return ShaderResult(
        fragment_source=fragment_source,
        vertex_source=VERTEX_SHADER,
        uniforms=uniforms,
    )


__all__ = [
    "py2glsl",
    "ShaderResult",
    "ShaderAnalysis",
    "GLSLType",
]
