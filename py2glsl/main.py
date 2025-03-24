"""Command line interface for py2glsl.

This module provides a command-line interface for transpiling Python shader functions
to GLSL and rendering them with various backends and output formats.
"""

import importlib.util
import inspect
import os
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from loguru import logger

from py2glsl.builtins import vec4
from py2glsl.render import animate, render_gif, render_image, render_video
from py2glsl.transpiler import transpile
from py2glsl.transpiler.backends.models import BackendType
from py2glsl.transpiler.core.interfaces import TargetLanguageType

app = typer.Typer(
    name="py2glsl",
    help="Transform Python functions into GLSL shaders with zero boilerplate.",
    add_completion=False,
)

# Create subcommands for different output formats
show_app = typer.Typer(help="Run interactive shader preview")
image_app = typer.Typer(help="Render shader to static image")
video_app = typer.Typer(help="Render shader to video file")
gif_app = typer.Typer(help="Render shader to animated GIF")
code_app = typer.Typer(help="Export shader code to file")

# Add the subcommands
app.add_typer(show_app, name="show")
app.add_typer(image_app, name="image")
app.add_typer(video_app, name="video")
app.add_typer(gif_app, name="gif")
app.add_typer(code_app, name="code")


def _find_shader_function(module: Any) -> tuple[Callable[..., Any], dict[str, Any]]:
    """Find the main shader function and global constants in a module.

    Args:
        module: The imported shader module

    Returns:
        Tuple containing:
        - Main shader function
        - Dictionary of global constants
    """
    # Track global constants with type annotations
    globals_dict: dict[str, Any] = {}
    
    # Track all helper functions
    helper_functions: dict[str, Any] = {}

    # Find the main shader function
    main_func = None

    # First collect all globals and functions
    for name, obj in inspect.getmembers(module):
        # Skip special methods and imported modules
        if name.startswith("__") or inspect.ismodule(obj):
            continue

        # Check if it's a function
        if inspect.isfunction(obj):
            # Check function signature
            sig = inspect.signature(obj)
            
            # Store all functions with return annotations as potential helpers
            if sig.return_annotation is not inspect.Signature.empty:
                helper_functions[name] = obj
                
                # If it returns vec4, it's a potential main function
                if sig.return_annotation == vec4:
                    # Check if it has standard parameters (vs_uv, u_time, u_aspect)
                    params = list(sig.parameters.keys())
                    if "vs_uv" in params and "u_time" in params:
                        # Priority for functions named main_shader, simple_shader, etc.
                        if name in ["main_shader", "simple_shader", "shader"]:
                            main_func = obj
                            logger.info(f"Found main shader function: {name}")
                            break

        # Check if it's a global constant with type annotation or a dataclass
        elif (
            (not callable(obj) and not inspect.ismodule(obj) and 
             hasattr(module, "__annotations__") and name in module.__annotations__) or
            hasattr(obj, "__dataclass_fields__")  # Check for dataclasses
        ):
            globals_dict[name] = obj
            logger.info(f"Found global constant: {name} = {obj}")

    # If we didn't find a main function with a preferred name, use any function returning vec4
    if main_func is None:
        for name, obj in helper_functions.items():
            sig = inspect.signature(obj)
            if sig.return_annotation == vec4:
                params = list(sig.parameters.keys())
                if "vs_uv" in params and "u_time" in params:
                    main_func = obj
                    logger.info(f"Using function {name} as main shader")
                    break

    if main_func is None:
        raise ValueError("No suitable shader function found in the module")
        
    # Add all helper functions to the globals dict so they're included in transpilation
    for name, func in helper_functions.items():
        if func != main_func:  # Don't include the main function twice
            globals_dict[name] = func

    return main_func, globals_dict


def _load_shader_module(file_path: str) -> Any:
    """Load a Python file as a module.

    Args:
        file_path: Path to the Python file

    Returns:
        Loaded module
    """
    # Get absolute path
    abs_path = os.path.abspath(file_path)
    module_dir = os.path.dirname(abs_path)
    module_name = os.path.splitext(os.path.basename(abs_path))[0]

    # Add module directory to path if not already there
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    # Load module
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")

    shader_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shader_module)

    return shader_module


def _map_target(target: str) -> tuple[TargetLanguageType, BackendType]:
    """Map a target string to target language and backend types.

    Args:
        target: Target string ("glsl", "shadertoy", etc.)

    Returns:
        Tuple of (target language type, backend type)
    """
    target = target.lower()
    if target in ("glsl", "standard"):
        return TargetLanguageType.GLSL, BackendType.STANDARD
    elif target == "shadertoy":
        return TargetLanguageType.SHADERTOY, BackendType.SHADERTOY
    else:
        logger.warning(f"Unknown target: {target}. Using GLSL as default.")
        return TargetLanguageType.GLSL, BackendType.STANDARD


def _get_transpiled_shader(
    shader_file: str,
    target: str = "glsl",
) -> tuple[str, BackendType, TargetLanguageType]:
    """Transpile a Python shader file to GLSL.

    Args:
        shader_file: Path to the Python shader file
        target: Target language (glsl or shadertoy)

    Returns:
        Tuple of (GLSL code, backend type, target type)
    """
    try:
        # Load the shader module
        shader_module = _load_shader_module(shader_file)

        # Find main shader function and global constants
        main_func, globals_dict = _find_shader_function(shader_module)

        # Map target string to enum values
        target_type, backend_type = _map_target(target)
        logger.info(f"Using {target_type.name} target language")

        # Transpile the shader - include the main function and helper functions
        logger.info(f"Transpiling main function: {main_func.__name__}")
        logger.info(f"Globals: {list(globals_dict.keys())}")

        # Extract functions and other globals for proper transpilation
        function_args = []
        other_globals = {}
        
        for name, item in globals_dict.items():
            # Include only actual functions, not classes or builtins
            if callable(item) and not name.startswith("__") and not isinstance(item, type):
                # Only include user-defined functions, not builtins
                if hasattr(item, "__module__") and not item.__module__.startswith("py2glsl.builtins"):
                    function_args.append(item)
            else:
                other_globals[name] = item
        
        # The main function should be first in the list for proper processing
        if main_func not in function_args:
            function_args.insert(0, main_func)
            
        logger.info(f"Including functions: {[func.__name__ for func in function_args]}")
            
        glsl_code, used_uniforms = transpile(
            *function_args,
            main_func=main_func.__name__,
            target_type=target_type,
            **other_globals,
        )

        logger.info(f"Used uniforms: {used_uniforms}")

        return glsl_code, backend_type, target_type

    except ImportError as e:
        logger.error(f"Failed to load shader module: {e}")
        raise typer.Exit(1) from e
    except ValueError as e:
        logger.error(f"Invalid shader function: {e}")
        raise typer.Exit(1) from e
    except Exception as e:
        logger.error(f"Transpilation error: {e}")
        raise typer.Exit(1) from e


def _get_size(width: int, height: int) -> tuple[int, int]:
    """Create a size tuple from width and height.

    Args:
        width: Width in pixels
        height: Height in pixels

    Returns:
        Size tuple (width, height)
    """
    return (width, height)


@show_app.command("run")
def show_shader(
    shader_file: str = typer.Argument(
        ..., help="Python file containing shader functions"
    ),
    target: str = typer.Option(
        "glsl", "--target", "-t", help="Target language (glsl, shadertoy)"
    ),
    width: int = typer.Option(800, "--width", "-w", help="Window width"),
    height: int = typer.Option(600, "--height", "-h", help="Window height"),
    fps: int = typer.Option(30, "--fps", help="Target framerate (0 for unlimited)"),
) -> None:
    """Run interactive shader preview."""
    # Get transpiled shader (error handling is inside the function)
    glsl_code, backend_type, _ = _get_transpiled_shader(shader_file, target)

    # Set output size
    size = _get_size(width, height)

    # Run interactive animation
    logger.info(f"Running interactive animation at {fps}fps (press ESC to exit)...")
    animate(
        shader_input=glsl_code,
        backend_type=backend_type,
        size=size,
        fps=fps,
    )


@image_app.command("render")
def render_shader_image(
    shader_file: str = typer.Argument(
        ..., help="Python file containing shader functions"
    ),
    output: Path = typer.Argument(..., help="Output image file path"),
    target: str = typer.Option(
        "glsl", "--target", "-t", help="Target language (glsl, shadertoy)"
    ),
    width: int = typer.Option(800, "--width", "-w", help="Image width"),
    height: int = typer.Option(600, "--height", "-h", help="Image height"),
    time: float = typer.Option(0.0, "--time", help="Time value for the image"),
) -> None:
    """Render shader to static image."""
    # Get transpiled shader
    glsl_code, backend_type, _ = _get_transpiled_shader(shader_file, target)

    # Set output size
    size = _get_size(width, height)

    # Render the image
    logger.info(f"Rendering still image to {output}...")
    render_image(
        shader_input=glsl_code,
        size=size,
        time=time,
        backend_type=backend_type,
        output_path=str(output),
    )
    logger.info(f"Image saved to {output}")


@video_app.command("render")
def render_shader_video(
    shader_file: str = typer.Argument(
        ..., help="Python file containing shader functions"
    ),
    output: Path = typer.Argument(..., help="Output video file path"),
    target: str = typer.Option(
        "glsl", "--target", "-t", help="Target language (glsl, shadertoy)"
    ),
    width: int = typer.Option(800, "--width", "-w", help="Video width"),
    height: int = typer.Option(600, "--height", "-h", help="Video height"),
    fps: int = typer.Option(30, "--fps", help="Frames per second"),
    duration: float = typer.Option(5.0, "--duration", "-d", help="Duration in seconds"),
    time_offset: float = typer.Option(
        0.0, "--time-offset", help="Starting time for animation"
    ),
    codec: str = typer.Option("h264", "--codec", help="Video codec (h264, vp9, etc.)"),
    quality: int = typer.Option(8, "--quality", "-q", help="Video quality (0-10)"),
) -> None:
    """Render shader to video."""
    # Get transpiled shader
    glsl_code, backend_type, _ = _get_transpiled_shader(shader_file, target)

    # Set output size
    size = _get_size(width, height)

    # Render the video
    logger.info(f"Rendering {duration}s video at {fps}fps to {output}...")
    render_video(
        shader_input=glsl_code,
        size=size,
        duration=duration,
        fps=fps,
        backend_type=backend_type,
        output_path=str(output),
        time_offset=time_offset,
        codec=codec,
        quality=quality,
    )
    logger.info(f"Video saved to {output}")


@gif_app.command("render")
def render_shader_gif(
    shader_file: str = typer.Argument(
        ..., help="Python file containing shader functions"
    ),
    output: Path = typer.Argument(..., help="Output GIF file path"),
    target: str = typer.Option(
        "glsl", "--target", "-t", help="Target language (glsl, shadertoy)"
    ),
    width: int = typer.Option(800, "--width", "-w", help="GIF width"),
    height: int = typer.Option(600, "--height", "-h", help="GIF height"),
    fps: int = typer.Option(30, "--fps", help="Frames per second"),
    duration: float = typer.Option(5.0, "--duration", "-d", help="Duration in seconds"),
    time_offset: float = typer.Option(
        0.0, "--time-offset", help="Starting time for animation"
    ),
) -> None:
    """Render shader to animated GIF."""
    # Get transpiled shader
    glsl_code, backend_type, _ = _get_transpiled_shader(shader_file, target)

    # Set output size
    size = _get_size(width, height)

    # Render the GIF
    logger.info(f"Rendering {duration}s GIF at {fps}fps to {output}...")
    render_gif(
        shader_input=glsl_code,
        size=size,
        duration=duration,
        fps=fps,
        backend_type=backend_type,
        output_path=str(output),
        time_offset=time_offset,
    )
    logger.info(f"GIF saved to {output}")


# List of Shadertoy built-in uniforms
SHADERTOY_UNIFORMS = [
    "uniform vec3 iResolution;",
    "uniform float iTime;",
    "uniform float iTimeDelta;",
    "uniform int iFrame;",
    "uniform vec4 iMouse;",
    "uniform vec4 iDate;",
    "uniform float iSampleRate;",
    "uniform sampler2D iChannel0;",
    "uniform sampler2D iChannel1;",
    "uniform sampler2D iChannel2;",
    "uniform sampler2D iChannel3;",
    "uniform float[4] iChannelTime;",
    "uniform vec3[4] iChannelResolution;",
]


def _remove_shadertoy_uniforms(code: str) -> str:
    """Remove Shadertoy built-in uniforms from code.

    Args:
        code: Source code

    Returns:
        Code with Shadertoy uniforms removed
    """
    lines = code.split("\n")
    filtered_lines = [
        line
        for line in lines
        if not any(uniform in line for uniform in SHADERTOY_UNIFORMS)
    ]
    return "\n".join(filtered_lines)


def _add_header_comments(
    code: str,
    source_file: str,
    target_type: TargetLanguageType,
    shadertoy_compatible: bool,
) -> str:
    """Add header comments to the code.

    Args:
        code: Source code
        source_file: Source Python file
        target_type: Target language type
        shadertoy_compatible: Whether Shadertoy compatibility is enabled

    Returns:
        Code with header comments
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"// Generated by py2glsl on {timestamp}\n"
    header += f"// Source file: {os.path.basename(source_file)}\n"
    header += f"// Target: {target_type.name}\n"

    if shadertoy_compatible and target_type == TargetLanguageType.SHADERTOY:
        header += "// Shadertoy-compatible: Built-in uniforms removed\n"

    header += "\n"
    return header + code


def _format_shader_code(
    code: str,
    format_type: str,
    target_type: TargetLanguageType,
    source_file: str,
    shadertoy_compatible: bool = False,
) -> str:
    """Format shader code for export.

    Args:
        code: Raw shader code
        format_type: Format type (plain, commented, wrapped)
        target_type: Target language type
        source_file: Source Python file
        shadertoy_compatible: If True, remove shadertoy uniforms for direct copy-paste

    Returns:
        Formatted shader code
    """
    formatted_code = code

    # Process for Shadertoy compatibility - remove uniform declarations
    if shadertoy_compatible and target_type == TargetLanguageType.SHADERTOY:
        formatted_code = _remove_shadertoy_uniforms(formatted_code)

    # Add header comments if requested
    if format_type in ("commented", "wrapped"):
        formatted_code = _add_header_comments(
            formatted_code, source_file, target_type, shadertoy_compatible
        )

    # For Shadertoy, wrap in HTML comment for easy copying
    if format_type == "wrapped" and target_type == TargetLanguageType.SHADERTOY:
        formatted_code = f"/*\n{formatted_code}\n*/\n"

    return formatted_code


# Functions above are useful - they improve code organization
# But we don't need shared variables for the CLI arguments


@code_app.command("export")
def export_shader_code(
    shader_file: str = typer.Argument(
        ..., help="Python file containing shader functions"
    ),
    output: Path = typer.Argument(..., help="Output code file path"),
    target: str = typer.Option(
        "glsl", "--target", "-t", help="Target language (glsl, shadertoy)"
    ),
    format: str = typer.Option(
        "plain", "--format", "-f", help="Code format (plain, commented, wrapped)"
    ),
    shadertoy_compatible: bool = typer.Option(
        False,
        "--shadertoy-compatible",
        "-s",
        help="Remove Shadertoy built-in uniforms for direct copy-paste",
    ),
) -> None:
    """Export shader code to file for copy-pasting."""
    # Get transpiled shader
    glsl_code, _, target_type = _get_transpiled_shader(shader_file, target)

    # Auto-suggest format for Shadertoy
    if target_type == TargetLanguageType.SHADERTOY and format == "plain":
        logger.info("Tip: Using --format wrapped helps with Shadertoy copy-pasting")

    # If target isn't shadertoy but flag is set, warn user
    if shadertoy_compatible and target_type != TargetLanguageType.SHADERTOY:
        logger.warning(
            "--shadertoy-compatible flag only applies to shadertoy target. Ignoring."
        )
        shadertoy_compatible = False

    # Format the code
    formatted_code = _format_shader_code(
        glsl_code, format, target_type, shader_file, shadertoy_compatible
    )

    # Write to file
    logger.info(f"Exporting shader code to {output}...")
    with open(output, "w") as f:
        f.write(formatted_code)

    logger.info(f"Shader code exported to {output}")

    # Show some usage hints
    if target_type == TargetLanguageType.SHADERTOY:
        if shadertoy_compatible:
            logger.info(
                "✓ Ready for direct copy-pasting to Shadertoy (uniforms removed)"
            )
        else:
            logger.info("✓ Ready for copy-pasting to Shadertoy")
    else:
        logger.info("✓ Ready for use with OpenGL/WebGL")


if __name__ == "__main__":
    app()
