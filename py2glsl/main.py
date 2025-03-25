"""Command line interface for py2glsl.

This module provides a command-line interface for transpiling Python shader functions
to GLSL and rendering them with various backends and output formats.
"""

import importlib.util
import inspect
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

import arrow
import typer
import watchdog.events
import watchdog.observers
from loguru import logger
from watchdog.events import FileSystemEventHandler

from py2glsl.builtins import vec4
from py2glsl.render import animate, render_gif, render_image, render_video
from py2glsl.transpiler import transpile
from py2glsl.transpiler.backends.models import BackendType
from py2glsl.transpiler.core.interfaces import TargetLanguageType

# Define type variables for TypedCallable
F = TypeVar("F", bound=Callable[..., Any])


# TypedCommand decorator helper
def typed_command(app_command: Any) -> Callable[[F], F]:
    """Wrap typer command with proper typing for mypy."""

    def decorator(func: F) -> F:
        return cast(F, app_command(func))

    return decorator


app = typer.Typer(
    name="py2glsl",
    help=(
        "Transform Python functions into GLSL shaders. "
        "Commands: show, watch, render-image, render-video, render-gif, export-code."
    ),
    add_completion=False,
)

# Typer doesn't support custom help text this way, need to use its built-in mechanisms


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

    # Track all functions with return annotation vec4
    vec4_functions: dict[str, Callable[..., Any]] = {}

    # Track all other helper functions
    helper_functions: dict[str, Callable[..., Any]] = {}

    # Find the main shader function
    main_func: Callable[..., Any] | None = None

    # First collect all globals and functions
    for name, obj in inspect.getmembers(module):
        # Skip special methods and imported modules
        if name.startswith("__") or inspect.ismodule(obj):
            continue

        # Check if it's a function
        if inspect.isfunction(obj):
            # Check function signature
            sig = inspect.signature(obj)

            # Check if it's a potential shader function
            if sig.return_annotation == vec4:
                # It returns vec4, definitely a shader function
                vec4_functions[name] = obj
            elif sig.return_annotation is inspect.Signature.empty:
                # No return annotation, could be a shader function
                # We'll accept this as a possible shader function
                vec4_functions[name] = obj
            else:
                # Has non-vec4 annotation, must be a helper
                helper_functions[name] = obj

        # Check if it's a global constant with type annotation or a dataclass
        elif (
            (
                not callable(obj)
                and not inspect.ismodule(obj)
                and hasattr(module, "__annotations__")
                and name in module.__annotations__
            )
            or hasattr(obj, "__dataclass_fields__")  # Check for dataclasses
        ):
            globals_dict[name] = obj
            logger.info(f"Found global constant: {name} = {obj}")

    # No shader functions found
    if not vec4_functions:
        raise ValueError("No shader functions returning vec4 found in the module")

    # If we found exactly one vec4 function, use it
    if len(vec4_functions) == 1:
        name, func = next(iter(vec4_functions.items()))
        main_func = func
        logger.info(f"Using the only vec4 function as main shader: {name}")
    else:
        # Multiple vec4 functions, try to find the most suitable one
        # First, check for functions with vs_uv and u_time parameters
        for name, func in vec4_functions.items():
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            if "vs_uv" in params and "u_time" in params:
                main_func = func
                logger.info(f"Selected shader with standard parameters: {name}")
                break

        # If still no main function, just use the first one
        if main_func is None:
            name, func = next(iter(vec4_functions.items()))
            main_func = func
            logger.info(f"Using first vec4 function as main shader: {name}")

    # Add all helper and vec4 functions to globals for transpilation
    for name, func in {**helper_functions, **vec4_functions}.items():
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


def _prepare_transpilation_args(
    globals_dict: dict[str, Any], main_func: Callable[..., Any]
) -> tuple[list[Callable[..., Any]], dict[str, Any]]:
    """Prepare arguments for transpilation.

    Args:
        globals_dict: Dictionary of global objects
        main_func: Main shader function

    Returns:
        Tuple containing:
        - List of functions to transpile (main function first)
        - Dictionary of other globals to pass as kwargs
    """
    function_args: list[Callable[..., Any]] = []
    other_globals: dict[str, Any] = {}

    for name, item in globals_dict.items():
        # Include only actual functions, not classes or builtins
        is_callable = callable(item) and not name.startswith("__")
        if is_callable and not isinstance(item, type):
            # Only include user-defined functions, not builtins
            has_module = hasattr(item, "__module__")
            if has_module:
                is_builtin = item.__module__.startswith("py2glsl.builtins")
            else:
                is_builtin = False
            not_builtin = has_module and not is_builtin
            if not_builtin:
                function_args.append(item)
        else:
            other_globals[name] = item

    # The main function should be first in the list for proper processing
    if main_func not in function_args:
        function_args.insert(0, main_func)

    return function_args, other_globals


def _get_transpiled_shader(
    shader_file: str,
    target: str = "glsl",
    main_function_name: str = "shader",
) -> tuple[str, BackendType, TargetLanguageType]:
    """Transpile a Python shader file to GLSL.

    Args:
        shader_file: Path to the Python shader file
        target: Target language (glsl or shadertoy)
        main_function_name: Name of the function to use as the shader entry point.
                           Should NOT be "main" as that conflicts with GLSL.
                           Default is "shader".

    Returns:
        Tuple of (GLSL code, backend type, target type)
    """
    # Enforce naming convention to avoid conflicts with GLSL reserved names
    if main_function_name == "main":
        logger.warning(
            'Function name "main" is reserved in GLSL. '
            'Use "shader" or another name instead.'
        )
        main_function_name = "shader"  # Default to "shader" if they tried to use "main"
    try:
        # Type annotation for the main function
        main_func: Callable[..., Any] | None = None

        # Load the shader module
        shader_module = _load_shader_module(shader_file)

        # Find shader functions and global constants
        # If main_function_name is provided, find that function specifically
        if main_function_name:
            # Try to find the specified function
            if hasattr(shader_module, main_function_name):
                func = getattr(shader_module, main_function_name)
                if inspect.isfunction(func):
                    sig = inspect.signature(func)
                    # Accept any function that either has a vec4 return annotation
                    # or has no return annotation but from context might return vec4
                    if sig.return_annotation in {vec4, inspect.Signature.empty}:
                        main_func = func
                        # We still need to collect other functions and globals
                        _, globals_dict = _find_shader_function(shader_module)
                        logger.info(f"Using specified function: {main_function_name}")
                    else:
                        msg = (
                            f"Function '{main_function_name}' must return vec4 "
                            "or have no return type annotation"
                        )
                        raise ValueError(msg)
                else:
                    msg = f"'{main_function_name}' is not a function"
                    raise ValueError(msg)
            else:
                raise ValueError(f"Function '{main_function_name}' not found in module")
        else:
            # Auto-detect main function
            main_func, globals_dict = _find_shader_function(shader_module)

        # Map target string to enum values
        target_type, backend_type = _map_target(target)
        logger.info(f"Using {target_type.name} target language")

        # Transpile the shader - include the main function and helper functions
        logger.info(f"Transpiling main function: {main_func.__name__}")
        logger.info(f"Globals: {list(globals_dict.keys())}")

        # Extract functions and other globals
        result = _prepare_transpilation_args(globals_dict, main_func)
        function_args, other_globals = result
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


@typed_command(app.command("show"))
def show_shader(
    shader_file: str = typer.Argument(
        ..., help="Python file containing shader functions"
    ),
    target: str = typer.Option(
        "glsl", "--target", "-t", help="Target language (glsl, shadertoy)"
    ),
    main_function: str = typer.Option(
        "", "--main", "-m", help="Specific shader function to use"
    ),
    width: int = typer.Option(800, "--width", "-w", help="Window width"),
    height: int = typer.Option(600, "--height", "-h", help="Window height"),
    fps: int = typer.Option(30, "--fps", help="Target framerate (0 for unlimited)"),
) -> None:
    """Display shader in an interactive window.

    The shader will run in realtime in a window, with the ability to close
    with ESC key.

    Example: py2glsl show examples/shader.py
    """
    # Get transpiled shader (error handling is inside the function)
    code = _get_transpiled_shader(shader_file, target, main_function)
    glsl_code, backend_type, _ = code

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


class ShaderChangeHandler(FileSystemEventHandler):  # type: ignore
    """Event handler for shader file changes."""

    def __init__(
        self,
        shader_file: str,
        target: str,
        main_function: str,
        width: int,
        height: int,
        fps: int,
    ):
        """Initialize shader change handler.

        Args:
            shader_file: Path to shader file
            target: Target language
            main_function: Main function name
            width: Window width
            height: Window height
            fps: Target framerate
        """
        self.shader_file = shader_file
        self.target = target
        self.main_function = main_function
        self.width = width
        self.height = height
        self.fps = fps
        self.is_running = False
        self.needs_reload = False

    def on_modified(self, event: watchdog.events.FileSystemEvent) -> None:
        """Handle file modified event.

        Args:
            event: File system event
        """
        if event.src_path == os.path.abspath(self.shader_file):
            logger.info(f"Detected changes in {self.shader_file}")
            self.needs_reload = True

    def run_shader(self) -> None:
        """Run the shader and reload on changes."""
        self.is_running = True

        try:
            # First time - try to transpile and run
            try:
                code = _get_transpiled_shader(
                    self.shader_file, self.target, self.main_function
                )
                glsl_code, backend_type, _ = code

                # Set output size
                size = _get_size(self.width, self.height)

                # Run animation with auto-reload
                logger.info(
                    f"Running shader in watch mode at {self.fps}fps "
                    "(press ESC to exit, shader will reload on file changes)..."
                )

                # Run with auto-reload callback
                def should_reload() -> bool:
                    if self.needs_reload:
                        self.needs_reload = False
                        return True
                    return False

                animate(
                    shader_input=glsl_code,
                    backend_type=backend_type,
                    size=size,
                    fps=self.fps,
                    reload_callback=should_reload,
                    reload_function=lambda: self._reload_shader(),
                )
            except Exception as e:
                logger.error(f"Error running shader: {e}")
                # Wait for file changes
                while self.is_running and not self.needs_reload:
                    time.sleep(0.1)

                if self.needs_reload:
                    self.needs_reload = False
                    self.run_shader()  # Recursive call to retry
        except KeyboardInterrupt:
            self.is_running = False
        finally:
            self.is_running = False

    def _reload_shader(self) -> tuple[str, BackendType]:
        """Reload the shader.

        Returns:
            Tuple of (GLSL code, backend type)
        """
        try:
            logger.info(f"Reloading shader from {self.shader_file}")
            code = _get_transpiled_shader(
                self.shader_file, self.target, self.main_function
            )
            glsl_code, backend_type, _ = code
            return glsl_code, backend_type
        except Exception as e:
            logger.error(f"Error reloading shader: {e}")
            # Return empty shader that won't crash but will display an error message
            error_shader = """
            void main() {
                gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);  // Red to indicate error
            }
            """
            return error_shader, BackendType.STANDARD


@typed_command(app.command("watch"))
def watch_shader(
    shader_file: str = typer.Argument(
        ..., help="Python file containing shader functions"
    ),
    target: str = typer.Option(
        "glsl", "--target", "-t", help="Target language (glsl, shadertoy)"
    ),
    main_function: str = typer.Option(
        "", "--main", "-m", help="Specific shader function to use"
    ),
    width: int = typer.Option(800, "--width", "-w", help="Window width"),
    height: int = typer.Option(600, "--height", "-h", help="Window height"),
    fps: int = typer.Option(30, "--fps", help="Target framerate (0 for unlimited)"),
) -> None:
    """Watch shader file and auto-reload on changes.

    Monitors the shader file and automatically reloads it when changes are
    detected, without having to restart the application.

    Example: py2glsl watch examples/shader.py
    """
    # Create file system observer for auto-reload
    observer = watchdog.observers.Observer()

    # Get absolute path
    abs_shader_file = os.path.abspath(shader_file)

    # Create handler
    handler = ShaderChangeHandler(
        abs_shader_file, target, main_function, width, height, fps
    )

    # Watch the file's directory, not the file itself
    directory = os.path.dirname(abs_shader_file)
    observer.schedule(handler, path=directory, recursive=False)
    observer.start()

    try:
        # Run the shader (will reload on changes)
        handler.run_shader()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping...")
    finally:
        observer.stop()
        observer.join()


# Define reusable argument
OUTPUT_IMAGE_ARG = typer.Argument(..., help="Output image file path")


@typed_command(app.command("render-image"))
def render_shader_image(
    shader_file: str = typer.Argument(
        ..., help="Python file containing shader functions"
    ),
    output: Path = OUTPUT_IMAGE_ARG,
    target: str = typer.Option(
        "glsl", "--target", "-t", help="Target language (glsl, shadertoy)"
    ),
    main_function: str = typer.Option(
        "", "--main", "-m", help="Specific shader function to use"
    ),
    width: int = typer.Option(800, "--width", "-w", help="Image width"),
    height: int = typer.Option(600, "--height", "-h", help="Image height"),
    time: float = typer.Option(0.0, "--time", help="Time value for the image"),
) -> None:
    """Render shader to static image file.

    Creates a still image at a specific time value.

    Example: py2glsl render-image examples/shader.py output.png --time 1.5
    """
    # Get transpiled shader
    code = _get_transpiled_shader(shader_file, target, main_function)
    glsl_code, backend_type, _ = code

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


# Define reusable argument
OUTPUT_VIDEO_ARG = typer.Argument(..., help="Output video file path")


@typed_command(app.command("render-video"))
def render_shader_video(
    shader_file: str = typer.Argument(
        ..., help="Python file containing shader functions"
    ),
    output: Path = OUTPUT_VIDEO_ARG,
    target: str = typer.Option(
        "glsl", "--target", "-t", help="Target language (glsl, shadertoy)"
    ),
    main_function: str = typer.Option(
        "", "--main", "-m", help="Specific shader function to use"
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
    """Render shader to video file.

    Creates a video animation of the shader running over time.

    Example: py2glsl render-video examples/shader.py output.mp4 --duration 10
    """
    # Get transpiled shader
    code = _get_transpiled_shader(shader_file, target, main_function)
    glsl_code, backend_type, _ = code

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


# Define reusable argument
OUTPUT_GIF_ARG = typer.Argument(..., help="Output GIF file path")


@typed_command(app.command("render-gif"))
def render_shader_gif(
    shader_file: str = typer.Argument(
        ..., help="Python file containing shader functions"
    ),
    output: Path = OUTPUT_GIF_ARG,
    target: str = typer.Option(
        "glsl", "--target", "-t", help="Target language (glsl, shadertoy)"
    ),
    main_function: str = typer.Option(
        "", "--main", "-m", help="Specific shader function to use"
    ),
    width: int = typer.Option(800, "--width", "-w", help="GIF width"),
    height: int = typer.Option(600, "--height", "-h", help="GIF height"),
    fps: int = typer.Option(30, "--fps", help="Frames per second"),
    duration: float = typer.Option(5.0, "--duration", "-d", help="Duration in seconds"),
    time_offset: float = typer.Option(
        0.0, "--time-offset", help="Starting time for animation"
    ),
) -> None:
    """Render shader to animated GIF file.

    Creates an animated GIF of the shader running over time.

    Example: py2glsl render-gif examples/shader.py output.gif --fps 15
    """
    # Get transpiled shader
    code = _get_transpiled_shader(shader_file, target, main_function)
    glsl_code, backend_type, _ = code

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


def _prepare_shadertoy_code(code: str) -> str:
    """Prepare GLSL code for Shadertoy by removing unnecessary elements.

    Removes:
    1. Shadertoy built-in uniforms
    2. GLSL version statements (#version)
    3. Precision statements (precision mediump float, etc.)
    4. The standard OpenGL main() function and its inputs/outputs
    5. Excess empty lines at beginning and end
    6. Header/metadata comments (if --format commented is used with
       --shadertoy-compatible)

    Args:
        code: Source code

    Returns:
        Code ready for direct copy-pasting to Shadertoy
    """
    lines = code.split("\n")
    filtered_lines = []
    skip_section = False

    # Skip header comment lines
    line_index = 0
    while line_index < len(lines) and lines[line_index].startswith("//"):
        line_index += 1

    # Skip blank line after header if present
    if line_index < len(lines) and lines[line_index].strip() == "":
        line_index += 1

    # Process remaining lines
    for i in range(line_index, len(lines)):
        line = lines[i]

        # Skip OpenGL entry point sections
        if "// Standard entry point for OpenGL" in line:
            skip_section = True
            continue
        elif skip_section and line.strip() == "}":
            skip_section = False
            continue
        elif skip_section:
            continue

        # Skip vertex shader input/output declarations
        line_trimmed = line.strip()
        if line_trimmed.startswith("in ") or line_trimmed.startswith("out "):
            continue
        # Skip lines containing Shadertoy built-in uniforms
        if any(uniform in line for uniform in SHADERTOY_UNIFORMS):
            continue

        # Skip version statements
        if line.strip().startswith("#version"):
            continue

        # Skip precision statements
        if line.strip().startswith("precision "):
            continue

        filtered_lines.append(line)

    # Clean up excess whitespace at beginning and end
    # First remove leading empty lines
    while filtered_lines and filtered_lines[0].strip() == "":
        filtered_lines.pop(0)

    # Then remove trailing empty lines
    while filtered_lines and filtered_lines[-1].strip() == "":
        filtered_lines.pop()

    # Add a single empty line at the end for clean formatting
    filtered_lines.append("")

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
    timestamp = arrow.utcnow().format("YYYY-MM-DD HH:mm:ss UTC")
    header = f"// Generated by py2glsl v{__import__('py2glsl').__version__}\n"
    header += f"// Generation time: {timestamp}\n"
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

    # Process for Shadertoy compatibility - prepare code for copy-pasting
    if shadertoy_compatible and target_type == TargetLanguageType.SHADERTOY:
        formatted_code = _prepare_shadertoy_code(formatted_code)

    # Always add header comments for Shadertoy exports, or when specifically requested
    if shadertoy_compatible or format_type in ("commented", "wrapped"):
        formatted_code = _add_header_comments(
            formatted_code, source_file, target_type, shadertoy_compatible
        )

    # For Shadertoy, wrap in HTML comment for easy copying
    if format_type == "wrapped" and target_type == TargetLanguageType.SHADERTOY:
        formatted_code = f"/*\n{formatted_code}\n*/\n"

    return formatted_code


# Functions above are useful - they improve code organization
# But we need to use variables for the CLI arguments to avoid B008 warnings


# Define reusable argument
OUTPUT_CODE_ARG = typer.Argument(..., help="Output code file path")


@typed_command(app.command("export-code"))
def export_shader_code(
    shader_file: str = typer.Argument(
        ..., help="Python file containing shader functions"
    ),
    output: Path = OUTPUT_CODE_ARG,
    target: str = typer.Option(
        "glsl", "--target", "-t", help="Target language (glsl, shadertoy)"
    ),
    main_function: str = typer.Option(
        "", "--main", "-m", help="Specific shader function to use"
    ),
    format: str = typer.Option(
        "plain", "--format", "-f", help="Code format (plain, commented, wrapped)"
    ),
    shadertoy_compatible: bool = typer.Option(
        False,
        "--shadertoy-compatible",
        "-s",
        help="Process code for direct Shadertoy paste (removes version and uniforms)",
    ),
) -> None:
    """Export shader to GLSL code file.

    Transpiles the Python shader to GLSL and exports the result to a file.

    Example: py2glsl export-code examples/shader.py shader.glsl --target shadertoy
    """
    # Get transpiled shader
    code = _get_transpiled_shader(shader_file, target, main_function)
    glsl_code, _, target_type = code

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
                "✓ Ready for direct Shadertoy paste (version and uniforms removed)"
            )
        else:
            logger.info("✓ Ready for Shadertoy (requires manual edits)")
    else:
        logger.info("✓ Ready for use with OpenGL/WebGL")


if __name__ == "__main__":
    app()
