"""Command line interface for py2glsl."""

import importlib.util
import inspect
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer
from loguru import logger

from py2glsl.builtins import vec4
from py2glsl.render import animate, render_gif, render_image, render_video
from py2glsl.transpiler import TargetType, transpile

app = typer.Typer(
    name="py2glsl",
    help="Transform Python functions into GLSL shaders with zero boilerplate.",
    add_completion=False,
)


def _assert_target_supports_rendering(target_type: TargetType) -> None:
    """Raise error if target doesn't support rendering."""
    if target_type == TargetType.SHADERTOY:
        logger.error(
            "Shadertoy target is for code export only and cannot be rendered.\n"
            "Use 'py2glsl export --target shadertoy' to export code.\n"
            "For rendering, use --target glsl (default) or --target opengl33."
        )
        raise typer.Exit(1)


def _returns_vec4(func: Callable[..., Any]) -> bool:
    """Check if a function returns vec4."""
    sig = inspect.signature(func)
    return_type = sig.return_annotation
    if return_type is inspect.Signature.empty:
        return False
    type_name = getattr(return_type, "__name__", None)
    return return_type in (vec4, "vec4") or type_name == "vec4"


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

            # Store all functions with return annotations as potential helpers
            if sig.return_annotation is not inspect.Signature.empty:
                if _returns_vec4(obj):
                    vec4_functions[name] = obj
                else:
                    helper_functions[name] = obj

        # Check if it's a global constant with type annotation or a dataclass
        # or a simple constant (int, float, bool) without annotation
        # or a regular class (for struct/method support)
        elif (
            (
                not callable(obj)
                and not inspect.ismodule(obj)
                and hasattr(module, "__annotations__")
                and name in module.__annotations__
            )
            or hasattr(obj, "__dataclass_fields__")  # Check for dataclasses
            or (
                # Regular classes (for struct/method support)
                inspect.isclass(obj)
                and not name.startswith("_")
                and hasattr(obj, "__init__")
            )
            or (
                # Simple constants without annotations (int, float, bool)
                not callable(obj)
                and not inspect.ismodule(obj)
                and isinstance(obj, int | float | bool)
                and not isinstance(obj, type)
            )
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
        # Multiple vec4 functions, try to find one with ShaderContext parameter
        for name, func in vec4_functions.items():
            sig = inspect.signature(func)
            params = list(sig.parameters.values())
            for param in params:
                ann = param.annotation
                ann_name = getattr(ann, "__name__", str(ann))
                if ann_name == "ShaderContext":
                    main_func = func
                    logger.info(f"Selected shader with ShaderContext: {name}")
                    break
            if main_func:
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
    # Register in sys.modules before exec - required for dataclass in Python 3.13+
    sys.modules[module_name] = shader_module
    try:
        spec.loader.exec_module(shader_module)
    except Exception:
        # Clean up on error
        sys.modules.pop(module_name, None)
        raise

    return shader_module


def _map_target(target_str: str) -> TargetType:
    """Map a target string to TargetType.

    Args:
        target_str: Target string ("glsl", "shadertoy", etc.)

    Returns:
        TargetType enum value
    """
    target_str = target_str.lower()
    mapping = {
        "glsl": TargetType.OPENGL46,
        "standard": TargetType.OPENGL46,
        "opengl46": TargetType.OPENGL46,
        "opengl33": TargetType.OPENGL33,
        "shadertoy": TargetType.SHADERTOY,
        "webgl": TargetType.WEBGL2,
        "webgl2": TargetType.WEBGL2,
    }
    if target_str in mapping:
        return mapping[target_str]
    logger.warning(f"Unknown target: {target_str}. Using OPENGL46 as default.")
    return TargetType.OPENGL46


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
        elif inspect.isclass(item) and not name.startswith("_"):
            # Include regular classes for struct/method support
            function_args.append(item)
        else:
            other_globals[name] = item

    # The main function should be first in the list for proper processing
    if main_func not in function_args:
        function_args.insert(0, main_func)

    return function_args, other_globals


def _get_transpiled_shader(
    shader_file: str,
    target_str: str = "glsl",
    main_function_name: str = "shader",
) -> tuple[str, TargetType]:
    """Transpile a Python shader file to GLSL.

    Args:
        shader_file: Path to the Python shader file
        target_str: Target platform string (glsl, shadertoy, etc.)
        main_function_name: Name of the function to use as the shader entry point.
                           Should NOT be "main" as that conflicts with GLSL.
                           Default is "shader".

    Returns:
        Tuple of (GLSL code, target type)
    """
    # Enforce naming convention to avoid conflicts with GLSL reserved names
    if main_function_name == "main":
        logger.warning(
            'Function name "main" is reserved in GLSL. '
            'Use "shader" or another name instead.'
        )
        main_function_name = "shader"
    try:
        main_func: Callable[..., Any] | None = None

        # Load the shader module
        shader_module = _load_shader_module(shader_file)

        # Find shader functions and global constants
        if main_function_name:
            # Try to find the specified function
            if hasattr(shader_module, main_function_name):
                func = getattr(shader_module, main_function_name)
                if not inspect.isfunction(func):
                    raise ValueError(f"'{main_function_name}' is not a function")
                if not _returns_vec4(func):
                    msg = f"Function '{main_function_name}' must return vec4"
                    raise ValueError(msg)
                main_func = func
                _, globals_dict = _find_shader_function(shader_module)
                logger.info(f"Using specified function: {main_function_name}")
            else:
                raise ValueError(f"Function '{main_function_name}' not found in module")
        else:
            # Auto-detect main function
            main_func, globals_dict = _find_shader_function(shader_module)

        # Map target string to TargetType
        target = _map_target(target_str)
        logger.info(f"Using {target.name} target")

        # Transpile the shader
        logger.info(f"Transpiling main function: {main_func.__name__}")
        logger.info(f"Globals: {list(globals_dict.keys())}")

        # Read source code directly from file to preserve class definitions
        with open(shader_file) as f:
            source_code = f.read()

        glsl_code, used_uniforms = transpile(
            source_code,
            main_func=main_func.__name__,
            target=target,
        )

        logger.info(f"Used uniforms: {used_uniforms}")

        return glsl_code, target

    except ImportError as e:
        logger.error(f"Failed to load shader module: {e}")
        raise typer.Exit(1) from e
    except ValueError as e:
        logger.error(f"Invalid shader function: {e}")
        raise typer.Exit(1) from e
    except Exception as e:
        logger.error(f"Transpilation error: {e}")
        raise typer.Exit(1) from e


@app.command()
def show(
    shader_file: str = typer.Argument(..., help="Python file containing shader"),
    target: str = typer.Option(
        "glsl", "--target", "-t", help="Target platform (glsl, opengl33, webgl)"
    ),
    main: str = typer.Option("", "--main", "-m", help="Main shader function name"),
    width: int = typer.Option(800, "--width", "-w", help="Window width"),
    height: int = typer.Option(600, "--height", "-h", help="Window height"),
    fps: int = typer.Option(30, "--fps", help="Target framerate (0 for unlimited)"),
) -> None:
    """Run interactive shader preview (press ESC to exit)."""
    target_type = _map_target(target)
    _assert_target_supports_rendering(target_type)
    glsl_code, _ = _get_transpiled_shader(shader_file, target, main)
    logger.info(f"Running interactive preview at {fps}fps...")
    animate(
        shader_input=glsl_code,
        target=target_type,
        size=(width, height),
        fps=fps,
    )


@app.command()
def image(
    shader_file: str = typer.Argument(..., help="Python file containing shader"),
    output: Path = typer.Argument(..., help="Output image file (.png, .jpg)"),
    target: str = typer.Option(
        "glsl", "--target", "-t", help="Target platform (glsl, opengl33, webgl)"
    ),
    main: str = typer.Option("", "--main", "-m", help="Main shader function name"),
    width: int = typer.Option(800, "--width", "-w", help="Image width"),
    height: int = typer.Option(600, "--height", "-h", help="Image height"),
    time: float = typer.Option(0.0, "--time", help="Time value for u_time uniform"),
) -> None:
    """Render shader to static image."""
    target_type = _map_target(target)
    _assert_target_supports_rendering(target_type)
    glsl_code, _ = _get_transpiled_shader(shader_file, target, main)
    logger.info(f"Rendering image to {output}...")
    render_image(
        shader_input=glsl_code,
        size=(width, height),
        time=time,
        target=target_type,
        output_path=str(output),
    )
    logger.info(f"Saved {output}")


@app.command()
def video(
    shader_file: str = typer.Argument(..., help="Python file containing shader"),
    output: Path = typer.Argument(..., help="Output video file (.mp4, .webm)"),
    target: str = typer.Option(
        "glsl", "--target", "-t", help="Target platform (glsl, opengl33, webgl)"
    ),
    main: str = typer.Option("", "--main", "-m", help="Main shader function name"),
    width: int = typer.Option(800, "--width", "-w", help="Video width"),
    height: int = typer.Option(600, "--height", "-h", help="Video height"),
    fps: int = typer.Option(30, "--fps", help="Frames per second"),
    duration: float = typer.Option(5.0, "--duration", "-d", help="Duration in seconds"),
    start: float = typer.Option(0.0, "--start", "-s", help="Start time"),
    codec: str = typer.Option("h264", "--codec", help="Video codec (h264, vp9)"),
    quality: int = typer.Option(8, "--quality", "-q", help="Quality 0-10"),
) -> None:
    """Render shader to video file."""
    target_type = _map_target(target)
    _assert_target_supports_rendering(target_type)
    glsl_code, _ = _get_transpiled_shader(shader_file, target, main)
    logger.info(f"Rendering {duration}s video at {fps}fps...")
    render_video(
        shader_input=glsl_code,
        size=(width, height),
        duration=duration,
        fps=fps,
        target=target_type,
        output_path=str(output),
        time_offset=start,
        codec=codec,
        quality=quality,
    )
    logger.info(f"Saved {output}")


@app.command()
def gif(
    shader_file: str = typer.Argument(..., help="Python file containing shader"),
    output: Path = typer.Argument(..., help="Output GIF file"),
    target: str = typer.Option(
        "glsl", "--target", "-t", help="Target platform (glsl, opengl33, webgl)"
    ),
    main: str = typer.Option("", "--main", "-m", help="Main shader function name"),
    width: int = typer.Option(800, "--width", "-w", help="GIF width"),
    height: int = typer.Option(600, "--height", "-h", help="GIF height"),
    fps: int = typer.Option(30, "--fps", help="Frames per second"),
    duration: float = typer.Option(5.0, "--duration", "-d", help="Duration in seconds"),
    start: float = typer.Option(0.0, "--start", "-s", help="Start time"),
) -> None:
    """Render shader to animated GIF."""
    target_type = _map_target(target)
    _assert_target_supports_rendering(target_type)
    glsl_code, _ = _get_transpiled_shader(shader_file, target, main)
    logger.info(f"Rendering {duration}s GIF at {fps}fps...")
    render_gif(
        shader_input=glsl_code,
        size=(width, height),
        duration=duration,
        fps=fps,
        target=target_type,
        output_path=str(output),
        time_offset=start,
    )
    logger.info(f"Saved {output}")


@app.command()
def export(
    shader_file: str = typer.Argument(..., help="Python file containing shader"),
    output: Path = typer.Argument(None, help="Output file (default: stdout)"),
    target: str = typer.Option(
        "glsl", "--target", "-t", help="Target platform (glsl, shadertoy, webgl)"
    ),
    main: str = typer.Option("", "--main", "-m", help="Main shader function name"),
) -> None:
    """Export transpiled GLSL code.

    Use --target shadertoy to export code for shadertoy.com.
    """
    glsl_code, _ = _get_transpiled_shader(shader_file, target, main)

    if output:
        with open(output, "w") as f:
            f.write(glsl_code)
        logger.info(f"Exported to {output}")
    else:
        print(glsl_code)


if __name__ == "__main__":
    app()
