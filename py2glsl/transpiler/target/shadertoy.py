"""Shadertoy-specific GLSL language implementation.

This module implements a Shadertoy-compatible GLSL dialect.
"""


from py2glsl.transpiler.core.interfaces import LanguageConfig
from py2glsl.transpiler.models import FunctionInfo
from py2glsl.transpiler.target.glsl import GLSLStandardDialect


class ShadertoyGLSLDialect(GLSLStandardDialect):
    """Shadertoy GLSL dialect implementation."""

    def __init__(self) -> None:
        """Initialize with Shadertoy-specific settings."""
        super().__init__(version="330 core")
        self.uniform_mapping: dict[str, str] = {
            "u_time": "iTime",
            "u_resolution": "iResolution.xy",
            "u_mouse_pos": "iMouse.xy",
            "u_aspect": "iResolution.x / iResolution.y",
        }

    def get_config(self) -> LanguageConfig:
        """Get Shadertoy language configuration."""
        config = super().get_config()
        config.name = "Shadertoy GLSL"
        return config

    def _generate_predefined_uniforms(self) -> tuple[list[str], set[str]]:
        """Generate Shadertoy's predefined uniform declarations."""
        uniforms = {
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

        lines = []
        uniform_names = set()

        # Add precision qualifiers for Shadertoy compatibility
        lines.append("precision mediump float;")
        lines.append("precision mediump int;")
        lines.append("precision mediump sampler2D;")
        lines.append("")

        for name, type_name in uniforms.items():
            lines.append(f"uniform {type_name} {name};")
            uniform_names.add(name)

        return lines, uniform_names

    def _generate_uniforms(
        self, main_func_info: FunctionInfo
    ) -> tuple[list[str], set[str]]:
        """Generate uniform declarations, mapping to Shadertoy uniforms where applicable."""
        lines = []
        used_uniforms = set()

        for i, arg in enumerate(main_func_info.node.args.args):
            if arg.arg == "vs_uv":  # vs_uv is handled specially
                continue

            # Check if this is a Shadertoy mapped uniform
            if arg.arg in self.uniform_mapping:
                # We don't declare it as a uniform because it's already in the predefined set
                used_uniforms.add(arg.arg)
            else:
                # Regular uniform
                param_type = main_func_info.param_types[i]
                if param_type is not None:
                    mapped_type = self.symbol_mapper.map_type(param_type)
                    lines.append(f"uniform {mapped_type} {arg.arg};")
                else:
                    # Default to float if no type is specified
                    lines.append(f"uniform float {arg.arg};")
                used_uniforms.add(arg.arg)

        return lines, used_uniforms

    def _generate_entry_point(
        self, main_func: str, main_func_info: FunctionInfo
    ) -> list[str]:
        """Generate Shadertoy entry point.

        For Shadertoy, we need mainImage() with a specific signature, as well as
        a regular main() for local rendering.
        """
        # Define mainImage function with proper Shadertoy signature
        lines = ["\n// Shadertoy compatible mainImage function"]
        lines.append("void mainImage(out vec4 fragColor, in vec2 fragCoord) {")
        lines.append("    // Transform Shadertoy coordinates to UV coordinates")
        lines.append("    vec2 vs_uv = fragCoord / iResolution.xy;")

        # Build args list with transformations
        args = []
        for arg in main_func_info.node.args.args:
            arg_name = arg.arg
            if arg_name == "vs_uv":
                args.append("vs_uv")
            elif arg_name in self.uniform_mapping:
                args.append(self.uniform_mapping[arg_name])
            else:
                args.append(arg_name)

        main_call_str = ", ".join(args)
        lines.append(f"    vec4 result = {main_func}({main_call_str});")
        lines.append("    fragColor = result;")
        lines.append("}")

        # Add regular main function to work with OpenGL
        lines.append("\n// Standard entry point for OpenGL")
        lines.append("in vec2 vs_uv;")  # Declare input from vertex shader
        lines.append("out vec4 fragColor;")  # Declare the output variable
        lines.append("void main() {")
        # When rendering locally, we convert the vs_uv to gl_FragCoord-like coordinates
        lines.append("    // Convert vs_uv to fragCoord-like coordinates for Shadertoy")
        lines.append("    vec2 fragCoord = vs_uv * iResolution.xy;")
        lines.append("    mainImage(fragColor, fragCoord);")
        lines.append("}")

        return lines
