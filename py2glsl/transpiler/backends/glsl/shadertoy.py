from py2glsl.transpiler.backends.base import (
    BackendConfig,
    EntryPointConfig,
    GLSLBackend,
)
from py2glsl.transpiler.models import FunctionInfo


class ShadertoyBackend(GLSLBackend):
    """Shadertoy-specific GLSL backend."""

    def __init__(self, config: BackendConfig):
        super().__init__(config)
        # Mapping from standard uniform names to Shadertoy equivalents
        self.uniform_mapping: dict[str, str] = {
            "u_time": "iTime",
            "u_resolution": "iResolution.xy",
            "u_mouse_pos": "iMouse.xy",
            "u_aspect": "iResolution.x / iResolution.y",
        }

    def generate_entry_point(
        self, main_func: str, main_func_info: FunctionInfo
    ) -> list[str]:
        """Generate Shadertoy entry point.

        For use in actual Shadertoy, we would generate mainImage(), but for local
        rendering we need to use a standard main() function that wraps it.

        In Shadertoy, mainImage() receives fragCoord and transforms it to UV
        coordinates, but for local rendering we pass vs_uv from the vertex shader
        to maintain consistency.
        """
        # Define mainImage function with proper Shadertoy signature
        # void mainImage(out vec4 fragColor, in vec2 fragCoord)
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


def create_shadertoy_backend() -> ShadertoyBackend:
    """Create a Shadertoy GLSL backend."""
    config = BackendConfig(
        name="shadertoy",
        version_directive="#version 330 core",
        entry_point=EntryPointConfig(
            input_variables={"fragCoord": "vec2"},
            output_variables={"fragColor": "vec4"},
        ),
        predefined_uniforms={
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
        },
        # Note: We keep GLSL ES style precision qualifiers in the source code
        # for Shadertoy compatibility, but use OpenGL 3.3 core for desktop rendering
        preprocessor_defines={
            "precision mediump float": None,
            "precision mediump int": None,
            "precision mediump sampler2D": None,
        },
    )
    return ShadertoyBackend(config)
