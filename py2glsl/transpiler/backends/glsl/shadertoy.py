from py2glsl.transpiler.backends.base import GLSLBackend
from py2glsl.transpiler.backends.models import BackendConfig, EntryPointConfig
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
        rendering
        we need to use a standard main() function that wraps it.
        """
        lines = ["\n// Shadertoy mainImage function"]
        lines.append("vec4 mainImage(vec2 fragCoord) {")

        # Convert fragCoord to vs_uv
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
        lines.append(f"    return {main_func}({main_call_str});")
        lines.append("}")

        # Add regular main function to work with OpenGL
        lines.append("\n// Standard entry point for OpenGL")
        lines.append("out vec4 fragColor;")  # Declare the output variable
        lines.append("void main() {")
        lines.append("    fragColor = mainImage(gl_FragCoord.xy);")
        lines.append("}")
        return lines


def create_shadertoy_backend() -> ShadertoyBackend:
    """Create a Shadertoy GLSL backend."""
    config = BackendConfig(
        name="shadertoy",
        version_directive="#version 300 es",
        entry_point=EntryPointConfig(
            input_variables={"fragCoord": "vec2"},
            output_variables={"fragColor": "vec4"},
        ),
        predefined_uniforms={
            "iTime": "float",
            "iResolution": "vec3",
            "iMouse": "vec4",
            "iChannel0": "sampler2D",
            "iChannel1": "sampler2D",
            "iChannel2": "sampler2D",
            "iChannel3": "sampler2D",
            "iFrame": "int",
        },
        preprocessor_defines={
            "precision mediump float": None,
            "precision mediump int": None,
            "precision mediump sampler2D": None,
        },
    )
    return ShadertoyBackend(config)
