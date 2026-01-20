from py2glsl.transpiler.backends.base import (
    BackendConfig,
    EntryPointConfig,
    GLSLBackend,
)
from py2glsl.transpiler.models import FunctionInfo


class StandardGLSLBackend(GLSLBackend):
    """Standard GLSL backend (current behavior)."""

    def generate_entry_point(
        self, main_func: str, main_func_info: FunctionInfo
    ) -> list[str]:
        """Generate standard GLSL entry point."""
        lines = ["\nin vec2 vs_uv;\nout vec4 fragColor;\n\nvoid main() {"]
        main_call_args = [arg.arg for arg in main_func_info.node.args.args]
        main_call_str = ", ".join(main_call_args)
        lines.append(f"    fragColor = {main_func}({main_call_str});")
        lines.append("}")
        return lines


def create_standard_backend() -> StandardGLSLBackend:
    """Create a standard GLSL backend."""
    config = BackendConfig(
        name="standard",
        version_directive="#version 460 core",
        entry_point=EntryPointConfig(
            input_variables={"vs_uv": "vec2"}, output_variables={"fragColor": "vec4"}
        ),
    )
    return StandardGLSLBackend(config)
