"""Target abstraction for shader compilation.

A Target encapsulates everything needed to compile a shader for a specific platform:
- Code generation rules (GLSL syntax, entry point, builtin mapping)
- Runtime configuration (OpenGL version, vertex shader)
- Builtin uniform definitions and mappings
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from py2glsl.transpiler.ir import (
    IRFunction,
    IRType,
    IRVariable,
    ShaderStage,
    StorageClass,
)

# =============================================================================
# Canonical Builtin Names (the py2glsl shader language)
# =============================================================================

BUILTIN_UNIFORMS: dict[str, str] = {
    "u_time": "float",
    "u_resolution": "vec2",
    "u_aspect": "float",
    "u_mouse_pos": "vec2",
    "u_mouse_uv": "vec2",
}

BUILTIN_INPUTS: dict[str, str] = {
    "vs_uv": "vec2",
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class RuntimeConfig:
    """Runtime configuration for rendering."""

    gl_version: tuple[int, int]
    gl_profile: str
    vertex_shader: str


@dataclass
class TranspileResult:
    """Result of transpilation."""

    code: str
    uniforms: set[str]
    uniform_mapping: dict[str, str]
    runtime_config: RuntimeConfig


# =============================================================================
# Target ABC
# =============================================================================


class Target(ABC):
    """Base class for all compilation targets.

    A Target defines how to generate code for a specific platform and
    provides runtime configuration for rendering.
    """

    # --- Code Generation: Syntax ---

    @abstractmethod
    def version_directive(self) -> str | None:
        """Return the version directive (e.g., '#version 460 core')."""
        ...

    def precision_qualifiers(self) -> list[str]:
        """Return precision qualifiers (for ES targets). Default: empty."""
        return []

    def type_name(self, ir_type: IRType | str) -> str:
        """Map IR type to target type name. Default: pass-through."""
        if isinstance(ir_type, IRType):
            return ir_type.base
        return ir_type

    def operator(self, op: str) -> str:
        """Map operator to target operator. Default: standard C-like operators."""
        mapping = {
            "and": "&&",
            "or": "||",
            "not": "!",
        }
        return mapping.get(op, op)

    def literal(self, value: Any, ir_type: IRType | str) -> str:
        """Format a literal value. Default: standard formatting."""
        type_str = ir_type.base if isinstance(ir_type, IRType) else ir_type
        if type_str == "bool":
            return "true" if value else "false"
        if type_str == "float":
            s = str(float(value))
            if "." not in s and "e" not in s.lower():
                s += ".0"
            return s
        if type_str == "int":
            return str(int(value))
        return str(value)

    def builtin_function(self, _name: str) -> str | None:
        """Map function name to target builtin. Default: None (use as-is)."""
        return None

    def storage_qualifier(self, storage: StorageClass) -> str:
        """Map storage class to qualifier string."""
        mapping = {
            StorageClass.UNIFORM: "uniform",
            StorageClass.INPUT: "in",
            StorageClass.OUTPUT: "out",
            StorageClass.CONST: "const",
            StorageClass.BUFFER: "buffer",
            StorageClass.SHARED: "shared",
        }
        return mapping.get(storage, "")

    # --- Code Generation: Entry Point ---

    @abstractmethod
    def entry_point_wrapper(
        self,
        stage: ShaderStage,
        entry_func: IRFunction,
        inputs: list[IRVariable],
        outputs: list[IRVariable],
    ) -> list[str]:
        """Generate the entry point wrapper (e.g., void main())."""
        ...

    # --- Builtin Mapping ---

    @abstractmethod
    def get_uniform_mapping(self) -> dict[str, str]:
        """Return mapping from canonical uniform names to target names.

        Example for Shadertoy: {'u_time': 'iTime', 'u_resolution': 'iResolution.xy'}
        """
        ...

    def map_builtin(self, canonical_name: str) -> str:
        """Map a canonical builtin name to target-specific name."""
        mapping = self.get_uniform_mapping()
        return mapping.get(canonical_name, canonical_name)

    # --- Runtime Configuration ---

    @abstractmethod
    def get_vertex_shader(self) -> str:
        """Return the vertex shader code for this target."""
        ...

    @abstractmethod
    def get_gl_version(self) -> tuple[int, int]:
        """Return OpenGL version as (major, minor)."""
        ...

    @abstractmethod
    def get_gl_profile(self) -> str:
        """Return OpenGL profile ('core' or 'es')."""
        ...

    def get_runtime_config(self) -> RuntimeConfig:
        """Get complete runtime configuration."""
        return RuntimeConfig(
            gl_version=self.get_gl_version(),
            gl_profile=self.get_gl_profile(),
            vertex_shader=self.get_vertex_shader(),
        )

    # --- Target Capabilities ---

    def is_export_only(self) -> bool:
        """Return True if this target is for code export only (not rendering).

        Export-only targets (like Shadertoy) generate code meant to be
        pasted into external environments. They cannot be rendered locally.
        """
        return False

    # --- Predefined Uniforms (for targets like Shadertoy) ---

    def get_predefined_uniforms(self) -> dict[str, str]:
        """Return predefined uniforms that the target provides.

        These are uniforms that exist in the target environment
        and don't need to be declared by the user.
        Example for Shadertoy: {'iTime': 'float', 'iResolution': 'vec3', ...}
        """
        return {}

    # --- Uniform Value Transformation (for runtime) ---

    def transform_uniform_values(self, values: dict[str, Any]) -> dict[str, Any]:
        """Transform uniform values from canonical to target format.

        Override this for targets that need special value transformation.
        Example: Shadertoy needs iResolution as vec3, iMouse as vec4.
        """
        mapping = self.get_uniform_mapping()
        result = {}
        for name, value in values.items():
            target_name = mapping.get(name, name)
            result[target_name] = value
        return result


# =============================================================================
# OpenGL 4.6 Target (Standard Desktop)
# =============================================================================


class OpenGL46Target(Target):
    """Standard OpenGL 4.6 desktop target."""

    def version_directive(self) -> str | None:
        return "#version 460 core"

    def get_uniform_mapping(self) -> dict[str, str]:
        # Direct mapping - canonical names are used as-is
        return {}

    def entry_point_wrapper(
        self,
        _stage: ShaderStage,
        entry_func: IRFunction,
        _inputs: list[IRVariable],
        outputs: list[IRVariable],
    ) -> list[str]:
        """Generate standard OpenGL main() wrapper."""
        lines = ["void main() {"]

        # Build argument list for entry function call
        args = [param.name for param in entry_func.params]
        args_str = ", ".join(args)

        if entry_func.return_type:
            # Assign result to output variable
            output_name = outputs[0].name if outputs else "fragColor"
            lines.append(f"    {output_name} = {entry_func.name}({args_str});")
        else:
            lines.append(f"    {entry_func.name}({args_str});")

        lines.append("}")
        return lines

    def get_vertex_shader(self) -> str:
        return """#version 460 core
in vec2 in_position;
out vec2 vs_uv;
void main() {
    vs_uv = (in_position + 1.0) * 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

    def get_gl_version(self) -> tuple[int, int]:
        return (4, 6)

    def get_gl_profile(self) -> str:
        return "core"


# =============================================================================
# OpenGL 3.3 Target (Older Desktop)
# =============================================================================


class OpenGL33Target(Target):
    """OpenGL 3.3 desktop target for older systems."""

    def version_directive(self) -> str | None:
        return "#version 330 core"

    def get_uniform_mapping(self) -> dict[str, str]:
        return {}

    def entry_point_wrapper(
        self,
        _stage: ShaderStage,
        entry_func: IRFunction,
        _inputs: list[IRVariable],
        outputs: list[IRVariable],
    ) -> list[str]:
        """Generate standard OpenGL main() wrapper."""
        lines = ["void main() {"]

        args = [param.name for param in entry_func.params]
        args_str = ", ".join(args)

        if entry_func.return_type:
            output_name = outputs[0].name if outputs else "fragColor"
            lines.append(f"    {output_name} = {entry_func.name}({args_str});")
        else:
            lines.append(f"    {entry_func.name}({args_str});")

        lines.append("}")
        return lines

    def get_vertex_shader(self) -> str:
        return """#version 330 core
in vec2 in_position;
out vec2 vs_uv;
void main() {
    vs_uv = (in_position + 1.0) * 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

    def get_gl_version(self) -> tuple[int, int]:
        return (3, 3)

    def get_gl_profile(self) -> str:
        return "core"


# =============================================================================
# Shadertoy Target
# =============================================================================


class ShadertoyTarget(Target):
    """Shadertoy platform target.

    Generates code that can be directly pasted into shadertoy.com:
    - No #version directive (Shadertoy adds this)
    - No uniform declarations (Shadertoy provides iResolution, iTime, etc.)
    - No in/out declarations
    - Only outputs user functions + mainImage wrapper

    This target is EXPORT-ONLY and cannot be used for local rendering.
    Use OpenGL46 or OpenGL33 for development, then export to Shadertoy.
    """

    def version_directive(self) -> str | None:
        # Shadertoy doesn't need version directive
        return None

    def is_export_only(self) -> bool:
        """Shadertoy target is for code export only, not rendering."""
        return True

    def get_uniform_mapping(self) -> dict[str, str]:
        return {
            "u_time": "iTime",
            "u_resolution": "iResolution.xy",
            "u_aspect": "(iResolution.x / iResolution.y)",
            "u_mouse_pos": "iMouse.xy",
            "u_mouse_uv": "(iMouse.xy / iResolution.xy)",
            "vs_uv": "(fragCoord / iResolution.xy)",
        }

    def entry_point_wrapper(
        self,
        _stage: ShaderStage,
        entry_func: IRFunction,
        _inputs: list[IRVariable],
        _outputs: list[IRVariable],
    ) -> list[str]:
        """Generate Shadertoy mainImage() wrapper only."""
        lines = []

        lines.append("void mainImage(out vec4 fragColor, in vec2 fragCoord) {")

        # Build argument list with mapped names
        mapping = self.get_uniform_mapping()
        args = [mapping.get(param.name, param.name) for param in entry_func.params]

        args_str = ", ".join(args)
        lines.append(f"    fragColor = {entry_func.name}({args_str});")
        lines.append("}")

        return lines

    def get_vertex_shader(self) -> str:
        # Not used - Shadertoy target is export-only
        raise NotImplementedError(
            "Shadertoy target is export-only and cannot be rendered locally. "
            "Use OpenGL46 or OpenGL33 for development."
        )

    def get_gl_version(self) -> tuple[int, int]:
        # Not used - Shadertoy target is export-only
        raise NotImplementedError(
            "Shadertoy target is export-only and cannot be rendered locally."
        )

    def get_gl_profile(self) -> str:
        # Not used - Shadertoy target is export-only
        raise NotImplementedError(
            "Shadertoy target is export-only and cannot be rendered locally."
        )


# =============================================================================
# WebGL 2.0 Target
# =============================================================================


class WebGL2Target(Target):
    """WebGL 2.0 target for browsers."""

    def version_directive(self) -> str | None:
        return "#version 300 es"

    def precision_qualifiers(self) -> list[str]:
        return ["precision highp float;", "precision highp int;"]

    def get_uniform_mapping(self) -> dict[str, str]:
        return {}

    def entry_point_wrapper(
        self,
        _stage: ShaderStage,
        entry_func: IRFunction,
        _inputs: list[IRVariable],
        outputs: list[IRVariable],
    ) -> list[str]:
        """Generate WebGL main() wrapper."""
        lines = ["void main() {"]

        args = [param.name for param in entry_func.params]
        args_str = ", ".join(args)

        if entry_func.return_type:
            output_name = outputs[0].name if outputs else "fragColor"
            lines.append(f"    {output_name} = {entry_func.name}({args_str});")
        else:
            lines.append(f"    {entry_func.name}({args_str});")

        lines.append("}")
        return lines

    def get_vertex_shader(self) -> str:
        return """#version 300 es
precision highp float;
in vec2 in_position;
out vec2 vs_uv;
void main() {
    vs_uv = (in_position + 1.0) * 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

    def get_gl_version(self) -> tuple[int, int]:
        return (3, 0)  # WebGL 2.0 maps to OpenGL ES 3.0

    def get_gl_profile(self) -> str:
        return "es"


# =============================================================================
# Target Type Enum and Factory
# =============================================================================


class TargetType(Enum):
    """Supported compilation targets."""

    OPENGL46 = auto()
    OPENGL33 = auto()
    SHADERTOY = auto()
    WEBGL2 = auto()

    def create(self) -> Target:
        """Create a Target instance for this type."""
        factories: dict[TargetType, type[Target]] = {
            TargetType.OPENGL46: OpenGL46Target,
            TargetType.OPENGL33: OpenGL33Target,
            TargetType.SHADERTOY: ShadertoyTarget,
            TargetType.WEBGL2: WebGL2Target,
        }
        return factories[self]()


# =============================================================================
# Convenience aliases
# =============================================================================

# Default target
DEFAULT_TARGET = TargetType.OPENGL46
