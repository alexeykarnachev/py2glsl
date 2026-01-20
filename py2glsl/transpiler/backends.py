"""Backend module for shader code generation."""

import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from py2glsl.transpiler.models import CollectedInfo, FunctionInfo, TranspilerError


class BackendType(Enum):
    """Supported GLSL backend types."""

    STANDARD = auto()
    SHADERTOY = auto()


@dataclass
class EntryPointConfig:
    """Configuration for shader entry point generation."""

    input_variables: dict[str, str] = field(default_factory=dict)
    output_variables: dict[str, str] = field(default_factory=dict)
    main_wrapper_template: str = ""


@dataclass
class BackendConfig:
    """Configuration for a GLSL backend."""

    name: str
    version_directive: str
    entry_point: EntryPointConfig
    predefined_uniforms: dict[str, str] = field(default_factory=dict)
    extensions: list[str] = field(default_factory=list)
    preprocessor_defines: dict[str, str | None] = field(default_factory=dict)
    additional_options: dict[str, Any] = field(default_factory=dict)


class Backend(ABC):
    """Base abstract interface for all shader backends."""

    @abstractmethod
    def generate_code(
        self, collected: CollectedInfo, main_func: str
    ) -> tuple[str, set[str]]:
        """Generate shader code from collected information."""
        pass


class GLSLBackend(Backend):
    """Base implementation for GLSL-based backends."""

    def __init__(self, config: BackendConfig):
        self.config = config

    def generate_code(
        self, collected: CollectedInfo, main_func: str
    ) -> tuple[str, set[str]]:
        """Generate GLSL code from collected information."""
        from loguru import logger

        logger.debug(f"Starting GLSL generation for main function: {main_func}")

        main_func_info = collected.functions[main_func]
        if not main_func_info.node.body:
            raise TranspilerError("Empty function body not supported in GLSL")

        version_lines = self._generate_version_directive()
        extension_lines = self._generate_extensions()
        define_lines = self._generate_preprocessor_defines()
        predefined_uniform_lines, predefined_uniforms = (
            self._generate_predefined_uniforms()
        )
        user_uniform_lines, user_uniforms = self._generate_uniforms(main_func_info)

        uniform_lines = predefined_uniform_lines + user_uniform_lines
        used_uniforms = predefined_uniforms.union(user_uniforms)

        global_lines = self._generate_globals(collected)
        struct_lines = self._generate_structs(collected)
        function_lines = self._generate_functions(collected, main_func)
        main_entry_lines = self.generate_entry_point(main_func, main_func_info)

        all_lines = (
            version_lines
            + extension_lines
            + define_lines
            + uniform_lines
            + global_lines
            + struct_lines
            + function_lines
            + main_entry_lines
        )

        return "\n".join(all_lines), used_uniforms

    def _generate_version_directive(self) -> list[str]:
        return [f"{self.config.version_directive}\n"]

    def _generate_extensions(self) -> list[str]:
        return [f"#extension {ext} : enable" for ext in self.config.extensions]

    def _generate_preprocessor_defines(self) -> list[str]:
        lines = []
        for name, value in self.config.preprocessor_defines.items():
            if name.startswith("precision "):
                lines.append(f"{name};")
            else:
                lines.append(f"#define {name} {value if value else ''}")
        return lines

    def _generate_predefined_uniforms(self) -> tuple[list[str], set[str]]:
        lines = []
        uniforms = set()
        for name, type_name in self.config.predefined_uniforms.items():
            lines.append(f"uniform {type_name} {name};")
            uniforms.add(name)
        return lines, uniforms

    @abstractmethod
    def generate_entry_point(
        self, main_func: str, main_func_info: FunctionInfo
    ) -> list[str]:
        """Generate shader entry point."""
        pass

    def _generate_uniforms(
        self, main_func_info: FunctionInfo
    ) -> tuple[list[str], set[str]]:
        lines = []
        used_uniforms = set()
        for i, arg in enumerate(main_func_info.node.args.args):
            if arg.arg != "vs_uv":
                param_type = main_func_info.param_types[i]
                lines.append(f"uniform {param_type} {arg.arg};")
                used_uniforms.add(arg.arg)
        return lines, used_uniforms

    def _generate_globals(self, collected: CollectedInfo) -> list[str]:
        if not collected.globals:
            return []
        lines = [""]
        for name, (type_name, value) in collected.globals.items():
            lines.append(f"const {type_name} {name} = {value};")
        return lines

    def _generate_structs(self, collected: CollectedInfo) -> list[str]:
        if not collected.structs:
            return []
        lines = [""]
        for struct_name, struct_def in collected.structs.items():
            lines.append(f"struct {struct_name} {{")
            for f in struct_def.fields:
                lines.append(f"    {f.type_name} {f.name};")
            lines.append("};")
        return lines

    def _generate_function(
        self,
        func_name: str,
        func_info: FunctionInfo,
        is_main: bool,
        collected: CollectedInfo,
    ) -> list[str]:
        if not func_info.return_type and not is_main:
            raise TranspilerError(
                f"Helper function '{func_name}' lacks return type annotation"
            )

        effective_return_type = (
            "vec4" if is_main and not func_info.return_type else func_info.return_type
        )

        node = func_info.node
        param_str = ", ".join(
            f"{p_type} {arg.arg}"
            for p_type, arg in zip(func_info.param_types, node.args.args, strict=False)
        )

        symbols = self._create_symbols_dict(func_info, collected)

        from py2glsl.transpiler.code_gen_stmt import generate_body

        body_lines = generate_body(node.body, symbols, collected)

        lines = []
        lines.append(f"{effective_return_type} {func_name}({param_str}) {{")
        for line in body_lines:
            lines.append(f"    {line}")
        lines.append("}")
        return lines

    def _create_symbols_dict(
        self, func_info: FunctionInfo, collected: CollectedInfo
    ) -> dict[str, str | None]:
        symbols = {
            arg.arg: p_type
            for arg, p_type in zip(
                func_info.node.args.args, func_info.param_types, strict=False
            )
        }
        for name, (type_name, _) in collected.globals.items():
            if type_name is not None:
                symbols[name] = type_name
        return symbols

    def _find_function_calls(
        self, ast_node: ast.AST, collected: CollectedInfo
    ) -> set[str]:
        called_functions = set()

        class FunctionCallVisitor(ast.NodeVisitor):
            def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
                if (
                    isinstance(node.func, ast.Name)
                    and node.func.id in collected.functions
                ):
                    called_functions.add(node.func.id)
                self.generic_visit(node)

        FunctionCallVisitor().visit(ast_node)
        return called_functions

    def _generate_functions(
        self, collected: CollectedInfo, main_func: str
    ) -> list[str]:
        lines = [""]
        dependencies = {}
        for func_name, func_info in collected.functions.items():
            dependencies[func_name] = self._find_function_calls(
                func_info.node, collected
            )

        emitted = set()
        function_lines = []

        def emit_function(name: str) -> None:
            if name in emitted:
                return
            for dep in dependencies.get(name, set()):
                emit_function(dep)
            is_main = name == main_func
            func_info = collected.functions[name]
            func_lines = self._generate_function(name, func_info, is_main, collected)
            function_lines.extend(func_lines)
            function_lines.append("")
            emitted.add(name)

        emit_function(main_func)
        for func_name in collected.functions:
            emit_function(func_name)

        lines.extend(function_lines)
        return lines


class StandardGLSLBackend(GLSLBackend):
    """Standard GLSL backend."""

    def generate_entry_point(
        self, main_func: str, main_func_info: FunctionInfo
    ) -> list[str]:
        lines = ["\nin vec2 vs_uv;\nout vec4 fragColor;\n\nvoid main() {"]
        main_call_args = [arg.arg for arg in main_func_info.node.args.args]
        main_call_str = ", ".join(main_call_args)
        lines.append(f"    fragColor = {main_func}({main_call_str});")
        lines.append("}")
        return lines


class ShadertoyBackend(GLSLBackend):
    """Shadertoy-specific GLSL backend."""

    def __init__(self, config: BackendConfig):
        super().__init__(config)
        self.uniform_mapping: dict[str, str] = {
            "u_time": "iTime",
            "u_resolution": "iResolution.xy",
            "u_mouse_pos": "iMouse.xy",
            "u_aspect": "iResolution.x / iResolution.y",
        }

    def generate_entry_point(
        self, main_func: str, main_func_info: FunctionInfo
    ) -> list[str]:
        lines = ["\n// Shadertoy compatible mainImage function"]
        lines.append("void mainImage(out vec4 fragColor, in vec2 fragCoord) {")
        lines.append("    vec2 vs_uv = fragCoord / iResolution.xy;")

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

        lines.append("\n// Standard entry point for OpenGL")
        lines.append("in vec2 vs_uv;")
        lines.append("out vec4 fragColor;")
        lines.append("void main() {")
        lines.append("    vec2 fragCoord = vs_uv * iResolution.xy;")
        lines.append("    mainImage(fragColor, fragCoord);")
        lines.append("}")
        return lines


def _create_standard_backend() -> StandardGLSLBackend:
    config = BackendConfig(
        name="standard",
        version_directive="#version 460 core",
        entry_point=EntryPointConfig(
            input_variables={"vs_uv": "vec2"}, output_variables={"fragColor": "vec4"}
        ),
    )
    return StandardGLSLBackend(config)


def _create_shadertoy_backend() -> ShadertoyBackend:
    config = BackendConfig(
        name="shadertoy",
        version_directive="#version 330 core",
        entry_point=EntryPointConfig(
            input_variables={"fragCoord": "vec2"},
            output_variables={"fragColor": "vec4"},
        ),
        predefined_uniforms={
            "iResolution": "vec3",
            "iTime": "float",
            "iTimeDelta": "float",
            "iFrame": "int",
            "iMouse": "vec4",
            "iDate": "vec4",
            "iSampleRate": "float",
            "iChannel0": "sampler2D",
            "iChannel1": "sampler2D",
            "iChannel2": "sampler2D",
            "iChannel3": "sampler2D",
            "iChannelTime": "float[4]",
            "iChannelResolution": "vec3[4]",
        },
        preprocessor_defines={
            "precision mediump float": None,
            "precision mediump int": None,
            "precision mediump sampler2D": None,
        },
    )
    return ShadertoyBackend(config)


def create_backend(backend_type: BackendType = BackendType.STANDARD) -> Backend:
    """Create a backend instance based on type."""
    if backend_type == BackendType.STANDARD:
        return _create_standard_backend()
    elif backend_type == BackendType.SHADERTOY:
        return _create_shadertoy_backend()
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}")
