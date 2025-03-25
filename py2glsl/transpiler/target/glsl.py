"""GLSL target language implementation.

This module implements the TargetLanguage interface for GLSL.
"""


from py2glsl.transpiler.core.ast_processor import DependencyResolver, SymbolTable
from py2glsl.transpiler.core.interfaces import (
    LanguageConfig,
    SymbolMapper,
    TargetLanguage,
    TypeMapping,
)
from py2glsl.transpiler.models import CollectedInfo, FunctionInfo


class GLSLSymbolMapper(SymbolMapper):
    """Symbol mapper for GLSL."""

    def __init__(self) -> None:
        """Initialize with GLSL type mappings."""
        self.type_mappings: dict[str, str] = {
            "float": "float",
            "int": "int",
            "bool": "bool",
            "vec2": "vec2",
            "vec3": "vec3",
            "vec4": "vec4",
            "mat2": "mat2",
            "mat3": "mat3",
            "mat4": "mat4",
        }

        self.function_mappings: dict[str, str] = {
            "sin": "sin",
            "cos": "cos",
            "tan": "tan",
            "asin": "asin",
            "acos": "acos",
            "atan": "atan",
            "length": "length",
            "normalize": "normalize",
            "dot": "dot",
            "cross": "cross",
            "mix": "mix",
            "clamp": "clamp",
            "min": "min",
            "max": "max",
            "abs": "abs",
            "pow": "pow",
            "sqrt": "sqrt",
            "floor": "floor",
            "ceil": "ceil",
            "fract": "fract",
            "mod": "mod",
            "step": "step",
            "smoothstep": "smoothstep",
            "radians": "radians",
            "degrees": "degrees",
        }

        self.operator_mappings: dict[str, str] = {
            "+": "+",
            "-": "-",
            "*": "*",
            "/": "/",
            "==": "==",
            "!=": "!=",
            "<": "<",
            ">": ">",
            "<=": "<=",
            ">=": ">=",
            "and": "&&",
            "or": "||",
            "not": "!",
        }

    def map_type(self, python_type: str) -> str:
        """Map a Python type to a GLSL type."""
        return self.type_mappings.get(python_type, python_type)

    def map_function(self, python_function: str) -> str:
        """Map a Python function to a GLSL function."""
        return self.function_mappings.get(python_function, python_function)

    def map_operator(self, python_operator: str) -> str:
        """Map a Python operator to a GLSL operator."""
        return self.operator_mappings.get(python_operator, python_operator)


class GLSLStandardDialect(TargetLanguage):
    """Standard GLSL target language implementation."""

    def __init__(self, version: str = "460 core") -> None:
        """Initialize with GLSL version.

        Args:
            version: GLSL version string
        """
        self.version = version
        self.symbol_mapper = GLSLSymbolMapper()

    def get_config(self) -> LanguageConfig:
        """Get GLSL language configuration."""
        type_mappings = {
            "float": TypeMapping("float", "float", "0.0"),
            "int": TypeMapping("int", "int", "0"),
            "bool": TypeMapping("bool", "bool", "false"),
            "vec2": TypeMapping("vec2", "vec2", "vec2(0.0)"),
            "vec3": TypeMapping("vec3", "vec3", "vec3(0.0)"),
            "vec4": TypeMapping("vec4", "vec4", "vec4(0.0)"),
            "mat2": TypeMapping("mat2", "mat2", "mat2(0.0)"),
            "mat3": TypeMapping("mat3", "mat3", "mat3(0.0)"),
            "mat4": TypeMapping("mat4", "mat4", "mat4(0.0)"),
        }

        return LanguageConfig(
            name="GLSL",
            file_extension=".glsl",
            version=self.version,
            type_mappings=type_mappings,
        )

    def get_symbol_mapper(self) -> SymbolMapper:
        """Get the GLSL symbol mapper."""
        return self.symbol_mapper

    def generate_code(
        self, collected: CollectedInfo, main_func: str
    ) -> tuple[str, set[str]]:
        """Generate GLSL code from collected information.

        Args:
            collected: Information about functions, structs, and globals
            main_func: Name of the main function to use as shader entry point

        Returns:
            Tuple of (generated GLSL code, set of used uniform variables)
        """
        # Validate main function
        main_func_info = collected.functions[main_func]
        if not main_func_info.node.body:
            from py2glsl.transpiler.errors import TranspilerError

            raise TranspilerError("Empty function body not supported in GLSL")

        # Generate each section of the shader
        version_lines = self._generate_version_directive()
        result = self._generate_predefined_uniforms()
        predefined_uniform_lines, predefined_uniforms = result
        user_uniform_lines, user_uniforms = self._generate_uniforms(main_func_info)

        # Combine all uniforms
        uniform_lines = predefined_uniform_lines + user_uniform_lines
        used_uniforms = predefined_uniforms.union(user_uniforms)

        global_lines = self._generate_globals(collected)
        struct_lines = self._generate_structs(collected)

        # Order functions by dependencies
        resolver = DependencyResolver(collected)
        ordered_functions = resolver.get_ordered_functions(main_func)
        function_lines = self._generate_functions(collected, ordered_functions)

        # Generate entry point
        main_entry_lines = self._generate_entry_point(main_func, main_func_info)

        # Combine all sections
        all_lines = (
            version_lines
            + uniform_lines
            + global_lines
            + struct_lines
            + function_lines
            + main_entry_lines
        )

        # Join lines into a single string
        glsl_code = "\n".join(all_lines)
        return glsl_code, used_uniforms

    def _generate_version_directive(self) -> list[str]:
        """Generate GLSL version directive."""
        return [f"#version {self.version}\n"]

    def _generate_predefined_uniforms(self) -> tuple[list[str], set[str]]:
        """Generate predefined uniform declarations."""
        # Standard GLSL has no predefined uniforms
        return [], set()

    def _generate_uniforms(
        self, main_func_info: FunctionInfo
    ) -> tuple[list[str], set[str]]:
        """Generate uniform variable declarations from main function parameters."""
        lines = []
        used_uniforms = set()

        for i, arg in enumerate(main_func_info.node.args.args):
            if arg.arg != "vs_uv":  # vs_uv is a special input variable, not a uniform
                param_type = main_func_info.param_types[i]
                if param_type is not None:
                    mapped_type = self.symbol_mapper.map_type(param_type)
                    lines.append(f"uniform {mapped_type} {arg.arg};")
                else:
                    # Default to float for parameters without type annotations
                    lines.append(f"uniform float {arg.arg};")
                used_uniforms.add(arg.arg)

        return lines, used_uniforms

    def _generate_globals(self, collected: CollectedInfo) -> list[str]:
        """Generate global constant declarations."""
        if not collected.globals:
            return []

        lines = [""]  # Start with blank line for readability
        for name, (type_name, value) in collected.globals.items():
            # Since type_name is str according to the model, we can map directly
            mapped_type = self.symbol_mapper.map_type(type_name)
            lines.append(f"const {mapped_type} {name} = {value};")

        return lines

    def _generate_structs(self, collected: CollectedInfo) -> list[str]:
        """Generate struct definitions."""
        if not collected.structs:
            return []

        lines = [""]  # Start with blank line for readability
        for struct_name, struct_def in collected.structs.items():
            lines.append(f"struct {struct_name} {{")
            for field in struct_def.fields:
                # Map the field type directly
                mapped_type = self.symbol_mapper.map_type(field.type_name)
                lines.append(f"    {mapped_type} {field.name};")
            lines.append("};")

        return lines

    def _generate_functions(
        self, collected: CollectedInfo, ordered_functions: list[str]
    ) -> list[str]:
        """Generate function definitions in dependency order."""
        lines = [""]  # Start with blank line for readability
        function_lines = []

        # Generate functions in dependency order
        for func_name in ordered_functions:
            func_info = collected.functions[func_name]
            is_main = func_name == ordered_functions[0]  # First function is main
            # Generate the function code
            func_lines = self._generate_function(
                func_name, func_info, is_main, collected
            )
            function_lines.extend(func_lines)
            function_lines.append("")  # Add blank line for readability

        lines.extend(function_lines)
        return lines

    def _generate_function(
        self,
        func_name: str,
        func_info: FunctionInfo,
        is_main: bool,
        collected: CollectedInfo,
    ) -> list[str]:
        """Generate a single function definition."""
        # Check if function has a return type annotation
        if not func_info.return_type and not is_main:
            from py2glsl.transpiler.errors import TranspilerError

            raise TranspilerError(
                f"Helper function '{func_name}' lacks return type annotation"
            )

        # Default to vec4 for main function without return type
        effective_return_type = (
            "vec4" if is_main and not func_info.return_type else func_info.return_type
        )
        # Make sure we have a non-None type
        if effective_return_type is None:
            effective_return_type = "vec4"
        mapped_return_type = self.symbol_mapper.map_type(effective_return_type)

        node = func_info.node
        # Build the parameter string with type checking
        params = []
        for p_type, arg in zip(func_info.param_types, node.args.args, strict=False):
            if p_type is not None:
                params.append(f"{self.symbol_mapper.map_type(p_type)} {arg.arg}")
            else:
                # Default to float if no type is specified
                params.append(f"float {arg.arg}")
        param_str = ", ".join(params)

        # Initialize symbol table
        symbols = SymbolTable()
        # Add function parameters
        for arg, p_type in zip(node.args.args, func_info.param_types, strict=False):
            symbols.add_symbol(arg.arg, p_type)
        # Add global constants
        for name, (type_name, _) in collected.globals.items():
            if type_name is not None:
                symbols.add_symbol(name, type_name)

        # Generate function body using the existing code generator
        from py2glsl.transpiler.code_gen_stmt import generate_body

        body_lines = generate_body(node.body, symbols.current_scope(), collected)

        # Format the function definition
        lines = []
        lines.append(f"{mapped_return_type} {func_name}({param_str}) {{")
        for line in body_lines:
            lines.append(f"    {line}")
        lines.append("}")

        return lines

    def _generate_entry_point(
        self, main_func: str, main_func_info: FunctionInfo
    ) -> list[str]:
        """Generate standard GLSL entry point."""
        lines = ["\nin vec2 vs_uv;\nout vec4 fragColor;\n\nvoid main() {"]
        main_call_args = [arg.arg for arg in main_func_info.node.args.args]
        main_call_str = ", ".join(main_call_args)
        lines.append(f"    fragColor = {main_func}({main_call_str});")
        lines.append("}")
        return lines
