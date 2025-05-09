import ast
from abc import ABC, abstractmethod

from py2glsl.transpiler.backends.models import BackendConfig
from py2glsl.transpiler.errors import TranspilerError
from py2glsl.transpiler.models import CollectedInfo, FunctionInfo


class Backend(ABC):
    """Base abstract interface for all shader backends."""

    @abstractmethod
    def generate_code(
        self, collected: CollectedInfo, main_func: str
    ) -> tuple[str, set[str]]:
        """Generate shader code from collected information.

        Args:
            collected: Information about functions, structs, and globals
            main_func: Name of the main function to use as shader entry point

        Returns:
            Tuple of (generated shader code, set of used uniform variables)
        """
        pass


class GLSLBackend(Backend):
    """Base implementation for GLSL-based backends."""

    def __init__(self, config: BackendConfig):
        self.config = config

    def generate_code(
        self, collected: CollectedInfo, main_func: str
    ) -> tuple[str, set[str]]:
        """Generate GLSL code from collected information.

        Args:
            collected: Information about functions, structs, and globals
            main_func: Name of the main function to use as shader entry point

        Returns:
            Tuple of (generated shader code, set of used uniform variables)
        """
        from loguru import logger

        logger.debug(f"Starting GLSL generation for main function: {main_func}")

        # Validate main function
        main_func_info = collected.functions[main_func]
        if not main_func_info.node.body:
            from py2glsl.transpiler.errors import TranspilerError

            raise TranspilerError("Empty function body not supported in GLSL")

        # Generate each section of the shader
        version_lines = self.generate_version_directive()
        extension_lines = self.generate_extensions()
        define_lines = self.generate_preprocessor_defines()

        # Get predefined uniforms from backend
        predefined_uniform_lines, predefined_uniforms = (
            self.generate_predefined_uniforms()
        )

        # Get function-specific uniforms
        user_uniform_lines, user_uniforms = self._generate_uniforms(main_func_info)

        # Combine all uniforms
        uniform_lines = predefined_uniform_lines + user_uniform_lines
        used_uniforms = predefined_uniforms.union(user_uniforms)

        global_lines = self._generate_globals(collected)
        struct_lines = self._generate_structs(collected)
        function_lines = self._generate_functions(collected, main_func)
        main_entry_lines = self.generate_entry_point(main_func, main_func_info)

        # Combine all sections
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

        # Join lines into a single string
        glsl_code = "\n".join(all_lines)
        return glsl_code, used_uniforms

    def generate_version_directive(self) -> list[str]:
        """Generate version directive."""
        return [f"{self.config.version_directive}\n"]

    def generate_extensions(self) -> list[str]:
        """Generate extension directives."""
        return [f"#extension {ext} : enable" for ext in self.config.extensions]

    def generate_preprocessor_defines(self) -> list[str]:
        """Generate preprocessor defines."""
        lines = []
        for name, value in self.config.preprocessor_defines.items():
            if name.startswith("precision "):
                # Handle precision statements directly (not as #define)
                lines.append(f"{name};")
            else:
                # Regular preprocessor define
                lines.append(f"#define {name} {value if value else ''}")
        return lines

    def generate_predefined_uniforms(self) -> tuple[list[str], set[str]]:
        """Generate predefined uniform declarations."""
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
        """Generate shader entry point.

        Args:
            main_func: The name of the main function
            main_func_info: Information about the main function

        Returns:
            List of lines for the entry point
        """
        pass

    def _generate_uniforms(
        self, main_func_info: FunctionInfo
    ) -> tuple[list[str], set[str]]:
        """Generate uniform variable declarations.

        Args:
            main_func_info: Information about the main function

        Returns:
            Tuple of (list of uniform declaration lines, set of uniform names)
        """
        lines = []
        used_uniforms = set()

        for i, arg in enumerate(main_func_info.node.args.args):
            if arg.arg != "vs_uv":  # vs_uv is a special input variable, not a uniform
                param_type = main_func_info.param_types[i]
                lines.append(f"uniform {param_type} {arg.arg};")
                used_uniforms.add(arg.arg)

        return lines, used_uniforms

    def _generate_globals(self, collected: CollectedInfo) -> list[str]:
        """Generate global constant declarations.

        Args:
            collected: Information about functions, structs, and globals

        Returns:
            List of global constant declaration lines
        """
        if not collected.globals:
            return []

        lines = [""]  # Start with blank line for readability
        for name, (type_name, value) in collected.globals.items():
            lines.append(f"const {type_name} {name} = {value};")

        return lines

    def _generate_structs(self, collected: CollectedInfo) -> list[str]:
        """Generate struct definitions.

        Args:
            collected: Information about functions, structs, and globals

        Returns:
            List of struct definition lines
        """
        if not collected.structs:
            return []

        lines = [""]  # Start with blank line for readability
        for struct_name, struct_def in collected.structs.items():
            lines.append(f"struct {struct_name} {{")
            for field in struct_def.fields:
                lines.append(f"    {field.type_name} {field.name};")
            lines.append("};")

        return lines

    def _generate_function(
        self,
        func_name: str,
        func_info: FunctionInfo,
        is_main: bool,
        collected: CollectedInfo,
    ) -> list[str]:
        """Generate a function definition.

        Args:
            func_name: Name of the function
            func_info: Information about the function
            is_main: Whether this is the main shader function
            collected: Information about functions, structs, and globals

        Returns:
            List of function definition lines

        Raises:
            TranspilerError: If the function doesn't have a return type annotation
        """
        # Check if function has a return type annotation
        if not func_info.return_type and not is_main:
            raise TranspilerError(
                f"Helper function '{func_name}' lacks return type annotation"
            )

        # Default to vec4 for main function without return type
        effective_return_type = (
            "vec4" if is_main and not func_info.return_type else func_info.return_type
        )

        node = func_info.node
        param_str = ", ".join(
            f"{p_type} {arg.arg}"
            for p_type, arg in zip(func_info.param_types, node.args.args, strict=False)
        )

        # Initialize symbols dictionary
        symbols = self._create_symbols_dict(func_info, collected)

        # Generate function body
        from py2glsl.transpiler.code_gen_stmt import generate_body

        body_lines = generate_body(node.body, symbols, collected)

        # Format the function definition
        lines = []
        lines.append(f"{effective_return_type} {func_name}({param_str}) {{")
        for line in body_lines:
            lines.append(f"    {line}")
        lines.append("}")

        return lines

    def _create_symbols_dict(
        self, func_info: FunctionInfo, collected: CollectedInfo
    ) -> dict[str, str | None]:
        """Create a symbols dictionary for a function.

        Args:
            func_info: Information about the function
            collected: Information about functions, structs, and globals

        Returns:
            Dictionary mapping variable names to their types
        """
        # Initialize with function parameters
        symbols = {
            arg.arg: p_type
            for arg, p_type in zip(
                func_info.node.args.args, func_info.param_types, strict=False
            )
        }

        # Add global constants to the symbols table
        for name, (type_name, _) in collected.globals.items():
            if type_name is not None:
                symbols[name] = type_name

        return symbols

    def _find_function_calls(
        self, ast_node: ast.AST, collected: CollectedInfo
    ) -> set[str]:
        """Find all function calls within an AST node.

        Args:
            ast_node: The AST node to search
            collected: Information about functions, structs, and globals

        Returns:
            Set of function names that are called within the node
        """
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
        """Generate all function definitions.

        Args:
            collected: Information about functions, structs, and globals
            main_func: Name of the main function

        Returns:
            List of function definition lines
        """
        lines = [""]  # Start with blank line for readability

        # Build dependency graph
        dependencies = {}
        for func_name, func_info in collected.functions.items():
            dependencies[func_name] = self._find_function_calls(
                func_info.node, collected
            )

        # Topological sort
        emitted = set()
        function_lines = []

        def emit_function(name: str) -> None:
            if name in emitted:
                return

            # First emit dependencies
            for dep in dependencies.get(name, set()):
                emit_function(dep)

            # Then emit this function
            is_main = name == main_func
            func_info = collected.functions[name]
            func_lines = self._generate_function(name, func_info, is_main, collected)
            function_lines.extend(func_lines)
            function_lines.append("")  # Add blank line for readability
            emitted.add(name)

        # Start with main function and its dependencies
        emit_function(main_func)

        # Add any remaining functions that weren't dependencies of main
        for func_name in collected.functions:
            emit_function(func_name)

        lines.extend(function_lines)
        return lines
