import ast
import inspect
import keyword
import re
from dataclasses import dataclass
from typing import Callable

from py2glsl.glsl.types import mat3, mat4, vec2, vec3, vec4
from py2glsl.transpiler.type_system import TypeInferer

from .utils import extract_function_body


@dataclass
class TranspilerResult:
    vertex_src: str
    fragment_src: str


class GLSLCodeError(ValueError):
    pass


class GLSLBuilder:
    def __init__(self):
        self.uniforms: list[str] = []
        self.vertex_functions: list[str] = []
        self.fragment_functions: list[str] = []
        self.vertex_interface: list[str] = []
        self.fragment_interface: list[str] = []
        self.outputs: list[str] = []
        self.structs: list[str] = []
        self.vertex_attributes: list[str] = []
        self.vertex_main_body: list[str] = []
        self.fragment_main_body: list[str] = []
        self.declared_names = set()

    def _validate_identifier(self, name: str):
        if not name:
            raise GLSLCodeError("Identifier cannot be empty")
        if keyword.iskeyword(name):
            raise GLSLCodeError(f"Reserved keyword cannot be used: {name}")
        if not (name[0].isalpha() or name[0] == "_"):
            raise GLSLCodeError(f"Identifier must start with letter/underscore: {name}")
        if not all(c.isalnum() or c == "_" for c in name):
            raise GLSLCodeError(f"Invalid characters in identifier: {name}")
        if name.startswith("gl_"):
            raise GLSLCodeError(f"Reserved GLSL prefix 'gl_' in identifier: {name}")

    def add_uniform(self, name: str, type_: str):
        self._validate_identifier(name)
        if name in {"float", "int", "vec2", "mat4"}:  # TODO: Add more reserved words
            raise GLSLCodeError(f"Reserved GLSL keyword: {name}")

        if name in self.declared_names:
            raise GLSLCodeError(f"Duplicate declaration: {name}")
        self.declared_names.add(name)

        self.uniforms.append(f"uniform {type_} {name};")

    def add_vertex_attribute(self, location: int, type_: str, name: str):
        if name.startswith("gl_"):
            raise GLSLCodeError(f"Reserved GLSL prefix 'gl_' in attribute '{name}'")
        self._validate_identifier(name)
        if name in self.declared_names:
            raise GLSLCodeError(f"Duplicate attribute: {name}")
        self.declared_names.add(name)

        self.vertex_attributes.append(
            f"layout(location = {location}) in {type_} {name};"
        )

    def add_interface_block(
        self, block_name: str, qualifier: str, members: dict[str, str]
    ):
        self._validate_identifier(block_name)
        members_str = "\n    ".join(
            [f"{type_} {name};" for name, type_ in members.items()]
        )
        block = f"{qualifier} {block_name} {{\n    {members_str}\n}} {block_name.lower()}Out;\n"

        if qualifier == "out":
            self.vertex_interface.append(block)
        elif qualifier == "in":
            self.fragment_interface.append(block)
        else:
            raise GLSLCodeError(f"Invalid interface qualifier: {qualifier}")

    def add_output(self, name: str, type_: str):
        self._validate_identifier(name)
        self.outputs.append(f"out {type_} {name};")

    def add_struct(self, name: str, members: dict[str, str]):
        self._validate_identifier(name)
        members_str = "\n    ".join(
            [f"{type_} {name};" for name, type_ in members.items()]
        )
        self.structs.append(f"struct {name} {{\n    {members_str}\n}};")

    def add_function(
        self,
        return_type: str,
        name: str,
        parameters: list[tuple[str, str]],
        body: list[str],
        shader_type: str,
    ):
        """Add a GLSL function with validation"""
        self._validate_identifier(name)

        # Process body lines to convert Python-style comments to GLSL
        processed_body = []
        for line in body:
            # Preserve indentation while converting comments
            stripped = line.lstrip()
            if stripped.startswith("#"):
                indent = line[: len(line) - len(stripped)]
                glsl_comment = f"{indent}//{stripped[1:]}"
                processed_body.append(glsl_comment)
            else:
                processed_body.append(line)

        self._validate_swizzle_operations(parameters, processed_body)

        params = self._format_parameters(parameters)
        body_str = self._format_body(processed_body)

        func_str = f"{return_type} {name}({params}) {{\n{body_str}\n}}"

        # Store in appropriate list
        if shader_type == "fragment":
            self.fragment_functions.append(func_str)
        else:
            self.vertex_functions.append(func_str)

    def _validate_swizzle_operations(
        self, parameters: list[tuple[str, str]], body: list[str]
    ):
        """Check for invalid swizzle patterns in function body"""
        component_indices = {"x": 0, "y": 1, "z": 2, "w": 3}
        param_types = {name: type_ for type_, name in parameters}

        for line in body:
            # Match swizzle patterns like .xyz or .rgba
            swizzle_match = re.search(r"\b(\w+)\.([xyzw]{2,})", line)
            if swizzle_match:
                var_name = swizzle_match.group(1)
                swizzle = swizzle_match.group(2)
                var_type = param_types.get(var_name, "")

                if var_type.startswith("vec"):
                    max_components = int(var_type[3])
                    if any(component_indices[c] >= max_components for c in swizzle):
                        raise GLSLCodeError(
                            f"Invalid swizzle '{swizzle}' for {var_type} in line: {line}"
                        )

    def _format_parameters(self, parameters: list[tuple[str, str]]) -> str:
        """Format function parameters with type annotations"""
        return ", ".join([f"{type_} {name}" for type_, name in parameters])

    def _format_body(self, body: list[str]) -> str:
        """Indent and join body lines with proper GLSL semicolons"""
        body_str = "    " + "\n    ".join(body)

        # Convert Python ** to GLSL pow()
        body_str = re.sub(r"(\w+)\s*\*\*\s*(\w+)", r"pow(\1, \2)", body_str)

        # Add missing semicolons using more comprehensive regex
        body_str = re.sub(r"(\w|\))(\s*(?=\n|//|$))", r"\1;", body_str)

        # Fix any double semicolons
        body_str = body_str.replace(";;", ";")

        return body_str

    def configure_shader_transpiler(
        self,
        uniforms: dict[str, str],
        attributes: dict[str, str],
        func_name: str,
        shader_body: list[str],
        extra_functions: list[Callable] | None = None,
    ):
        """Configure the builder for shader transpilation with dependencies."""
        # Add vertex attribute for position
        self.add_vertex_attribute(0, "vec2", "a_pos")

        # Vertex outputs -> fragment inputs
        for name, type_ in attributes.items():
            self._validate_identifier(name)
            self.vertex_interface.append(f"out {type_} {name};")
            self.fragment_interface.append(f"in {type_} {name};")

        # Uniform declarations
        for name, type_ in uniforms.items():
            self.add_uniform(name, type_)

        # Final output
        self.add_output("fs_color", "vec4")

        # Add dependent functions first
        for func in extra_functions or []:
            self._add_function_from_callable(func)

        # Main shader function parameters
        params = [(type_, name) for name, type_ in attributes.items()]
        params += [(type_, name) for name, type_ in uniforms.items()]

        # Add main shader function
        self.add_function(
            return_type="vec4",
            name=func_name,
            parameters=params,
            body=shader_body,
            shader_type="fragment",
        )

        # Vertex main body
        self.vertex_main_body = [
            "gl_Position = vec4(a_pos, 0.0, 1.0);",
            *[f"{name} = a_pos * 0.5 + 0.5;" for name in attributes],
        ]

        # Fragment main body
        args = list(attributes.keys()) + list(uniforms.keys())
        self.fragment_main_body = [f"fs_color = {func_name}({', '.join(args)});"]

    def _add_function_from_callable(self, func: Callable):
        """Add a function from Python callable to GLSL builder."""
        sig = inspect.signature(func)

        # Get return type
        return_type = sig.return_annotation.__name__

        # Process parameters
        params = []
        for param in sig.parameters.values():
            param_type = param.annotation.__name__
            params.append((param_type, param.name))

        # Get typed function body
        body = extract_function_body(func)
        processed_body = []
        inferer = TypeInferer()

        for line in body:
            try:
                tree = ast.parse(line)
                inferer.visit(tree)

                if isinstance(tree.body[0], ast.AnnAssign):
                    # Convert Python type hints to GLSL declarations
                    ann = tree.body[0].annotation
                    target = tree.body[0].target.id
                    glsl_type = self._pytype_to_glsl(inferer._resolve_annotation(ann))
                    value = ast.unparse(tree.body[0].value)
                    processed_body.append(f"{glsl_type} {target} = {value};")
                else:
                    processed_body.append(line)
            except:
                processed_body.append(line)

        self.add_function(
            return_type=return_type,
            name=func.__name__,
            parameters=params,
            body=processed_body,
            shader_type="fragment",
        )

    def _pytype_to_glsl(self, py_type: type) -> str:
        """Get GLSL type name from Python type using __name__ attribute"""
        if py_type in (float, int, bool):
            return py_type.__name__
        return getattr(py_type, "__name__", "float")

    def build_vertex_shader(self) -> str:
        components = [
            "#version 460 core",
            *self.vertex_attributes,
            *self.vertex_interface,
            *self.uniforms,
            *self.vertex_functions,
            "void main() {",
            *[f"    {line}" for line in self.vertex_main_body],
            "}",
        ]
        return "\n".join(components)

    def build_fragment_shader(self) -> str:
        components = [
            "#version 460 core",
            "precision mediump float;",
            *self.fragment_interface,
            *self.uniforms,
            *self.outputs,
            *self.fragment_functions,
            "void main() {",
            *[f"    {line}" for line in self.fragment_main_body],
            "}",
        ]
        return "\n".join(components)

    def build(self) -> TranspilerResult:
        return TranspilerResult(
            vertex_src=self.build_vertex_shader(),
            fragment_src=self.build_fragment_shader(),
        )
