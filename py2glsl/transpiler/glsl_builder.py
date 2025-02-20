import keyword
from typing import Dict, List, Tuple


class GLSLCodeError(ValueError):
    pass


class GLSLBuilder:
    def __init__(self):
        self.uniforms: List[str] = []
        self.vertex_interface: List[str] = []
        self.fragment_interface: List[str] = []
        self.outputs: List[str] = []
        self.structs: List[str] = []
        self.functions: List[str] = []
        self.vertex_attributes: List[str] = []
        self.vertex_main_body: List[str] = []
        self.fragment_main_body: List[str] = []
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
        self._validate_identifier(name)
        self.vertex_attributes.append(
            f"layout(location = {location}) in {type_} {name};"
        )

    def add_interface_block(
        self, block_name: str, qualifier: str, members: Dict[str, str]
    ):
        self._validate_identifier(block_name)
        members_str = "\n    ".join(
            [f"{type_} {name};" for name, type_ in members.items()]
        )
        block = f"{qualifier} {block_name} {{\n    {members_str}\n}};"

        if qualifier == "out":
            self.vertex_interface.append(block)
        elif qualifier == "in":
            self.fragment_interface.append(block)
        else:
            raise GLSLCodeError(f"Invalid interface qualifier: {qualifier}")

    def add_output(self, name: str, type_: str):
        self._validate_identifier(name)
        self.outputs.append(f"out {type_} {name};")

    def add_struct(self, name: str, members: Dict[str, str]):
        self._validate_identifier(name)
        members_str = "\n    ".join(
            [f"{type_} {name};" for name, type_ in members.items()]
        )
        self.structs.append(f"struct {name} {{\n    {members_str}\n}};")

    def add_function(
        self,
        return_type: str,
        name: str,
        parameters: List[Tuple[str, str]],
        body: List[str],
    ):
        """Add a GLSL function with validation"""
        self._validate_identifier(name)
        self._validate_swizzle_operations(parameters, body)

        params = self._format_parameters(parameters)
        body_str = self._format_body(body)
        self.functions.append(f"{return_type} {name}({params}) {{\n{body_str}\n}}")

    def _validate_swizzle_operations(
        self, parameters: List[Tuple[str, str]], body: List[str]
    ):
        """Check for invalid swizzle patterns in function body"""
        param_types = {name: type_ for type_, name in parameters}

        for line in body:
            if "." in line and any(c in "xyzw" for c in line.split(".")[1]):
                var_name = line.split(".")[0].strip()
                var_type = param_types.get(var_name, "")

                if var_type.startswith("vec"):
                    max_components = int(var_type[3])
                    swizzle = line.split(".")[1].split()[0]  # Get swizzle components
                    if any(ord(c) - ord("x") >= max_components for c in swizzle):
                        raise GLSLCodeError(
                            f"Invalid swizzle '{swizzle}' for {var_type}"
                        )

    def _format_parameters(self, parameters: List[Tuple[str, str]]) -> str:
        """Format function parameters with type annotations"""
        return ", ".join([f"{type_} {name}" for type_, name in parameters])

    def _format_body(self, body: List[str]) -> str:
        """Indent and join body lines"""
        return "    " + "\n    ".join(body)

    def configure_shader_transpiler(
        self,
        uniforms: Dict[str, str],
        attributes: Dict[str, str],
        func_name: str,
        shader_body: List[str],
    ):
        # Add vertex attributes
        for idx, (name, type_) in enumerate(attributes.items()):
            self.add_vertex_attribute(idx, type_, name)

        # Create interface blocks
        self.add_interface_block("VertexData", "out", attributes)
        self.add_interface_block("VertexData", "in", attributes)
        self.add_output("fs_color", "vec4")

        # Add uniforms
        for name, type_ in uniforms.items():
            self.add_uniform(name, type_)

        # Build shader function
        params = [(type_, name) for name, type_ in attributes.items()] + [
            (type_, name) for name, type_ in uniforms.items()
        ]

        self.add_function(
            return_type="vec4", name=func_name, parameters=params, body=shader_body
        )

        # Vertex main
        self.vertex_main_body = [
            *[f"VertexData.{name} = {name};" for name in attributes],
            (
                "gl_Position = vec4(VertexData.vs_uv, 0.0, 1.0);"
                if "vs_uv" in attributes
                else "gl_Position = vec4(0.0);"
            ),
        ]

        # Fragment main
        args = [f"VertexData.{name}" for name in attributes] + [
            name for name in uniforms
        ]
        self.fragment_main_body = [f"fs_color = {func_name}({', '.join(args)});"]

    def build_vertex_shader(self) -> str:
        components = [
            "#version 460 core",
            *self.vertex_attributes,
            *self.vertex_interface,
            *self.structs,
            *self.functions,
            "void main() {",
            *[f"    {line}" for line in self.vertex_main_body],
            "}",
        ]
        return "\n".join(components)

    def build_fragment_shader(self) -> str:
        components = [
            "#version 460 core",
            *self.uniforms,
            *self.fragment_interface,
            *self.outputs,
            *self.structs,
            *self.functions,
            "void main() {",
            *[f"    {line}" for line in self.fragment_main_body],
            "}",
        ]
        return "\n".join(components)
