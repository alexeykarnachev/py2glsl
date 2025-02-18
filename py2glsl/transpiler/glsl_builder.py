from dataclasses import dataclass, field
from typing import Dict, List, Optional


class GLSLCodeError(Exception):
    pass


@dataclass
class GLSLStruct:
    name: str
    fields: Dict[str, str]


@dataclass
class GLSLInterfaceBlock:
    struct_name: str
    direction: str  # 'in' or 'out'
    fields: Dict[str, str]


@dataclass
class GLSLFunction:
    return_type: str
    name: str
    parameters: List[str]
    body: List[str]


@dataclass
class GLSLBuilder:
    version: str = "#version 460 core"
    uniforms: Dict[str, str] = field(default_factory=dict)
    structs: Dict[str, GLSLStruct] = field(default_factory=dict)
    interfaces: Dict[str, GLSLInterfaceBlock] = field(default_factory=dict)
    functions: List[GLSLFunction] = field(default_factory=list)
    vertex_attributes: Dict[int, str] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)
    main_body: List[str] = field(default_factory=list)

    def add_uniform(self, name: str, glsl_type: str) -> None:
        if not name.startswith("u_"):
            raise GLSLCodeError(f"Uniform name '{name}' must start with 'u_' prefix")
        self.uniforms[name] = glsl_type

    def add_struct(self, name: str, fields: Dict[str, str]) -> None:
        if name in self.structs:
            raise GLSLCodeError(f"Struct {name} already defined")
        self.structs[name] = GLSLStruct(name=name, fields=fields)

    def add_interface_block(
        self, struct_name: str, direction: str, fields: Dict[str, str]
    ) -> None:
        if direction not in ("in", "out"):
            raise GLSLCodeError("Interface direction must be 'in' or 'out'")

        self.interfaces[struct_name] = GLSLInterfaceBlock(
            struct_name=struct_name, direction=direction, fields=fields
        )

    def add_vertex_attribute(self, location: int, glsl_type: str, name: str) -> None:
        if not name.startswith("a_"):
            raise GLSLCodeError(f"Attribute name '{name}' must start with 'a_' prefix")
        self.vertex_attributes[location] = f"{glsl_type} {name}"

    def add_function(
        self, return_type: str, name: str, parameters: List[str], body: List[str]
    ) -> None:
        self.functions.append(
            GLSLFunction(
                return_type=return_type, name=name, parameters=parameters, body=body
            )
        )

    def add_output(self, name: str, glsl_type: str) -> None:
        if not name.startswith("fs_"):
            raise GLSLCodeError(f"Output name '{name}' must start with 'fs_' prefix")
        self.outputs[name] = glsl_type

    def build_vertex_shader(self) -> str:
        sections = [self.version]

        # Vertex attributes
        for loc, attr in sorted(self.vertex_attributes.items()):
            sections.append(f"layout(location = {loc}) in {attr};")

        # Interface blocks
        for block in self.interfaces.values():
            if block.direction == "out":
                sections.append(self._build_interface_block(block))

        # Main function
        sections.extend(
            [
                "void main() {",
                "    vs_uv = a_pos * 0.5 + 0.5;",
                "    gl_Position = vec4(a_pos, 0.0, 1.0);",
                "}",
            ]
        )

        return "\n".join(sections)

    def build_fragment_shader(self, entry_point: str) -> str:
        sections = [self.version]

        # Uniforms
        if self.uniforms:
            sections.extend(
                [f"uniform {typ} {name};" for name, typ in self.uniforms.items()]
            )

        # Interface blocks
        for block in self.interfaces.values():
            if block.direction == "in":
                sections.append(self._build_interface_block(block))

        # Outputs
        if self.outputs:
            sections.extend(
                [f"out {typ} {name};" for name, typ in self.outputs.items()]
            )

        # Functions
        func_src = []
        for func in self.functions:
            params = ", ".join(func.parameters)
            body = "\n".join(f"    {line}" for line in func.body)
            func_src.append(f"{func.return_type} {func.name}({params}) {{\n{body}\n}}")
        sections.extend(func_src)

        # Main function
        main_body = "\n".join(f"    {line}" for line in self.main_body)
        sections.append(f"void main() {{\n{main_body}\n}}")

        return "\n".join(sections)

    def _build_interface_block(self, block: GLSLInterfaceBlock) -> str:
        fields = "\n".join(f"    {typ} {name};" for name, typ in block.fields.items())
        return f"{block.direction} {block.struct_name} {{\n{fields}\n}};"

    def _build_struct(self, struct: GLSLStruct) -> str:
        fields = "\n".join(f"    {typ} {name};" for name, typ in struct.fields.items())
        return f"struct {struct.name} {{\n{fields}\n}};"
