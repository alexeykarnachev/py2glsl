from dataclasses import dataclass
from typing import Dict, List


class GLSLCodeError(ValueError):
    pass


@dataclass
class GLSLInterfaceBlock:
    name: str
    qualifier: str
    fields: Dict[str, str]


@dataclass
class GLSLFunction:
    return_type: str
    name: str
    parameters: List[str]
    body: List[str]


@dataclass
class GLSLStruct:
    name: str
    fields: Dict[str, str]


class GLSLBuilder:
    def __init__(self):
        self.version = "#version 460 core"
        self.uniforms: Dict[str, str] = {}
        self.structs: Dict[str, GLSLStruct] = {}
        self.interfaces: Dict[str, GLSLInterfaceBlock] = {}
        self.functions: List[GLSLFunction] = []
        self.vertex_attributes: Dict[int, tuple[str, str]] = {}
        self.outputs: Dict[str, str] = {}
        self.main_body: List[str] = []

    def add_uniform(self, name: str, type_: str):
        self._validate_identifier(name, "uniform")
        if not name.startswith("u_"):
            raise GLSLCodeError(f"Uniform name '{name}' must start with 'u_'")
        self.uniforms[name] = type_

    def add_struct(self, name: str, fields: Dict[str, str]):
        if name in self.structs:
            raise GLSLCodeError(f"Struct {name} already defined")
        self.structs[name] = GLSLStruct(name, fields)

    def add_interface_block(self, name: str, qualifier: str, fields: Dict[str, str]):
        self._validate_identifier(name, "interface block")
        self.interfaces[name] = GLSLInterfaceBlock(name, qualifier, fields)

    def add_function(
        self, return_type: str, name: str, parameters: List[str], body: List[str]
    ):
        self._validate_identifier(name, "function")
        self.functions.append(GLSLFunction(return_type, name, parameters, body))

    def add_vertex_attribute(self, location: int, type_: str, name: str):
        self._validate_identifier(name, "attribute")
        if not name.startswith("a_"):
            raise GLSLCodeError(f"Attribute name '{name}' must start with 'a_'")
        self.vertex_attributes[location] = (type_, name)

    def add_output(self, name: str, type_: str):
        self._validate_identifier(name, "output")
        if not name.startswith("fs_"):
            raise GLSLCodeError(f"Fragment output name '{name}' must start with 'fs_'")
        self.outputs[name] = type_

    def _validate_identifier(self, name: str, context: str):
        if not name.isidentifier():
            raise GLSLCodeError(f"Invalid {context} name: '{name}'")
        if any(c.isupper() for c in name):
            raise GLSLCodeError(
                f"{context.capitalize()} name '{name}' must be lowercase"
            )

    def build_vertex_shader(self) -> str:
        code = [self.version]

        # Vertex attributes
        for loc, (type_, name) in sorted(self.vertex_attributes.items()):
            code.append(f"layout(location = {loc}) in {type_} {name};")

        # Interfaces
        for block in self.interfaces.values():
            code.append(f"out {block.name} {{")
            for field, ftype in block.fields.items():
                code.append(f"    {ftype} {field};")
            code.append("};")

        # Main function
        code.append("void main() {")
        code.extend(f"    {line}" for line in self.main_body)
        code.append("}")

        return "\n".join(code)

    def build_fragment_shader(self, entry_point: str) -> str:
        code = [self.version]

        # Structs
        for struct in self.structs.values():
            code.append(f"struct {struct.name} {{")
            for field, ftype in struct.fields.items():
                code.append(f"    {ftype} {field};")
            code.append("};")

        # Uniforms
        for name, type_ in self.uniforms.items():
            code.append(f"uniform {type_} {name};")

        # Interfaces
        for block in self.interfaces.values():
            code.append(f"{block.qualifier} {block.name} {{")
            for field, ftype in block.fields.items():
                code.append(f"    {ftype} {field};")
            code.append("};")

        # Outputs
        for name, type_ in self.outputs.items():
            code.append(f"out {type_} {name};")

        # Functions
        for func in self.functions:
            params = ", ".join(func.parameters)
            code.append(f"{func.return_type} {func.name}({params}) {{")
            code.extend(f"    {line}" for line in func.body)
            code.append("}")

        # Main function
        code.append("void main() {")
        code.extend(f"    {line}" for line in self.main_body)
        code.append("}")

        return "\n".join(code)
