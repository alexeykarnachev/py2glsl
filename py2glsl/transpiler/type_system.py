import ast
from ast import (
    AST,
    Assign,
    BinOp,
    Constant,
    FunctionDef,
    Name,
    NodeVisitor,
    Return,
    Subscript,
)
from dataclasses import dataclass
from typing import Dict, List, Optional, Union


class GlslTypeError(Exception):
    def __init__(self, node: AST, message: str):
        self.lineno = getattr(node, "lineno", 0)
        self.col_offset = getattr(node, "col_offset", 0)
        super().__init__(
            f"[GLSL Type Error] Line {self.lineno}:{self.col_offset} - {message}"
        )


@dataclass
class TypeInfo:
    glsl_type: str
    is_constant: bool = False
    vector_size: Optional[int] = None
    matrix_dim: Optional[tuple] = None  # (rows, cols)

    def __eq__(self, other):
        return (
            self.glsl_type == other.glsl_type
            and self.vector_size == other.vector_size
            and self.matrix_dim == other.matrix_dim
        )

    def promotes_to(self, other: "TypeInfo") -> bool:
        if self.glsl_type == other.glsl_type:
            return True
        return (self.glsl_type, other.glsl_type) in {("int", "float")}


class TypeInferer(NodeVisitor):
    def __init__(self):
        self.symbols: Dict[str, TypeInfo] = {}
        self.current_fn_return: Optional[TypeInfo] = None
        self._in_function = False

    def visit_Module(self, node: ast.Module) -> None:
        for stmt in node.body:
            self.visit(stmt)

    def visit_FunctionDef(self, node: FunctionDef) -> None:
        self._in_function = True
        self.current_fn_return = None

        # Process return type annotation
        if node.returns:
            self.current_fn_return = self._parse_annotation(node.returns)

        # Process arguments
        for arg in node.args.args:
            if arg.annotation:
                arg_type = self._parse_annotation(arg.annotation)
                self.symbols[arg.arg] = arg_type
            else:
                raise GlslTypeError(
                    arg, f"Parameter '{arg.arg}' requires type annotation"
                )

        # Visit body to infer return type if not annotated
        for stmt in node.body:
            self.visit(stmt)

        if not self.current_fn_return:
            raise GlslTypeError(
                node, "Could not infer return type for unannotated function"
            )

        self._in_function = False

    def visit_Return(self, node: Return) -> None:
        if not self._in_function:
            raise GlslTypeError(node, "Return outside function")

        ret_type = self.visit(node.value) if node.value else TypeInfo("void")

        if not self.current_fn_return:
            self.current_fn_return = ret_type
        elif self.current_fn_return != ret_type:
            raise GlslTypeError(
                node,
                f"Return type mismatch: {ret_type.glsl_type} vs "
                f"{self.current_fn_return.glsl_type}",
            )

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if not isinstance(node.target, ast.Name):
            raise GlslTypeError(node, "Complex assignments not supported")

        target_id = node.target.id
        value_type = self.visit(node.value) if node.value else None
        annotated_type = self._parse_annotation(node.annotation)

        if value_type and annotated_type != value_type:
            raise GlslTypeError(
                node,
                f"Annotation mismatch for '{target_id}': "
                f"{annotated_type.glsl_type} vs {value_type.glsl_type}",
            )

        self.symbols[target_id] = annotated_type

    def visit_Assign(self, node: Assign) -> None:
        if len(node.targets) != 1:
            raise GlslTypeError(node, "Multiple assignment not supported")

        target = node.targets[0]
        if not isinstance(target, Name):
            raise GlslTypeError(node, "Complex assignments not supported")

        value_type = self.visit(node.value)

        if target.id in self.symbols:
            existing_type = self.symbols[target.id]
            if existing_type != value_type:
                raise GlslTypeError(
                    node,
                    f"Type mismatch for '{target.id}': "
                    f"{existing_type.glsl_type} vs {value_type.glsl_type}",
                )
        else:
            self.symbols[target.id] = value_type

    def visit_BinOp(self, node: ast.BinOp) -> TypeInfo:
        left_type = self.visit(node.left)
        right_type = self.visit(node.right)

        # Handle matrix-vector multiplication
        if isinstance(node.op, ast.Mult):
            # Matrix * Vector
            if left_type.glsl_type.startswith(
                "mat"
            ) and right_type.glsl_type.startswith("vec"):
                if left_type.matrix_dim[1] == right_type.vector_size:
                    vec_dim = left_type.matrix_dim[0]
                    return TypeInfo(f"vec{vec_dim}", vector_size=vec_dim)
            # Vector * Matrix
            elif right_type.glsl_type.startswith(
                "mat"
            ) and left_type.glsl_type.startswith("vec"):
                if right_type.matrix_dim[0] == left_type.vector_size:
                    vec_dim = right_type.matrix_dim[1]
                    return TypeInfo(f"vec{vec_dim}", vector_size=vec_dim)

        # Existing type promotion logic
        if left_type.glsl_type != right_type.glsl_type:
            if left_type.promotes_to(right_type):
                return right_type
            if right_type.promotes_to(left_type):
                return left_type
            raise GlslTypeError(
                node,
                f"Type mismatch in operation: {left_type.glsl_type} vs "
                f"{right_type.glsl_type}",
            )

        return left_type

    def visit_Constant(self, node: Constant) -> TypeInfo:
        if isinstance(node.value, (int, float)):
            return TypeInfo("float", is_constant=True)
        if isinstance(node.value, bool):
            return TypeInfo("bool", is_constant=True)
        raise GlslTypeError(node, f"Unsupported constant type {type(node.value)}")

    def visit_Name(self, node: Name) -> TypeInfo:
        if node.id not in self.symbols:
            raise GlslTypeError(node, f"Undefined variable '{node.id}'")
        return self.symbols[node.id]

    def _parse_annotation(self, node: AST) -> TypeInfo:
        if isinstance(node, Name):
            return self._handle_basic_type(node.id)

        if isinstance(node, Subscript):
            if isinstance(node.value, Name):
                base_type = node.value.id
                if base_type == "Vector":
                    return self._parse_vector_annotation(node)
                if base_type == "Matrix":
                    return self._parse_matrix_annotation(node)

        raise GlslTypeError(node, f"Unsupported type annotation")

    def _handle_basic_type(self, type_name: str) -> TypeInfo:
        basic_types = {
            "float": TypeInfo("float"),
            "int": TypeInfo("float"),  # Treat int as float in GLSL
            "bool": TypeInfo("bool"),
            "vec2": TypeInfo("vec2", vector_size=2),
            "vec3": TypeInfo("vec3", vector_size=3),
            "vec4": TypeInfo("vec4", vector_size=4),
            "mat2": TypeInfo("mat2", matrix_dim=(2, 2)),
            "mat3": TypeInfo("mat3", matrix_dim=(3, 3)),
            "mat4": TypeInfo("mat4", matrix_dim=(4, 4)),
        }
        if type_name not in basic_types:
            raise ValueError(f"Unknown basic type {type_name}")
        return basic_types[type_name]

    def _parse_vector_annotation(self, node: Subscript) -> TypeInfo:
        if isinstance(node.slice, ast.Tuple):
            dim = len(node.slice.elts)
            if dim not in {2, 3, 4}:
                raise GlslTypeError(node, f"Invalid vector dimension {dim}")
            return TypeInfo(f"vec{dim}", vector_size=dim)
        raise GlslTypeError(node, "Invalid vector annotation")

    def _parse_matrix_annotation(self, node: Subscript) -> TypeInfo:
        if isinstance(node.slice, ast.Tuple):
            dims = tuple(elt.value for elt in node.slice.elts)
            if dims not in {(2, 2), (3, 3), (4, 4)}:
                raise GlslTypeError(node, f"Invalid matrix dimensions {dims}")
            return TypeInfo(f"mat{dims[0]}", matrix_dim=dims)
        raise GlslTypeError(node, "Invalid matrix annotation")
