import ast
from ast import AST, Assign, Constant, FunctionDef, Name, NodeVisitor, Return, Subscript
from dataclasses import dataclass
from typing import cast


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
    vector_size: int | None = None
    matrix_dim: tuple[int, int] | None = None  # (rows, cols)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TypeInfo):
            return False
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
    def __init__(self) -> None:
        self.symbols: dict[str, TypeInfo] = {}
        self.current_fn_return: TypeInfo | None = None
        self._in_function = False

    def visit_Module(self, node: ast.Module) -> None:  # noqa: N802
        for stmt in node.body:
            self.visit(stmt)

    def visit_FunctionDef(self, node: FunctionDef) -> None:  # noqa: N802
        self._in_function = True
        self.current_fn_return = (
            self._parse_annotation(node.returns) if node.returns else None
        )
        self.symbols = {}

        for arg in node.args.args:
            if not arg.annotation:
                raise GlslTypeError(
                    arg, f"Parameter '{arg.arg}' requires type annotation"
                )
            arg_type = self._parse_annotation(arg.annotation)
            self.symbols[arg.arg] = arg_type

        for stmt in node.body:
            self.visit(stmt)

        if not self.current_fn_return:
            raise GlslTypeError(
                node, "Could not infer return type for unannotated function"
            )

        self._in_function = False

    def visit_Return(self, node: Return) -> None:  # noqa: N802
        if not self._in_function:
            raise GlslTypeError(node, "Return outside function")

        ret_type = self.visit(node.value) if node.value else TypeInfo("void")

        if not self.current_fn_return:
            self.current_fn_return = ret_type
        elif self.current_fn_return != ret_type:
            raise GlslTypeError(
                node,
                "Return type mismatch: "
                f"{ret_type.glsl_type} vs {self.current_fn_return.glsl_type}",
            )

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        if not isinstance(node.target, ast.Name):
            raise GlslTypeError(node, "Complex assignments not supported")

        target_id = node.target.id
        annotated_type = self._parse_annotation(node.annotation)
        value_type = self.visit(node.value) if node.value else annotated_type

        if annotated_type != value_type:
            raise GlslTypeError(
                node,
                f"Annotation mismatch for '{target_id}': "
                f"{annotated_type.glsl_type} vs {value_type.glsl_type}",
            )

        self.symbols[target_id] = annotated_type

    def visit_Assign(self, node: Assign) -> None:  # noqa: N802
        if len(node.targets) != 1:
            raise GlslTypeError(node, "Multiple assignment not supported")

        target = node.targets[0]
        if not isinstance(target, Name):
            raise GlslTypeError(node, "Complex assignments not supported")

        value_type = self.visit(node.value)
        self.symbols[target.id] = value_type

    def visit_BinOp(self, node: ast.BinOp) -> TypeInfo:  # noqa: N802
        left_type = self.visit(node.left)
        right_type = self.visit(node.right)

        # Try vector/scalar operations first
        vector_result = self._check_vector_scalar_op(node.op, left_type, right_type)
        if vector_result:
            return vector_result

        # Try matrix/vector multiplication
        matrix_result = self._check_matrix_mult(node.op, left_type, right_type)
        if matrix_result:
            return matrix_result

        # Handle type promotion
        return self._check_type_promotion(node, left_type, right_type)

    def _check_vector_scalar_op(
        self, op: ast.operator, left: TypeInfo, right: TypeInfo
    ) -> TypeInfo | None:
        """Handle vector-scalar operations with explicit typing"""
        if not isinstance(op, (ast.Mult | ast.Add | ast.Sub | ast.Div)):
            return None

        if left.vector_size and right.glsl_type == "float":
            return left
        if right.vector_size and left.glsl_type == "float":
            return right

        return None

    def _check_matrix_mult(
        self, op: ast.operator, left: TypeInfo, right: TypeInfo
    ) -> TypeInfo | None:
        """Validate matrix-vector multiplication dimensions"""
        if not isinstance(op, ast.Mult):
            return None

        # Matrix * Vector case
        if (
            left.glsl_type.startswith("mat")
            and right.glsl_type.startswith("vec")
            and left.matrix_dim is not None
            and left.matrix_dim[1] == right.vector_size
        ):
            return TypeInfo(f"vec{left.matrix_dim[0]}", vector_size=left.matrix_dim[0])

        # Vector * Matrix case
        if (
            right.glsl_type.startswith("mat")
            and left.glsl_type.startswith("vec")
            and right.matrix_dim is not None
            and right.matrix_dim[0] == left.vector_size
        ):
            return TypeInfo(
                f"vec{right.matrix_dim[1]}", vector_size=right.matrix_dim[1]
            )

        return None

    def _check_type_promotion(
        self, node: ast.BinOp, left: TypeInfo, right: TypeInfo
    ) -> TypeInfo:
        """Handle type promotion with guaranteed TypeInfo return"""
        if left.glsl_type == right.glsl_type:
            return left

        if left.promotes_to(right):
            return right
        if right.promotes_to(left):
            return left

        raise GlslTypeError(
            node, f"Type mismatch: {left.glsl_type} vs {right.glsl_type}"
        )

    def visit_BoolOp(self, node: ast.BoolOp) -> TypeInfo:  # noqa: N802
        for value in node.values:
            value_type = self.visit(value)
            if value_type.glsl_type != "bool":
                raise GlslTypeError(node, "Boolean ops require bool operands")
        return TypeInfo("bool")

    def visit_Compare(self, node: ast.Compare) -> TypeInfo:  # noqa: N802
        left_type = self.visit(node.left)
        for _, comparator in zip(node.ops, node.comparators, strict=False):
            right_type = self.visit(comparator)
            if left_type != right_type:
                raise GlslTypeError(
                    node,
                    f"Comparison mismatch: {left_type.glsl_type} vs "
                    f"{right_type.glsl_type}",
                )
        return TypeInfo("bool")

    def visit_Constant(self, node: Constant) -> TypeInfo:  # noqa: N802
        if isinstance(node.value, bool | int | float):
            return TypeInfo(
                "bool" if isinstance(node.value, bool) else "float", is_constant=True
            )
        raise GlslTypeError(node, f"Unsupported constant {type(node.value)}")

    def visit_Call(self, node: ast.Call) -> TypeInfo:  # noqa: N802
        if isinstance(node.func, ast.Name) and node.func.id.startswith("vec"):
            dim = int(node.func.id[3])
            for arg in node.args:
                self.visit(arg)
            return TypeInfo(f"vec{dim}", vector_size=dim)
        raise GlslTypeError(node, f"Unsupported call: {getattr(node.func, 'id', '')}")

    def visit_Name(self, node: Name) -> TypeInfo:  # noqa: N802
        if node.id not in self.symbols:
            raise GlslTypeError(node, f"Undefined variable '{node.id}'")
        return self.symbols[node.id]

    def _parse_annotation(self, node: AST) -> TypeInfo:
        if isinstance(node, Name):
            return self._handle_basic_type(node.id)
        if isinstance(node, Subscript) and isinstance(node.value, Name):
            base_type = node.value.id
            if base_type == "Vector":
                return self._parse_vector_annotation(node)
            if base_type == "Matrix":
                return self._parse_matrix_annotation(node)
        raise GlslTypeError(node, "Invalid type annotation")

    def _handle_basic_type(self, type_name: str) -> TypeInfo:
        basic_types = {
            "float": TypeInfo("float"),
            "int": TypeInfo("float"),
            "bool": TypeInfo("bool"),
            "vec2": TypeInfo("vec2", vector_size=2),
            "vec3": TypeInfo("vec3", vector_size=3),
            "vec4": TypeInfo("vec4", vector_size=4),
            "mat2": TypeInfo("mat2", matrix_dim=(2, 2)),
            "mat3": TypeInfo("mat3", matrix_dim=(3, 3)),
            "mat4": TypeInfo("mat4", matrix_dim=(4, 4)),
        }
        if type_name not in basic_types:
            raise ValueError(f"Unknown type: {type_name}")
        return basic_types[type_name]

    def _parse_vector_annotation(self, node: Subscript) -> TypeInfo:
        if isinstance(node.slice, ast.Tuple):
            dim = len(node.slice.elts)
            if dim not in {2, 3, 4}:
                raise GlslTypeError(node, f"Invalid vector dimension {dim}")
            return TypeInfo(f"vec{dim}", vector_size=dim)
        raise GlslTypeError(node, "Vector annotation requires tuple of types")

    def _parse_matrix_annotation(self, node: Subscript) -> TypeInfo:
        if isinstance(node.slice, ast.Tuple):
            dims = []
            for elt in node.slice.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, int):
                    dims.append(elt.value)
                else:
                    raise GlslTypeError(node, "Matrix dimensions must be integers")

            if len(dims) != 2:  # noqa: PLR2004
                raise GlslTypeError(node, "Matrix requires 2 dimensions")

            dims_tuple = cast(tuple[int, int], tuple(dims))
            if dims_tuple not in {(2, 2), (3, 3), (4, 4)}:
                raise GlslTypeError(node, f"Invalid matrix dimensions {dims_tuple}")

            return TypeInfo(f"mat{dims_tuple[0]}", matrix_dim=dims_tuple)

        raise GlslTypeError(node, "Matrix annotation requires tuple of dimensions")
