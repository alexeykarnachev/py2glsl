import ast
from ast import AST, Assign, Constant, FunctionDef, Name, NodeVisitor, Return, Subscript
from dataclasses import dataclass
from typing import cast


# ------------------------------------------------------------------------
# Core Types
@dataclass
class TypeInfo:
    glsl_type: str
    is_constant: bool = False
    vector_size: int | None = None
    matrix_dim: tuple[int, int] | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TypeInfo):
            return False
        return (
            self.glsl_type == other.glsl_type
            and self.vector_size == other.vector_size
            and self.matrix_dim == other.matrix_dim
        )

    def promotes_to(self, other: "TypeInfo") -> bool:
        return self.glsl_type == other.glsl_type or (
            self.glsl_type == "int" and other.glsl_type == "float"
        )


class GlslTypeError(Exception):
    def __init__(self, node: AST, message: str):
        self.lineno = getattr(node, "lineno", 0)
        self.col_offset = getattr(node, "col_offset", 0)
        super().__init__(
            f"[GLSL Type Error] Line {self.lineno}:{self.col_offset} - {message}"
        )


# ------------------------------------------------------------------------
# Type Parsing
def parse_annotation(node: AST) -> TypeInfo:
    """Convert Python type annotations to GLSL TypeInfo"""
    if isinstance(node, Name):
        return _parse_basic_type(node.id)
    if isinstance(node, Subscript) and isinstance(node.value, Name):
        return _parse_container_type(node)
    raise GlslTypeError(node, "Invalid type annotation")


def _parse_basic_type(type_name: str) -> TypeInfo:
    match type_name:
        case "float" | "int" | "bool":
            return TypeInfo("float" if type_name == "int" else type_name)
        case _ if type_name.startswith("vec"):
            return _parse_vector_type(type_name)
        case _ if type_name.startswith("mat"):
            return _parse_matrix_type(type_name)
        case _:
            raise ValueError(f"Unknown type: {type_name}")


def _parse_container_type(node: Subscript) -> TypeInfo:
    base_type = node.value.id
    match base_type:
        case "Vector":
            return _parse_vector_annotation(node)
        case "Matrix":
            return _parse_matrix_annotation(node)
        case _:
            raise GlslTypeError(node, f"Unsupported container {base_type}")


def _parse_vector_type(type_name: str) -> TypeInfo:
    size = int(type_name[3:])
    if size not in {2, 3, 4}:
        raise ValueError(f"Invalid vector size: {size}")
    return TypeInfo(type_name, vector_size=size)


def _parse_matrix_type(type_name: str) -> TypeInfo:
    parts = type_name[3:].split("x")

    if len(parts) == 1:
        rows = cols = int(parts[0])
    else:
        rows, cols = map(int, parts)

    if rows not in range(2, 5) or cols not in range(2, 5):
        raise ValueError(f"Invalid matrix dimensions: {rows}x{cols}")

    return TypeInfo(
        f"mat{rows}" if rows == cols else f"mat{rows}x{cols}", matrix_dim=(rows, cols)
    )


def _parse_vector_annotation(node: Subscript) -> TypeInfo:
    if not isinstance(node.slice, ast.Tuple):
        raise GlslTypeError(node, "Vector annotation requires type tuple")

    size = len(node.slice.elts)
    if size not in {2, 3, 4}:
        raise GlslTypeError(node, f"Invalid vector size {size}")

    return TypeInfo(f"vec{size}", vector_size=size)


def _parse_matrix_annotation(node: Subscript) -> TypeInfo:
    if not isinstance(node.slice, ast.Tuple):
        raise GlslTypeError(node, "Matrix requires dimension tuple")

    dims = [n.value for n in node.slice.elts if isinstance(n, ast.Constant)]
    if len(dims) != 2 or any(d not in range(2, 5) for d in dims):
        raise GlslTypeError(node, f"Invalid matrix dimensions {dims}")

    return TypeInfo(f"mat{dims[0]}x{dims[1]}", matrix_dim=tuple(dims))


# ------------------------------------------------------------------------
# Type Operations
def check_bin_op(node: ast.BinOp, left: TypeInfo, right: TypeInfo) -> TypeInfo:
    """Handle binary operations with type validation"""
    if result := check_vector_scalar_op(node.op, left, right):
        return result
    if result := check_matrix_op(node, node.op, left, right):
        return result
    return check_type_promotion(node, left, right)


def check_vector_scalar_op(
    op: ast.operator, left: TypeInfo, right: TypeInfo
) -> TypeInfo | None:
    if not isinstance(op, (ast.Mult, ast.Add, ast.Sub, ast.Div)):
        return None

    if left.vector_size and right.glsl_type == "float":
        return left
    if right.vector_size and left.glsl_type == "float":
        return right
    return None


def check_matrix_op(
    bin_node: AST, op: ast.operator, left: TypeInfo, right: TypeInfo
) -> TypeInfo | None:
    if not isinstance(op, ast.Mult):
        return None

    if left.matrix_dim and right.matrix_dim:
        return check_matrix_matrix_mult(bin_node, left, right)
    if left.matrix_dim and right.vector_size:
        return check_matrix_vector_mult(bin_node, left, right)
    return None


def check_matrix_matrix_mult(
    node: AST,
    left: TypeInfo,
    right: TypeInfo,
) -> TypeInfo:
    lrows, lcols = left.matrix_dim
    rrows, rcols = right.matrix_dim

    if lcols != rrows:
        raise GlslTypeError(
            node, f"Matrix dimension mismatch: {lrows}x{lcols} vs {rrows}x{rcols}"
        )
    return TypeInfo(
        f"mat{lrows}x{rcols}" if lrows != rcols else f"mat{lrows}",
        matrix_dim=(lrows, rcols),
    )


def check_matrix_vector_mult(
    node: AST,
    left: TypeInfo,
    right: TypeInfo,
) -> TypeInfo:
    mrows, mcols = left.matrix_dim
    if mcols != right.vector_size:
        raise GlslTypeError(
            node, f"Matrix-vector dimension mismatch: {mcols} vs {right.vector_size}"
        )
    return TypeInfo(f"vec{mrows}", vector_size=mrows)


def check_type_promotion(node: AST, a: TypeInfo, b: TypeInfo) -> TypeInfo:
    if a == b:
        return a
    if a.promotes_to(b):
        return b
    if b.promotes_to(a):
        return a
    raise GlslTypeError(
        node,
        f"Type mismatch in operation: {a.glsl_type} vs {b.glsl_type}",
    )


# ------------------------------------------------------------------------
# Type Inference
class TypeInferer(NodeVisitor):
    def __init__(self) -> None:
        self.symbols: dict[str, TypeInfo] = {}
        self.current_fn_return: TypeInfo | None = None
        self._in_function = False

    def visit_Module(self, node: ast.Module) -> None:
        for stmt in node.body:
            self.visit(stmt)

    def visit_FunctionDef(self, node: FunctionDef) -> None:
        self._in_function = True
        self.current_fn_return = (
            parse_annotation(node.returns) if node.returns else None
        )
        self.symbols = {}

        for arg in node.args.args:
            if not arg.annotation:
                raise GlslTypeError(
                    arg, f"Parameter '{arg.arg}' requires type annotation"
                )
            self.symbols[arg.arg] = parse_annotation(arg.annotation)

        for stmt in node.body:
            self.visit(stmt)

        if not self.current_fn_return:
            raise GlslTypeError(node, "Could not infer return type")
        self._in_function = False

    def visit_Return(self, node: Return) -> None:
        if not self._in_function:
            raise GlslTypeError(node, "Return outside function")

        ret_type = self.visit(node.value) if node.value else TypeInfo("void")

        if self.current_fn_return:
            if ret_type != self.current_fn_return:
                raise GlslTypeError(
                    node,
                    f"Return type mismatch: {ret_type.glsl_type} vs "
                    f"{self.current_fn_return.glsl_type}",
                )
        else:
            self.current_fn_return = ret_type

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if not isinstance(node.target, Name):
            raise GlslTypeError(node, "Complex assignments not supported")

        target_id = node.target.id
        annotated = parse_annotation(node.annotation)
        value_type = self.visit(node.value) if node.value else annotated

        if annotated != value_type:
            raise GlslTypeError(
                node,
                f"Annotation mismatch '{target_id}': {annotated.glsl_type} vs {value_type.glsl_type}",
            )
        self.symbols[target_id] = annotated

    def visit_Assign(self, node: Assign) -> None:
        if len(node.targets) != 1 or not isinstance(node.targets[0], Name):
            raise GlslTypeError(node, "Invalid assignment format")
        target = node.targets[0].id
        self.symbols[target] = self.visit(node.value)

    def visit_BinOp(self, node: ast.BinOp) -> TypeInfo:
        return check_bin_op(node, self.visit(node.left), self.visit(node.right))

    def visit_BoolOp(self, node: ast.BoolOp) -> TypeInfo:
        for value in node.values:
            if (val_type := self.visit(value)).glsl_type != "bool":
                raise GlslTypeError(
                    node, f"Non-bool in boolean op: {val_type.glsl_type}"
                )
        return TypeInfo("bool")

    def visit_Compare(self, node: ast.Compare) -> TypeInfo:
        base_type = self.visit(node.left)
        for comparator in node.comparators:
            if (cmp_type := self.visit(comparator)) != base_type:
                raise GlslTypeError(
                    node, f"Compare type mismatch: {base_type} vs {cmp_type}"
                )
        return TypeInfo("bool")

    def visit_Constant(self, node: Constant) -> TypeInfo:
        match node.value:
            case bool():
                return TypeInfo("bool", is_constant=True)
            case int() | float():
                return TypeInfo("float", is_constant=True)
            case _:
                raise GlslTypeError(node, f"Unsupported constant {type(node.value)}")

    def visit_Call(self, node: ast.Call) -> TypeInfo:
        if isinstance(node.func, Name):
            return self._handle_constructor_call(node)
        raise GlslTypeError(node, "Unsupported call type")

    def _handle_constructor_call(self, node: ast.Call) -> TypeInfo:
        name = node.func.id
        if name.startswith("vec"):
            return self._parse_vector_constructor(node)
        if name.startswith("mat"):
            return self._parse_matrix_constructor(node)
        raise GlslTypeError(node, f"Unsupported constructor {name}")

    def _parse_vector_constructor(self, node: ast.Call) -> TypeInfo:
        size = int(node.func.id[3:])
        if size not in {2, 3, 4}:
            raise GlslTypeError(node, f"Invalid vector size: {size}")

        if len(node.args) == 1:
            arg_type = self.visit(node.args[0])
            if arg_type.vector_size and arg_type.vector_size != size:
                raise GlslTypeError(
                    node, f"Cannot convert {arg_type.glsl_type} to vec{size}"
                )
        else:
            if len(node.args) != size:
                raise GlslTypeError(
                    node, f"vec{size} requires {size} arguments, got {len(node.args)}"
                )
            for arg in node.args:
                arg_type = self.visit(arg)
                if arg_type.glsl_type not in ("float", "int"):
                    raise GlslTypeError(
                        node,
                        f"Vector component must be scalar, got {arg_type.glsl_type}",
                    )

        return TypeInfo(f"vec{size}", vector_size=size)

    def _parse_matrix_constructor(self, node: ast.Call) -> TypeInfo:
        if len(node.args) != 1:
            raise GlslTypeError(node, "Matrix constructor requires single argument")

        arg_type = self.visit(node.args[0])
        if arg_type.glsl_type not in ("float", "int"):
            raise GlslTypeError(node, f"Invalid matrix argument: {arg_type}")

        parts = node.func.id[3:].split("x")
        dims = (
            tuple(map(int, parts))
            if len(parts) == 2
            else (int(parts[0]), int(parts[0]))
        )
        glsl_type = f"mat{dims[0]}" if dims[0] == dims[1] else f"mat{dims[0]}x{dims[1]}"
        return TypeInfo(glsl_type, matrix_dim=dims)

    def visit_Attribute(self, node: ast.Attribute) -> TypeInfo:
        base_type = self.visit(node.value)
        if not base_type.vector_size:
            raise GlslTypeError(
                node, f"Swizzle on non-vector type {base_type.glsl_type}"
            )

        swizzle = node.attr
        valid_channels = {"rgba", "stpq", "xyzw"}
        component_map = {
            "r": 0,
            "g": 1,
            "b": 2,
            "a": 3,
            "s": 0,
            "t": 1,
            "p": 2,
            "q": 3,
            "x": 0,
            "y": 1,
            "z": 2,
            "w": 3,
        }

        # Validate swizzle pattern
        if len(swizzle) < 1 or len(swizzle) > 4:
            raise GlslTypeError(node, f"Invalid swizzle length {len(swizzle)}")

        # Check for mixed channel types
        used_sets = set()
        for c in swizzle:
            for channel_set in valid_channels:
                if c in channel_set:
                    used_sets.add(channel_set)
                    break
            else:
                raise GlslTypeError(node, f"Invalid swizzle character '{c}'")

        if len(used_sets) > 1:
            raise GlslTypeError(node, "Mixed swizzle channels (rgba and stpq)")

        # Check component bounds
        max_dim = base_type.vector_size
        for c in swizzle:
            if component_map[c] >= max_dim:
                raise GlslTypeError(
                    node,
                    f"Swizzle component '{c}' out of bounds for {base_type.glsl_type}",
                )

        return TypeInfo(f"vec{len(swizzle)}", vector_size=len(swizzle))

    def visit_Name(self, node: Name) -> TypeInfo:
        if (t := self.symbols.get(node.id)) is None:
            raise GlslTypeError(node, f"Undefined variable {node.id}")
        return t
