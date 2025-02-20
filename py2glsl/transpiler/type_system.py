import ast
from ast import AST, Constant, FunctionDef, Name, NodeVisitor, Return, Subscript
from dataclasses import dataclass
from enum import Enum, auto
from typing import ClassVar


class GLSLTypeKind(Enum):
    SCALAR = auto()
    VECTOR = auto()
    MATRIX = auto()
    VOID = auto()


@dataclass(frozen=True, slots=True)
class TypeInfo:
    kind: GLSLTypeKind
    glsl_name: str
    size: tuple[int, ...] = ()

    # Predefined type instances
    FLOAT: ClassVar["TypeInfo"]
    BOOL: ClassVar["TypeInfo"]
    VEC2: ClassVar["TypeInfo"]
    VEC3: ClassVar["TypeInfo"]
    VEC4: ClassVar["TypeInfo"]
    MAT3: ClassVar["TypeInfo"]
    MAT4: ClassVar["TypeInfo"]

    def __post_init__(self):
        if self.kind == GLSLTypeKind.VECTOR and len(self.size) != 1:
            raise ValueError(f"Vector needs 1 dimension, got {self.size}")
        if self.kind == GLSLTypeKind.MATRIX and len(self.size) != 2:
            raise ValueError(f"Matrix needs 2 dimensions, got {self.size}")

    @classmethod
    def from_annotation(cls, node: AST) -> "TypeInfo":
        if isinstance(node, Name):
            return _parse_basic_type(node.id)
        if isinstance(node, Subscript) and isinstance(node.value, Name):
            return _parse_container_type(node)
        raise GLSLTypeError(node, "Invalid type annotation")

    def promotes_to(self, other: "TypeInfo") -> bool:
        if self == other:
            return True
        return self.glsl_name == "int" and other.glsl_name == "float"

    def __str__(self):
        return self.glsl_name


TypeInfo.FLOAT = TypeInfo(GLSLTypeKind.SCALAR, "float")
TypeInfo.BOOL = TypeInfo(GLSLTypeKind.SCALAR, "bool")
TypeInfo.VEC2 = TypeInfo(GLSLTypeKind.VECTOR, "vec2", (2,))
TypeInfo.VEC3 = TypeInfo(GLSLTypeKind.VECTOR, "vec3", (3,))
TypeInfo.VEC4 = TypeInfo(GLSLTypeKind.VECTOR, "vec4", (4,))
TypeInfo.MAT3 = TypeInfo(GLSLTypeKind.MATRIX, "mat3", (3, 3))
TypeInfo.MAT4 = TypeInfo(GLSLTypeKind.MATRIX, "mat4", (4, 4))


class GLSLTypeError(Exception):
    def __init__(self, node: AST, message: str):
        lineno = getattr(node, "lineno", "?")
        super().__init__(f"Line {lineno}: {message}")


def _parse_basic_type(name: str) -> TypeInfo:
    match name:
        case "float":
            return TypeInfo.FLOAT
        case "int":
            return TypeInfo.FLOAT  # GLSL int->float promotion
        case "bool":
            return TypeInfo.BOOL
        case "vec2":
            return TypeInfo.VEC2
        case "vec3":
            return TypeInfo.VEC3
        case "vec4":
            return TypeInfo.VEC4
        case "mat3":
            return TypeInfo.MAT3
        case "mat4":
            return TypeInfo.MAT4
        case _:
            raise ValueError(f"Unknown GLSL type: {name}")


def _parse_container_type(node: Subscript) -> TypeInfo:
    base = node.value.id
    if base == "Vector":
        return _parse_vector_type(node)
    if base == "Matrix":
        return _parse_matrix_type(node)
    raise GLSLTypeError(node, f"Unsupported container {base}")


def _parse_vector_type(node: Subscript) -> TypeInfo:
    if not isinstance(node.slice, ast.Tuple):
        raise GLSLTypeError(node, "Vector needs size annotation")

    size = len(node.slice.elts)
    match size:
        case 2:
            return TypeInfo.VEC2
        case 3:
            return TypeInfo.VEC3
        case 4:
            return TypeInfo.VEC4
        case _:
            raise GLSLTypeError(node, f"Invalid vector size: {size}")


def _parse_matrix_type(node: Subscript) -> TypeInfo:
    if not isinstance(node.slice, ast.Tuple):
        raise GLSLTypeError(node, "Matrix needs dimensions annotation")

    dims = [n.value for n in node.slice.elts if isinstance(n, Constant)]
    if len(dims) != 2 or dims[0] != dims[1]:
        raise GLSLTypeError(node, "Only square matrices supported")

    match dims[0]:
        case 3:
            return TypeInfo.MAT3
        case 4:
            return TypeInfo.MAT4
        case _:
            raise GLSLTypeError(node, f"Unsupported matrix size: {dims[0]}")


class TypeInferer(NodeVisitor):
    def __init__(self):
        self.symbols: dict[str, TypeInfo] = {}
        self.current_return: TypeInfo | None = None

    def visit_Module(self, node: ast.Module) -> None:  # noqa: N802
        for stmt in node.body:
            self.visit(stmt)

    def visit_FunctionDef(self, node: FunctionDef) -> None:
        # Process return type annotation
        self.current_return = (
            TypeInfo.from_annotation(node.returns) if node.returns else None
        )

        # Clear symbols for this function scope
        self.symbols.clear()

        # Register all parameters (including posonlyargs)
        for param in node.args.posonlyargs + node.args.args:
            param_name = param.arg
            if not param.annotation:
                raise GLSLTypeError(
                    param, f"Parameter '{param_name}' needs type annotation"
                )
            param_type = TypeInfo.from_annotation(param.annotation)
            self.symbols[param_name] = param_type

        # Process function body
        for stmt in node.body:
            self.visit(stmt)

        # Validate return type consistency
        if not self.current_return:
            raise GLSLTypeError(node, "Could not infer return type")

    def visit_Return(self, node: Return) -> None:  # noqa: N802
        ret_type = (
            self.visit(node.value)
            if node.value
            else TypeInfo(GLSLTypeKind.VOID, "void")
        )

        if not self.current_return:
            self.current_return = ret_type
        elif ret_type != self.current_return:
            raise GLSLTypeError(
                node, f"Return type mismatch: {ret_type} vs {self.current_return}"
            )

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        target = node.target
        if not isinstance(target, Name):
            raise GLSLTypeError(node, "Complex assignments not supported")

        annotated = TypeInfo.from_annotation(node.annotation)
        value_type = self.visit(node.value) if node.value else annotated

        if annotated != value_type:
            raise GLSLTypeError(
                node, f"Annotation mismatch '{target.id}': {annotated} vs {value_type}"
            )
        self.symbols[target.id] = annotated

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        if len(node.targets) != 1 or not isinstance(node.targets[0], Name):
            raise GLSLTypeError(node, "Complex assignments not supported")

        target = node.targets[0].id
        value_type = self.visit(node.value)
        self.symbols[target] = value_type

    def visit_BinOp(self, node: ast.BinOp) -> TypeInfo:  # noqa: N802
        left = self.visit(node.left)
        right = self.visit(node.right)

        # Vector-scalar operations
        if left.kind == GLSLTypeKind.VECTOR and right.glsl_name == "float":
            return left
        if right.kind == GLSLTypeKind.VECTOR and left.glsl_name == "float":
            return right

        # Matrix multiplication
        if isinstance(node.op, ast.MatMult):
            # Matrix-matrix
            if left.kind == GLSLTypeKind.MATRIX and right.kind == GLSLTypeKind.MATRIX:
                if left.size != right.size:
                    raise GLSLTypeError(
                        node, f"Matrix size mismatch: {left} vs {right}"
                    )
                return left

            # Matrix-vector
            if left.kind == GLSLTypeKind.MATRIX and right.kind == GLSLTypeKind.VECTOR:
                if left.size[1] != right.size[0]:
                    raise GLSLTypeError(
                        node, f"Matrix-vector dimension mismatch: {left} vs {right}"
                    )
                return TypeInfo(
                    GLSLTypeKind.VECTOR, f"vec{left.size[0]}", (left.size[0],)
                )

            # Vector-matrix
            if left.kind == GLSLTypeKind.VECTOR and right.kind == GLSLTypeKind.MATRIX:
                if left.size[0] != right.size[1]:
                    raise GLSLTypeError(
                        node, f"Vector-matrix dimension mismatch: {left} vs {right}"
                    )
                return TypeInfo(
                    GLSLTypeKind.VECTOR, f"vec{right.size[0]}", (right.size[0],)
                )

            raise GLSLTypeError(node, "Matrix multiply requires matrix/vector operands")

        # Type promotion
        if left.promotes_to(right):
            return right
        if right.promotes_to(left):
            return left
        raise GLSLTypeError(node, f"Type mismatch: {left} vs {right}")

    def visit_BoolOp(self, node: ast.BoolOp) -> TypeInfo:  # noqa: N802
        for value in node.values:
            value_type = self.visit(value)
            if value_type != TypeInfo.BOOL:
                raise GLSLTypeError(node, f"Non-bool in boolean op: {value_type}")
        return TypeInfo.BOOL

    def visit_Constant(self, node: Constant) -> TypeInfo:  # noqa: N802
        match node.value:
            case bool():
                return TypeInfo.BOOL
            case int() | float():
                return TypeInfo.FLOAT
            case _:
                raise GLSLTypeError(node, f"Unsupported constant: {type(node.value)}")

    def visit_Call(self, node: ast.Call) -> TypeInfo:  # noqa: N802
        if isinstance(node.func, Name):
            if node.func.id.startswith("vec"):
                size = int(node.func.id[3:])
                return getattr(TypeInfo, f"VEC{size}")
            if node.func.id.startswith("mat"):
                size = int(node.func.id[3:])
                return getattr(TypeInfo, f"MAT{size}")
        raise GLSLTypeError(node, f"Unsupported call: {ast.dump(node)}")

    def visit_Name(self, node: Name) -> TypeInfo:  # noqa: N802
        if (t := self.symbols.get(node.id)) is None:
            raise GLSLTypeError(node, f"Undefined variable: {node.id}")
        return t

    def visit_Subscript(self, node: Subscript) -> TypeInfo:  # noqa: N802
        value_type = self.visit(node.value)

        if value_type.kind not in (GLSLTypeKind.VECTOR, GLSLTypeKind.MATRIX):
            raise GLSLTypeError(
                node, f"Subscript on non-vector/matrix type {value_type}"
            )

        return (
            TypeInfo.FLOAT if value_type.kind == GLSLTypeKind.VECTOR else TypeInfo.VEC3
        )

    def visit_UnaryOp(self, node: ast.UnaryOp) -> TypeInfo:  # noqa: N802
        operand_type = self.visit(node.operand)

        if isinstance(node.op, (ast.USub, ast.UAdd)):
            return operand_type

        raise GLSLTypeError(node, f"Unsupported unary operator {type(node.op)}")

    def visit_IfExp(self, node: ast.IfExp) -> TypeInfo:  # noqa: N802
        test_type = self.visit(node.test)
        body_type = self.visit(node.body)
        orelse_type = self.visit(node.orelse)

        if test_type != TypeInfo.BOOL:
            raise GLSLTypeError(node.test, "Condition must be boolean")

        if body_type != orelse_type:
            raise GLSLTypeError(
                node, f"Conditional branches mismatch: {body_type} vs {orelse_type}"
            )

        return body_type

    def visit_Attribute(self, node: ast.Attribute) -> TypeInfo:  # noqa: N802
        base_type = self.visit(node.value)
        if base_type.kind != GLSLTypeKind.VECTOR:
            raise GLSLTypeError(node, f"Swizzle on non-vector type {base_type}")

        swizzle = node.attr
        if not all(c in "xyzw" for c in swizzle):
            raise GLSLTypeError(node, f"Invalid swizzle characters: {swizzle}")

        if len(swizzle) < 1 or len(swizzle) > 4:
            raise GLSLTypeError(node, f"Invalid swizzle length {len(swizzle)}")

        # Validate component indices
        max_components = base_type.size[0]
        component_map = {"x": 0, "y": 1, "z": 2, "w": 3}
        for c in swizzle:
            if component_map.get(c, -1) >= max_components:
                raise GLSLTypeError(
                    node,
                    f"Swizzle component '{c}' out of bounds for {base_type.glsl_name}",
                )

        return TypeInfo(GLSLTypeKind.VECTOR, f"vec{len(swizzle)}", (len(swizzle),))

    def visit_Compare(self, node: ast.Compare) -> TypeInfo:  # noqa: N802
        left_type = self.visit(node.left)

        for _, comparator in zip(node.ops, node.comparators, strict=False):
            comparator_type = self.visit(comparator)

            if left_type != comparator_type:
                raise GLSLTypeError(
                    node, f"Comparison type mismatch: {left_type} vs {comparator_type}"
                )

        return TypeInfo.BOOL
