from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, ClassVar, Optional, Union, get_type_hints

import numpy as np

from py2glsl.glsl.types import mat3, mat4, vec2, vec3, vec4


class GLSLTypeKind(Enum):
    SCALAR = 1
    VECTOR = 2
    MATRIX = 3
    SAMPLER = 4


@dataclass(frozen=True, slots=True)
class TypeInfo:
    kind: GLSLTypeKind
    glsl_name: str
    size: tuple[int, ...] = ()
    is_polymorphic: bool = False

    @property
    def is_vector_like(self) -> bool:
        return self.is_vector or self.is_matrix

    @property
    def matrix_cols(self) -> int:
        return self.size[1] if self.is_matrix else 0

    @property
    def matrix_rows(self) -> int:
        return self.size[0] if self.is_matrix else 0

    @property
    def vector_size(self) -> int:
        return self.size[0] if self.is_vector else 0

    @classmethod
    def from_pytype(cls, py_type: type) -> "TypeInfo":
        type_map = {
            float: cls.FLOAT,
            int: cls.FLOAT,
            bool: cls.BOOL,
            vec2: cls.VEC2,
            vec3: cls.VEC3,
            vec4: cls.VEC4,
            mat3: cls.MAT3,
            mat4: cls.MAT4,
        }
        return type_map.get(py_type, cls.FLOAT)

    @classmethod
    def from_swizzle(cls, base_type: TypeInfo, components: str) -> TypeInfo:
        if not base_type.is_vector:
            raise GLSLTypeError(
                None, f"Cannot swizzle non-vector type {base_type.glsl_name}"
            )

        max_size = base_type.vector_size
        if len(components) > 4 or any(c not in "xyzw"[:max_size] for c in components):
            raise GLSLTypeError(
                None, f"Invalid swizzle '{components}' for {base_type.glsl_name}"
            )

        return _VECTOR_SIZE_MAP.get(len(components), TypeInfo.FLOAT)

    def __post_init__(self):
        if self.kind == GLSLTypeKind.VECTOR and not self.is_polymorphic:
            if self.size[0] not in {2, 3, 4}:
                raise ValueError(f"Invalid vector size: {self.size[0]}")
        elif self.kind == GLSLTypeKind.MATRIX and not self.is_polymorphic:
            if self.size[0] not in {3, 4} or self.size[0] != self.size[1]:
                raise ValueError(f"Invalid matrix size: {self.size}")

    def _validate_vector(self):
        if self.is_polymorphic:
            return
        if len(self.size) != 1:
            raise ValueError(f"Vector needs 1D size, got {self.size}")
        if self.size[0] not in {2, 3, 4}:
            raise ValueError(f"Invalid vector size: {self.size[0]}")

    def _validate_matrix(self):
        if self.is_polymorphic:
            return
        if len(self.size) != 2:
            raise ValueError(f"Matrix needs 2D size, got {self.size}")
        if self.size[0] not in {3, 4} or self.size[0] != self.size[1]:
            raise ValueError(f"Invalid matrix size: {self.size}")

    @property
    def is_scalar(self) -> bool:
        return self.kind == GLSLTypeKind.SCALAR

    @property
    def is_vector(self) -> bool:
        return self.kind == GLSLTypeKind.VECTOR

    @property
    def is_matrix(self) -> bool:
        return self.kind == GLSLTypeKind.MATRIX

    def resolve_size(self, ctx_size: tuple[int, ...]) -> TypeInfo:
        if not self.is_polymorphic:
            return self
        if not ctx_size:
            raise GLSLTypeError(
                None, "Cannot resolve polymorphic type without context size"
            )
        return TypeInfo(
            kind=self.kind,
            glsl_name=self.glsl_name.replace("n", str(ctx_size[0])),
            size=ctx_size,
            is_polymorphic=False,
        )

    # Predefined types
    FLOAT: ClassVar[TypeInfo]
    INT: ClassVar[TypeInfo]
    BOOL: ClassVar[TypeInfo]
    VEC2: ClassVar[TypeInfo]
    VEC3: ClassVar[TypeInfo]
    VEC4: ClassVar[TypeInfo]
    VECn: ClassVar[TypeInfo]
    MAT3: ClassVar[TypeInfo]
    MAT4: ClassVar[TypeInfo]
    MATn: ClassVar[TypeInfo]


TypeInfo.FLOAT = TypeInfo(GLSLTypeKind.SCALAR, "float")
TypeInfo.BOOL = TypeInfo(GLSLTypeKind.SCALAR, "bool")
TypeInfo.VEC2 = TypeInfo(GLSLTypeKind.VECTOR, "vec2", (2,))
TypeInfo.VEC3 = TypeInfo(GLSLTypeKind.VECTOR, "vec3", (3,))
TypeInfo.VEC4 = TypeInfo(GLSLTypeKind.VECTOR, "vec4", (4,))
TypeInfo.VECn = TypeInfo(GLSLTypeKind.VECTOR, "vecn", is_polymorphic=True)
TypeInfo.MAT3 = TypeInfo(GLSLTypeKind.MATRIX, "mat3", (3, 3))
TypeInfo.MAT4 = TypeInfo(GLSLTypeKind.MATRIX, "mat4", (4, 4))
TypeInfo.MATn = TypeInfo(GLSLTypeKind.MATRIX, "matn", is_polymorphic=True)

_VECTOR_SIZE_MAP = {
    1: TypeInfo.FLOAT,
    2: TypeInfo.VEC2,
    3: TypeInfo.VEC3,
    4: TypeInfo.VEC4,
}


class GLSLTypeError(Exception):
    def __init__(self, node: Optional[ast.AST], message: str):
        super().__init__(message)
        self.node = node
        self.message = message

    def __str__(self):
        if self.node and hasattr(self.node, "lineno"):
            return f"Line {self.node.lineno}: {self.message}"
        return self.message


class TypeInferer(ast.NodeVisitor):
    def __init__(self):
        # Predefined symbols and built-ins
        self.symbols = {
            # GLSL built-in variables
            "gl_Position": TypeInfo.VEC4,
            "gl_FragCoord": TypeInfo.VEC4,
            "gl_FragColor": TypeInfo.VEC4,
            # Common shader inputs
            "vs_uv": TypeInfo.VEC2,
            "u_time": TypeInfo.FLOAT,
            "u_resolution": TypeInfo.VEC2,
            # Default vertex shader attribute
            "a_pos": TypeInfo.VEC2,
        }
        self._type_registry = {
            "float": TypeInfo.FLOAT,
            "int": TypeInfo.FLOAT,
            "bool": TypeInfo.BOOL,
            "vec2": TypeInfo.VEC2,
            "vec3": TypeInfo.VEC3,
            "vec4": TypeInfo.VEC4,
            "mat3": TypeInfo.MAT3,
            "mat4": TypeInfo.MAT4,
        }
        self.current_return: Optional[TypeInfo] = None
        self._size_stack: list[tuple[int, ...]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Process return type
        self.current_return = TypeInfo.FLOAT  # Default
        if node.returns:
            self.current_return = self._resolve_annotation(node.returns)
        else:
            raise GLSLTypeError(node, "Missing return type annotation")

        # Process parameters
        for arg in node.args.args:
            arg_type = self._resolve_annotation(arg.annotation)
            self.symbols[arg.arg] = arg_type

        # Visit body
        for stmt in node.body:
            self.visit(stmt)

    def _resolve_annotation(self, node: ast.expr) -> TypeInfo:
        """Resolve type annotations without adding to symbols"""
        if isinstance(node, ast.Name):
            if node.id == "bool":
                return TypeInfo.BOOL
            if node.id in self._type_registry:
                return self._type_registry[node.id]
            raise GLSLTypeError(node, f"Undefined type '{node.id}'")
        if isinstance(node, ast.Subscript):  # Handle Optional[] cases
            return self._resolve_annotation(node.value)
        raise GLSLTypeError(node, f"Unsupported annotation: {ast.dump(node)}")

    def visit_Assign(self, node: ast.Assign):
        target = node.targets[0]
        if isinstance(target, ast.Name):
            target_name = target.id
            value_type = self.visit(node.value)
            self.symbols[target_name] = value_type
        return value_type

    def visit_AnnAssign(self, node: ast.AnnAssign):
        target_name = node.target.id
        ann_type = self._resolve_annotation(node.annotation)
        value_type = self.visit(node.value) if node.value else ann_type

        if ann_type != value_type:
            raise GLSLTypeError(
                node,
                f"Assignment type mismatch: {ann_type.glsl_name} vs {value_type.glsl_name}",
            )

        self.symbols[target_name] = ann_type
        return ann_type

    def visit_Return(self, node: ast.Return):
        return_type = self.visit(node.value)
        if self.current_return:
            if return_type != self.current_return:
                raise GLSLTypeError(
                    node,
                    f"Return type mismatch: {return_type.glsl_name} vs {self.current_return.glsl_name}",
                )
        return return_type

    def visit_Name(self, node: ast.Name) -> TypeInfo:
        if node.id in self.symbols:
            return self.symbols[node.id]
        raise GLSLTypeError(node, f"Undefined variable '{node.id}'")

    def visit_Call(self, node: ast.Call):
        return self._handle_constructor(node)

    def _handle_constructor(self, node: ast.Call) -> TypeInfo:
        name = node.func.id
        args = [self.visit(arg) for arg in node.args]

        if name.startswith("vec"):
            size = int(name[3:])
            if len(args) == 1 and args[0].is_scalar:
                return self._type_registry[name]

            # Validate vector components
            total_components = sum(
                arg.vector_size if arg.is_vector else 1 for arg in args
            )
            if total_components != size:
                raise GLSLTypeError(
                    node,
                    f"vec{size} constructor needs {size} components, got {total_components}",
                )
            return self._type_registry[name]

        if name.startswith("mat"):
            size = int(name[3:])
            if not all(a.is_scalar for a in args):
                raise GLSLTypeError(node, "Matrix components must be scalars")
            return TypeInfo(GLSLTypeKind.MATRIX, name, (size, size))

        raise GLSLTypeError(node, f"Unknown constructor {name}")

    def visit_BinOp(self, node: ast.BinOp):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = type(node.op)

        if op == ast.MatMult:
            return self._handle_matrix_mult(left, right)

        handler = {
            ast.Add: self._handle_add,
            ast.Sub: self._handle_sub,
            ast.Mult: self._handle_mult,
            ast.Div: self._handle_div,
        }.get(op)

        if handler:
            return handler(left, right)

        raise GLSLTypeError(node, f"Unsupported operator {op.__name__}")

    def _resolve_operation_size(
        self, a: TypeInfo, b: TypeInfo, node: ast.AST
    ) -> tuple[int, ...]:
        if a.is_vector and b.is_vector:
            if a.size != b.size:
                raise GLSLTypeError(node, f"Vector size mismatch {a.size} vs {b.size}")
            return a.size
        return a.size if a.is_vector else b.size

    def _handle_add(self, left: TypeInfo, right: TypeInfo) -> TypeInfo:
        """Handle addition: supports scalar/vector/matrix with type promotion"""
        if left == right:
            return left

        # Scalar promotion
        if left.is_scalar and right.is_vector:
            return right
        if right.is_scalar and left.is_vector:
            return left

        raise GLSLTypeError(
            None,
            f"Invalid addition: {left.glsl_name} + {right.glsl_name}\n"
            f"Require matching types or scalar promotion",
        )

    def _handle_sub(self, left: TypeInfo, right: TypeInfo) -> TypeInfo:
        """Handle subtraction: same rules as addition"""
        return self._handle_add(left, right)

    def _handle_mult(self, left: TypeInfo, right: TypeInfo) -> TypeInfo:
        """Handle multiplication rules including matrix/vector"""
        # Scalar multiplication
        if left.is_scalar:
            return right
        if right.is_scalar:
            return left

        # Matrix-vector multiplication (linear algebra)
        if left.is_matrix and right.is_vector:
            if left.matrix_cols == right.vector_size:
                return right
        if right.is_matrix and left.is_vector:
            if right.matrix_rows == left.vector_size:
                return left

        # Component-wise vector multiplication
        if left.is_vector and right.is_vector:
            if left == right:
                return left
            raise GLSLTypeError(
                None,
                f"Component-wise vector multiply requires matching types\n"
                f"Got: {left.glsl_name} * {right.glsl_name}",
            )

        # Matrix-matrix multiplication
        if left.is_matrix and right.is_matrix:
            if left.matrix_cols == right.matrix_rows:
                return TypeInfo.MAT4 if left.matrix_rows == 4 else TypeInfo.MAT3

        raise GLSLTypeError(
            None, f"Invalid multiplication: {left.glsl_name} * {right.glsl_name}"
        )

    def _handle_div(self, left: TypeInfo, right: TypeInfo) -> TypeInfo:
        """Handle division: right operand must be scalar for vector/matrix"""
        if not right.is_scalar:
            raise GLSLTypeError(
                None,
                f"Division requires scalar denominator for "
                f"{left.glsl_name} / {right.glsl_name}",
            )
        return left

    def _handle_matmult(
        self, left: TypeInfo, right: TypeInfo, size: tuple[int, ...]
    ) -> TypeInfo:
        if left.is_matrix and right.is_matrix:
            if left.size != right.size:
                raise GLSLTypeError(
                    None,
                    f"Matrix multiplication size mismatch {left.size} vs {right.size}",
                )
            return left
        if left.is_matrix and right.is_vector:
            if left.size[0] != right.size[0]:
                raise GLSLTypeError(
                    None, f"Matrix-vector size mismatch {left.size} vs {right.size}"
                )
            return right
        raise GLSLTypeError(None, "Invalid operands for matrix multiplication")

    def _handle_matrix_mult(self, left: TypeInfo, right: TypeInfo) -> TypeInfo:
        """Handle matrix multiplication rules for GLSL"""
        # Matrix-vector multiplication
        if left.is_matrix and right.is_vector:
            if left.matrix_cols != right.vector_size:
                raise GLSLTypeError(
                    None,
                    f"Matrix-vector dimension mismatch: "
                    f"matrix cols ({left.matrix_cols}) != "
                    f"vector size ({right.vector_size})",
                )
            return right  # Returns vector of same dimension as input

        # Matrix-matrix multiplication
        if left.is_matrix and right.is_matrix:
            if left.matrix_cols != right.matrix_rows:
                raise GLSLTypeError(
                    None,
                    f"Matrix multiplication dimension mismatch: "
                    f"{left.matrix_rows}x{left.matrix_cols} * "
                    f"{right.matrix_rows}x{right.matrix_cols}",
                )
            # Return matrix with left's rows and right's columns
            return TypeInfo.MAT4 if left.matrix_rows == 4 else TypeInfo.MAT3

        # Vector-matrix multiplication (invalid in GLSL)
        if left.is_vector and right.is_matrix:
            raise GLSLTypeError(
                None,
                "Vector-matrix multiplication not supported. "
                "Use matrix * vector instead.",
            )

        raise GLSLTypeError(None, "Invalid matrix multiplication operands")

    def visit_Attribute(self, node: ast.Attribute):
        base_type = self.visit(node.value)
        components = node.attr

        # Validate swizzle pattern
        if len(components) > 4 or any(c not in "xyzw" for c in components):
            raise GLSLTypeError(node, f"Invalid swizzle pattern '{components}'")

        # Validate component bounds
        max_component = len("xyzw"[: base_type.vector_size])
        if any(ord(c) - ord("x") >= max_component for c in components):
            raise GLSLTypeError(
                node, f"Swizzle out of bounds for {base_type.glsl_name}"
            )

        return TypeInfo.from_swizzle(base_type, components)

    def visit_Subscript(self, node: ast.Subscript):
        base_type = self.visit(node.value)
        if not base_type.is_vector_like:
            raise GLSLTypeError(node, "Subscript on non-vector/matrix type")

        if isinstance(node.slice, ast.Constant):
            idx = node.slice.value
            if not (0 <= idx < base_type.vector_size):
                raise GLSLTypeError(
                    node, f"Index {idx} out of bounds for {base_type.glsl_name}"
                )
            return TypeInfo.FLOAT

        raise GLSLTypeError(node, "Unsupported subscript type")

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, (int, float)):
            return TypeInfo.FLOAT
        if isinstance(node.value, bool):
            return TypeInfo.BOOL
        raise GLSLTypeError(node, f"Unsupported constant type {type(node.value)}")

    def visit_BoolOp(self, node: ast.BoolOp) -> TypeInfo:
        """Handle boolean operations (and/or)"""
        for value in node.values:
            value_type = self.visit(value)
            if value_type != TypeInfo.BOOL:
                raise GLSLTypeError(
                    node,
                    f"Boolean operation requires bool operands, got {value_type.glsl_name}",
                )
        return TypeInfo.BOOL

    def visit_Compare(self, node: ast.Compare) -> TypeInfo:
        """Handle comparison operations"""
        left_type = self.visit(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            comp_type = self.visit(comparator)
            if left_type != comp_type:
                raise GLSLTypeError(
                    node,
                    f"Comparison type mismatch: {left_type.glsl_name} vs {comp_type.glsl_name}",
                )
        return TypeInfo.BOOL

    def visit_UnaryOp(self, node: ast.UnaryOp) -> TypeInfo:
        """Handle unary operations like negation"""
        operand_type = self.visit(node.operand)

        if isinstance(node.op, ast.USub):
            # Handle numeric negation
            if (
                operand_type.is_scalar
                or operand_type.is_vector
                or operand_type.is_matrix
            ):
                return operand_type
            raise GLSLTypeError(
                node, f"Cannot negate non-numeric type {operand_type.glsl_name}"
            )

        if isinstance(node.op, ast.Not):
            # Handle boolean NOT
            if operand_type != TypeInfo.BOOL:
                raise GLSLTypeError(
                    node,
                    f"Boolean operation requires bool operand, got {operand_type.glsl_name}",
                )
            return TypeInfo.BOOL

        raise GLSLTypeError(
            node, f"Unsupported unary operator {type(node.op).__name__}"
        )

    def visit_IfExp(self, node: ast.IfExp) -> TypeInfo:
        """Handle conditional expressions (a if test else b)"""
        test_type = self.visit(node.test)
        if test_type != TypeInfo.BOOL:
            raise GLSLTypeError(node, "Condition must be a boolean")

        if_type = self.visit(node.body)
        else_type = self.visit(node.orelse)

        if if_type != else_type:
            raise GLSLTypeError(
                node,
                f"Conditional branches must match types: {if_type.glsl_name} vs {else_type.glsl_name}",
            )

        return if_type


def infer_types(node: ast.AST) -> dict[str, TypeInfo]:
    inferer = TypeInferer()
    inferer.visit(node)
    return inferer.symbols
