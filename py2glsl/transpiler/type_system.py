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

    @classmethod
    def from_pytype(cls, py_type: type) -> "TypeInfo":
        type_map = {
            float: cls.FLOAT,
            int: cls.FLOAT,  # in GLSL we usually treat int as float
            bool: cls.BOOL,
            vec2: cls.VEC2,
            vec3: cls.VEC3,
            vec4: cls.VEC4,
            mat3: cls.MAT3,
            mat4: cls.MAT4,
        }
        return type_map.get(py_type, cls.FLOAT)

    def __post_init__(self):
        if self.kind == GLSLTypeKind.VECTOR:
            self._validate_vector()
        elif self.kind == GLSLTypeKind.MATRIX:
            self._validate_matrix()

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
        return TypeInfo(
            kind=self.kind,
            glsl_name=self.glsl_name.replace("n", str(ctx_size[0])),
            size=ctx_size,
            is_polymorphic=False,
        )

    # Predefined types
    FLOAT: ClassVar[TypeInfo]
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
        self.symbols: dict[str, TypeInfo] = {}
        self.current_return: Optional[TypeInfo] = None
        self._size_stack: list[tuple[int, ...]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        type_hints = get_type_hints(node, include_extras=True)

        # Process return type
        self.current_return = TypeInfo.from_pytype(
            type_hints.get("return", TypeInfo.FLOAT)
        )

        # Process parameters
        for param in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
            if param.annotation:
                param_type = eval(
                    ast.unparse(param.annotation),
                    {t.__name__: t for t in [vec2, vec3, vec4, mat3, mat4]},
                )
                self.symbols[param.arg] = TypeInfo.from_pytype(param_type)

        self.generic_visit(node)
        self.current_return = None

    def visit_Assign(self, node: ast.Assign) -> None:
        value_type = self.visit(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.symbols[target.id] = value_type

    def visit_Name(self, node: ast.Name) -> TypeInfo:
        if node.id in self.symbols:
            return self.symbols[node.id]
        raise GLSLTypeError(node, f"Undefined variable {node.id}")

    def visit_Call(self, node: ast.Call) -> TypeInfo:
        # Handle type constructors first
        if isinstance(node.func, ast.Name) and node.func.id.startswith(("vec", "mat")):
            return self._handle_constructor(node)

        # Handle builtin functions
        func_name = node.func.id if isinstance(node.func, ast.Name) else None
        args = [self.visit(arg) for arg in node.args]

        try:
            from py2glsl.glsl import builtins

            glsl_func = getattr(builtins, func_name)
            meta = glsl_func.__glsl_metadata__
        except (AttributeError, ImportError):
            raise GLSLTypeError(node, f"Undefined function: {func_name}")

        # Resolve argument types
        resolved_args = []
        for spec, arg in zip(meta.arg_types, args):
            if callable(spec):
                resolved = spec(args)
            elif spec.is_polymorphic:
                resolved = arg
            else:
                resolved = spec
            resolved_args.append(resolved)

        # Validate arguments
        for spec, arg in zip(meta.arg_types, resolved_args):
            if not spec.is_polymorphic and spec != arg:
                raise GLSLTypeError(
                    node, f"Argument type mismatch. Expected {spec}, got {arg}"
                )

        # Determine return type
        if callable(meta.return_type):
            return_type = meta.return_type(resolved_args)
        elif meta.return_type.is_polymorphic:
            return_type = resolved_args[0]
        else:
            return_type = meta.return_type

        # Apply size context for polymorphic types
        if return_type.is_polymorphic:
            if return_type.is_vector and resolved_args:
                ctx_size = resolved_args[0].size
                return_type = return_type.resolve_size(ctx_size)
            elif return_type.is_matrix and resolved_args:
                ctx_size = resolved_args[0].size
                return_type = return_type.resolve_size(ctx_size)

        return return_type

    def _handle_constructor(self, node: ast.Call) -> TypeInfo:
        name = node.func.id
        args = [self.visit(arg) for arg in node.args]

        if name.startswith("vec"):
            size = int(name[3:])
            if not all(a.is_scalar for a in args):
                raise GLSLTypeError(node, "Vector components must be scalars")
            return TypeInfo(GLSLTypeKind.VECTOR, name, (size,))

        if name.startswith("mat"):
            size = int(name[3:])
            if not all(a.is_scalar for a in args):
                raise GLSLTypeError(node, "Matrix components must be scalars")
            return TypeInfo(GLSLTypeKind.MATRIX, name, (size, size))

        raise GLSLTypeError(node, f"Unknown constructor {name}")

    def visit_BinOp(self, node: ast.BinOp) -> TypeInfo:
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = type(node.op)

        # Vector/matrix size propagation
        size = self._resolve_operation_size(left, right, node)

        try:
            handler = {
                ast.Add: self._handle_add,
                ast.Sub: self._handle_sub,
                ast.Mult: self._handle_mult,
                ast.Div: self._handle_div,
                ast.MatMult: self._handle_matmult,
            }[op]
            return handler(left, right, size)
        except KeyError:
            raise GLSLTypeError(node, f"Unsupported operator {op.__name__}")

    def _resolve_operation_size(
        self, a: TypeInfo, b: TypeInfo, node: ast.AST
    ) -> tuple[int, ...]:
        if a.is_vector and b.is_vector:
            if a.size != b.size:
                raise GLSLTypeError(node, f"Vector size mismatch {a.size} vs {b.size}")
            return a.size
        return a.size if a.is_vector else b.size

    def _handle_add(
        self, left: TypeInfo, right: TypeInfo, size: tuple[int, ...]
    ) -> TypeInfo:
        if left.is_scalar and right.is_scalar:
            return TypeInfo.FLOAT
        return TypeInfo.VECn.resolve_size(size)

    def _handle_sub(
        self, left: TypeInfo, right: TypeInfo, size: tuple[int, ...]
    ) -> TypeInfo:
        return self._handle_add(left, right, size)

    def _handle_mult(
        self, left: TypeInfo, right: TypeInfo, size: tuple[int, ...]
    ) -> TypeInfo:
        if left.is_matrix and right.is_matrix:
            if left.size != right.size:
                raise GLSLTypeError(
                    None, f"Matrix size mismatch {left.size} vs {right.size}"
                )
            return left
        if left.is_matrix or right.is_matrix:
            return self._handle_matmult(left, right, size)
        return TypeInfo.VECn.resolve_size(size)

    def _handle_div(
        self, left: TypeInfo, right: TypeInfo, size: tuple[int, ...]
    ) -> TypeInfo:
        if not right.is_scalar:
            raise GLSLTypeError(None, "Can only divide by scalar values")
        return TypeInfo.VECn.resolve_size(size)

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

    def visit_Attribute(self, node: ast.Attribute) -> TypeInfo:
        base_type = self.visit(node.value)

        # Handle vector components
        if base_type.is_vector and node.attr in {"x", "y", "z", "w"}:
            return TypeInfo.FLOAT

        # Handle swizzle operations
        if base_type.is_vector and len(node.attr) > 1:
            if not all(c in {"x", "y", "z", "w"} for c in node.attr):
                raise GLSLTypeError(node, f"Invalid swizzle pattern {node.attr}")
            return TypeInfo.VECn.resolve_size((len(node.attr),))

        return base_type

    def visit_Constant(self, node: ast.Constant) -> TypeInfo:
        if isinstance(node.value, (float, int)):
            return TypeInfo.FLOAT
        if isinstance(node.value, bool):
            return TypeInfo.BOOL
        raise GLSLTypeError(node, f"Unsupported constant type {type(node.value)}")


def infer_types(node: ast.AST) -> dict[str, TypeInfo]:
    inferer = TypeInferer()
    inferer.visit(node)
    return inferer.symbols
