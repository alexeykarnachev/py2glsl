"""Type system for GLSL code generation."""

import ast
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Optional, Set, Union

from py2glsl.transpiler.constants import (
    BOOL_TYPE,
    FLOAT_TYPE,
    INT_TYPE,
    VEC2_TYPE,
    VEC3_TYPE,
    VEC4_TYPE,
)


class GLSLContext(Enum):
    """Context for GLSL code generation."""

    DEFAULT = auto()
    LOOP_BOUND = auto()  # For loop bounds and indices
    ARITHMETIC = auto()  # For general math expressions
    VECTOR_INIT = auto()  # Keep existing vector context


@dataclass
class GLSLType:
    """Represents a GLSL type with optional qualifiers."""

    name: str
    is_uniform: bool = False
    is_varying: bool = False
    is_attribute: bool = False
    array_size: Optional[int] = None

    def __str__(self) -> str:
        """Convert to GLSL type declaration."""
        result = []
        if self.is_uniform:
            result.append("uniform")
        if self.is_varying:
            result.append("varying")
        if self.is_attribute:
            result.append("attribute")
        result.append(self.name)
        if self.array_size is not None:
            result.append(f"[{self.array_size}]")
        return " ".join(result)


class TypeInference:
    """GLSL type inference system."""

    def __init__(self) -> None:
        """Initialize type inference system."""
        self.type_map: Dict[str, GLSLType] = {}
        self.current_context = GLSLContext.DEFAULT

    def get_type(self, node: ast.AST) -> GLSLType:
        """Infer GLSL type from AST node."""
        if isinstance(node, ast.Name) and node.id == "vs_uv":
            return GLSLType(VEC2_TYPE)

        if isinstance(node, ast.Name):
            return self.type_map.get(node.id, GLSLType(FLOAT_TYPE))

        elif isinstance(node, ast.Num):
            if isinstance(node.n, bool):
                return GLSLType(BOOL_TYPE)
            elif isinstance(node.n, int):
                return GLSLType(INT_TYPE)
            return GLSLType(FLOAT_TYPE)

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ("vec2", "Vec2"):
                    return GLSLType(VEC2_TYPE)
                elif node.func.id in ("vec3", "Vec3"):
                    return GLSLType(VEC3_TYPE)
                elif node.func.id in ("vec4", "Vec4"):
                    return GLSLType(VEC4_TYPE)

        elif isinstance(node, ast.BinOp):
            left_type = self.get_type(node.left)
            right_type = self.get_type(node.right)
            # Vector operations take precedence
            if left_type.name.startswith("vec") or right_type.name.startswith("vec"):
                return left_type if left_type.name.startswith("vec") else right_type
            return GLSLType(FLOAT_TYPE)

        elif isinstance(node, ast.Attribute):
            base_type = self.get_type(node.value)
            if base_type.name.startswith("vec"):
                swizzle = node.attr
                if len(swizzle) == 1:
                    return GLSLType(FLOAT_TYPE)
                return GLSLType(f"vec{len(swizzle)}")

        return GLSLType(FLOAT_TYPE)

    def register_type(self, name: str, glsl_type: GLSLType) -> None:
        """Register type for a variable."""
        self.type_map[name] = glsl_type

    def get_declaration(self, name: str) -> str:
        """Get GLSL declaration for a variable."""
        if name in self.type_map:
            return f"{str(self.type_map[name])} {name}"
        return f"{FLOAT_TYPE} {name}"


class TypeValidator:
    """Validates GLSL type compatibility."""

    @staticmethod
    def is_numeric(type_: GLSLType) -> bool:
        """Check if type is numeric (float, int, or vector)."""
        return type_.name in (FLOAT_TYPE, INT_TYPE) or type_.name.startswith("vec")

    @staticmethod
    def is_vector(type_: GLSLType) -> bool:
        """Check if type is a vector type."""
        return type_.name.startswith("vec")

    @staticmethod
    def get_vector_size(type_: GLSLType) -> int:
        """Get size of vector type."""
        if not type_.name.startswith("vec"):
            return 0
        return int(type_.name[3])

    def validate_binary_op(
        self, left: GLSLType, right: GLSLType, op: ast.operator
    ) -> GLSLType:
        """Validate binary operation and return result type."""
        if not (self.is_numeric(left) and self.is_numeric(right)):
            raise TypeError("Binary operations require numeric types")

        # Vector operations
        if self.is_vector(left) or self.is_vector(right):
            if self.is_vector(left) and self.is_vector(right):
                if self.get_vector_size(left) != self.get_vector_size(right):
                    raise TypeError("Vector operations require same dimensions")
                return left
            return left if self.is_vector(left) else right

        # Scalar operations
        return GLSLType(FLOAT_TYPE)
