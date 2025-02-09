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
        parts = []
        if self.is_uniform:
            parts.append("uniform")
        if self.is_varying:
            parts.append("varying")
        if self.is_attribute:
            parts.append("attribute")
        parts.append(self.name)
        if self.array_size is not None:
            parts[-1] = f"{parts[-1]}[{self.array_size}]"
        return " ".join(parts)

    def is_vector(self) -> bool:
        return self.name.startswith("vec")

    def vector_size(self) -> int:
        return int(self.name[3]) if self.is_vector() else 0


class TypeInference:
    """GLSL type inference system."""

    def __init__(self) -> None:
        self.type_map: Dict[str, GLSLType] = {}
        self.current_context = GLSLContext.DEFAULT

    def get_type(self, node: ast.AST) -> GLSLType:
        """Infer GLSL type from AST node."""
        # Special case for shader parameter
        if isinstance(node, ast.Name) and node.id == "vs_uv":
            return GLSLType(VEC2_TYPE)

        # Handle variable names
        if isinstance(node, ast.Name):
            return self.type_map.get(node.id, GLSLType(FLOAT_TYPE))

        # Handle constants
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return GLSLType(BOOL_TYPE)
            elif isinstance(node.value, int):
                return GLSLType(INT_TYPE)
            return GLSLType(FLOAT_TYPE)

        # Handle vector constructors and function calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ("vec2", "Vec2"):
                    return GLSLType(VEC2_TYPE)
                elif node.func.id in ("vec3", "Vec3"):
                    return GLSLType(VEC3_TYPE)
                elif node.func.id in ("vec4", "Vec4"):
                    return GLSLType(VEC4_TYPE)

        # Handle binary operations
        if isinstance(node, ast.BinOp):
            left_type = self.get_type(node.left)
            right_type = self.get_type(node.right)

            # Vector operations take precedence
            if left_type.is_vector() or right_type.is_vector():
                return left_type if left_type.is_vector() else right_type

            # Integer operations in loop context
            if self.current_context == GLSLContext.LOOP_BOUND:
                if left_type.name == INT_TYPE and right_type.name == INT_TYPE:
                    return GLSLType(INT_TYPE)

            return GLSLType(FLOAT_TYPE)

        # Handle vector swizzling
        if isinstance(node, ast.Attribute):
            base_type = self.get_type(node.value)
            if base_type.is_vector():
                swizzle = node.attr
                if len(swizzle) == 1:
                    return GLSLType(FLOAT_TYPE)
                return GLSLType(f"vec{len(swizzle)}")

        return GLSLType(FLOAT_TYPE)

    def register_type(self, name: str, glsl_type: GLSLType) -> None:
        """Register type for a variable."""
        self.type_map[name] = glsl_type

    def set_context(self, context: GLSLContext) -> None:
        """Set current type inference context."""
        self.current_context = context


def validate_types(left: GLSLType, right: GLSLType, op: ast.operator) -> GLSLType:
    """Validate type compatibility and return result type."""
    # Vector operations
    if left.is_vector() or right.is_vector():
        if left.is_vector() and right.is_vector():
            if left.vector_size() != right.vector_size():
                raise TypeError("Vector operations require matching dimensions")
            return left
        return left if left.is_vector() else right

    # Numeric operations
    if left.name in (FLOAT_TYPE, INT_TYPE) and right.name in (FLOAT_TYPE, INT_TYPE):
        return GLSLType(FLOAT_TYPE)

    raise TypeError(f"Invalid operation between {left.name} and {right.name}")
