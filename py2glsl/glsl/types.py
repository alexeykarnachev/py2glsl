from __future__ import annotations

from typing import TypeVar, Union

import numpy as np

T = TypeVar("T", bound="GLSLVector")
GLSLValue = Union[float, "GLSLVector", "mat3", "mat4"]


class GLSLVector:
    """Base class for GLSL vector types"""

    _size: int
    _fields: tuple

    _component_map = {
        'r': 'x', 'g': 'y', 'b': 'z', 'a': 'w',
        's': 'x', 't': 'y', 'p': 'z', 'q': 'w'
    }

    def __init__(self, *args: float | list | tuple | np.ndarray | GLSLVector):
        if len(args) == 1 and isinstance(args[0], GLSLVector):
            self.data: np.ndarray = args[0].data.copy()
        else:
            if len(args) == 1 and isinstance(args[0], (int, float)):
                values = [float(args[0])] * self._size
            elif len(args) == 1 and isinstance(args[0], (list, tuple, np.ndarray)):
                values = args[0]
            else:
                values = args

            self.data: np.ndarray = np.asarray(values, dtype=np.float32).flatten()

            if len(self.data) != self._size:
                raise ValueError(
                    f"Invalid input size for {self.__class__.__name__}. "
                    f"Expected {self._size}, got {len(self.data)}"
                )

    def __repr__(self) -> str:
        vals = ", ".join(f"{x:.3f}" for x in self.data)
        return f"{self.__class__.__name__}({vals})"

    # Operator overloading
    def __add__(self: T, other: GLSLValue) -> T:
        return self._apply_op(other, lambda a, b: a + b)

    def __sub__(self: T, other: GLSLValue) -> T:
        return self._apply_op(other, lambda a, b: a - b)

    def __mul__(self: T, other: GLSLValue) -> T:
        return self._apply_op(other, lambda a, b: a * b)

    def __rmul__(self: T, other: GLSLValue) -> T:
        return self.__mul__(other)

    def __truediv__(self: T, other: GLSLValue) -> T:
        return self._apply_op(other, lambda a, b: a / b)

    def _apply_op(self: T, other: GLSLValue, op) -> T:
        """Apply operation with scalar/vector broadcasting"""
        if isinstance(other, GLSLVector):
            if self._size != other._size:
                raise ValueError("Vector size mismatch")
            return self.__class__(op(self.data, other.data))
        elif isinstance(other, (int, float)):
            return self.__class__(op(self.data, other))
        else:
            raise TypeError(f"Unsupported operand type: {type(other)}")

    # Swizzle operations
    def __getattr__(self, name: str):
        """Handle swizzle patterns only"""
        if not all(c in "xyzw" for c in name) or not 1 <= len(name) <= 4:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

        components = []
        for c in name:
            idx = "xyzw".index(c)
            if idx >= self._size:
                raise AttributeError(
                    f"Component {c} not available for {self.__class__.__name__}"
                )
            components.append(self.data[idx])

        return self._create_swizzle_result(components, name)

    def _create_swizzle_result(self, components: list, name: str):
        if len(components) == 1:
            return components[0]
        return _VECTOR_CLS_MAP[len(components)](components)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return np.array_equal(self.data, other.data)

    def to_array(self) -> np.ndarray:
        return self.data.copy()


class vec2(GLSLVector):
    _size = 2
    _fields = ("x", "y")


class vec3(GLSLVector):
    _size = 3
    _fields = ("x", "y", "z")


class vec4(GLSLVector):
    _size = 4
    _fields = ("x", "y", "z", "w")


_VECTOR_CLS_MAP = {2: vec2, 3: vec3, 4: vec4}


class mat3:
    """GLSL-style 3x3 matrix"""

    def __init__(self, values: list | np.ndarray):
        self.data = np.asarray(values, dtype=np.float32).reshape(3, 3)

    def __matmul__(self, other: mat3 | vec3) -> mat3 | vec3:
        if isinstance(other, mat3):
            return mat3(self.data @ other.data)
        if isinstance(other, vec3):
            return vec3(self.data @ other.data)
        raise TypeError("Unsupported operand type for matrix multiplication")

    def transpose(self) -> mat3:
        return mat3(self.data.T)

    def __repr__(self) -> str:
        rows = []
        for row in self.data:
            formatted_row = ", ".join(f"{x:.3f}" for x in row)
            rows.append(f"[{formatted_row}]")
        return f"mat3([{', '.join(rows)}])"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, mat3):
            return False
        return np.allclose(self.data, other.data)


class mat4:
    """GLSL-style 4x4 matrix"""

    def __init__(self, values: list | np.ndarray):
        self.data = np.asarray(values, dtype=np.float32).reshape(4, 4)

    def __matmul__(self, other: mat4 | vec4) -> mat4 | vec4:
        if isinstance(other, mat4):
            return mat4(self.data @ other.data)
        if isinstance(other, vec4):
            return vec4(self.data @ other.data)
        raise TypeError("Unsupported operand type for matrix multiplication")

    def transpose(self) -> mat4:
        return mat4(self.data.T)

    def __repr__(self) -> str:
        rows = []
        for row in self.data:
            formatted_row = ", ".join(f"{x:.3f}" for x in row)
            rows.append(f"[{formatted_row}]")
        return f"mat4([{', '.join(rows)}])"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, mat4):
            return False
        return np.allclose(self.data, other.data)
