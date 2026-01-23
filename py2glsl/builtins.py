"""GLSL builtin types and functions for Python runtime.

This module provides Python implementations of GLSL types
(vec2, vec3, vec4, mat2, mat3, mat4) and builtin functions
(sin, cos, mix, etc.) that match GLSL semantics.
"""

from typing import TypeVar, overload

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]

# Type variable for array element type
E = TypeVar("E")


class vec2:
    """GLSL vec2 type - 2-component float vector."""

    __slots__ = ("data",)

    @overload
    def __init__(self, x: float, y: float) -> None: ...
    @overload
    def __init__(self, x: float) -> None: ...
    @overload
    def __init__(self, x: "vec2") -> None: ...

    def __init__(self, x: "float | vec2" = 0.0, y: float | None = None) -> None:
        if isinstance(x, vec2):
            self.data = x.data.copy()
        elif y is None:
            self.data = np.array([x, x], dtype=np.float32)
        else:
            self.data = np.array([x, y], dtype=np.float32)

    # Component accessors
    @property
    def x(self) -> float:
        return float(self.data[0])

    @x.setter
    def x(self, value: float) -> None:
        self.data[0] = value

    @property
    def y(self) -> float:
        return float(self.data[1])

    @y.setter
    def y(self, value: float) -> None:
        self.data[1] = value

    # Aliases r,g for x,y
    @property
    def r(self) -> float:
        return self.x

    @r.setter
    def r(self, value: float) -> None:
        self.x = value

    @property
    def g(self) -> float:
        return self.y

    @g.setter
    def g(self, value: float) -> None:
        self.y = value

    # Swizzle properties - vec2 from vec2
    @property
    def xx(self) -> "vec2":
        return vec2(self.x, self.x)

    @property
    def xy(self) -> "vec2":
        return vec2(self.x, self.y)

    @property
    def yx(self) -> "vec2":
        return vec2(self.y, self.x)

    @property
    def yy(self) -> "vec2":
        return vec2(self.y, self.y)

    # Arithmetic operators
    @overload
    def __add__(self, other: "vec2") -> "vec2": ...
    @overload
    def __add__(self, other: float) -> "vec2": ...

    def __add__(self, other: "vec2 | float") -> "vec2":
        if isinstance(other, vec2):
            return vec2(self.x + other.x, self.y + other.y)
        return vec2(self.x + other, self.y + other)

    def __radd__(self, other: float) -> "vec2":
        return vec2(other + self.x, other + self.y)

    @overload
    def __sub__(self, other: "vec2") -> "vec2": ...
    @overload
    def __sub__(self, other: float) -> "vec2": ...

    def __sub__(self, other: "vec2 | float") -> "vec2":
        if isinstance(other, vec2):
            return vec2(self.x - other.x, self.y - other.y)
        return vec2(self.x - other, self.y - other)

    def __rsub__(self, other: float) -> "vec2":
        return vec2(other - self.x, other - self.y)

    @overload
    def __mul__(self, other: "vec2") -> "vec2": ...
    @overload
    def __mul__(self, other: float) -> "vec2": ...

    def __mul__(self, other: "vec2 | float") -> "vec2":
        if isinstance(other, vec2):
            return vec2(self.x * other.x, self.y * other.y)
        return vec2(self.x * other, self.y * other)

    def __rmul__(self, other: float) -> "vec2":
        return vec2(other * self.x, other * self.y)

    @overload
    def __truediv__(self, other: "vec2") -> "vec2": ...
    @overload
    def __truediv__(self, other: float) -> "vec2": ...

    def __truediv__(self, other: "vec2 | float") -> "vec2":
        if isinstance(other, vec2):
            return vec2(self.x / other.x, self.y / other.y)
        return vec2(self.x / other, self.y / other)

    def __rtruediv__(self, other: float) -> "vec2":
        return vec2(other / self.x, other / self.y)

    def __neg__(self) -> "vec2":
        return vec2(-self.x, -self.y)

    def __pos__(self) -> "vec2":
        return vec2(self.x, self.y)

    def __getitem__(self, index: int) -> float:
        return float(self.data[index])

    def __setitem__(self, index: int, value: float) -> None:
        self.data[index] = value

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter([self.x, self.y])

    def __repr__(self) -> str:
        return f"vec2({self.x}, {self.y})"

    def __str__(self) -> str:
        return f"vec2({self.x}, {self.y})"


class vec3:
    """GLSL vec3 type - 3-component float vector."""

    __slots__ = ("data",)

    @overload
    def __init__(self, x: float, y: float, z: float) -> None: ...
    @overload
    def __init__(self, x: float) -> None: ...
    @overload
    def __init__(self, x: vec2, y: float) -> None: ...
    @overload
    def __init__(self, x: float, y: vec2) -> None: ...
    @overload
    def __init__(self, x: "vec3") -> None: ...

    def __init__(
        self,
        x: "float | vec2 | vec3" = 0.0,
        y: "float | vec2 | None" = None,
        z: float | None = None,
    ) -> None:
        if isinstance(x, vec3):
            self.data = x.data.copy()
        elif isinstance(x, vec2) and isinstance(y, int | float):
            self.data = np.array([x.x, x.y, y], dtype=np.float32)
        elif isinstance(x, int | float) and isinstance(y, vec2):
            self.data = np.array([x, y.x, y.y], dtype=np.float32)
        elif y is None:
            self.data = np.array([x, x, x], dtype=np.float32)
        else:
            self.data = np.array([x, y, z], dtype=np.float32)

    # Component accessors
    @property
    def x(self) -> float:
        return float(self.data[0])

    @x.setter
    def x(self, value: float) -> None:
        self.data[0] = value

    @property
    def y(self) -> float:
        return float(self.data[1])

    @y.setter
    def y(self, value: float) -> None:
        self.data[1] = value

    @property
    def z(self) -> float:
        return float(self.data[2])

    @z.setter
    def z(self, value: float) -> None:
        self.data[2] = value

    # Aliases r,g,b for x,y,z
    @property
    def r(self) -> float:
        return self.x

    @r.setter
    def r(self, value: float) -> None:
        self.x = value

    @property
    def g(self) -> float:
        return self.y

    @g.setter
    def g(self, value: float) -> None:
        self.y = value

    @property
    def b(self) -> float:
        return self.z

    @b.setter
    def b(self, value: float) -> None:
        self.z = value

    # Swizzle properties - vec2 from vec3
    @property
    def xx(self) -> vec2:
        return vec2(self.x, self.x)

    @property
    def xy(self) -> vec2:
        return vec2(self.x, self.y)

    @property
    def xz(self) -> vec2:
        return vec2(self.x, self.z)

    @property
    def yx(self) -> vec2:
        return vec2(self.y, self.x)

    @property
    def yy(self) -> vec2:
        return vec2(self.y, self.y)

    @property
    def yz(self) -> vec2:
        return vec2(self.y, self.z)

    @property
    def zx(self) -> vec2:
        return vec2(self.z, self.x)

    @property
    def zy(self) -> vec2:
        return vec2(self.z, self.y)

    @property
    def zz(self) -> vec2:
        return vec2(self.z, self.z)

    # Swizzle properties - vec3 from vec3
    @property
    def xxx(self) -> "vec3":
        return vec3(self.x, self.x, self.x)

    @property
    def xxy(self) -> "vec3":
        return vec3(self.x, self.x, self.y)

    @property
    def xxz(self) -> "vec3":
        return vec3(self.x, self.x, self.z)

    @property
    def xyx(self) -> "vec3":
        return vec3(self.x, self.y, self.x)

    @property
    def xyy(self) -> "vec3":
        return vec3(self.x, self.y, self.y)

    @property
    def xyz(self) -> "vec3":
        return vec3(self.x, self.y, self.z)

    @property
    def xzx(self) -> "vec3":
        return vec3(self.x, self.z, self.x)

    @property
    def xzy(self) -> "vec3":
        return vec3(self.x, self.z, self.y)

    @property
    def xzz(self) -> "vec3":
        return vec3(self.x, self.z, self.z)

    @property
    def yxx(self) -> "vec3":
        return vec3(self.y, self.x, self.x)

    @property
    def yxy(self) -> "vec3":
        return vec3(self.y, self.x, self.y)

    @property
    def yxz(self) -> "vec3":
        return vec3(self.y, self.x, self.z)

    @property
    def yyx(self) -> "vec3":
        return vec3(self.y, self.y, self.x)

    @property
    def yyy(self) -> "vec3":
        return vec3(self.y, self.y, self.y)

    @property
    def yyz(self) -> "vec3":
        return vec3(self.y, self.y, self.z)

    @property
    def yzx(self) -> "vec3":
        return vec3(self.y, self.z, self.x)

    @property
    def yzy(self) -> "vec3":
        return vec3(self.y, self.z, self.y)

    @property
    def yzz(self) -> "vec3":
        return vec3(self.y, self.z, self.z)

    @property
    def zxx(self) -> "vec3":
        return vec3(self.z, self.x, self.x)

    @property
    def zxy(self) -> "vec3":
        return vec3(self.z, self.x, self.y)

    @property
    def zxz(self) -> "vec3":
        return vec3(self.z, self.x, self.z)

    @property
    def zyx(self) -> "vec3":
        return vec3(self.z, self.y, self.x)

    @property
    def zyy(self) -> "vec3":
        return vec3(self.z, self.y, self.y)

    @property
    def zyz(self) -> "vec3":
        return vec3(self.z, self.y, self.z)

    @property
    def zzx(self) -> "vec3":
        return vec3(self.z, self.z, self.x)

    @property
    def zzy(self) -> "vec3":
        return vec3(self.z, self.z, self.y)

    @property
    def zzz(self) -> "vec3":
        return vec3(self.z, self.z, self.z)

    # RGB swizzles
    @property
    def rgb(self) -> "vec3":
        return vec3(self.x, self.y, self.z)

    @property
    def rbg(self) -> "vec3":
        return vec3(self.x, self.z, self.y)

    @property
    def grb(self) -> "vec3":
        return vec3(self.y, self.x, self.z)

    @property
    def gbr(self) -> "vec3":
        return vec3(self.y, self.z, self.x)

    @property
    def brg(self) -> "vec3":
        return vec3(self.z, self.x, self.y)

    @property
    def bgr(self) -> "vec3":
        return vec3(self.z, self.y, self.x)

    # Arithmetic operators
    @overload
    def __add__(self, other: "vec3") -> "vec3": ...
    @overload
    def __add__(self, other: float) -> "vec3": ...

    def __add__(self, other: "vec3 | float") -> "vec3":
        if isinstance(other, vec3):
            return vec3(self.x + other.x, self.y + other.y, self.z + other.z)
        return vec3(self.x + other, self.y + other, self.z + other)

    def __radd__(self, other: float) -> "vec3":
        return vec3(other + self.x, other + self.y, other + self.z)

    @overload
    def __sub__(self, other: "vec3") -> "vec3": ...
    @overload
    def __sub__(self, other: float) -> "vec3": ...

    def __sub__(self, other: "vec3 | float") -> "vec3":
        if isinstance(other, vec3):
            return vec3(self.x - other.x, self.y - other.y, self.z - other.z)
        return vec3(self.x - other, self.y - other, self.z - other)

    def __rsub__(self, other: float) -> "vec3":
        return vec3(other - self.x, other - self.y, other - self.z)

    @overload
    def __mul__(self, other: "vec3") -> "vec3": ...
    @overload
    def __mul__(self, other: float) -> "vec3": ...

    def __mul__(self, other: "vec3 | float") -> "vec3":
        if isinstance(other, vec3):
            return vec3(self.x * other.x, self.y * other.y, self.z * other.z)
        return vec3(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other: float) -> "vec3":
        return vec3(other * self.x, other * self.y, other * self.z)

    @overload
    def __truediv__(self, other: "vec3") -> "vec3": ...
    @overload
    def __truediv__(self, other: float) -> "vec3": ...

    def __truediv__(self, other: "vec3 | float") -> "vec3":
        if isinstance(other, vec3):
            return vec3(self.x / other.x, self.y / other.y, self.z / other.z)
        return vec3(self.x / other, self.y / other, self.z / other)

    def __rtruediv__(self, other: float) -> "vec3":
        return vec3(other / self.x, other / self.y, other / self.z)

    def __neg__(self) -> "vec3":
        return vec3(-self.x, -self.y, -self.z)

    def __pos__(self) -> "vec3":
        return vec3(self.x, self.y, self.z)

    def __getitem__(self, index: int) -> float:
        return float(self.data[index])

    def __setitem__(self, index: int, value: float) -> None:
        self.data[index] = value

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter([self.x, self.y, self.z])

    def __repr__(self) -> str:
        return f"vec3({self.x}, {self.y}, {self.z})"

    def __str__(self) -> str:
        return f"vec3({self.x}, {self.y}, {self.z})"


class vec4:
    """GLSL vec4 type - 4-component float vector."""

    __slots__ = ("data",)

    @overload
    def __init__(self, x: float, y: float, z: float, w: float) -> None: ...
    @overload
    def __init__(self, x: float) -> None: ...
    @overload
    def __init__(self, x: vec3, y: float) -> None: ...
    @overload
    def __init__(self, x: float, y: vec3) -> None: ...
    @overload
    def __init__(self, x: vec2, y: vec2) -> None: ...
    @overload
    def __init__(self, x: vec2, y: float, z: float) -> None: ...
    @overload
    def __init__(self, x: "vec4") -> None: ...

    def __init__(
        self,
        x: "float | vec2 | vec3 | vec4" = 0.0,
        y: "float | vec2 | vec3 | None" = None,
        z: "float | vec2 | None" = None,
        w: float | None = None,
    ) -> None:
        if isinstance(x, vec4):
            self.data = x.data.copy()
        elif isinstance(x, vec3) and isinstance(y, int | float):
            self.data = np.array([x.x, x.y, x.z, y], dtype=np.float32)
        elif isinstance(x, int | float) and isinstance(y, vec3):
            self.data = np.array([x, y.x, y.y, y.z], dtype=np.float32)
        elif isinstance(x, vec2) and isinstance(y, vec2):
            self.data = np.array([x.x, x.y, y.x, y.y], dtype=np.float32)
        elif (
            isinstance(x, vec2)
            and isinstance(y, int | float)
            and isinstance(z, int | float)
        ):
            self.data = np.array([x.x, x.y, y, z], dtype=np.float32)
        elif y is None:
            self.data = np.array([x, x, x, x], dtype=np.float32)
        else:
            self.data = np.array([x, y, z, w], dtype=np.float32)

    # Component accessors
    @property
    def x(self) -> float:
        return float(self.data[0])

    @x.setter
    def x(self, value: float) -> None:
        self.data[0] = value

    @property
    def y(self) -> float:
        return float(self.data[1])

    @y.setter
    def y(self, value: float) -> None:
        self.data[1] = value

    @property
    def z(self) -> float:
        return float(self.data[2])

    @z.setter
    def z(self, value: float) -> None:
        self.data[2] = value

    @property
    def w(self) -> float:
        return float(self.data[3])

    @w.setter
    def w(self, value: float) -> None:
        self.data[3] = value

    # Aliases r,g,b,a for x,y,z,w
    @property
    def r(self) -> float:
        return self.x

    @r.setter
    def r(self, value: float) -> None:
        self.x = value

    @property
    def g(self) -> float:
        return self.y

    @g.setter
    def g(self, value: float) -> None:
        self.y = value

    @property
    def b(self) -> float:
        return self.z

    @b.setter
    def b(self, value: float) -> None:
        self.z = value

    @property
    def a(self) -> float:
        return self.w

    @a.setter
    def a(self, value: float) -> None:
        self.w = value

    # Swizzle properties - vec2 from vec4
    @property
    def xx(self) -> vec2:
        return vec2(self.x, self.x)

    @property
    def xy(self) -> vec2:
        return vec2(self.x, self.y)

    @property
    def xz(self) -> vec2:
        return vec2(self.x, self.z)

    @property
    def xw(self) -> vec2:
        return vec2(self.x, self.w)

    @property
    def yx(self) -> vec2:
        return vec2(self.y, self.x)

    @property
    def yy(self) -> vec2:
        return vec2(self.y, self.y)

    @property
    def yz(self) -> vec2:
        return vec2(self.y, self.z)

    @property
    def yw(self) -> vec2:
        return vec2(self.y, self.w)

    @property
    def zx(self) -> vec2:
        return vec2(self.z, self.x)

    @property
    def zy(self) -> vec2:
        return vec2(self.z, self.y)

    @property
    def zz(self) -> vec2:
        return vec2(self.z, self.z)

    @property
    def zw(self) -> vec2:
        return vec2(self.z, self.w)

    @property
    def wx(self) -> vec2:
        return vec2(self.w, self.x)

    @property
    def wy(self) -> vec2:
        return vec2(self.w, self.y)

    @property
    def wz(self) -> vec2:
        return vec2(self.w, self.z)

    @property
    def ww(self) -> vec2:
        return vec2(self.w, self.w)

    # Swizzle properties - vec3 from vec4
    @property
    def xyz(self) -> vec3:
        return vec3(self.x, self.y, self.z)

    @property
    def xyw(self) -> vec3:
        return vec3(self.x, self.y, self.w)

    @property
    def xzy(self) -> vec3:
        return vec3(self.x, self.z, self.y)

    @property
    def xzw(self) -> vec3:
        return vec3(self.x, self.z, self.w)

    @property
    def xwy(self) -> vec3:
        return vec3(self.x, self.w, self.y)

    @property
    def xwz(self) -> vec3:
        return vec3(self.x, self.w, self.z)

    @property
    def yxz(self) -> vec3:
        return vec3(self.y, self.x, self.z)

    @property
    def yxw(self) -> vec3:
        return vec3(self.y, self.x, self.w)

    @property
    def yzx(self) -> vec3:
        return vec3(self.y, self.z, self.x)

    @property
    def yzw(self) -> vec3:
        return vec3(self.y, self.z, self.w)

    @property
    def ywx(self) -> vec3:
        return vec3(self.y, self.w, self.x)

    @property
    def ywz(self) -> vec3:
        return vec3(self.y, self.w, self.z)

    @property
    def zxy(self) -> vec3:
        return vec3(self.z, self.x, self.y)

    @property
    def zxw(self) -> vec3:
        return vec3(self.z, self.x, self.w)

    @property
    def zyx(self) -> vec3:
        return vec3(self.z, self.y, self.x)

    @property
    def zyw(self) -> vec3:
        return vec3(self.z, self.y, self.w)

    @property
    def zwx(self) -> vec3:
        return vec3(self.z, self.w, self.x)

    @property
    def zwy(self) -> vec3:
        return vec3(self.z, self.w, self.y)

    @property
    def wxy(self) -> vec3:
        return vec3(self.w, self.x, self.y)

    @property
    def wxz(self) -> vec3:
        return vec3(self.w, self.x, self.z)

    @property
    def wyx(self) -> vec3:
        return vec3(self.w, self.y, self.x)

    @property
    def wyz(self) -> vec3:
        return vec3(self.w, self.y, self.z)

    @property
    def wzx(self) -> vec3:
        return vec3(self.w, self.z, self.x)

    @property
    def wzy(self) -> vec3:
        return vec3(self.w, self.z, self.y)

    # RGB(A) swizzles
    @property
    def rgb(self) -> vec3:
        return vec3(self.x, self.y, self.z)

    @property
    def rgba(self) -> "vec4":
        return vec4(self.x, self.y, self.z, self.w)

    # Arithmetic operators
    @overload
    def __add__(self, other: "vec4") -> "vec4": ...
    @overload
    def __add__(self, other: float) -> "vec4": ...

    def __add__(self, other: "vec4 | float") -> "vec4":
        if isinstance(other, vec4):
            return vec4(
                self.x + other.x, self.y + other.y, self.z + other.z, self.w + other.w
            )
        return vec4(self.x + other, self.y + other, self.z + other, self.w + other)

    def __radd__(self, other: float) -> "vec4":
        return vec4(other + self.x, other + self.y, other + self.z, other + self.w)

    @overload
    def __sub__(self, other: "vec4") -> "vec4": ...
    @overload
    def __sub__(self, other: float) -> "vec4": ...

    def __sub__(self, other: "vec4 | float") -> "vec4":
        if isinstance(other, vec4):
            return vec4(
                self.x - other.x, self.y - other.y, self.z - other.z, self.w - other.w
            )
        return vec4(self.x - other, self.y - other, self.z - other, self.w - other)

    def __rsub__(self, other: float) -> "vec4":
        return vec4(other - self.x, other - self.y, other - self.z, other - self.w)

    @overload
    def __mul__(self, other: "vec4") -> "vec4": ...
    @overload
    def __mul__(self, other: float) -> "vec4": ...

    def __mul__(self, other: "vec4 | float") -> "vec4":
        if isinstance(other, vec4):
            return vec4(
                self.x * other.x, self.y * other.y, self.z * other.z, self.w * other.w
            )
        return vec4(self.x * other, self.y * other, self.z * other, self.w * other)

    def __rmul__(self, other: float) -> "vec4":
        return vec4(other * self.x, other * self.y, other * self.z, other * self.w)

    @overload
    def __truediv__(self, other: "vec4") -> "vec4": ...
    @overload
    def __truediv__(self, other: float) -> "vec4": ...

    def __truediv__(self, other: "vec4 | float") -> "vec4":
        if isinstance(other, vec4):
            return vec4(
                self.x / other.x, self.y / other.y, self.z / other.z, self.w / other.w
            )
        return vec4(self.x / other, self.y / other, self.z / other, self.w / other)

    def __rtruediv__(self, other: float) -> "vec4":
        return vec4(other / self.x, other / self.y, other / self.z, other / self.w)

    def __neg__(self) -> "vec4":
        return vec4(-self.x, -self.y, -self.z, -self.w)

    def __pos__(self) -> "vec4":
        return vec4(self.x, self.y, self.z, self.w)

    def __getitem__(self, index: int) -> float:
        return float(self.data[index])

    def __setitem__(self, index: int, value: float) -> None:
        self.data[index] = value

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter([self.x, self.y, self.z, self.w])

    def __repr__(self) -> str:
        return f"vec4({self.x}, {self.y}, {self.z}, {self.w})"

    def __str__(self) -> str:
        return f"vec4({self.x}, {self.y}, {self.z}, {self.w})"


# Matrix classes
class mat2:
    """GLSL mat2 type - 2x2 float matrix."""

    __slots__ = ("data",)

    def __init__(self, *args: float) -> None:
        mat2_size = 4
        if len(args) == mat2_size:
            self.data = np.array(
                [[args[0], args[1]], [args[2], args[3]]], dtype=np.float32
            )
        else:
            self.data = np.eye(2, dtype=np.float32)

    def __getitem__(self, index: int) -> vec2:
        row = self.data[index]
        return vec2(float(row[0]), float(row[1]))


class mat3:
    """GLSL mat3 type - 3x3 float matrix."""

    __slots__ = ("data",)

    def __init__(self, *args: float) -> None:
        mat3_size = 9
        if len(args) == mat3_size:
            self.data = np.array(
                [
                    [args[0], args[1], args[2]],
                    [args[3], args[4], args[5]],
                    [args[6], args[7], args[8]],
                ],
                dtype=np.float32,
            )
        else:
            self.data = np.eye(3, dtype=np.float32)

    def __getitem__(self, index: int) -> vec3:
        row = self.data[index]
        return vec3(float(row[0]), float(row[1]), float(row[2]))


class mat4:
    """GLSL mat4 type - 4x4 float matrix."""

    __slots__ = ("data",)

    def __init__(self, *args: float) -> None:
        mat4_size = 16
        if len(args) == mat4_size:
            self.data = np.array(
                [
                    [args[0], args[1], args[2], args[3]],
                    [args[4], args[5], args[6], args[7]],
                    [args[8], args[9], args[10], args[11]],
                    [args[12], args[13], args[14], args[15]],
                ],
                dtype=np.float32,
            )
        else:
            self.data = np.eye(4, dtype=np.float32)

    def __getitem__(self, index: int) -> vec4:
        row = self.data[index]
        return vec4(float(row[0]), float(row[1]), float(row[2]), float(row[3]))


# =============================================================================
# GLSL built-in functions
# =============================================================================


def sin(x: float) -> float:
    return float(np.sin(x))


def cos(x: float) -> float:
    return float(np.cos(x))


def tan(x: float) -> float:
    return float(np.tan(x))


def asin(x: float) -> float:
    return float(np.arcsin(x))


def acos(x: float) -> float:
    return float(np.arccos(x))


@overload
def atan(y: float) -> float: ...
@overload
def atan(y: float, x: float) -> float: ...


def atan(y: float, x: float | None = None) -> float:
    """Arctangent. Single arg: atan(y). Two args: atan(y, x) like atan2."""
    if x is None:
        return float(np.arctan(y))
    return float(np.arctan2(y, x))


# abs - redefined builtin
@overload
def abs(x: float) -> float: ...
@overload
def abs(x: vec2) -> vec2: ...
@overload
def abs(x: vec3) -> vec3: ...
@overload
def abs(x: vec4) -> vec4: ...


def abs(x: float | vec2 | vec3 | vec4) -> float | vec2 | vec3 | vec4:
    if isinstance(x, vec2):
        return vec2(float(np.abs(x.x)), float(np.abs(x.y)))
    if isinstance(x, vec3):
        return vec3(float(np.abs(x.x)), float(np.abs(x.y)), float(np.abs(x.z)))
    if isinstance(x, vec4):
        return vec4(
            float(np.abs(x.x)),
            float(np.abs(x.y)),
            float(np.abs(x.z)),
            float(np.abs(x.w)),
        )
    return float(np.abs(x))


# length
@overload
def length(v: vec2) -> float: ...
@overload
def length(v: vec3) -> float: ...
@overload
def length(v: vec4) -> float: ...


def length(v: vec2 | vec3 | vec4) -> float:
    return float(np.linalg.norm(v.data))


# distance
@overload
def distance(p0: vec2, p1: vec2) -> float: ...
@overload
def distance(p0: vec3, p1: vec3) -> float: ...
@overload
def distance(p0: vec4, p1: vec4) -> float: ...


def distance(p0: vec2 | vec3 | vec4, p1: vec2 | vec3 | vec4) -> float:
    return float(np.linalg.norm(p0.data - p1.data))


# min - supports 2 or 3 arguments
@overload
def min(a: float, b: float) -> float: ...
@overload
def min(a: float, b: float, c: float) -> float: ...
@overload
def min(a: vec2, b: vec2) -> vec2: ...
@overload
def min(a: vec2, b: float) -> vec2: ...
@overload
def min(a: vec3, b: vec3) -> vec3: ...
@overload
def min(a: vec3, b: float) -> vec3: ...
@overload
def min(a: vec4, b: vec4) -> vec4: ...
@overload
def min(a: vec4, b: float) -> vec4: ...


def min(
    a: float | vec2 | vec3 | vec4,
    b: float | vec2 | vec3 | vec4,
    c: float | None = None,
) -> float | vec2 | vec3 | vec4:
    # Handle 3-argument case (floats only)
    if c is not None:
        if isinstance(a, int | float) and isinstance(b, int | float):
            return float(np.minimum(np.minimum(a, b), c))
        raise TypeError("min() with 3 args only supports floats")

    # vec2
    if isinstance(a, vec2):
        if isinstance(b, vec2):
            return vec2(
                a.x if a.x < b.x else b.x,
                a.y if a.y < b.y else b.y,
            )
        if isinstance(b, int | float):
            bf = float(b)
            return vec2(a.x if a.x < bf else bf, a.y if a.y < bf else bf)

    # vec3
    if isinstance(a, vec3):
        if isinstance(b, vec3):
            return vec3(
                a.x if a.x < b.x else b.x,
                a.y if a.y < b.y else b.y,
                a.z if a.z < b.z else b.z,
            )
        if isinstance(b, int | float):
            bf = float(b)
            return vec3(
                a.x if a.x < bf else bf,
                a.y if a.y < bf else bf,
                a.z if a.z < bf else bf,
            )

    # vec4
    if isinstance(a, vec4):
        if isinstance(b, vec4):
            return vec4(
                a.x if a.x < b.x else b.x,
                a.y if a.y < b.y else b.y,
                a.z if a.z < b.z else b.z,
                a.w if a.w < b.w else b.w,
            )
        if isinstance(b, int | float):
            bf = float(b)
            return vec4(
                a.x if a.x < bf else bf,
                a.y if a.y < bf else bf,
                a.z if a.z < bf else bf,
                a.w if a.w < bf else bf,
            )

    # float
    if isinstance(a, int | float) and isinstance(b, int | float):
        return float(a) if a < b else float(b)

    raise TypeError(f"min() received incompatible types: {type(a)} and {type(b)}")


# max - supports 2 or 3 arguments
@overload
def max(a: float, b: float) -> float: ...
@overload
def max(a: float, b: float, c: float) -> float: ...
@overload
def max(a: vec2, b: vec2) -> vec2: ...
@overload
def max(a: vec2, b: float) -> vec2: ...
@overload
def max(a: vec3, b: vec3) -> vec3: ...
@overload
def max(a: vec3, b: float) -> vec3: ...
@overload
def max(a: vec4, b: vec4) -> vec4: ...
@overload
def max(a: vec4, b: float) -> vec4: ...


def max(
    a: float | vec2 | vec3 | vec4,
    b: float | vec2 | vec3 | vec4,
    c: float | None = None,
) -> float | vec2 | vec3 | vec4:
    # Handle 3-argument case (floats only)
    if c is not None:
        if isinstance(a, int | float) and isinstance(b, int | float):
            return float(np.maximum(np.maximum(a, b), c))
        raise TypeError("max() with 3 args only supports floats")

    # vec2
    if isinstance(a, vec2):
        if isinstance(b, vec2):
            return vec2(
                a.x if a.x > b.x else b.x,
                a.y if a.y > b.y else b.y,
            )
        if isinstance(b, int | float):
            bf = float(b)
            return vec2(a.x if a.x > bf else bf, a.y if a.y > bf else bf)

    # vec3
    if isinstance(a, vec3):
        if isinstance(b, vec3):
            return vec3(
                a.x if a.x > b.x else b.x,
                a.y if a.y > b.y else b.y,
                a.z if a.z > b.z else b.z,
            )
        if isinstance(b, int | float):
            bf = float(b)
            return vec3(
                a.x if a.x > bf else bf,
                a.y if a.y > bf else bf,
                a.z if a.z > bf else bf,
            )

    # vec4
    if isinstance(a, vec4):
        if isinstance(b, vec4):
            return vec4(
                a.x if a.x > b.x else b.x,
                a.y if a.y > b.y else b.y,
                a.z if a.z > b.z else b.z,
                a.w if a.w > b.w else b.w,
            )
        if isinstance(b, int | float):
            bf = float(b)
            return vec4(
                a.x if a.x > bf else bf,
                a.y if a.y > bf else bf,
                a.z if a.z > bf else bf,
                a.w if a.w > bf else bf,
            )

    # float
    if isinstance(a, int | float) and isinstance(b, int | float):
        return float(a) if a > b else float(b)

    raise TypeError(f"max() received incompatible types: {type(a)} and {type(b)}")


# normalize
@overload
def normalize(v: vec2) -> vec2: ...
@overload
def normalize(v: vec3) -> vec3: ...
@overload
def normalize(v: vec4) -> vec4: ...


def normalize(v: vec2 | vec3 | vec4) -> vec2 | vec3 | vec4:
    norm = float(np.linalg.norm(v.data))
    if norm == 0:
        return v

    if isinstance(v, vec2):
        return vec2(v.x / norm, v.y / norm)
    if isinstance(v, vec3):
        return vec3(v.x / norm, v.y / norm, v.z / norm)
    return vec4(v.x / norm, v.y / norm, v.z / norm, v.w / norm)


# cross
def cross(a: vec3, b: vec3) -> vec3:
    result = np.cross(a.data, b.data)
    return vec3(float(result[0]), float(result[1]), float(result[2]))


# dot
@overload
def dot(a: vec2, b: vec2) -> float: ...
@overload
def dot(a: vec3, b: vec3) -> float: ...
@overload
def dot(a: vec4, b: vec4) -> float: ...


def dot(a: vec2 | vec3 | vec4, b: vec2 | vec3 | vec4) -> float:
    return float(np.dot(a.data, b.data))


# mix
@overload
def mix(x: float, y: float, a: float) -> float: ...
@overload
def mix(x: vec2, y: vec2, a: float) -> vec2: ...
@overload
def mix(x: vec2, y: vec2, a: vec2) -> vec2: ...
@overload
def mix(x: vec3, y: vec3, a: float) -> vec3: ...
@overload
def mix(x: vec3, y: vec3, a: vec3) -> vec3: ...
@overload
def mix(x: vec4, y: vec4, a: float) -> vec4: ...
@overload
def mix(x: vec4, y: vec4, a: vec4) -> vec4: ...


def mix(
    x: float | vec2 | vec3 | vec4,
    y: float | vec2 | vec3 | vec4,
    a: float | vec2 | vec3 | vec4,
) -> float | vec2 | vec3 | vec4:
    # float, float, float
    if (
        isinstance(x, int | float)
        and isinstance(y, int | float)
        and isinstance(a, int | float)
    ):
        return float(x * (1 - a) + y * a)

    # vec2
    if isinstance(x, vec2) and isinstance(y, vec2):
        if isinstance(a, vec2):
            return vec2(
                x.x * (1 - a.x) + y.x * a.x,
                x.y * (1 - a.y) + y.y * a.y,
            )
        if isinstance(a, int | float):
            af = float(a)
            return vec2(
                x.x * (1 - af) + y.x * af,
                x.y * (1 - af) + y.y * af,
            )

    # vec3
    if isinstance(x, vec3) and isinstance(y, vec3):
        if isinstance(a, vec3):
            return vec3(
                x.x * (1 - a.x) + y.x * a.x,
                x.y * (1 - a.y) + y.y * a.y,
                x.z * (1 - a.z) + y.z * a.z,
            )
        if isinstance(a, int | float):
            af = float(a)
            return vec3(
                x.x * (1 - af) + y.x * af,
                x.y * (1 - af) + y.y * af,
                x.z * (1 - af) + y.z * af,
            )

    # vec4
    if isinstance(x, vec4) and isinstance(y, vec4):
        if isinstance(a, vec4):
            return vec4(
                x.x * (1 - a.x) + y.x * a.x,
                x.y * (1 - a.y) + y.y * a.y,
                x.z * (1 - a.z) + y.z * a.z,
                x.w * (1 - a.w) + y.w * a.w,
            )
        if isinstance(a, int | float):
            af = float(a)
            return vec4(
                x.x * (1 - af) + y.x * af,
                x.y * (1 - af) + y.y * af,
                x.z * (1 - af) + y.z * af,
                x.w * (1 - af) + y.w * af,
            )

    raise TypeError(
        f"mix() received incompatible types: {type(x)}, {type(y)}, {type(a)}"
    )


# smoothstep
@overload
def smoothstep(edge0: float, edge1: float, x: float) -> float: ...
@overload
def smoothstep(edge0: vec2, edge1: vec2, x: vec2) -> vec2: ...
@overload
def smoothstep(edge0: vec3, edge1: vec3, x: vec3) -> vec3: ...
@overload
def smoothstep(edge0: vec4, edge1: vec4, x: vec4) -> vec4: ...
@overload
def smoothstep(edge0: float, edge1: float, x: vec2) -> vec2: ...
@overload
def smoothstep(edge0: float, edge1: float, x: vec3) -> vec3: ...
@overload
def smoothstep(edge0: float, edge1: float, x: vec4) -> vec4: ...


def smoothstep(
    edge0: float | vec2 | vec3 | vec4,
    edge1: float | vec2 | vec3 | vec4,
    x: float | vec2 | vec3 | vec4,
) -> float | vec2 | vec3 | vec4:
    def _ss(e0: float, e1: float, v: float) -> float:
        t = float(np.clip((v - e0) / (e1 - e0), 0.0, 1.0))
        return t * t * (3.0 - 2.0 * t)

    # float, float, float
    if (
        isinstance(edge0, int | float)
        and isinstance(edge1, int | float)
        and isinstance(x, int | float)
    ):
        return _ss(edge0, edge1, x)

    # float edges with vector x
    if isinstance(edge0, int | float) and isinstance(edge1, int | float):
        if isinstance(x, vec2):
            return vec2(_ss(edge0, edge1, x.x), _ss(edge0, edge1, x.y))
        if isinstance(x, vec3):
            return vec3(
                _ss(edge0, edge1, x.x), _ss(edge0, edge1, x.y), _ss(edge0, edge1, x.z)
            )
        if isinstance(x, vec4):
            return vec4(
                _ss(edge0, edge1, x.x),
                _ss(edge0, edge1, x.y),
                _ss(edge0, edge1, x.z),
                _ss(edge0, edge1, x.w),
            )

    # vec2
    if isinstance(edge0, vec2) and isinstance(edge1, vec2) and isinstance(x, vec2):
        return vec2(
            _ss(edge0.x, edge1.x, x.x),
            _ss(edge0.y, edge1.y, x.y),
        )

    # vec3
    if isinstance(edge0, vec3) and isinstance(edge1, vec3) and isinstance(x, vec3):
        return vec3(
            _ss(edge0.x, edge1.x, x.x),
            _ss(edge0.y, edge1.y, x.y),
            _ss(edge0.z, edge1.z, x.z),
        )

    # vec4
    if isinstance(edge0, vec4) and isinstance(edge1, vec4) and isinstance(x, vec4):
        return vec4(
            _ss(edge0.x, edge1.x, x.x),
            _ss(edge0.y, edge1.y, x.y),
            _ss(edge0.z, edge1.z, x.z),
            _ss(edge0.w, edge1.w, x.w),
        )

    raise TypeError(f"smoothstep() types: {type(edge0)}, {type(edge1)}, {type(x)}")


def radians(degrees: float) -> float:
    return float(np.radians(degrees))


def sqrt(x: float) -> float:
    return float(np.sqrt(x))


def pow(x: float, y: float) -> float:
    return float(np.power(x, y))


# fract
@overload
def fract(x: float) -> float: ...
@overload
def fract(x: vec2) -> vec2: ...
@overload
def fract(x: vec3) -> vec3: ...
@overload
def fract(x: vec4) -> vec4: ...


def fract(x: float | vec2 | vec3 | vec4) -> float | vec2 | vec3 | vec4:
    if isinstance(x, vec2):
        return vec2(x.x - np.floor(x.x), x.y - np.floor(x.y))
    if isinstance(x, vec3):
        return vec3(x.x - np.floor(x.x), x.y - np.floor(x.y), x.z - np.floor(x.z))
    if isinstance(x, vec4):
        return vec4(
            x.x - np.floor(x.x),
            x.y - np.floor(x.y),
            x.z - np.floor(x.z),
            x.w - np.floor(x.w),
        )
    return float(x - np.floor(x))


# floor
@overload
def floor(x: float) -> float: ...
@overload
def floor(x: vec2) -> vec2: ...
@overload
def floor(x: vec3) -> vec3: ...
@overload
def floor(x: vec4) -> vec4: ...


def floor(x: float | vec2 | vec3 | vec4) -> float | vec2 | vec3 | vec4:
    if isinstance(x, vec2):
        return vec2(float(np.floor(x.x)), float(np.floor(x.y)))
    if isinstance(x, vec3):
        return vec3(float(np.floor(x.x)), float(np.floor(x.y)), float(np.floor(x.z)))
    if isinstance(x, vec4):
        return vec4(
            float(np.floor(x.x)),
            float(np.floor(x.y)),
            float(np.floor(x.z)),
            float(np.floor(x.w)),
        )
    return float(np.floor(x))


# ceil
@overload
def ceil(x: float) -> float: ...
@overload
def ceil(x: vec2) -> vec2: ...
@overload
def ceil(x: vec3) -> vec3: ...
@overload
def ceil(x: vec4) -> vec4: ...


def ceil(x: float | vec2 | vec3 | vec4) -> float | vec2 | vec3 | vec4:
    if isinstance(x, vec2):
        return vec2(float(np.ceil(x.x)), float(np.ceil(x.y)))
    if isinstance(x, vec3):
        return vec3(float(np.ceil(x.x)), float(np.ceil(x.y)), float(np.ceil(x.z)))
    if isinstance(x, vec4):
        return vec4(
            float(np.ceil(x.x)),
            float(np.ceil(x.y)),
            float(np.ceil(x.z)),
            float(np.ceil(x.w)),
        )
    return float(np.ceil(x))


# sign
@overload
def sign(x: float) -> float: ...
@overload
def sign(x: vec2) -> vec2: ...
@overload
def sign(x: vec3) -> vec3: ...
@overload
def sign(x: vec4) -> vec4: ...


def sign(x: float | vec2 | vec3 | vec4) -> float | vec2 | vec3 | vec4:
    if isinstance(x, vec2):
        return vec2(float(np.sign(x.x)), float(np.sign(x.y)))
    if isinstance(x, vec3):
        return vec3(float(np.sign(x.x)), float(np.sign(x.y)), float(np.sign(x.z)))
    if isinstance(x, vec4):
        return vec4(
            float(np.sign(x.x)),
            float(np.sign(x.y)),
            float(np.sign(x.z)),
            float(np.sign(x.w)),
        )
    return float(np.sign(x))


def exp(x: float) -> float:
    return float(np.exp(x))


def exp2(x: float) -> float:
    return float(np.exp2(x))


def log(x: float) -> float:
    return float(np.log(x))


def log2(x: float) -> float:
    return float(np.log2(x))


# mod
@overload
def mod(x: float, y: float) -> float: ...
@overload
def mod(x: vec2, y: float) -> vec2: ...
@overload
def mod(x: vec2, y: vec2) -> vec2: ...
@overload
def mod(x: vec3, y: float) -> vec3: ...
@overload
def mod(x: vec3, y: vec3) -> vec3: ...
@overload
def mod(x: vec4, y: float) -> vec4: ...
@overload
def mod(x: vec4, y: vec4) -> vec4: ...


def mod(
    x: float | vec2 | vec3 | vec4, y: float | vec2 | vec3 | vec4
) -> float | vec2 | vec3 | vec4:
    """GLSL mod: x - y * floor(x/y)"""
    if isinstance(x, vec2):
        if isinstance(y, vec2):
            return vec2(
                x.x - y.x * float(np.floor(x.x / y.x)),
                x.y - y.y * float(np.floor(x.y / y.y)),
            )
        if isinstance(y, int | float):
            yf = float(y)
            return vec2(
                x.x - yf * float(np.floor(x.x / yf)),
                x.y - yf * float(np.floor(x.y / yf)),
            )
    if isinstance(x, vec3):
        if isinstance(y, vec3):
            return vec3(
                x.x - y.x * float(np.floor(x.x / y.x)),
                x.y - y.y * float(np.floor(x.y / y.y)),
                x.z - y.z * float(np.floor(x.z / y.z)),
            )
        if isinstance(y, int | float):
            yf = float(y)
            return vec3(
                x.x - yf * float(np.floor(x.x / yf)),
                x.y - yf * float(np.floor(x.y / yf)),
                x.z - yf * float(np.floor(x.z / yf)),
            )
    if isinstance(x, vec4):
        if isinstance(y, vec4):
            return vec4(
                x.x - y.x * float(np.floor(x.x / y.x)),
                x.y - y.y * float(np.floor(x.y / y.y)),
                x.z - y.z * float(np.floor(x.z / y.z)),
                x.w - y.w * float(np.floor(x.w / y.w)),
            )
        if isinstance(y, int | float):
            yf = float(y)
            return vec4(
                x.x - yf * float(np.floor(x.x / yf)),
                x.y - yf * float(np.floor(x.y / yf)),
                x.z - yf * float(np.floor(x.z / yf)),
                x.w - yf * float(np.floor(x.w / yf)),
            )
    if isinstance(x, int | float) and isinstance(y, int | float):
        xf = float(x)
        yf = float(y)
        return xf - yf * float(np.floor(xf / yf))
    raise TypeError(f"mod() received incompatible types: {type(x)} and {type(y)}")


# step
@overload
def step(edge: float, x: float) -> float: ...
@overload
def step(edge: float, x: vec2) -> vec2: ...
@overload
def step(edge: float, x: vec3) -> vec3: ...
@overload
def step(edge: float, x: vec4) -> vec4: ...
@overload
def step(edge: vec2, x: vec2) -> vec2: ...
@overload
def step(edge: vec3, x: vec3) -> vec3: ...
@overload
def step(edge: vec4, x: vec4) -> vec4: ...


def step(
    edge: float | vec2 | vec3 | vec4, x: float | vec2 | vec3 | vec4
) -> float | vec2 | vec3 | vec4:
    """Returns 0.0 if x < edge, else 1.0"""

    def _step(e: float, v: float) -> float:
        return 0.0 if v < e else 1.0

    if isinstance(x, vec2):
        if isinstance(edge, vec2):
            return vec2(_step(edge.x, x.x), _step(edge.y, x.y))
        if isinstance(edge, int | float):
            ef = float(edge)
            return vec2(_step(ef, x.x), _step(ef, x.y))
    if isinstance(x, vec3):
        if isinstance(edge, vec3):
            return vec3(_step(edge.x, x.x), _step(edge.y, x.y), _step(edge.z, x.z))
        if isinstance(edge, int | float):
            ef = float(edge)
            return vec3(_step(ef, x.x), _step(ef, x.y), _step(ef, x.z))
    if isinstance(x, vec4):
        if isinstance(edge, vec4):
            return vec4(
                _step(edge.x, x.x),
                _step(edge.y, x.y),
                _step(edge.z, x.z),
                _step(edge.w, x.w),
            )
        if isinstance(edge, int | float):
            ef = float(edge)
            return vec4(_step(ef, x.x), _step(ef, x.y), _step(ef, x.z), _step(ef, x.w))
    if isinstance(edge, int | float) and isinstance(x, int | float):
        return _step(float(edge), float(x))
    raise TypeError(f"step() received incompatible types: {type(edge)} and {type(x)}")


# clamp
@overload
def clamp(x: float, min_val: float, max_val: float) -> float: ...
@overload
def clamp(x: vec2, min_val: vec2, max_val: vec2) -> vec2: ...
@overload
def clamp(x: vec2, min_val: float, max_val: float) -> vec2: ...
@overload
def clamp(x: vec3, min_val: vec3, max_val: vec3) -> vec3: ...
@overload
def clamp(x: vec3, min_val: float, max_val: float) -> vec3: ...
@overload
def clamp(x: vec4, min_val: vec4, max_val: vec4) -> vec4: ...
@overload
def clamp(x: vec4, min_val: float, max_val: float) -> vec4: ...


def clamp(
    x: float | vec2 | vec3 | vec4,
    min_val: float | vec2 | vec3 | vec4,
    max_val: float | vec2 | vec3 | vec4,
) -> float | vec2 | vec3 | vec4:
    def _clamp(v: float, lo: float, hi: float) -> float:
        return float(np.clip(v, lo, hi))

    if isinstance(x, vec2):
        if isinstance(min_val, vec2) and isinstance(max_val, vec2):
            return vec2(
                _clamp(x.x, min_val.x, max_val.x), _clamp(x.y, min_val.y, max_val.y)
            )
        if isinstance(min_val, int | float) and isinstance(max_val, int | float):
            return vec2(_clamp(x.x, min_val, max_val), _clamp(x.y, min_val, max_val))
    if isinstance(x, vec3):
        if isinstance(min_val, vec3) and isinstance(max_val, vec3):
            return vec3(
                _clamp(x.x, min_val.x, max_val.x),
                _clamp(x.y, min_val.y, max_val.y),
                _clamp(x.z, min_val.z, max_val.z),
            )
        if isinstance(min_val, int | float) and isinstance(max_val, int | float):
            return vec3(
                _clamp(x.x, min_val, max_val),
                _clamp(x.y, min_val, max_val),
                _clamp(x.z, min_val, max_val),
            )
    if isinstance(x, vec4):
        if isinstance(min_val, vec4) and isinstance(max_val, vec4):
            return vec4(
                _clamp(x.x, min_val.x, max_val.x),
                _clamp(x.y, min_val.y, max_val.y),
                _clamp(x.z, min_val.z, max_val.z),
                _clamp(x.w, min_val.w, max_val.w),
            )
        if isinstance(min_val, int | float) and isinstance(max_val, int | float):
            return vec4(
                _clamp(x.x, min_val, max_val),
                _clamp(x.y, min_val, max_val),
                _clamp(x.z, min_val, max_val),
                _clamp(x.w, min_val, max_val),
            )
    if (
        isinstance(x, int | float)
        and isinstance(min_val, int | float)
        and isinstance(max_val, int | float)
    ):
        return _clamp(x, min_val, max_val)

    raise TypeError(f"clamp() types: {type(x)}, {type(min_val)}, {type(max_val)}")


# reflect
@overload
def reflect(I: vec2, N: vec2) -> vec2: ...
@overload
def reflect(I: vec3, N: vec3) -> vec3: ...
@overload
def reflect(I: vec4, N: vec4) -> vec4: ...


def reflect(I: vec2 | vec3 | vec4, N: vec2 | vec3 | vec4) -> vec2 | vec3 | vec4:
    """Reflect incident vector I around normal N."""
    if isinstance(I, vec2) and isinstance(N, vec2):
        d = dot(N, I)
        return I - 2.0 * d * N
    if isinstance(I, vec3) and isinstance(N, vec3):
        d = dot(N, I)
        return I - 2.0 * d * N
    if isinstance(I, vec4) and isinstance(N, vec4):
        d = dot(N, I)
        return I - 2.0 * d * N
    raise TypeError(
        f"reflect() requires matching vector types, got {type(I)} and {type(N)}"
    )


# refract
def refract(I: vec3, N: vec3, eta: float) -> vec3:
    """Refract incident vector I through surface with normal N and ratio eta."""
    d = dot(N, I)
    k = 1.0 - eta * eta * (1.0 - d * d)
    if k < 0.0:
        return vec3(0.0, 0.0, 0.0)
    return eta * I - (eta * d + sqrt(k)) * N
