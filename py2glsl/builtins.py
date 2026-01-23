"""GLSL builtin types and functions for Python runtime.

This module provides Python implementations of GLSL types
(vec2, vec3, vec4, mat2, mat3, mat4) and builtin functions
(sin, cos, mix, etc.) that match GLSL semantics.
"""

import itertools
from collections.abc import Callable
from typing import TypeVar, overload

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]

E = TypeVar("E")

# =============================================================================
# Vector Types
# =============================================================================

# Component name mappings
_XYZW = "xyzw"
_RGBA = "rgba"


def _make_component_property(index: int) -> property:
    """Create a property for accessing vector component by index."""

    def getter(self: "vec2 | vec3 | vec4") -> float:
        return float(self.data[index])

    def setter(self: "vec2 | vec3 | vec4", value: float) -> None:
        self.data[index] = value

    return property(getter, setter)


def _make_swizzle_property(indices: tuple[int, ...], result_type: type) -> property:
    """Create a swizzle property that returns a new vector."""

    def getter(self: "vec2 | vec3 | vec4") -> "vec2 | vec3 | vec4":
        return result_type(*(float(self.data[i]) for i in indices))

    return property(getter)


class vec2:
    """GLSL vec2 type - 2-component float vector."""

    __slots__ = ("data",)
    _size = 2

    # Component access (set dynamically below)
    x: float
    y: float
    r: float
    g: float

    # Common swizzles (set dynamically below)
    xy: "vec2"
    yx: "vec2"
    xx: "vec2"
    yy: "vec2"
    rg: "vec2"
    gr: "vec2"
    rr: "vec2"
    gg: "vec2"
    # 3-component swizzles
    xxx: "vec3"
    xxy: "vec3"
    xyx: "vec3"
    xyy: "vec3"
    yxx: "vec3"
    yxy: "vec3"
    yyx: "vec3"
    yyy: "vec3"
    rrr: "vec3"
    rrg: "vec3"
    rgr: "vec3"
    rgg: "vec3"
    grr: "vec3"
    grg: "vec3"
    ggr: "vec3"
    ggg: "vec3"
    # 4-component swizzles
    xxxx: "vec4"
    xxxy: "vec4"
    xxyx: "vec4"
    xxyy: "vec4"
    xyxx: "vec4"
    xyxy: "vec4"
    xyyx: "vec4"
    xyyy: "vec4"
    yxxx: "vec4"
    yxxy: "vec4"
    yxyx: "vec4"
    yxyy: "vec4"
    yyxx: "vec4"
    yyxy: "vec4"
    yyyx: "vec4"
    yyyy: "vec4"

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

    def __add__(self, other: "vec2 | float") -> "vec2":
        if isinstance(other, vec2):
            return vec2(self.data[0] + other.data[0], self.data[1] + other.data[1])
        return vec2(self.data[0] + other, self.data[1] + other)

    def __radd__(self, other: float) -> "vec2":
        return vec2(other + self.data[0], other + self.data[1])

    def __sub__(self, other: "vec2 | float") -> "vec2":
        if isinstance(other, vec2):
            return vec2(self.data[0] - other.data[0], self.data[1] - other.data[1])
        return vec2(self.data[0] - other, self.data[1] - other)

    def __rsub__(self, other: float) -> "vec2":
        return vec2(other - self.data[0], other - self.data[1])

    def __mul__(self, other: "vec2 | float") -> "vec2":
        if isinstance(other, vec2):
            return vec2(self.data[0] * other.data[0], self.data[1] * other.data[1])
        return vec2(self.data[0] * other, self.data[1] * other)

    def __rmul__(self, other: float) -> "vec2":
        return vec2(other * self.data[0], other * self.data[1])

    def __truediv__(self, other: "vec2 | float") -> "vec2":
        if isinstance(other, vec2):
            return vec2(self.data[0] / other.data[0], self.data[1] / other.data[1])
        return vec2(self.data[0] / other, self.data[1] / other)

    def __rtruediv__(self, other: float) -> "vec2":
        return vec2(other / self.data[0], other / self.data[1])

    def __neg__(self) -> "vec2":
        return vec2(-self.data[0], -self.data[1])

    def __pos__(self) -> "vec2":
        return vec2(self.data[0], self.data[1])

    def __getitem__(self, index: int) -> float:
        return float(self.data[index])

    def __setitem__(self, index: int, value: float) -> None:
        self.data[index] = value

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter([float(self.data[0]), float(self.data[1])])

    def __repr__(self) -> str:
        return f"vec2({self.data[0]}, {self.data[1]})"

    def __str__(self) -> str:
        return f"vec2({self.data[0]}, {self.data[1]})"


# Add component properties to vec2
for i, (c, col) in enumerate(zip(_XYZW[:2], _RGBA[:2], strict=False)):
    setattr(vec2, c, _make_component_property(i))
    setattr(vec2, col, _make_component_property(i))


class vec3:
    """GLSL vec3 type - 3-component float vector."""

    __slots__ = ("data",)
    _size = 3

    # Component access (set dynamically below)
    x: float
    y: float
    z: float
    r: float
    g: float
    b: float

    # Common 2-component swizzles
    xy: vec2
    xz: vec2
    yx: vec2
    yz: vec2
    zx: vec2
    zy: vec2
    xx: vec2
    yy: vec2
    zz: vec2
    rg: vec2
    rb: vec2
    gr: vec2
    gb: vec2
    br: vec2
    bg: vec2
    rr: vec2
    gg: vec2
    bb: vec2
    # Common 3-component swizzles
    xyz: "vec3"
    xzy: "vec3"
    yxz: "vec3"
    yzx: "vec3"
    zxy: "vec3"
    zyx: "vec3"
    xxx: "vec3"
    yyy: "vec3"
    zzz: "vec3"
    xxy: "vec3"
    xxz: "vec3"
    xyx: "vec3"
    xyy: "vec3"
    xzx: "vec3"
    xzz: "vec3"
    yxx: "vec3"
    yxy: "vec3"
    yyx: "vec3"
    yyz: "vec3"
    yzy: "vec3"
    yzz: "vec3"
    zxx: "vec3"
    zxz: "vec3"
    zyy: "vec3"
    zyz: "vec3"
    zzx: "vec3"
    zzy: "vec3"
    rgb: "vec3"
    rbg: "vec3"
    grb: "vec3"
    gbr: "vec3"
    brg: "vec3"
    bgr: "vec3"
    rrr: "vec3"
    ggg: "vec3"
    bbb: "vec3"
    # Common 4-component swizzles
    xxxx: "vec4"
    yyyy: "vec4"
    zzzz: "vec4"
    xyzx: "vec4"
    xyzy: "vec4"
    xyzz: "vec4"

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
            self.data = np.array([x.data[0], x.data[1], y], dtype=np.float32)
        elif isinstance(x, int | float) and isinstance(y, vec2):
            self.data = np.array([x, y.data[0], y.data[1]], dtype=np.float32)
        elif y is None:
            self.data = np.array([x, x, x], dtype=np.float32)
        else:
            self.data = np.array([x, y, z], dtype=np.float32)

    def __add__(self, other: "vec3 | float") -> "vec3":
        if isinstance(other, vec3):
            return vec3(
                self.data[0] + other.data[0],
                self.data[1] + other.data[1],
                self.data[2] + other.data[2],
            )
        return vec3(self.data[0] + other, self.data[1] + other, self.data[2] + other)

    def __radd__(self, other: float) -> "vec3":
        return vec3(other + self.data[0], other + self.data[1], other + self.data[2])

    def __sub__(self, other: "vec3 | float") -> "vec3":
        if isinstance(other, vec3):
            return vec3(
                self.data[0] - other.data[0],
                self.data[1] - other.data[1],
                self.data[2] - other.data[2],
            )
        return vec3(self.data[0] - other, self.data[1] - other, self.data[2] - other)

    def __rsub__(self, other: float) -> "vec3":
        return vec3(other - self.data[0], other - self.data[1], other - self.data[2])

    def __mul__(self, other: "vec3 | float") -> "vec3":
        if isinstance(other, vec3):
            return vec3(
                self.data[0] * other.data[0],
                self.data[1] * other.data[1],
                self.data[2] * other.data[2],
            )
        return vec3(self.data[0] * other, self.data[1] * other, self.data[2] * other)

    def __rmul__(self, other: float) -> "vec3":
        return vec3(other * self.data[0], other * self.data[1], other * self.data[2])

    def __truediv__(self, other: "vec3 | float") -> "vec3":
        if isinstance(other, vec3):
            return vec3(
                self.data[0] / other.data[0],
                self.data[1] / other.data[1],
                self.data[2] / other.data[2],
            )
        return vec3(self.data[0] / other, self.data[1] / other, self.data[2] / other)

    def __rtruediv__(self, other: float) -> "vec3":
        return vec3(other / self.data[0], other / self.data[1], other / self.data[2])

    def __neg__(self) -> "vec3":
        return vec3(-self.data[0], -self.data[1], -self.data[2])

    def __pos__(self) -> "vec3":
        return vec3(self.data[0], self.data[1], self.data[2])

    def __getitem__(self, index: int) -> float:
        return float(self.data[index])

    def __setitem__(self, index: int, value: float) -> None:
        self.data[index] = value

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter([float(self.data[0]), float(self.data[1]), float(self.data[2])])

    def __repr__(self) -> str:
        return f"vec3({self.data[0]}, {self.data[1]}, {self.data[2]})"

    def __str__(self) -> str:
        return f"vec3({self.data[0]}, {self.data[1]}, {self.data[2]})"


# Add component properties to vec3
for i, (c, col) in enumerate(zip(_XYZW[:3], _RGBA[:3], strict=False)):
    setattr(vec3, c, _make_component_property(i))
    setattr(vec3, col, _make_component_property(i))


class vec4:
    """GLSL vec4 type - 4-component float vector."""

    __slots__ = ("data",)
    _size = 4

    # Component access (set dynamically below)
    x: float
    y: float
    z: float
    w: float
    r: float
    g: float
    b: float
    a: float

    # Common 2-component swizzles
    xy: vec2
    xz: vec2
    xw: vec2
    yx: vec2
    yz: vec2
    yw: vec2
    zx: vec2
    zy: vec2
    zw: vec2
    wx: vec2
    wy: vec2
    wz: vec2
    xx: vec2
    yy: vec2
    zz: vec2
    ww: vec2
    rg: vec2
    rb: vec2
    ra: vec2
    gr: vec2
    gb: vec2
    ga: vec2
    br: vec2
    bg: vec2
    ba: vec2
    ar: vec2
    ag: vec2
    ab: vec2
    rr: vec2
    gg: vec2
    bb: vec2
    aa: vec2
    # Common 3-component swizzles
    xyz: vec3
    xyw: vec3
    xzy: vec3
    xzw: vec3
    xwy: vec3
    xwz: vec3
    yxz: vec3
    yxw: vec3
    yzx: vec3
    yzw: vec3
    ywx: vec3
    ywz: vec3
    zxy: vec3
    zxw: vec3
    zyx: vec3
    zyw: vec3
    zwx: vec3
    zwy: vec3
    wxy: vec3
    wxz: vec3
    wyx: vec3
    wyz: vec3
    wzx: vec3
    wzy: vec3
    xxx: vec3
    yyy: vec3
    zzz: vec3
    www: vec3
    rgb: vec3
    rga: vec3
    rbg: vec3
    rba: vec3
    rag: vec3
    rab: vec3
    grb: vec3
    gra: vec3
    gbr: vec3
    gba: vec3
    gar: vec3
    gab: vec3
    brg: vec3
    bra: vec3
    bgr: vec3
    bga: vec3
    bar: vec3
    bag: vec3
    arg: vec3
    arb: vec3
    agr: vec3
    agb: vec3
    abr: vec3
    abg: vec3
    rrr: vec3
    ggg: vec3
    bbb: vec3
    aaa: vec3
    # Common 4-component swizzles
    xyzw: "vec4"
    xywz: "vec4"
    xzyw: "vec4"
    xzwy: "vec4"
    xwyz: "vec4"
    xwzy: "vec4"
    yxzw: "vec4"
    yxwz: "vec4"
    yzxw: "vec4"
    yzwx: "vec4"
    ywxz: "vec4"
    ywzx: "vec4"
    zxyw: "vec4"
    zxwy: "vec4"
    zyxw: "vec4"
    zywx: "vec4"
    zwxy: "vec4"
    zwyx: "vec4"
    wxyz: "vec4"
    wxzy: "vec4"
    wyxz: "vec4"
    wyzx: "vec4"
    wzxy: "vec4"
    wzyx: "vec4"
    xxxx: "vec4"
    yyyy: "vec4"
    zzzz: "vec4"
    wwww: "vec4"
    rgba: "vec4"
    rgab: "vec4"
    rbga: "vec4"
    rbag: "vec4"
    ragb: "vec4"
    rabg: "vec4"
    grba: "vec4"
    grab: "vec4"
    gbra: "vec4"
    gbar: "vec4"
    garb: "vec4"
    gabr: "vec4"
    brga: "vec4"
    brag: "vec4"
    bgra: "vec4"
    bgar: "vec4"
    barg: "vec4"
    bagr: "vec4"
    argb: "vec4"
    arbg: "vec4"
    agrb: "vec4"
    agbr: "vec4"
    abrg: "vec4"
    abgr: "vec4"
    rrrr: "vec4"
    gggg: "vec4"
    bbbb: "vec4"
    aaaa: "vec4"

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
            self.data = np.array([x.data[0], x.data[1], x.data[2], y], dtype=np.float32)
        elif isinstance(x, int | float) and isinstance(y, vec3):
            self.data = np.array([x, y.data[0], y.data[1], y.data[2]], dtype=np.float32)
        elif isinstance(x, vec2) and isinstance(y, vec2):
            self.data = np.array(
                [x.data[0], x.data[1], y.data[0], y.data[1]], dtype=np.float32
            )
        elif (
            isinstance(x, vec2)
            and isinstance(y, int | float)
            and isinstance(z, int | float)
        ):
            self.data = np.array([x.data[0], x.data[1], y, z], dtype=np.float32)
        elif y is None:
            self.data = np.array([x, x, x, x], dtype=np.float32)
        else:
            self.data = np.array([x, y, z, w], dtype=np.float32)

    def __add__(self, other: "vec4 | float") -> "vec4":
        if isinstance(other, vec4):
            return vec4(
                self.data[0] + other.data[0],
                self.data[1] + other.data[1],
                self.data[2] + other.data[2],
                self.data[3] + other.data[3],
            )
        return vec4(
            self.data[0] + other,
            self.data[1] + other,
            self.data[2] + other,
            self.data[3] + other,
        )

    def __radd__(self, other: float) -> "vec4":
        return vec4(
            other + self.data[0],
            other + self.data[1],
            other + self.data[2],
            other + self.data[3],
        )

    def __sub__(self, other: "vec4 | float") -> "vec4":
        if isinstance(other, vec4):
            return vec4(
                self.data[0] - other.data[0],
                self.data[1] - other.data[1],
                self.data[2] - other.data[2],
                self.data[3] - other.data[3],
            )
        return vec4(
            self.data[0] - other,
            self.data[1] - other,
            self.data[2] - other,
            self.data[3] - other,
        )

    def __rsub__(self, other: float) -> "vec4":
        return vec4(
            other - self.data[0],
            other - self.data[1],
            other - self.data[2],
            other - self.data[3],
        )

    def __mul__(self, other: "vec4 | float") -> "vec4":
        if isinstance(other, vec4):
            return vec4(
                self.data[0] * other.data[0],
                self.data[1] * other.data[1],
                self.data[2] * other.data[2],
                self.data[3] * other.data[3],
            )
        return vec4(
            self.data[0] * other,
            self.data[1] * other,
            self.data[2] * other,
            self.data[3] * other,
        )

    def __rmul__(self, other: float) -> "vec4":
        return vec4(
            other * self.data[0],
            other * self.data[1],
            other * self.data[2],
            other * self.data[3],
        )

    def __truediv__(self, other: "vec4 | float") -> "vec4":
        if isinstance(other, vec4):
            return vec4(
                self.data[0] / other.data[0],
                self.data[1] / other.data[1],
                self.data[2] / other.data[2],
                self.data[3] / other.data[3],
            )
        return vec4(
            self.data[0] / other,
            self.data[1] / other,
            self.data[2] / other,
            self.data[3] / other,
        )

    def __rtruediv__(self, other: float) -> "vec4":
        return vec4(
            other / self.data[0],
            other / self.data[1],
            other / self.data[2],
            other / self.data[3],
        )

    def __neg__(self) -> "vec4":
        return vec4(-self.data[0], -self.data[1], -self.data[2], -self.data[3])

    def __pos__(self) -> "vec4":
        return vec4(self.data[0], self.data[1], self.data[2], self.data[3])

    def __getitem__(self, index: int) -> float:
        return float(self.data[index])

    def __setitem__(self, index: int, value: float) -> None:
        self.data[index] = value

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(
            [
                float(self.data[0]),
                float(self.data[1]),
                float(self.data[2]),
                float(self.data[3]),
            ]
        )

    def __repr__(self) -> str:
        return f"vec4({self.data[0]}, {self.data[1]}, {self.data[2]}, {self.data[3]})"

    def __str__(self) -> str:
        return f"vec4({self.data[0]}, {self.data[1]}, {self.data[2]}, {self.data[3]})"


# Add component properties to vec4
for i, (c, col) in enumerate(zip(_XYZW[:4], _RGBA[:4], strict=False)):
    setattr(vec4, c, _make_component_property(i))
    setattr(vec4, col, _make_component_property(i))


# =============================================================================
# Generate Swizzle Properties
# =============================================================================


def _generate_swizzles() -> None:
    """Generate all swizzle properties for vec2, vec3, vec4."""
    vec_types = {2: vec2, 3: vec3, 4: vec4}

    # Generate swizzles for each source vector size
    for src_size, src_type in vec_types.items():
        components = _XYZW[:src_size]
        color_components = _RGBA[:src_size]

        # Generate 2, 3, 4 component swizzles
        for dst_size in [2, 3, 4]:
            dst_type = vec_types[dst_size]

            # Generate all combinations using xyzw
            for combo in itertools.product(range(src_size), repeat=dst_size):
                name = "".join(components[i] for i in combo)
                if not hasattr(src_type, name):
                    setattr(src_type, name, _make_swizzle_property(combo, dst_type))

                # Also generate rgba version
                color_name = "".join(color_components[i] for i in combo)
                if color_name != name and not hasattr(src_type, color_name):
                    setattr(
                        src_type, color_name, _make_swizzle_property(combo, dst_type)
                    )


_generate_swizzles()


# =============================================================================
# Matrix Types
# =============================================================================


class mat2:
    """GLSL mat2 type - 2x2 float matrix."""

    __slots__ = ("data",)

    def __init__(self, *args: float) -> None:
        if len(args) == 4:
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
        if len(args) == 9:
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
        if len(args) == 16:
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
# GLSL Built-in Functions
# =============================================================================

# Type aliases for function signatures
Vec = vec2 | vec3 | vec4
VecOrFloat = Vec | float


def _apply_elementwise(x: VecOrFloat, func: Callable[[float], float]) -> VecOrFloat:
    """Apply a function elementwise to a scalar or vector."""
    if isinstance(x, vec2):
        return vec2(func(x.data[0]), func(x.data[1]))
    if isinstance(x, vec3):
        return vec3(func(x.data[0]), func(x.data[1]), func(x.data[2]))
    if isinstance(x, vec4):
        return vec4(func(x.data[0]), func(x.data[1]), func(x.data[2]), func(x.data[3]))
    return func(float(x))


def _apply_binary_elementwise(
    a: VecOrFloat, b: VecOrFloat, func: Callable[[float, float], float]
) -> VecOrFloat:
    """Apply a binary function elementwise."""
    if isinstance(a, vec2):
        if isinstance(b, vec2):
            return vec2(func(a.data[0], b.data[0]), func(a.data[1], b.data[1]))
        if isinstance(b, int | float):
            bf = float(b)
            return vec2(func(a.data[0], bf), func(a.data[1], bf))
    if isinstance(a, vec3):
        if isinstance(b, vec3):
            return vec3(
                func(a.data[0], b.data[0]),
                func(a.data[1], b.data[1]),
                func(a.data[2], b.data[2]),
            )
        if isinstance(b, int | float):
            bf = float(b)
            return vec3(func(a.data[0], bf), func(a.data[1], bf), func(a.data[2], bf))
    if isinstance(a, vec4):
        if isinstance(b, vec4):
            return vec4(
                func(a.data[0], b.data[0]),
                func(a.data[1], b.data[1]),
                func(a.data[2], b.data[2]),
                func(a.data[3], b.data[3]),
            )
        if isinstance(b, int | float):
            bf = float(b)
            return vec4(
                func(a.data[0], bf),
                func(a.data[1], bf),
                func(a.data[2], bf),
                func(a.data[3], bf),
            )
    if isinstance(a, int | float) and isinstance(b, int | float):
        return func(float(a), float(b))
    msg = f"Incompatible types: {type(a)} and {type(b)}"
    raise TypeError(msg)


# Trigonometric functions
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


# Math functions
@overload
def abs(x: float) -> float: ...
@overload
def abs(x: vec2) -> vec2: ...
@overload
def abs(x: vec3) -> vec3: ...
@overload
def abs(x: vec4) -> vec4: ...


def abs(x: VecOrFloat) -> VecOrFloat:
    return _apply_elementwise(x, lambda v: float(np.abs(v)))


@overload
def fract(x: float) -> float: ...
@overload
def fract(x: vec2) -> vec2: ...
@overload
def fract(x: vec3) -> vec3: ...
@overload
def fract(x: vec4) -> vec4: ...


def fract(x: VecOrFloat) -> VecOrFloat:
    return _apply_elementwise(x, lambda v: float(v - np.floor(v)))


@overload
def floor(x: float) -> float: ...
@overload
def floor(x: vec2) -> vec2: ...
@overload
def floor(x: vec3) -> vec3: ...
@overload
def floor(x: vec4) -> vec4: ...


def floor(x: VecOrFloat) -> VecOrFloat:
    return _apply_elementwise(x, lambda v: float(np.floor(v)))


@overload
def ceil(x: float) -> float: ...
@overload
def ceil(x: vec2) -> vec2: ...
@overload
def ceil(x: vec3) -> vec3: ...
@overload
def ceil(x: vec4) -> vec4: ...


def ceil(x: VecOrFloat) -> VecOrFloat:
    return _apply_elementwise(x, lambda v: float(np.ceil(v)))


@overload
def sign(x: float) -> float: ...
@overload
def sign(x: vec2) -> vec2: ...
@overload
def sign(x: vec3) -> vec3: ...
@overload
def sign(x: vec4) -> vec4: ...


def sign(x: VecOrFloat) -> VecOrFloat:
    return _apply_elementwise(x, lambda v: float(np.sign(v)))


def sqrt(x: float) -> float:
    return float(np.sqrt(x))


def pow(x: float, y: float) -> float:
    return float(np.power(x, y))


def exp(x: float) -> float:
    return float(np.exp(x))


def exp2(x: float) -> float:
    return float(np.exp2(x))


def log(x: float) -> float:
    return float(np.log(x))


def log2(x: float) -> float:
    return float(np.log2(x))


def radians(degrees: float) -> float:
    return float(np.radians(degrees))


# Vector functions
@overload
def length(v: vec2) -> float: ...
@overload
def length(v: vec3) -> float: ...
@overload
def length(v: vec4) -> float: ...


def length(v: Vec) -> float:
    return float(np.linalg.norm(v.data))


@overload
def distance(p0: vec2, p1: vec2) -> float: ...
@overload
def distance(p0: vec3, p1: vec3) -> float: ...
@overload
def distance(p0: vec4, p1: vec4) -> float: ...


def distance(p0: Vec, p1: Vec) -> float:
    return float(np.linalg.norm(p0.data - p1.data))


@overload
def dot(a: vec2, b: vec2) -> float: ...
@overload
def dot(a: vec3, b: vec3) -> float: ...
@overload
def dot(a: vec4, b: vec4) -> float: ...


def dot(a: Vec, b: Vec) -> float:
    return float(np.dot(a.data, b.data))


def cross(a: vec3, b: vec3) -> vec3:
    result = np.cross(a.data, b.data)
    return vec3(float(result[0]), float(result[1]), float(result[2]))


@overload
def normalize(v: vec2) -> vec2: ...
@overload
def normalize(v: vec3) -> vec3: ...
@overload
def normalize(v: vec4) -> vec4: ...


def normalize(v: Vec) -> Vec:
    norm = float(np.linalg.norm(v.data))
    if norm == 0:
        return v
    if isinstance(v, vec2):
        return vec2(v.data[0] / norm, v.data[1] / norm)
    if isinstance(v, vec3):
        return vec3(v.data[0] / norm, v.data[1] / norm, v.data[2] / norm)
    return vec4(v.data[0] / norm, v.data[1] / norm, v.data[2] / norm, v.data[3] / norm)


# Min/max functions
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


def min(a: VecOrFloat, b: VecOrFloat, c: float | None = None) -> VecOrFloat:
    if c is not None:
        if isinstance(a, int | float) and isinstance(b, int | float):
            return float(np.minimum(np.minimum(a, b), c))
        raise TypeError("min() with 3 args only supports floats")
    return _apply_binary_elementwise(a, b, lambda x, y: x if x < y else y)


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


def max(a: VecOrFloat, b: VecOrFloat, c: float | None = None) -> VecOrFloat:
    if c is not None:
        if isinstance(a, int | float) and isinstance(b, int | float):
            return float(np.maximum(np.maximum(a, b), c))
        raise TypeError("max() with 3 args only supports floats")
    return _apply_binary_elementwise(a, b, lambda x, y: x if x > y else y)


# Interpolation functions
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


def mix(x: VecOrFloat, y: VecOrFloat, a: VecOrFloat) -> VecOrFloat:
    def _mix(xv: float, yv: float, av: float) -> float:
        return xv * (1 - av) + yv * av

    if (
        isinstance(x, int | float)
        and isinstance(y, int | float)
        and isinstance(a, int | float)
    ):
        return _mix(float(x), float(y), float(a))

    if isinstance(x, vec2) and isinstance(y, vec2):
        if isinstance(a, vec2):
            return vec2(
                _mix(x.data[0], y.data[0], a.data[0]),
                _mix(x.data[1], y.data[1], a.data[1]),
            )
        if isinstance(a, int | float):
            af = float(a)
            return vec2(
                _mix(x.data[0], y.data[0], af),
                _mix(x.data[1], y.data[1], af),
            )

    if isinstance(x, vec3) and isinstance(y, vec3):
        if isinstance(a, vec3):
            return vec3(
                _mix(x.data[0], y.data[0], a.data[0]),
                _mix(x.data[1], y.data[1], a.data[1]),
                _mix(x.data[2], y.data[2], a.data[2]),
            )
        if isinstance(a, int | float):
            af = float(a)
            return vec3(
                _mix(x.data[0], y.data[0], af),
                _mix(x.data[1], y.data[1], af),
                _mix(x.data[2], y.data[2], af),
            )

    if isinstance(x, vec4) and isinstance(y, vec4):
        if isinstance(a, vec4):
            return vec4(
                _mix(x.data[0], y.data[0], a.data[0]),
                _mix(x.data[1], y.data[1], a.data[1]),
                _mix(x.data[2], y.data[2], a.data[2]),
                _mix(x.data[3], y.data[3], a.data[3]),
            )
        if isinstance(a, int | float):
            af = float(a)
            return vec4(
                _mix(x.data[0], y.data[0], af),
                _mix(x.data[1], y.data[1], af),
                _mix(x.data[2], y.data[2], af),
                _mix(x.data[3], y.data[3], af),
            )

    msg = f"mix() received incompatible types: {type(x)}, {type(y)}, {type(a)}"
    raise TypeError(msg)


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


def smoothstep(edge0: VecOrFloat, edge1: VecOrFloat, x: VecOrFloat) -> VecOrFloat:
    def _ss(e0: float, e1: float, v: float) -> float:
        t = float(np.clip((v - e0) / (e1 - e0), 0.0, 1.0))
        return t * t * (3.0 - 2.0 * t)

    # All floats
    if isinstance(edge0, int | float) and isinstance(edge1, int | float):
        if isinstance(x, int | float):
            return _ss(float(edge0), float(edge1), float(x))
        e0, e1 = float(edge0), float(edge1)
        if isinstance(x, vec2):
            return vec2(_ss(e0, e1, x.data[0]), _ss(e0, e1, x.data[1]))
        if isinstance(x, vec3):
            return vec3(
                _ss(e0, e1, x.data[0]),
                _ss(e0, e1, x.data[1]),
                _ss(e0, e1, x.data[2]),
            )
        if isinstance(x, vec4):
            return vec4(
                _ss(e0, e1, x.data[0]),
                _ss(e0, e1, x.data[1]),
                _ss(e0, e1, x.data[2]),
                _ss(e0, e1, x.data[3]),
            )

    # Vector edges
    if isinstance(edge0, vec2) and isinstance(edge1, vec2) and isinstance(x, vec2):
        return vec2(
            _ss(edge0.data[0], edge1.data[0], x.data[0]),
            _ss(edge0.data[1], edge1.data[1], x.data[1]),
        )
    if isinstance(edge0, vec3) and isinstance(edge1, vec3) and isinstance(x, vec3):
        return vec3(
            _ss(edge0.data[0], edge1.data[0], x.data[0]),
            _ss(edge0.data[1], edge1.data[1], x.data[1]),
            _ss(edge0.data[2], edge1.data[2], x.data[2]),
        )
    if isinstance(edge0, vec4) and isinstance(edge1, vec4) and isinstance(x, vec4):
        return vec4(
            _ss(edge0.data[0], edge1.data[0], x.data[0]),
            _ss(edge0.data[1], edge1.data[1], x.data[1]),
            _ss(edge0.data[2], edge1.data[2], x.data[2]),
            _ss(edge0.data[3], edge1.data[3], x.data[3]),
        )

    raise TypeError(f"smoothstep() types: {type(edge0)}, {type(edge1)}, {type(x)}")


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


def step(edge: VecOrFloat, x: VecOrFloat) -> VecOrFloat:
    def _step(e: float, v: float) -> float:
        return 0.0 if v < e else 1.0

    if isinstance(edge, int | float):
        ef = float(edge)
        if isinstance(x, int | float):
            return _step(ef, float(x))
        if isinstance(x, vec2):
            return vec2(_step(ef, x.data[0]), _step(ef, x.data[1]))
        if isinstance(x, vec3):
            return vec3(
                _step(ef, x.data[0]),
                _step(ef, x.data[1]),
                _step(ef, x.data[2]),
            )
        if isinstance(x, vec4):
            return vec4(
                _step(ef, x.data[0]),
                _step(ef, x.data[1]),
                _step(ef, x.data[2]),
                _step(ef, x.data[3]),
            )

    if isinstance(edge, vec2) and isinstance(x, vec2):
        return vec2(
            _step(edge.data[0], x.data[0]),
            _step(edge.data[1], x.data[1]),
        )
    if isinstance(edge, vec3) and isinstance(x, vec3):
        return vec3(
            _step(edge.data[0], x.data[0]),
            _step(edge.data[1], x.data[1]),
            _step(edge.data[2], x.data[2]),
        )
    if isinstance(edge, vec4) and isinstance(x, vec4):
        return vec4(
            _step(edge.data[0], x.data[0]),
            _step(edge.data[1], x.data[1]),
            _step(edge.data[2], x.data[2]),
            _step(edge.data[3], x.data[3]),
        )

    raise TypeError(f"step() received incompatible types: {type(edge)} and {type(x)}")


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


def mod(x: VecOrFloat, y: VecOrFloat) -> VecOrFloat:
    """GLSL mod: x - y * floor(x/y)"""

    def _mod(xv: float, yv: float) -> float:
        return xv - yv * float(np.floor(xv / yv))

    return _apply_binary_elementwise(x, y, _mod)


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


def clamp(x: VecOrFloat, min_val: VecOrFloat, max_val: VecOrFloat) -> VecOrFloat:
    def _clamp(v: float, lo: float, hi: float) -> float:
        return float(np.clip(v, lo, hi))

    if (
        isinstance(x, int | float)
        and isinstance(min_val, int | float)
        and isinstance(max_val, int | float)
    ):
        return _clamp(float(x), float(min_val), float(max_val))

    if isinstance(x, vec2):
        if isinstance(min_val, vec2) and isinstance(max_val, vec2):
            return vec2(
                _clamp(x.data[0], min_val.data[0], max_val.data[0]),
                _clamp(x.data[1], min_val.data[1], max_val.data[1]),
            )
        if isinstance(min_val, int | float) and isinstance(max_val, int | float):
            lo, hi = float(min_val), float(max_val)
            return vec2(_clamp(x.data[0], lo, hi), _clamp(x.data[1], lo, hi))

    if isinstance(x, vec3):
        if isinstance(min_val, vec3) and isinstance(max_val, vec3):
            return vec3(
                _clamp(x.data[0], min_val.data[0], max_val.data[0]),
                _clamp(x.data[1], min_val.data[1], max_val.data[1]),
                _clamp(x.data[2], min_val.data[2], max_val.data[2]),
            )
        if isinstance(min_val, int | float) and isinstance(max_val, int | float):
            lo, hi = float(min_val), float(max_val)
            return vec3(
                _clamp(x.data[0], lo, hi),
                _clamp(x.data[1], lo, hi),
                _clamp(x.data[2], lo, hi),
            )

    if isinstance(x, vec4):
        if isinstance(min_val, vec4) and isinstance(max_val, vec4):
            return vec4(
                _clamp(x.data[0], min_val.data[0], max_val.data[0]),
                _clamp(x.data[1], min_val.data[1], max_val.data[1]),
                _clamp(x.data[2], min_val.data[2], max_val.data[2]),
                _clamp(x.data[3], min_val.data[3], max_val.data[3]),
            )
        if isinstance(min_val, int | float) and isinstance(max_val, int | float):
            lo, hi = float(min_val), float(max_val)
            return vec4(
                _clamp(x.data[0], lo, hi),
                _clamp(x.data[1], lo, hi),
                _clamp(x.data[2], lo, hi),
                _clamp(x.data[3], lo, hi),
            )

    raise TypeError(f"clamp() types: {type(x)}, {type(min_val)}, {type(max_val)}")


# Reflection/refraction
@overload
def reflect(I: vec2, N: vec2) -> vec2: ...
@overload
def reflect(I: vec3, N: vec3) -> vec3: ...
@overload
def reflect(I: vec4, N: vec4) -> vec4: ...


def reflect(I: Vec, N: Vec) -> Vec:
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
    raise TypeError(f"reflect() types: {type(I)}, {type(N)}")


def refract(I: vec3, N: vec3, eta: float) -> vec3:
    """Refract incident vector I through surface with normal N and ratio eta."""
    d = dot(N, I)
    k = 1.0 - eta * eta * (1.0 - d * d)
    if k < 0.0:
        return vec3(0.0, 0.0, 0.0)
    return eta * I - (eta * d + sqrt(k)) * N
