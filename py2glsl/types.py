from dataclasses import dataclass
from typing import Any, Union

Number = Union[float, "vec2", "vec3", "vec4"]


@dataclass
class glsl_type:
    """Base for all GLSL types"""

    def __add__(self, other: Any) -> Any:
        raise NotImplementedError

    def __mul__(self, other: Any) -> Any:
        raise NotImplementedError


@dataclass
class vec2(glsl_type):
    x: float
    y: float

    def __add__(self, other: Number) -> "vec2":
        if isinstance(other, (int, float)):
            return vec2(self.x + other, self.y + other)
        elif isinstance(other, vec2):
            return vec2(self.x + other.x, self.y + other.y)
        raise TypeError(f"Cannot add vec2 and {type(other)}")

    def __mul__(self, other: Number) -> "vec2":
        if isinstance(other, (int, float)):
            return vec2(self.x * other, self.y * other)
        elif isinstance(other, vec2):
            return vec2(self.x * other.x, self.y * other.y)
        raise TypeError(f"Cannot multiply vec2 and {type(other)}")

    def __sub__(self, other: Number) -> "vec2":
        if isinstance(other, (int, float)):
            return vec2(self.x - other, self.y - other)
        elif isinstance(other, vec2):
            return vec2(self.x - other.x, self.y - other.y)
        raise TypeError(f"Cannot subtract vec2 and {type(other)}")

    def __truediv__(self, other: Number) -> "vec2":
        if isinstance(other, (int, float)):
            return vec2(self.x / other, self.y / other)
        elif isinstance(other, vec2):
            return vec2(self.x / other.x, self.y / other.y)
        raise TypeError(f"Cannot divide vec2 and {type(other)}")

    @property
    def xy(self) -> "vec2":
        return vec2(self.x, self.y)

    @property
    def yx(self) -> "vec2":
        return vec2(self.y, self.x)


@dataclass
class vec3(glsl_type):
    x: float
    y: float
    z: float

    def __add__(self, other: Number) -> "vec3":
        if isinstance(other, (int, float)):
            return vec3(self.x + other, self.y + other, self.z + other)
        elif isinstance(other, vec3):
            return vec3(self.x + other.x, self.y + other.y, self.z + other.z)
        raise TypeError(f"Cannot add vec3 and {type(other)}")

    def __mul__(self, other: Number) -> "vec3":
        if isinstance(other, (int, float)):
            return vec3(self.x * other, self.y * other, self.z * other)
        elif isinstance(other, vec3):
            return vec3(self.x * other.x, self.y * other.y, self.z * other.z)
        raise TypeError(f"Cannot multiply vec3 and {type(other)}")

    def __sub__(self, other: Number) -> "vec3":
        if isinstance(other, (int, float)):
            return vec3(self.x - other, self.y - other, self.z - other)
        elif isinstance(other, vec3):
            return vec3(self.x - other.x, self.y - other.y, self.z - other.z)
        raise TypeError(f"Cannot subtract vec3 and {type(other)}")

    def __truediv__(self, other: Number) -> "vec3":
        if isinstance(other, (int, float)):
            return vec3(self.x / other, self.y / other, self.z / other)
        elif isinstance(other, vec3):
            return vec3(self.x / other.x, self.y / other.y, self.z / other.z)
        raise TypeError(f"Cannot divide vec3 and {type(other)}")

    # Swizzling
    @property
    def xy(self) -> vec2:
        return vec2(self.x, self.y)

    @property
    def xz(self) -> vec2:
        return vec2(self.x, self.z)

    @property
    def yz(self) -> vec2:
        return vec2(self.y, self.z)

    @property
    def rgb(self) -> "vec3":
        return vec3(self.x, self.y, self.z)


@dataclass
class vec4(glsl_type):
    x: float
    y: float
    z: float
    w: float

    def __add__(self, other: Number) -> "vec4":
        if isinstance(other, (int, float)):
            return vec4(self.x + other, self.y + other, self.z + other, self.w + other)
        elif isinstance(other, vec4):
            return vec4(
                self.x + other.x, self.y + other.y, self.z + other.z, self.w + other.w
            )
        raise TypeError(f"Cannot add vec4 and {type(other)}")

    def __mul__(self, other: Number) -> "vec4":
        if isinstance(other, (int, float)):
            return vec4(self.x * other, self.y * other, self.z * other, self.w * other)
        elif isinstance(other, vec4):
            return vec4(
                self.x * other.x, self.y * other.y, self.z * other.z, self.w * other.w
            )
        raise TypeError(f"Cannot multiply vec4 and {type(other)}")

    def __sub__(self, other: Number) -> "vec4":
        if isinstance(other, (int, float)):
            return vec4(self.x - other, self.y - other, self.z - other, self.w - other)
        elif isinstance(other, vec4):
            return vec4(
                self.x - other.x, self.y - other.y, self.z - other.z, self.w - other.w
            )
        raise TypeError(f"Cannot subtract vec4 and {type(other)}")

    def __truediv__(self, other: Number) -> "vec4":
        if isinstance(other, (int, float)):
            return vec4(self.x / other, self.y / other, self.z / other, self.w / other)
        elif isinstance(other, vec4):
            return vec4(
                self.x / other.x, self.y / other.y, self.z / other.z, self.w / other.w
            )
        raise TypeError(f"Cannot divide vec4 and {type(other)}")

    # Swizzling
    @property
    def xyz(self) -> vec3:
        return vec3(self.x, self.y, self.z)

    @property
    def rgb(self) -> vec3:
        return vec3(self.x, self.y, self.z)

    @property
    def xy(self) -> vec2:
        return vec2(self.x, self.y)

    @property
    def zw(self) -> vec2:
        return vec2(self.z, self.w)
