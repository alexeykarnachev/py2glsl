from typing import NamedTuple, Tuple, Union

import numpy as np


# Vector classes
class vec2:
    def __init__(self, x: float, y: float):
        self.data = np.array([x, y], dtype=np.float32)

    @property
    def x(self) -> float:
        return self.data[0]

    @property
    def y(self) -> float:
        return self.data[1]

    @x.setter
    def x(self, value: float):
        self.data[0] = value

    @y.setter
    def y(self, value: float):
        self.data[1] = value

    def __add__(self, other: "vec2") -> "vec2":
        return vec2(*(self.data + other.data))

    def __sub__(self, other: "vec2") -> "vec2":
        return vec2(*(self.data - other.data))

    def __mul__(self, scalar: float) -> "vec2":
        return vec2(*(self.data * scalar))

    def __rmul__(self, scalar: float) -> "vec2":
        return self.__mul__(scalar)

    def __str__(self) -> str:
        return f"vec2({self.x}, {self.y})"


class vec3:
    def __init__(self, x: float, y: float, z: float):
        self.data = np.array([x, y, z], dtype=np.float32)

    @property
    def x(self) -> float:
        return self.data[0]

    @property
    def y(self) -> float:
        return self.data[1]

    @property
    def z(self) -> float:
        return self.data[2]

    @x.setter
    def x(self, value: float):
        self.data[0] = value

    @y.setter
    def y(self, value: float):
        self.data[1] = value

    @z.setter
    def z(self, value: float):
        self.data[2] = value

    @property
    def xyy(self) -> "vec3":
        return vec3(self.x, self.y, self.y)

    @property
    def yxy(self) -> "vec3":
        return vec3(self.y, self.x, self.y)

    @property
    def yyx(self) -> "vec3":
        return vec3(self.y, self.y, self.x)

    def __add__(self, other: "vec3") -> "vec3":
        return vec3(*(self.data + other.data))

    def __sub__(self, other: "vec3") -> "vec3":
        return vec3(*(self.data - other.data))

    def __mul__(self, scalar: float) -> "vec3":
        return vec3(*(self.data * scalar))

    def __rmul__(self, scalar: float) -> "vec3":
        return self.__mul__(scalar)

    def __str__(self) -> str:
        return f"vec3({self.x}, {self.y}, {self.z})"


class vec4:
    def __init__(self, x: float, y: float, z: float, w: float):
        self.data = np.array([x, y, z, w], dtype=np.float32)

    @property
    def x(self) -> float:
        return self.data[0]

    @property
    def y(self) -> float:
        return self.data[1]

    @property
    def z(self) -> float:
        return self.data[2]

    @property
    def w(self) -> float:
        return self.data[3]

    @x.setter
    def x(self, value: float):
        self.data[0] = value

    @y.setter
    def y(self, value: float):
        self.data[1] = value

    @z.setter
    def z(self, value: float):
        self.data[2] = value

    @w.setter
    def w(self, value: float):
        self.data[3] = value

    def __add__(self, other: "vec4") -> "vec4":
        return vec4(*(self.data + other.data))

    def __sub__(self, other: "vec4") -> "vec4":
        return vec4(*(self.data - other.data))

    def __mul__(self, scalar: float) -> "vec4":
        return vec4(*(self.data * scalar))

    def __rmul__(self, scalar: float) -> "vec4":
        return self.__mul__(scalar)

    def __str__(self) -> str:
        return f"vec4({self.x}, {self.y}, {self.z}, {self.w})"


# Matrix classes
class mat2:
    def __init__(self, *args: float):
        if len(args) == 4:  # 2x2 matrix
            self.data = np.array(
                [[args[0], args[1]], [args[2], args[3]]], dtype=np.float32
            )
        else:
            self.data = np.eye(2, dtype=np.float32)  # Identity matrix


class mat3:
    def __init__(self, *args: float):
        if len(args) == 9:  # 3x3 matrix
            self.data = np.array(
                [
                    [args[0], args[1], args[2]],
                    [args[3], args[4], args[5]],
                    [args[6], args[7], args[8]],
                ],
                dtype=np.float32,
            )
        else:
            self.data = np.eye(3, dtype=np.float32)  # Identity matrix


class mat4:
    def __init__(self, *args: float):
        if len(args) == 16:  # 4x4 matrix
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
            self.data = np.eye(4, dtype=np.float32)  # Identity matrix


# GLSL built-in functions
def sin(x: float) -> float:
    return np.sin(x)


def cos(x: float) -> float:
    return np.cos(x)


def tan(x: float) -> float:
    return np.tan(x)


def abs(x: Union[float, vec2, vec3, vec4]) -> Union[float, vec2, vec3, vec4]:
    if isinstance(x, (vec2, vec3, vec4)):
        return type(x)(*np.abs(x.data))
    return np.abs(x)


def length(v: Union[vec2, vec3, vec4]) -> float:
    return np.linalg.norm(v.data)


def distance(p0: Union[vec2, vec3], p1: Union[vec2, vec3]) -> float:
    return np.linalg.norm(p0.data - p1.data)


def min(
    a: Union[float, vec2, vec3, vec4], b: Union[float, vec2, vec3, vec4]
) -> Union[float, vec2, vec3, vec4]:
    if isinstance(a, (vec2, vec3, vec4)):
        return type(a)(*np.minimum(a.data, b.data))
    return np.minimum(a, b)


def max(
    a: Union[float, vec2, vec3, vec4], b: Union[float, vec2, vec3, vec4]
) -> Union[float, vec2, vec3, vec4]:
    if isinstance(a, (vec2, vec3, vec4)):
        return type(a)(*np.maximum(a.data, b.data))
    return np.maximum(a, b)


def normalize(v: Union[vec2, vec3, vec4]) -> Union[vec2, vec3, vec4]:
    norm = np.linalg.norm(v.data)
    return type(v)(*(v.data / norm)) if norm != 0 else v


def cross(a: vec3, b: vec3) -> vec3:
    return vec3(*np.cross(a.data, b.data))


def mix(x: vec3, y: vec3, a: float) -> vec3:
    return vec3(*(x.data * (1 - a) + y.data * a))


def radians(degrees: float) -> float:
    return np.radians(degrees)


def sqrt(x: float) -> float:
    return np.sqrt(x)


def fract(v: Union[vec2, vec3, vec4]) -> Union[vec2, vec3, vec4]:
    return type(v)(*(v.data - np.floor(v.data)))
