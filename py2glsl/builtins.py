from typing import (
    Any, Callable, Generic, NamedTuple, Protocol, Tuple, 
    TypeVar, Union, cast, overload
)
from typing_extensions import Self  # For Python 3.11 and older
import numpy as np
from numpy.typing import NDArray

# Define type variables for vector types
T = TypeVar("T", bound=Union[float, "vec2", "vec3", "vec4"])
VecType = TypeVar("VecType", bound=Union["vec2", "vec3", "vec4"])
FloatArray = NDArray[np.float32]


# Vector classes
class vec2:
    data: FloatArray
    _x: property
    _y: property
    
    def __init__(self, x: float, y: float):
        self.data = np.array([x, y], dtype=np.float32)

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

    def __add__(self, other: "vec2") -> "vec2":
        result = self.data + other.data
        return vec2(float(result[0]), float(result[1]))

    def __sub__(self, other: "vec2") -> "vec2":
        result = self.data - other.data
        return vec2(float(result[0]), float(result[1]))

    def __mul__(self, scalar: float) -> "vec2":
        result = self.data * scalar
        return vec2(float(result[0]), float(result[1]))

    def __rmul__(self, scalar: float) -> "vec2":
        return self.__mul__(scalar)

    def __str__(self) -> str:
        return f"vec2({self.x}, {self.y})"


class vec3:
    data: FloatArray
    _x: property
    _y: property
    _z: property
    
    def __init__(self, x: float, y: float, z: float):
        self.data = np.array([x, y, z], dtype=np.float32)

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
    def xyy(self) -> "vec3":
        return vec3(self.x, self.y, self.y)

    @property
    def yxy(self) -> "vec3":
        return vec3(self.y, self.x, self.y)

    @property
    def yyx(self) -> "vec3":
        return vec3(self.y, self.y, self.x)

    def __add__(self, other: "vec3") -> "vec3":
        result = self.data + other.data
        return vec3(float(result[0]), float(result[1]), float(result[2]))

    def __sub__(self, other: "vec3") -> "vec3":
        result = self.data - other.data
        return vec3(float(result[0]), float(result[1]), float(result[2]))

    def __mul__(self, scalar: float) -> "vec3":
        result = self.data * scalar
        return vec3(float(result[0]), float(result[1]), float(result[2]))

    def __rmul__(self, scalar: float) -> "vec3":
        return self.__mul__(scalar)

    def __str__(self) -> str:
        return f"vec3({self.x}, {self.y}, {self.z})"


class vec4:
    data: FloatArray
    _x: property
    _y: property
    _z: property
    _w: property
    
    def __init__(self, x: float, y: float, z: float, w: float):
        self.data = np.array([x, y, z, w], dtype=np.float32)

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

    def __add__(self, other: "vec4") -> "vec4":
        result = self.data + other.data
        return vec4(float(result[0]), float(result[1]), float(result[2]), float(result[3]))

    def __sub__(self, other: "vec4") -> "vec4":
        result = self.data - other.data
        return vec4(float(result[0]), float(result[1]), float(result[2]), float(result[3]))

    def __mul__(self, scalar: float) -> "vec4":
        result = self.data * scalar
        return vec4(float(result[0]), float(result[1]), float(result[2]), float(result[3]))

    def __rmul__(self, scalar: float) -> "vec4":
        return self.__mul__(scalar)

    def __str__(self) -> str:
        return f"vec4({self.x}, {self.y}, {self.z}, {self.w})"


# Matrix classes
class mat2:
    data: FloatArray
    
    def __init__(self, *args: float):
        if len(args) == 4:  # 2x2 matrix
            self.data = np.array(
                [[args[0], args[1]], [args[2], args[3]]], dtype=np.float32
            )
        else:
            self.data = np.eye(2, dtype=np.float32)  # Identity matrix


class mat3:
    data: FloatArray
    
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
    data: FloatArray
    
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
    return float(np.sin(x))


def cos(x: float) -> float:
    return float(np.cos(x))


def tan(x: float) -> float:
    return float(np.tan(x))


@overload
def abs(x: float) -> float: ...

@overload
def abs(x: vec2) -> vec2: ...

@overload
def abs(x: vec3) -> vec3: ...

@overload
def abs(x: vec4) -> vec4: ...

def abs(x: Union[float, vec2, vec3, vec4]) -> Union[float, vec2, vec3, vec4]:
    if isinstance(x, vec2):
        result = np.abs(x.data)
        return vec2(float(result[0]), float(result[1]))
    elif isinstance(x, vec3):
        result = np.abs(x.data)
        return vec3(float(result[0]), float(result[1]), float(result[2]))
    elif isinstance(x, vec4):
        result = np.abs(x.data)
        return vec4(float(result[0]), float(result[1]), float(result[2]), float(result[3]))
    return float(np.abs(x))


@overload
def length(v: vec2) -> float: ...

@overload
def length(v: vec3) -> float: ...

@overload
def length(v: vec4) -> float: ...

def length(v: Union[vec2, vec3, vec4]) -> float:
    return float(np.linalg.norm(v.data))


@overload
def distance(p0: vec2, p1: vec2) -> float: ...

@overload
def distance(p0: vec3, p1: vec3) -> float: ...

@overload
def distance(p0: vec4, p1: vec4) -> float: ...

def distance(p0: Union[vec2, vec3, vec4], p1: Union[vec2, vec3, vec4]) -> float:
    if type(p0) != type(p1):
        raise TypeError(f"distance() requires matching vector types, got {type(p0)} and {type(p1)}")
    return float(np.linalg.norm(p0.data - p1.data))


@overload
def min(a: float, b: float) -> float: ...

@overload
def min(a: vec2, b: vec2) -> vec2: ...

@overload
def min(a: vec3, b: vec3) -> vec3: ...

@overload
def min(a: vec4, b: vec4) -> vec4: ...

def min(
    a: Union[float, vec2, vec3, vec4], b: Union[float, vec2, vec3, vec4]
) -> Union[float, vec2, vec3, vec4]:
    if isinstance(a, vec2) and isinstance(b, vec2):
        result = np.minimum(a.data, b.data)
        return vec2(float(result[0]), float(result[1]))
    elif isinstance(a, vec3) and isinstance(b, vec3):
        result = np.minimum(a.data, b.data)
        return vec3(float(result[0]), float(result[1]), float(result[2]))
    elif isinstance(a, vec4) and isinstance(b, vec4):
        result = np.minimum(a.data, b.data)
        return vec4(float(result[0]), float(result[1]), float(result[2]), float(result[3]))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(np.minimum(a, b))
    else:
        raise TypeError(f"min() received incompatible types: {type(a)} and {type(b)}")


@overload
def max(a: float, b: float) -> float: ...

@overload
def max(a: vec2, b: vec2) -> vec2: ...

@overload
def max(a: vec3, b: vec3) -> vec3: ...

@overload
def max(a: vec4, b: vec4) -> vec4: ...

def max(
    a: Union[float, vec2, vec3, vec4], b: Union[float, vec2, vec3, vec4]
) -> Union[float, vec2, vec3, vec4]:
    if isinstance(a, vec2) and isinstance(b, vec2):
        result = np.maximum(a.data, b.data)
        return vec2(float(result[0]), float(result[1]))
    elif isinstance(a, vec3) and isinstance(b, vec3):
        result = np.maximum(a.data, b.data)
        return vec3(float(result[0]), float(result[1]), float(result[2]))
    elif isinstance(a, vec4) and isinstance(b, vec4):
        result = np.maximum(a.data, b.data)
        return vec4(float(result[0]), float(result[1]), float(result[2]), float(result[3]))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(np.maximum(a, b))
    else:
        raise TypeError(f"max() received incompatible types: {type(a)} and {type(b)}")


@overload
def normalize(v: vec2) -> vec2: ...

@overload
def normalize(v: vec3) -> vec3: ...

@overload
def normalize(v: vec4) -> vec4: ...

def normalize(v: Union[vec2, vec3, vec4]) -> Union[vec2, vec3, vec4]:
    norm = float(np.linalg.norm(v.data))
    if norm == 0:
        return v
        
    if isinstance(v, vec2):
        result = v.data / norm
        return vec2(float(result[0]), float(result[1]))
    elif isinstance(v, vec3):
        result = v.data / norm
        return vec3(float(result[0]), float(result[1]), float(result[2]))
    else:  # vec4
        result = v.data / norm
        return vec4(float(result[0]), float(result[1]), float(result[2]), float(result[3]))


def cross(a: vec3, b: vec3) -> vec3:
    result = np.cross(a.data, b.data)
    return vec3(float(result[0]), float(result[1]), float(result[2]))


@overload
def mix(x: float, y: float, a: float) -> float: ...

@overload
def mix(x: vec2, y: vec2, a: float) -> vec2: ...

@overload
def mix(x: vec3, y: vec3, a: float) -> vec3: ...

@overload
def mix(x: vec4, y: vec4, a: float) -> vec4: ...

@overload
def mix(x: vec2, y: vec2, a: vec2) -> vec2: ...

@overload
def mix(x: vec3, y: vec3, a: vec3) -> vec3: ...

@overload
def mix(x: vec4, y: vec4, a: vec4) -> vec4: ...

def mix(
    x: Union[float, vec2, vec3, vec4], 
    y: Union[float, vec2, vec3, vec4], 
    a: Union[float, vec2, vec3, vec4]
) -> Union[float, vec2, vec3, vec4]:
    # Handle float, float, float → float
    if isinstance(x, (int, float)) and isinstance(y, (int, float)) and isinstance(a, (int, float)):
        return float(x * (1 - a) + y * a)
    
    # Handle vec2, vec2, float → vec2
    if isinstance(x, vec2) and isinstance(y, vec2) and isinstance(a, (int, float)):
        result = x.data * (1 - a) + y.data * a
        return vec2(float(result[0]), float(result[1]))
    
    # Handle vec3, vec3, float → vec3
    if isinstance(x, vec3) and isinstance(y, vec3) and isinstance(a, (int, float)):
        result = x.data * (1 - a) + y.data * a
        return vec3(float(result[0]), float(result[1]), float(result[2]))
    
    # Handle vec4, vec4, float → vec4
    if isinstance(x, vec4) and isinstance(y, vec4) and isinstance(a, (int, float)):
        result = x.data * (1 - a) + y.data * a
        return vec4(float(result[0]), float(result[1]), float(result[2]), float(result[3]))
    
    # Handle vec2, vec2, vec2 → vec2
    if isinstance(x, vec2) and isinstance(y, vec2) and isinstance(a, vec2):
        result = x.data * (1 - a.data) + y.data * a.data
        return vec2(float(result[0]), float(result[1]))
    
    # Handle vec3, vec3, vec3 → vec3
    if isinstance(x, vec3) and isinstance(y, vec3) and isinstance(a, vec3):
        result = x.data * (1 - a.data) + y.data * a.data
        return vec3(float(result[0]), float(result[1]), float(result[2]))
    
    # Handle vec4, vec4, vec4 → vec4
    if isinstance(x, vec4) and isinstance(y, vec4) and isinstance(a, vec4):
        result = x.data * (1 - a.data) + y.data * a.data
        return vec4(float(result[0]), float(result[1]), float(result[2]), float(result[3]))
    
    raise TypeError(f"mix() received incompatible types: {type(x)}, {type(y)}, {type(a)}")


@overload
def smoothstep(edge0: float, edge1: float, x: float) -> float: ...

@overload
def smoothstep(edge0: vec2, edge1: vec2, x: vec2) -> vec2: ...

@overload
def smoothstep(edge0: vec3, edge1: vec3, x: vec3) -> vec3: ...

@overload
def smoothstep(edge0: vec4, edge1: vec4, x: vec4) -> vec4: ...

def smoothstep(
    edge0: Union[float, vec2, vec3, vec4],
    edge1: Union[float, vec2, vec3, vec4],
    x: Union[float, vec2, vec3, vec4]
) -> Union[float, vec2, vec3, vec4]:
    # Handle float, float, float → float
    if isinstance(edge0, (int, float)) and isinstance(edge1, (int, float)) and isinstance(x, (int, float)):
        t = float(np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0))
        return t * t * (3.0 - 2.0 * t)
    
    # Handle vec2, vec2, vec2 → vec2
    if isinstance(edge0, vec2) and isinstance(edge1, vec2) and isinstance(x, vec2):
        t_array = np.clip((x.data - edge0.data) / (edge1.data - edge0.data), 0.0, 1.0)
        result_array = t_array * t_array * (3.0 - 2.0 * t_array)
        return vec2(float(result_array[0]), float(result_array[1]))
    
    # Handle vec3, vec3, vec3 → vec3
    if isinstance(edge0, vec3) and isinstance(edge1, vec3) and isinstance(x, vec3):
        t_array = np.clip((x.data - edge0.data) / (edge1.data - edge0.data), 0.0, 1.0)
        result_array = t_array * t_array * (3.0 - 2.0 * t_array)
        return vec3(float(result_array[0]), float(result_array[1]), float(result_array[2]))
    
    # Handle vec4, vec4, vec4 → vec4
    if isinstance(edge0, vec4) and isinstance(edge1, vec4) and isinstance(x, vec4):
        t_array = np.clip((x.data - edge0.data) / (edge1.data - edge0.data), 0.0, 1.0)
        result_array = t_array * t_array * (3.0 - 2.0 * t_array)
        return vec4(float(result_array[0]), float(result_array[1]), float(result_array[2]), float(result_array[3]))
    
    raise TypeError(f"smoothstep() received incompatible types: {type(edge0)}, {type(edge1)}, {type(x)}")


def radians(degrees: float) -> float:
    return float(np.radians(degrees))


def sqrt(x: float) -> float:
    return float(np.sqrt(x))


@overload
def fract(v: vec2) -> vec2: ...

@overload
def fract(v: vec3) -> vec3: ...

@overload
def fract(v: vec4) -> vec4: ...

def fract(v: Union[vec2, vec3, vec4]) -> Union[vec2, vec3, vec4]:
    if isinstance(v, vec2):
        result = v.data - np.floor(v.data)
        return vec2(float(result[0]), float(result[1]))
    elif isinstance(v, vec3):
        result = v.data - np.floor(v.data)
        return vec3(float(result[0]), float(result[1]), float(result[2]))
    else:  # vec4
        result = v.data - np.floor(v.data)
        return vec4(float(result[0]), float(result[1]), float(result[2]), float(result[3]))
