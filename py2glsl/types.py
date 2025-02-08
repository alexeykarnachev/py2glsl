from dataclasses import dataclass
from typing import Any, Union


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

    @property
    def xy(self) -> vec2:
        return vec2(self.x, self.y)

    @property
    def rgb(self) -> "vec3":
        return vec3(self.x, self.y, self.z)


@dataclass
class vec4(glsl_type):
    x: float
    y: float
    z: float
    w: float

    @property
    def rgb(self) -> vec3:
        return vec3(self.x, self.y, self.z)
