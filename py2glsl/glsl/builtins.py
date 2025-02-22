import inspect
from functools import wraps
from typing import Any, Callable, NamedTuple, Union, get_type_hints

import numpy as np

from py2glsl.glsl.types import mat3, mat4, vec2, vec3, vec4
from py2glsl.transpiler.type_system import TypeInfo


class GLSLFuncMeta:
    def __init__(self, template: str, arg_types: list, return_type: Any):
        self.template = template
        self.arg_types = arg_types
        self.return_type = return_type


def glsl_template(template: str):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            np_args = [a.data if hasattr(a, "data") else a for a in args]
            result = func(*np_args, **kwargs)

            if isinstance(result, np.ndarray) and args:
                return type(args[0])(result)
            return result

        # Generate metadata from type hints
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        arg_types = []
        for param in sig.parameters.values():
            hint = type_hints.get(param.name, float)
            arg_types.append(TypeInfo.from_pytype(hint))

        return_type = TypeInfo.from_pytype(type_hints.get("return", float))

        # Attach metadata
        wrapper.__glsl_metadata__ = GLSLFuncMeta(
            template=template, arg_types=arg_types, return_type=return_type
        )
        return wrapper

    return decorator


# ==================================================
# Type Resolution Functions
# ==================================================
def resolve_math_arg_types(args: list[TypeInfo]) -> list[TypeInfo]:
    """For functions like sin/cos that return same type as input"""
    if not args:
        return [TypeInfo.FLOAT]
    return [args[0]] * len(args)


def resolve_math_return_type(args: list[TypeInfo]) -> TypeInfo:
    return args[0] if args else TypeInfo.FLOAT


def resolve_clamp_types(args: list[TypeInfo]) -> list[TypeInfo]:
    if len(args) != 3:
        raise TypeError("Clamp requires 3 arguments")
    if args[0] != args[1] or args[0] != args[2]:
        raise TypeError("Clamp arguments must be same type")
    return [args[0]] * 3


def resolve_mix_types(args: list[TypeInfo]) -> list[TypeInfo]:
    if args[0] != args[1]:
        raise TypeError("Mix arguments 0 and 1 must match")
    if not (args[2].is_scalar or args[2] == args[0]):
        raise TypeError("Mix argument 2 must be scalar or match arguments 0/1")
    return [args[0], args[0], args[2]]


def resolve_vector_op_types(args: list[TypeInfo]) -> list[TypeInfo]:
    if args[0] != args[1]:
        raise TypeError("Vector operations require matching types")
    if not args[0].is_vector:
        raise TypeError("Vector operations require vector types")
    return [args[0]] * 2


# ==================================================
# Angle Conversion
# ==================================================
@glsl_template("radians({degrees})")
def radians(degrees: float) -> float:
    return np.radians(degrees)


radians.__glsl_metadata__.return_type = resolve_math_return_type


@glsl_template("degrees({radians})")
def degrees(radians: float) -> float:
    return np.degrees(radians)


degrees.__glsl_metadata__.return_type = resolve_math_return_type


# ==================================================
# Trigonometry
# ==================================================
@glsl_template("sin({x})")
def sin(x: float) -> float:
    return np.sin(x)


sin.__glsl_metadata__.arg_types = [resolve_math_arg_types]
sin.__glsl_metadata__.return_type = resolve_math_return_type


@glsl_template("cos({x})")
def cos(x: float) -> float:
    return np.cos(x)


cos.__glsl_metadata__.arg_types = [resolve_math_arg_types]
cos.__glsl_metadata__.return_type = resolve_math_return_type


@glsl_template("tan({x})")
def tan(x: float) -> float:
    return np.tan(x)


tan.__glsl_metadata__.arg_types = [resolve_math_arg_types]
tan.__glsl_metadata__.return_type = resolve_math_return_type


@glsl_template("asin({x})")
def asin(x: float) -> float:
    return np.arcsin(x)


asin.__glsl_metadata__.arg_types = [resolve_math_arg_types]
asin.__glsl_metadata__.return_type = resolve_math_return_type


@glsl_template("acos({x})")
def acos(x: float) -> float:
    return np.arccos(x)


acos.__glsl_metadata__.arg_types = [resolve_math_arg_types]
acos.__glsl_metadata__.return_type = resolve_math_return_type


@glsl_template("atan({y}, {x})")
def atan(y: float, x: float) -> float:
    return np.arctan2(y, x)


atan.__glsl_metadata__.arg_types = [
    lambda args: args[0],  # Inherit from first arg
    lambda args: args[0],  # Match first arg type
]
atan.__glsl_metadata__.return_type = resolve_math_return_type


# ==================================================
# Hyperbolic
# ==================================================
@glsl_template("sinh({x})")
def sinh(x: float) -> float:
    return np.sinh(x)


sinh.__glsl_metadata__.arg_types = [resolve_math_arg_types]
sinh.__glsl_metadata__.return_type = resolve_math_return_type


@glsl_template("cosh({x})")
def cosh(x: float) -> float:
    return np.cosh(x)


cosh.__glsl_metadata__.arg_types = [resolve_math_arg_types]
cosh.__glsl_metadata__.return_type = resolve_math_return_type


@glsl_template("tanh({x})")
def tanh(x: float) -> float:
    return np.tanh(x)


tanh.__glsl_metadata__.arg_types = [resolve_math_arg_types]
tanh.__glsl_metadata__.return_type = resolve_math_return_type


@glsl_template("asinh({x})")
def asinh(x: float) -> float:
    return np.arcsinh(x)


asinh.__glsl_metadata__.arg_types = [resolve_math_arg_types]
asinh.__glsl_metadata__.return_type = resolve_math_return_type


@glsl_template("acosh({x})")
def acosh(x: float) -> float:
    return np.arccosh(x)


acosh.__glsl_metadata__.arg_types = [resolve_math_arg_types]
acosh.__glsl_metadata__.return_type = resolve_math_return_type


@glsl_template("atanh({x})")
def atanh(x: float) -> float:
    return np.arctanh(x)


atanh.__glsl_metadata__.arg_types = [resolve_math_arg_types]
atanh.__glsl_metadata__.return_type = resolve_math_return_type


# ==================================================
# Exponential
# ==================================================
@glsl_template("pow({x}, {y})")
def pow(x: float, y: float) -> float:
    return np.power(x, y)


pow.__glsl_metadata__.arg_types = [resolve_math_arg_types, resolve_math_arg_types]
pow.__glsl_metadata__.return_type = resolve_math_return_type


@glsl_template("exp({x})")
def exp(x: float) -> float:
    return np.exp(x)


exp.__glsl_metadata__.arg_types = [resolve_math_arg_types]
exp.__glsl_metadata__.return_type = resolve_math_return_type


@glsl_template("log({x})")
def log(x: float) -> float:
    return np.log(x)


log.__glsl_metadata__.arg_types = [resolve_math_arg_types]
log.__glsl_metadata__.return_type = resolve_math_return_type


@glsl_template("exp2({x})")
def exp2(x: float) -> float:
    return np.exp2(x)


exp2.__glsl_metadata__.arg_types = [resolve_math_arg_types]
exp2.__glsl_metadata__.return_type = resolve_math_return_type


@glsl_template("log2({x})")
def log2(x: float) -> float:
    return np.log2(x)


log2.__glsl_metadata__.arg_types = [resolve_math_arg_types]
log2.__glsl_metadata__.return_type = resolve_math_return_type


@glsl_template("sqrt({x})")
def sqrt(x: float) -> float:
    return np.sqrt(x)


sqrt.__glsl_metadata__.arg_types = [resolve_math_arg_types]
sqrt.__glsl_metadata__.return_type = resolve_math_return_type


@glsl_template("inversesqrt({x})")
def inversesqrt(x: float) -> float:
    return 1.0 / np.sqrt(x)


inversesqrt.__glsl_metadata__.arg_types = [resolve_math_arg_types]
inversesqrt.__glsl_metadata__.return_type = resolve_math_return_type


# ==================================================
# Common
# ==================================================
@glsl_template("abs({x})")
def abs(x: float) -> float:
    return np.abs(x)


abs.__glsl_metadata__.arg_types = [resolve_math_arg_types]
abs.__glsl_metadata__.return_type = resolve_math_return_type


@glsl_template("sign({x})")
def sign(x: float) -> float:
    return np.sign(x)


sign.__glsl_metadata__.arg_types = [resolve_math_arg_types]
sign.__glsl_metadata__.return_type = resolve_math_return_type


@glsl_template("floor({x})")
def floor(x: float) -> float:
    return np.floor(x)


floor.__glsl_metadata__.arg_types = [resolve_math_arg_types]
floor.__glsl_metadata__.return_type = resolve_math_return_type


@glsl_template("ceil({x})")
def ceil(x: float) -> float:
    return np.ceil(x)


ceil.__glsl_metadata__.arg_types = [resolve_math_arg_types]
ceil.__glsl_metadata__.return_type = resolve_math_return_type


@glsl_template("round({x})")
def round(x: float) -> float:
    return np.round(x)


round.__glsl_metadata__.arg_types = [resolve_math_arg_types]
round.__glsl_metadata__.return_type = resolve_math_return_type


@glsl_template("trunc({x})")
def trunc(x: float) -> float:
    return np.trunc(x)


trunc.__glsl_metadata__.arg_types = [resolve_math_arg_types]
trunc.__glsl_metadata__.return_type = resolve_math_return_type


@glsl_template("fract({x})")
def fract(x: float) -> float:
    return x - np.floor(x)


fract.__glsl_metadata__.arg_types = [resolve_math_arg_types]
fract.__glsl_metadata__.return_type = resolve_math_return_type


@glsl_template("mod({x}, {y})")
def mod(x: float, y: float) -> float:
    return np.mod(x, y)


mod.__glsl_metadata__.arg_types = [
    resolve_math_arg_types,
    lambda args: args[0],  # Match first arg type
]
mod.__glsl_metadata__.return_type = resolve_math_return_type


@glsl_template("min({x}, {y})")
def min(x: float, y: float) -> float:
    return np.minimum(x, y)


min.__glsl_metadata__.arg_types = [lambda args: args[0], lambda args: args[0]]
min.__glsl_metadata__.return_type = resolve_math_return_type


@glsl_template("max({x}, {y})")
def max(x: float, y: float) -> float:
    return np.maximum(x, y)


max.__glsl_metadata__.arg_types = [lambda args: args[0], lambda args: args[0]]
max.__glsl_metadata__.return_type = resolve_math_return_type


@glsl_template("clamp({x}, {min_val}, {max_val})")
def clamp(x: float, min_val: float, max_val: float) -> float:
    return np.clip(x, min_val, max_val)


clamp.__glsl_metadata__.arg_types = [resolve_clamp_types]
clamp.__glsl_metadata__.return_type = lambda args: args[0]


@glsl_template("mix({x}, {y}, {a})")
def mix(
    x: Union[float, vec2, vec3, vec4],
    y: Union[float, vec2, vec3, vec4],
    a: Union[float, vec2, vec3, vec4],
) -> Union[float, vec2, vec3, vec4]:
    return x * (1.0 - a) + y * a


# Explicit metadata setup
mix.__glsl_metadata__ = GLSLFuncMeta(
    template="mix({x}, {y}, {a})",
    arg_types=[
        TypeInfo.VECn,  # Polymorphic first argument
        lambda args: args[0],  # Match first argument type
        lambda args: (
            TypeInfo.FLOAT if args[0].is_scalar else args[0]
        ),  # Scalar or matching vector
    ],
    return_type=lambda args: args[0],  # Return type matches first argument
)


@glsl_template("step({edge}, {x})")
def step(edge: float, x: float) -> float:
    return np.where(x < edge, 0.0, 1.0)


step.__glsl_metadata__.arg_types = [resolve_math_arg_types, lambda args: args[0]]
step.__glsl_metadata__.return_type = resolve_math_return_type


@glsl_template("smoothstep({edge0}, {edge1}, {x})")
def smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


smoothstep.__glsl_metadata__.arg_types = [
    resolve_math_arg_types,
    lambda args: args[0],
    lambda args: args[0],
]
smoothstep.__glsl_metadata__.return_type = resolve_math_return_type


# ==================================================
# Geometric
# ==================================================
@glsl_template("length({v})")
def length(v: vec2) -> float:
    return np.linalg.norm(v)


length.__glsl_metadata__.arg_types = [TypeInfo.VECn]
length.__glsl_metadata__.return_type = TypeInfo.FLOAT


@glsl_template("distance({a}, {b})")
def distance(a: vec2, b: vec2) -> float:
    return np.linalg.norm(a - b)


distance.__glsl_metadata__.arg_types = [TypeInfo.VECn, TypeInfo.VECn]
distance.__glsl_metadata__.return_type = TypeInfo.FLOAT


@glsl_template("dot({x}, {y})")
def dot(x: vec2, y: vec2) -> float:
    return np.dot(x.flatten(), y.flatten())


dot.__glsl_metadata__.arg_types = [resolve_vector_op_types]
dot.__glsl_metadata__.return_type = TypeInfo.FLOAT


@glsl_template("cross({x}, {y})")
def cross(x: vec3, y: vec3) -> vec3:
    return np.cross(x, y)


cross.__glsl_metadata__.arg_types = [TypeInfo.VEC3, TypeInfo.VEC3]
cross.__glsl_metadata__.return_type = TypeInfo.VEC3


@glsl_template("normalize({v})")
def normalize(v: vec3) -> vec3:
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v


normalize.__glsl_metadata__.arg_types = [TypeInfo.VECn]
normalize.__glsl_metadata__.return_type = lambda args: args[0]


@glsl_template("faceforward({N}, {I}, {Nref})")
def faceforward(N: vec3, I: vec3, Nref: vec3) -> vec3:
    dot_val = np.dot(Nref.flatten(), I.flatten())
    return np.where(dot_val < 0.0, N, -N)


faceforward.__glsl_metadata__.arg_types = [TypeInfo.VECn, TypeInfo.VECn, TypeInfo.VECn]
faceforward.__glsl_metadata__.return_type = lambda args: args[0]


@glsl_template("reflect({I}, {N})")
def reflect(I: vec3, N: vec3) -> vec3:
    return I - 2.0 * np.dot(I.flatten(), N.flatten()) * N


reflect.__glsl_metadata__.arg_types = [TypeInfo.VECn, TypeInfo.VECn]
reflect.__glsl_metadata__.return_type = lambda args: args[0]


@glsl_template("refract({I}, {N}, {eta})")
def refract(I: vec3, N: vec3, eta: float) -> vec3:
    dot_val = np.dot(I.flatten(), N.flatten())
    k = 1.0 - eta**2 * (1.0 - dot_val**2)
    return np.where(k < 0.0, 0.0, eta * I - (eta * dot_val + np.sqrt(k)) * N)


refract.__glsl_metadata__.arg_types = [TypeInfo.VECn, TypeInfo.VECn, TypeInfo.FLOAT]
refract.__glsl_metadata__.return_type = lambda args: args[0]


# ==================================================
# Matrix
# ==================================================
@glsl_template("transpose({m})")
def transpose(m: mat4) -> mat4:
    return np.transpose(m)


transpose.__glsl_metadata__.arg_types = [TypeInfo.MATn]
transpose.__glsl_metadata__.return_type = lambda args: args[0]
