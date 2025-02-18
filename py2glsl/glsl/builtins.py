from functools import wraps

import numpy as np


def glsl_template(template: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Convert inputs to numpy arrays
            np_args = [a.data if hasattr(a, "data") else a for a in args]
            result = func(*np_args, **kwargs)

            # Convert back to original type if applicable
            if isinstance(result, np.ndarray) and args:
                return type(args[0])(result)
            return result

        wrapper.__glsl_template__ = template
        return wrapper

    return decorator


# --------------------------------------------------
# Angle Conversion
# --------------------------------------------------
@glsl_template("radians({degrees})")
def radians(degrees):
    return np.radians(degrees)


@glsl_template("degrees({radians})")
def degrees(radians):
    return np.degrees(radians)


# --------------------------------------------------
# Trigonometry
# --------------------------------------------------
@glsl_template("sin({x})")
def sin(x):
    return np.sin(x)


@glsl_template("cos({x})")
def cos(x):
    return np.cos(x)


@glsl_template("tan({x})")
def tan(x):
    return np.tan(x)


@glsl_template("asin({x})")
def asin(x):
    return np.arcsin(x)


@glsl_template("acos({x})")
def acos(x):
    return np.arccos(x)


@glsl_template("atan({y_over_x})")
def atan(y_over_x):
    return np.arctan(y_over_x)


@glsl_template("atan({y}, {x})")
def atan2(y, x):
    return np.arctan2(y, x)


# --------------------------------------------------
# Hyperbolic
# --------------------------------------------------
@glsl_template("sinh({x})")
def sinh(x):
    return np.sinh(x)


@glsl_template("cosh({x})")
def cosh(x):
    return np.cosh(x)


@glsl_template("tanh({x})")
def tanh(x):
    return np.tanh(x)


@glsl_template("asinh({x})")
def asinh(x):
    return np.arcsinh(x)


@glsl_template("acosh({x})")
def acosh(x):
    return np.arccosh(x)


@glsl_template("atanh({x})")
def atanh(x):
    return np.arctanh(x)


# --------------------------------------------------
# Exponential
# --------------------------------------------------
@glsl_template("pow({x}, {y})")
def pow(x, y):
    return np.power(x, y)


@glsl_template("exp({x})")
def exp(x):
    return np.exp(x)


@glsl_template("log({x})")
def log(x):
    return np.log(x)


@glsl_template("exp2({x})")
def exp2(x):
    return np.exp2(x)


@glsl_template("log2({x})")
def log2(x):
    return np.log2(x)


@glsl_template("sqrt({x})")
def sqrt(x):
    return np.sqrt(x)


@glsl_template("inversesqrt({x})")
def inversesqrt(x):
    return 1.0 / np.sqrt(x)


# --------------------------------------------------
# Common
# --------------------------------------------------
@glsl_template("abs({x})")
def abs(x):
    return np.abs(x)


@glsl_template("sign({x})")
def sign(x):
    return np.sign(x)


@glsl_template("floor({x})")
def floor(x):
    return np.floor(x)


@glsl_template("ceil({x})")
def ceil(x):
    return np.ceil(x)


@glsl_template("round({x})")
def round(x):
    return np.round(x)


@glsl_template("trunc({x})")
def trunc(x):
    return np.trunc(x)


@glsl_template("fract({x})")
def fract(x):
    return x - np.floor(x)


@glsl_template("mod({x}, {y})")
def mod(x, y):
    return np.mod(x, y)


@glsl_template("min({x}, {y})")
def min(x, y):
    return np.minimum(x, y)


@glsl_template("max({x}, {y})")
def max(x, y):
    return np.maximum(x, y)


@glsl_template("clamp({x}, {min_val}, {max_val})")
def clamp(x, min_val, max_val):
    return np.clip(x, min_val, max_val)


@glsl_template("mix({x}, {y}, {a})")
def mix(x, y, a):
    return x * (1.0 - a) + y * a


@glsl_template("step({edge}, {x})")
def step(edge, x):
    return np.where(x < edge, 0.0, 1.0)


@glsl_template("smoothstep({edge0}, {edge1}, {x})")
def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# --------------------------------------------------
# Geometric
# --------------------------------------------------
@glsl_template("length({v})")
def length(v):
    return np.linalg.norm(v)


@glsl_template("distance({a}, {b})")
def distance(a, b):
    return np.linalg.norm(a - b)


@glsl_template("dot({x}, {y})")
def dot(x, y):
    return np.dot(x.flatten(), y.flatten())


@glsl_template("cross({x}, {y})")
def cross(x, y):
    return np.cross(x, y)


@glsl_template("normalize({v})")
def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v


@glsl_template("faceforward({N}, {I}, {Nref})")
def faceforward(N, I, Nref):
    dot_val = np.dot(Nref.flatten(), I.flatten())
    return np.where(dot_val < 0.0, N, -N)


@glsl_template("reflect({I}, {N})")
def reflect(I, N):
    return I - 2.0 * np.dot(I.flatten(), N.flatten()) * N


@glsl_template("refract({I}, {N}, {eta})")
def refract(I, N, eta):
    dot_val = np.dot(I.flatten(), N.flatten())
    k = 1.0 - eta**2 * (1.0 - dot_val**2)
    return np.where(k < 0.0, 0.0, eta * I - (eta * dot_val + np.sqrt(k)) * N)


# --------------------------------------------------
# Matrix
# --------------------------------------------------
@glsl_template("transpose({m})")
def transpose(m):
    return np.transpose(m)
