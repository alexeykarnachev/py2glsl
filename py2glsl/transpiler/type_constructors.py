"""GLSL type constructors and runtime type implementations."""

from typing import TypeAlias

import numpy as np

from py2glsl.transpiler.type_system import GLSLTypeError

# Runtime type aliases
Vec2: TypeAlias = np.ndarray  # shape (2,)
Vec3: TypeAlias = np.ndarray  # shape (3,)
Vec4: TypeAlias = np.ndarray  # shape (4,)
Mat2: TypeAlias = np.ndarray  # shape (2, 2)
Mat3: TypeAlias = np.ndarray  # shape (3, 3)
Mat4: TypeAlias = np.ndarray  # shape (4, 4)
IVec2: TypeAlias = np.ndarray  # shape (2,) dtype=int32
IVec3: TypeAlias = np.ndarray  # shape (3,) dtype=int32
IVec4: TypeAlias = np.ndarray  # shape (4,) dtype=int32
BVec2: TypeAlias = np.ndarray  # shape (2,) dtype=bool
BVec3: TypeAlias = np.ndarray  # shape (3,) dtype=bool
BVec4: TypeAlias = np.ndarray  # shape (4,) dtype=bool


def vec2(*args) -> Vec2:
    """Create a 2D float vector."""
    if len(args) == 0:
        raise ValueError("vec2 requires at least 1 argument")
    if len(args) > 2:
        raise TypeError("vec2 requires 1 or 2 arguments")

    if len(args) == 1:
        if isinstance(args[0], (Vec2, IVec2, BVec2)):
            return np.array(args[0], dtype=np.float32)
        val = float(args[0])
        return np.array([val, val], dtype=np.float32)

    return np.array([float(args[0]), float(args[1])], dtype=np.float32)


def vec3(*args) -> Vec3:
    """Create a 3D float vector."""
    if len(args) == 0:
        raise ValueError("vec3 requires at least 1 argument")
    if len(args) > 3:
        raise TypeError("vec3 requires 1 to 3 arguments")

    if len(args) == 1:
        if isinstance(args[0], (Vec3, IVec3, BVec3)):
            return np.array(args[0], dtype=np.float32)
        val = float(args[0])
        return np.array([val, val, val], dtype=np.float32)

    if len(args) == 2:
        if isinstance(args[0], (Vec2, IVec2, BVec2)):
            return np.array([*args[0], float(args[1])], dtype=np.float32)
        raise TypeError("First argument must be vec2 when using 2 arguments")

    return np.array([float(args[0]), float(args[1]), float(args[2])], dtype=np.float32)


def vec4(*args) -> Vec4:
    """Create a 4D float vector."""
    if len(args) == 0:
        raise ValueError("vec4 requires at least 1 argument")
    if len(args) > 4:
        raise TypeError("vec4 requires 1 to 4 arguments")

    if len(args) == 1:
        if isinstance(args[0], (Vec4, IVec4, BVec4)):
            return np.array(args[0], dtype=np.float32)
        val = float(args[0])
        return np.array([val, val, val, val], dtype=np.float32)

    if len(args) == 2:
        if isinstance(args[0], (Vec3, IVec3, BVec3)):
            return np.array([*args[0], float(args[1])], dtype=np.float32)
        if isinstance(args[0], (Vec2, IVec2, BVec2)):
            val = float(args[1])
            return np.array([*args[0], val, val], dtype=np.float32)
        raise TypeError("First argument must be vec2 or vec3 when using 2 arguments")

    if len(args) == 3:
        if isinstance(args[0], (Vec2, IVec2, BVec2)):
            return np.array(
                [*args[0], float(args[1]), float(args[2])], dtype=np.float32
            )
        raise TypeError("First argument must be vec2 when using 3 arguments")

    return np.array(
        [float(args[0]), float(args[1]), float(args[2]), float(args[3])],
        dtype=np.float32,
    )


def ivec2(*args) -> IVec2:
    """Create a 2D integer vector."""
    if len(args) == 0:
        raise ValueError("ivec2 requires at least 1 argument")
    if len(args) > 2:
        raise TypeError("ivec2 requires 1 or 2 arguments")

    if len(args) == 1:
        if isinstance(args[0], (Vec2, IVec2, BVec2)):
            return np.array(args[0], dtype=np.int32)
        val = int(args[0])
        return np.array([val, val], dtype=np.int32)

    return np.array([int(args[0]), int(args[1])], dtype=np.int32)


def ivec3(*args) -> IVec3:
    """Create a 3D integer vector."""
    if len(args) == 0:
        raise ValueError("ivec3 requires at least 1 argument")
    if len(args) > 3:
        raise TypeError("ivec3 requires 1 to 3 arguments")

    if len(args) == 1:
        if isinstance(args[0], (Vec3, IVec3, BVec3)):
            return np.array(args[0], dtype=np.int32)
        val = int(args[0])
        return np.array([val, val, val], dtype=np.int32)

    if len(args) == 2:
        if isinstance(args[0], (Vec2, IVec2, BVec2)):
            return np.array([*args[0], int(args[1])], dtype=np.int32)
        raise TypeError("First argument must be vec2 when using 2 arguments")

    return np.array([int(args[0]), int(args[1]), int(args[2])], dtype=np.int32)


def ivec4(*args) -> IVec4:
    """Create a 4D integer vector."""
    if len(args) == 0:
        raise ValueError("ivec4 requires at least 1 argument")
    if len(args) > 4:
        raise TypeError("ivec4 requires 1 to 4 arguments")

    if len(args) == 1:
        if isinstance(args[0], (Vec4, IVec4, BVec4)):
            return np.array(args[0], dtype=np.int32)
        val = int(args[0])
        return np.array([val, val, val, val], dtype=np.int32)

    if len(args) == 2:
        if isinstance(args[0], (Vec3, IVec3, BVec3)):
            return np.array([*args[0], int(args[1])], dtype=np.int32)
        if isinstance(args[0], (Vec2, IVec2, BVec2)):
            val = int(args[1])
            return np.array([*args[0], val, val], dtype=np.int32)
        raise TypeError("First argument must be vec2 or vec3 when using 2 arguments")

    if len(args) == 3:
        if isinstance(args[0], (Vec2, IVec2, BVec2)):
            return np.array([*args[0], int(args[1]), int(args[2])], dtype=np.int32)
        raise TypeError("First argument must be vec2 when using 3 arguments")

    return np.array(
        [int(args[0]), int(args[1]), int(args[2]), int(args[3])], dtype=np.int32
    )


def bvec2(*args) -> BVec2:
    """Create a 2D boolean vector."""
    if len(args) == 0:
        raise ValueError("bvec2 requires at least 1 argument")
    if len(args) > 2:
        raise TypeError("bvec2 requires 1 or 2 arguments")

    if len(args) == 1:
        if isinstance(args[0], (Vec2, IVec2, BVec2)):
            return np.array(args[0], dtype=bool)
        val = bool(args[0])
        return np.array([val, val], dtype=bool)

    return np.array([bool(args[0]), bool(args[1])], dtype=bool)


def bvec3(*args) -> BVec3:
    """Create a 3D boolean vector."""
    if len(args) == 0:
        raise ValueError("bvec3 requires at least 1 argument")
    if len(args) > 3:
        raise TypeError("bvec3 requires 1 to 3 arguments")

    if len(args) == 1:
        if isinstance(args[0], (Vec3, IVec3, BVec3)):
            return np.array(args[0], dtype=bool)
        val = bool(args[0])
        return np.array([val, val, val], dtype=bool)

    if len(args) == 2:
        if isinstance(args[0], (Vec2, IVec2, BVec2)):
            return np.array([*args[0], bool(args[1])], dtype=bool)
        raise TypeError("First argument must be vec2 when using 2 arguments")

    return np.array([bool(args[0]), bool(args[1]), bool(args[2])], dtype=bool)


def bvec4(*args) -> BVec4:
    """Create a 4D boolean vector."""
    if len(args) == 0:
        raise ValueError("bvec4 requires at least 1 argument")
    if len(args) > 4:
        raise TypeError("bvec4 requires 1 to 4 arguments")

    if len(args) == 1:
        if isinstance(args[0], (Vec4, IVec4, BVec4)):
            return np.array(args[0], dtype=bool)
        val = bool(args[0])
        return np.array([val, val, val, val], dtype=bool)

    if len(args) == 2:
        if isinstance(args[0], (Vec3, IVec3, BVec3)):
            return np.array([*args[0], bool(args[1])], dtype=bool)
        if isinstance(args[0], (Vec2, IVec2, BVec2)):
            val = bool(args[1])
            return np.array([*args[0], val, val], dtype=bool)
        raise TypeError("First argument must be vec2 or vec3 when using 2 arguments")

    if len(args) == 3:
        if isinstance(args[0], (Vec2, IVec2, BVec2)):
            return np.array([*args[0], bool(args[1]), bool(args[2])], dtype=bool)
        raise TypeError("First argument must be vec2 when using 3 arguments")

    return np.array(
        [bool(args[0]), bool(args[1]), bool(args[2]), bool(args[3])], dtype=bool
    )
