from typing import TypeAlias

import numpy as np

Vec2: TypeAlias = np.ndarray  # shape (2,)
Vec3: TypeAlias = np.ndarray  # shape (3,)
Vec4: TypeAlias = np.ndarray  # shape (4,)


def vec2(x: float, y: float) -> Vec2:
    return np.array([x, y], dtype=np.float32)


def vec3(x: float, y: float, z: float) -> Vec3:
    return np.array([x, y, z], dtype=np.float32)


def vec4(x: float, y: float, z: float, w: float) -> Vec4:
    return np.array([x, y, z, w], dtype=np.float32)
