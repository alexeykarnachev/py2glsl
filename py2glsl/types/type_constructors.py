"""GLSL type constructors and runtime type implementations."""

from enum import Enum, auto
from typing import Any, Callable, TypeAlias, Union

import numpy as np
from loguru import logger

# Runtime type aliases
Vec2: TypeAlias = np.ndarray  # shape (2,)
Vec3: TypeAlias = np.ndarray  # shape (3,)
Vec4: TypeAlias = np.ndarray  # shape (4,)
IVec2: TypeAlias = np.ndarray  # shape (2,) dtype=int32
IVec3: TypeAlias = np.ndarray  # shape (3,) dtype=int32
IVec4: TypeAlias = np.ndarray  # shape (4,) dtype=int32
BVec2: TypeAlias = np.ndarray  # shape (2,) dtype=bool
BVec3: TypeAlias = np.ndarray  # shape (3,) dtype=bool
BVec4: TypeAlias = np.ndarray  # shape (4,) dtype=bool

VectorType = Union[Vec2, Vec3, Vec4, IVec2, IVec3, IVec4, BVec2, BVec3, BVec4]


class VectorKind(Enum):
    """Vector type kinds."""

    FLOAT = auto()
    INT = auto()
    BOOL = auto()


class VectorConfig:
    """Configuration for vector construction."""

    def __init__(
        self,
        kind: VectorKind,
        size: int,
        dtype: np.dtype,
        converter: Callable[[Any], Union[float, int, bool]],
    ):
        """Initialize vector configuration.

        Args:
            kind: Vector type kind (float, int, bool)
            size: Vector size (2, 3, or 4)
            dtype: NumPy dtype for the vector
            converter: Function to convert values to target type
        """
        self.kind = kind
        self.size = size
        self.dtype = dtype
        self.converter = converter
        self.name = f"{'vec' if kind == VectorKind.FLOAT else 'ivec' if kind == VectorKind.INT else 'bvec'}{size}"

    def __str__(self) -> str:
        """Get string representation."""
        return self.name


def _convert_to_float(value: Any) -> float:
    """Convert value to float with validation."""
    try:
        if isinstance(value, str):
            raise TypeError(f"Cannot convert <class 'str'> to float")
        return float(value)
    except (TypeError, ValueError) as e:
        logger.error(f"Failed to convert {value} to float: {e}")
        if isinstance(value, str):
            raise TypeError(f"Cannot convert <class 'str'> to float")
        raise TypeError(f"Cannot convert {type(value)} to float") from e


def _convert_to_int(value: Any) -> int:
    """Convert value to int with validation."""
    try:
        return int(value)
    except (TypeError, ValueError) as e:
        logger.error(f"Failed to convert {value} to int: {e}")
        raise TypeError(f"Cannot convert {type(value)} to int") from e


def _convert_to_bool(value: Any) -> bool:
    """Convert value to bool with validation."""
    try:
        return bool(value)
    except (TypeError, ValueError) as e:
        logger.error(f"Failed to convert {value} to bool: {e}")
        raise TypeError(f"Cannot convert {type(value)} to bool") from e


def create_vector(config: VectorConfig, *args) -> np.ndarray:
    """Generic vector constructor."""
    logger.debug(f"Creating {config} from args: {args}")

    if not args:
        msg = f"{config} requires at least 1 argument"
        logger.error(msg)
        raise TypeError(msg)

    if len(args) > config.size:
        msg = f"{config} requires 1 to {config.size} arguments"
        logger.error(msg)
        raise TypeError(msg)

    try:
        # Single argument case
        if len(args) == 1:
            if isinstance(args[0], np.ndarray):
                return np.array(args[0], dtype=config.dtype)
            val = config.converter(args[0])
            return np.full(config.size, val, dtype=config.dtype)

        # Handle vec2/vec3 + scalar cases for vec3/vec4
        if len(args) == 2:
            if config.size > 2:
                if isinstance(args[0], np.ndarray):
                    if args[0].shape[0] == config.size - 1:
                        return np.array(
                            [*args[0], config.converter(args[1])], dtype=config.dtype
                        )
                    if args[0].shape[0] == 2 and config.size == 4:
                        val = config.converter(args[1])
                        return np.array([*args[0], val, val], dtype=config.dtype)
                if config.size == 3:
                    msg = "First argument must be vec2 when using 2 arguments"
                    logger.error(msg)
                    raise TypeError(msg)
                msg = "First argument must be vec2 or vec3 when using 2 arguments"
                logger.error(msg)
                raise TypeError(msg)

        # Handle vec2 + scalar + scalar case for vec4
        if len(args) == 3 and config.size == 4:
            if not isinstance(args[0], np.ndarray) or args[0].shape[0] != 2:
                msg = "First argument must be vec2 when using 3 arguments"
                logger.error(msg)
                raise TypeError(msg)
            return np.array(
                [*args[0], config.converter(args[1]), config.converter(args[2])],
                dtype=config.dtype,
            )

        # Direct component initialization
        return np.array([config.converter(arg) for arg in args], dtype=config.dtype)

    except TypeError as e:
        # Re-raise conversion errors with original message
        raise TypeError(str(e)) from e


# Vector configurations
_VEC2_CONFIG = VectorConfig(VectorKind.FLOAT, 2, np.float32, _convert_to_float)
_VEC3_CONFIG = VectorConfig(VectorKind.FLOAT, 3, np.float32, _convert_to_float)
_VEC4_CONFIG = VectorConfig(VectorKind.FLOAT, 4, np.float32, _convert_to_float)
_IVEC2_CONFIG = VectorConfig(VectorKind.INT, 2, np.int32, _convert_to_int)
_IVEC3_CONFIG = VectorConfig(VectorKind.INT, 3, np.int32, _convert_to_int)
_IVEC4_CONFIG = VectorConfig(VectorKind.INT, 4, np.int32, _convert_to_int)
_BVEC2_CONFIG = VectorConfig(VectorKind.BOOL, 2, bool, _convert_to_bool)
_BVEC3_CONFIG = VectorConfig(VectorKind.BOOL, 3, bool, _convert_to_bool)
_BVEC4_CONFIG = VectorConfig(VectorKind.BOOL, 4, bool, _convert_to_bool)


def vec2(*args) -> Vec2:
    """Create a 2D float vector."""
    return create_vector(_VEC2_CONFIG, *args)


def vec3(*args) -> Vec3:
    """Create a 3D float vector."""
    return create_vector(_VEC3_CONFIG, *args)


def vec4(*args) -> Vec4:
    """Create a 4D float vector."""
    return create_vector(_VEC4_CONFIG, *args)


def ivec2(*args) -> IVec2:
    """Create a 2D integer vector."""
    return create_vector(_IVEC2_CONFIG, *args)


def ivec3(*args) -> IVec3:
    """Create a 3D integer vector."""
    return create_vector(_IVEC3_CONFIG, *args)


def ivec4(*args) -> IVec4:
    """Create a 4D integer vector."""
    return create_vector(_IVEC4_CONFIG, *args)


def bvec2(*args) -> BVec2:
    """Create a 2D boolean vector."""
    return create_vector(_BVEC2_CONFIG, *args)


def bvec3(*args) -> BVec3:
    """Create a 3D boolean vector."""
    return create_vector(_BVEC3_CONFIG, *args)


def bvec4(*args) -> BVec4:
    """Create a 4D boolean vector."""
    return create_vector(_BVEC4_CONFIG, *args)
