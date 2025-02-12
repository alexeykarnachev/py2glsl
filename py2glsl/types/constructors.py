"""GLSL type constructors using NumPy."""

from collections.abc import Callable
from enum import Enum, auto
from typing import Any

import numpy as np
from loguru import logger

from .errors import GLSLTypeError


class TypeKind(Enum):
    """Vector and matrix type kinds."""

    FLOAT = auto()
    INT = auto()
    BOOL = auto()


class VectorConfig:
    """Configuration for vector construction."""

    def __init__(
        self,
        kind: TypeKind,
        size: int,
        dtype: np.dtype,
        converter: Callable[[Any], float | int | bool],
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
        self.name = f"{'vec' if kind == TypeKind.FLOAT else 'ivec' if kind == TypeKind.INT else 'bvec'}{size}"

    def __str__(self) -> str:
        """Get string representation."""
        return self.name


class MatrixConfig:
    """Configuration for matrix construction."""

    def __init__(self, size: int):
        """Initialize matrix configuration.

        Args:
            size: Matrix size (2, 3, or 4)
        """
        self.size = size
        self.name = f"mat{size}"
        self.components = size * size
        self.dtype = np.float32

    def __str__(self) -> str:
        """Get string representation."""
        return self.name


def _convert_to_float(value: Any) -> float:
    """Convert value to float with validation."""
    try:
        if isinstance(value, str):
            raise TypeError("Cannot convert <class 'str'> to float")
        return float(value)
    except (TypeError, ValueError) as e:
        logger.error(f"Failed to convert {value} to float: {e}")
        if isinstance(value, str):
            raise TypeError("Cannot convert <class 'str'> to float")
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


def create_matrix(config: MatrixConfig, *args) -> np.ndarray:
    """Generic matrix constructor.

    Supports:
    - No args: identity matrix
    - Single scalar: scaled identity matrix
    - Single array/matrix: converted to matrix
    - N*N components: matrix from components
    """
    logger.debug(f"Creating {config} from args: {args}")

    try:
        if len(args) == 0:
            # Identity matrix
            return np.eye(config.size, dtype=config.dtype)

        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, (int, float)):
                # Scaled identity matrix
                return np.eye(config.size, dtype=config.dtype) * float(arg)
            if isinstance(arg, np.ndarray):
                # Convert existing array
                result = np.array(arg, dtype=config.dtype)
                if result.shape != (config.size, config.size):
                    if result.size == config.components:
                        # Reshape flat array to matrix
                        result = result.reshape(config.size, config.size)
                    else:
                        raise GLSLTypeError(
                            f"Cannot convert array of shape {result.shape} to {config}"
                        )
                return result
            if isinstance(arg, (list, tuple)):
                # Convert sequence
                result = np.array(arg, dtype=config.dtype)
                if result.size == config.components:
                    return result.reshape(config.size, config.size)
                raise GLSLTypeError(
                    f"Expected {config.components} components, got {result.size}"
                )
            raise GLSLTypeError(f"Cannot convert {type(arg)} to {config}")

        if len(args) == config.components:
            # Individual components
            return np.array(args, dtype=config.dtype).reshape(config.size, config.size)

        raise GLSLTypeError(
            f"{config} requires 0, 1 or {config.components} arguments, got {len(args)}"
        )

    except (ValueError, TypeError) as e:
        logger.error(f"Failed to create {config}: {e}")
        raise GLSLTypeError(f"Cannot create {config} from arguments: {args}") from e


# Vector configurations
_VEC2_CONFIG = VectorConfig(TypeKind.FLOAT, 2, np.float32, _convert_to_float)
_VEC3_CONFIG = VectorConfig(TypeKind.FLOAT, 3, np.float32, _convert_to_float)
_VEC4_CONFIG = VectorConfig(TypeKind.FLOAT, 4, np.float32, _convert_to_float)
_IVEC2_CONFIG = VectorConfig(TypeKind.INT, 2, np.int32, _convert_to_int)
_IVEC3_CONFIG = VectorConfig(TypeKind.INT, 3, np.int32, _convert_to_int)
_IVEC4_CONFIG = VectorConfig(TypeKind.INT, 4, np.int32, _convert_to_int)
_BVEC2_CONFIG = VectorConfig(TypeKind.BOOL, 2, bool, _convert_to_bool)
_BVEC3_CONFIG = VectorConfig(TypeKind.BOOL, 3, bool, _convert_to_bool)
_BVEC4_CONFIG = VectorConfig(TypeKind.BOOL, 4, bool, _convert_to_bool)

# Matrix configurations
_MAT2_CONFIG = MatrixConfig(2)
_MAT3_CONFIG = MatrixConfig(3)
_MAT4_CONFIG = MatrixConfig(4)


def vec2(*args) -> np.ndarray:
    """Create a 2D float vector."""
    return create_vector(_VEC2_CONFIG, *args)


def vec3(*args) -> np.ndarray:
    """Create a 3D float vector."""
    return create_vector(_VEC3_CONFIG, *args)


def vec4(*args) -> np.ndarray:
    """Create a 4D float vector."""
    return create_vector(_VEC4_CONFIG, *args)


def ivec2(*args) -> np.ndarray:
    """Create a 2D integer vector."""
    return create_vector(_IVEC2_CONFIG, *args)


def ivec3(*args) -> np.ndarray:
    """Create a 3D integer vector."""
    return create_vector(_IVEC3_CONFIG, *args)


def ivec4(*args) -> np.ndarray:
    """Create a 4D integer vector."""
    return create_vector(_IVEC4_CONFIG, *args)


def bvec2(*args) -> np.ndarray:
    """Create a 2D boolean vector."""
    return create_vector(_BVEC2_CONFIG, *args)


def bvec3(*args) -> np.ndarray:
    """Create a 3D boolean vector."""
    return create_vector(_BVEC3_CONFIG, *args)


def bvec4(*args) -> np.ndarray:
    """Create a 4D boolean vector."""
    return create_vector(_BVEC4_CONFIG, *args)


def mat2(*args) -> np.ndarray:
    """Create a 2x2 float matrix."""
    return create_matrix(_MAT2_CONFIG, *args)


def mat3(*args) -> np.ndarray:
    """Create a 3x3 float matrix."""
    return create_matrix(_MAT3_CONFIG, *args)


def mat4(*args) -> np.ndarray:
    """Create a 4x4 float matrix."""
    return create_matrix(_MAT4_CONFIG, *args)
