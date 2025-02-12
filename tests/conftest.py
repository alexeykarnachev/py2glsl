from typing import Dict

import numpy as np
import pytest

from py2glsl.types.type_system import GLSLType, TypeKind


@pytest.fixture
def temp_dir(tmp_path):
    """Provide temporary directory for test files."""
    return tmp_path


@pytest.fixture
def scalar_types() -> Dict[str, GLSLType]:
    """Provide scalar GLSL types."""
    return {
        "void": GLSLType(TypeKind.VOID),
        "bool": GLSLType(TypeKind.BOOL),
        "int": GLSLType(TypeKind.INT),
        "float": GLSLType(TypeKind.FLOAT),
    }


@pytest.fixture
def vector_types() -> Dict[str, GLSLType]:
    """Provide vector GLSL types."""
    return {
        "vec2": GLSLType(TypeKind.VEC2),
        "vec3": GLSLType(TypeKind.VEC3),
        "vec4": GLSLType(TypeKind.VEC4),
        "ivec2": GLSLType(TypeKind.IVEC2),
        "ivec3": GLSLType(TypeKind.IVEC3),
        "ivec4": GLSLType(TypeKind.IVEC4),
        "bvec2": GLSLType(TypeKind.BVEC2),
        "bvec3": GLSLType(TypeKind.BVEC3),
        "bvec4": GLSLType(TypeKind.BVEC4),
    }


@pytest.fixture
def matrix_types() -> Dict[str, GLSLType]:
    """Provide matrix GLSL types."""
    return {
        "mat2": GLSLType(TypeKind.MAT2),
        "mat3": GLSLType(TypeKind.MAT3),
        "mat4": GLSLType(TypeKind.MAT4),
    }


@pytest.fixture
def qualified_types() -> Dict[str, GLSLType]:
    """Provide types with qualifiers."""
    return {
        "uniform_float": GLSLType(TypeKind.FLOAT, is_uniform=True),
        "const_vec3": GLSLType(TypeKind.VEC3, is_const=True),
        "attribute_vec4": GLSLType(TypeKind.VEC4, is_attribute=True),
        "uniform_const_vec2": GLSLType(TypeKind.VEC2, is_uniform=True, is_const=True),
    }


@pytest.fixture
def array_types() -> Dict[str, GLSLType]:
    """Provide array types."""
    return {
        "float_array2": GLSLType(TypeKind.FLOAT, array_size=2),
        "float_array4": GLSLType(TypeKind.FLOAT, array_size=4),
        "vec2_array3": GLSLType(TypeKind.VEC2, array_size=3),
        "vec3_array2": GLSLType(TypeKind.VEC3, array_size=2),
    }


@pytest.fixture
def vector_values() -> Dict[str, np.ndarray]:
    """Provide sample vector values."""
    return {
        "vec2_zero": np.zeros(2, dtype=np.float32),
        "vec2_one": np.ones(2, dtype=np.float32),
        "vec3_123": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "vec4_1234": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "ivec2_12": np.array([1, 2], dtype=np.int32),
        "ivec3_123": np.array([1, 2, 3], dtype=np.int32),
        "ivec4_1234": np.array([1, 2, 3, 4], dtype=np.int32),
        "bvec2_tf": np.array([True, False], dtype=bool),
        "bvec3_tft": np.array([True, False, True], dtype=bool),
        "bvec4_tfft": np.array([True, False, False, True], dtype=bool),
    }


@pytest.fixture
def matrix_values() -> Dict[str, np.ndarray]:
    """Provide sample matrix values."""
    return {
        "mat2_identity": np.eye(2, dtype=np.float32),
        "mat3_identity": np.eye(3, dtype=np.float32),
        "mat4_identity": np.eye(4, dtype=np.float32),
        "mat2_ones": np.ones((2, 2), dtype=np.float32),
        "mat3_ones": np.ones((3, 3), dtype=np.float32),
        "mat4_ones": np.ones((4, 4), dtype=np.float32),
    }


@pytest.fixture
def operators() -> Dict[str, list[str]]:
    """Provide operator groups."""
    return {
        "arithmetic": ["+", "-", "*", "/", "%"],
        "logical": ["&&", "||"],
        "comparison": ["==", "!=", "<", ">", "<=", ">="],
    }


@pytest.fixture
def swizzle_sets() -> Dict[str, set[str]]:
    """Provide swizzle component sets."""
    return {
        "position": {"x", "y", "z", "w"},
        "color": {"r", "g", "b", "a"},
        "texture": {"s", "t", "p", "q"},
    }


@pytest.fixture
def all_types(
    scalar_types: Dict[str, GLSLType],
    vector_types: Dict[str, GLSLType],
    matrix_types: Dict[str, GLSLType],
) -> Dict[str, GLSLType]:
    """Provide all basic GLSL types."""
    return {
        **scalar_types,
        **vector_types,
        **matrix_types,
    }


@pytest.fixture
def numeric_types(all_types: Dict[str, GLSLType]) -> Dict[str, GLSLType]:
    """Provide all numeric GLSL types."""
    return {name: type_ for name, type_ in all_types.items() if type_.is_numeric}


@pytest.fixture
def vector_size_map() -> Dict[str, int]:
    """Provide mapping of vector types to their sizes."""
    return {
        "vec2": 2,
        "vec3": 3,
        "vec4": 4,
        "ivec2": 2,
        "ivec3": 3,
        "ivec4": 4,
        "bvec2": 2,
        "bvec3": 3,
        "bvec4": 4,
    }


@pytest.fixture
def matrix_size_map() -> Dict[str, int]:
    """Provide mapping of matrix types to their sizes."""
    return {
        "mat2": 2,
        "mat3": 3,
        "mat4": 4,
    }
