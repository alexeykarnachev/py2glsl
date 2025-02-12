"""Tests for GLSL type system."""

import numpy as np
import pytest

from py2glsl.types import (
    BOOL,
    BVEC2,
    BVEC3,
    BVEC4,
    FLOAT,
    INT,
    IVEC2,
    IVEC3,
    IVEC4,
    MAT2,
    MAT3,
    MAT4,
    VEC2,
    VEC3,
    VEC4,
    VOID,
    GLSLSwizzleError,
    GLSLType,
    GLSLTypeError,
    TypeKind,
    ivec2,
    ivec3,
    ivec4,
    vec2,
    vec3,
    vec4,
)


def test_basic_types():
    """Test basic type properties."""
    assert str(VOID) == "void"
    assert str(BOOL) == "bool"
    assert str(INT) == "int"
    assert str(FLOAT) == "float"
    assert str(VEC2) == "vec2"
    assert str(VEC3) == "vec3"
    assert str(VEC4) == "vec4"
    assert str(MAT2) == "mat2"
    assert str(MAT3) == "mat3"
    assert str(MAT4) == "mat4"


def test_type_qualifiers():
    """Test type qualifiers."""
    uniform_float = GLSLType(TypeKind.FLOAT, is_uniform=True)
    assert str(uniform_float) == "uniform float"

    const_vec3 = GLSLType(TypeKind.VEC3, is_const=True)
    assert str(const_vec3) == "const vec3"

    attribute_vec4 = GLSLType(TypeKind.VEC4, is_attribute=True)
    assert str(attribute_vec4) == "attribute vec4"


def test_array_types():
    """Test array type declarations."""
    float_array = GLSLType(TypeKind.FLOAT, array_size=4)
    assert str(float_array) == "float[4]"

    with pytest.raises(GLSLTypeError):
        GLSLType(TypeKind.FLOAT, array_size=0)

    with pytest.raises(GLSLTypeError):
        GLSLType(TypeKind.FLOAT, array_size=-1)


def test_type_properties():
    """Test type property checks."""
    assert FLOAT.is_numeric
    assert VEC3.is_numeric
    assert MAT4.is_numeric
    assert not BOOL.is_numeric
    assert not VOID.is_numeric

    assert VEC2.is_vector
    assert VEC3.is_vector
    assert VEC4.is_vector
    assert not FLOAT.is_vector
    assert not MAT3.is_vector

    assert MAT2.is_matrix
    assert MAT3.is_matrix
    assert MAT4.is_matrix
    assert not VEC3.is_matrix
    assert not FLOAT.is_matrix


def test_vector_constructors():
    """Test vector constructor functions."""
    # vec2
    v2a = vec2(1.0, 2.0)
    assert isinstance(v2a, np.ndarray)
    assert v2a.shape == (2,)
    np.testing.assert_array_equal(v2a, [1.0, 2.0])

    v2b = vec2(3.0)
    np.testing.assert_array_equal(v2b, [3.0, 3.0])

    # vec3
    v3a = vec3(1.0, 2.0, 3.0)
    assert v3a.shape == (3,)
    np.testing.assert_array_equal(v3a, [1.0, 2.0, 3.0])

    v3b = vec3(2.0)
    np.testing.assert_array_equal(v3b, [2.0, 2.0, 2.0])

    v3c = vec3(vec2(1.0, 2.0), 3.0)
    np.testing.assert_array_equal(v3c, [1.0, 2.0, 3.0])

    # vec4
    v4a = vec4(1.0, 2.0, 3.0, 4.0)
    assert v4a.shape == (4,)
    np.testing.assert_array_equal(v4a, [1.0, 2.0, 3.0, 4.0])

    v4b = vec4(2.0)
    np.testing.assert_array_equal(v4b, [2.0, 2.0, 2.0, 2.0])

    v4c = vec4(vec2(1.0, 2.0), 3.0, 4.0)
    np.testing.assert_array_equal(v4c, [1.0, 2.0, 3.0, 4.0])

    v4d = vec4(vec3(1.0, 2.0, 3.0), 4.0)
    np.testing.assert_array_equal(v4d, [1.0, 2.0, 3.0, 4.0])


def test_swizzle_operations():
    """Test vector swizzle operations."""
    # Basic swizzles
    assert VEC4.validate_swizzle("x") == FLOAT
    assert VEC4.validate_swizzle("xy") == VEC2
    assert VEC4.validate_swizzle("xyz") == VEC3
    assert VEC4.validate_swizzle("xyzw") == VEC4

    # Color components
    assert VEC4.validate_swizzle("r") == FLOAT
    assert VEC4.validate_swizzle("rg") == VEC2
    assert VEC4.validate_swizzle("rgb") == VEC3
    assert VEC4.validate_swizzle("rgba") == VEC4

    # Texture coordinates
    assert VEC4.validate_swizzle("s") == FLOAT
    assert VEC4.validate_swizzle("st") == VEC2
    assert VEC4.validate_swizzle("stp") == VEC3
    assert VEC4.validate_swizzle("stpq") == VEC4

    # Invalid swizzles
    with pytest.raises(GLSLSwizzleError):
        VEC3.validate_swizzle("xyzw")  # w not available in vec3

    with pytest.raises(GLSLSwizzleError):
        VEC4.validate_swizzle("xrs")  # mixed component sets

    with pytest.raises(GLSLSwizzleError):
        VEC2.validate_swizzle("xyz")  # too many components

    with pytest.raises(GLSLSwizzleError):
        FLOAT.validate_swizzle("x")  # non-vector type


def test_type_operations():
    """Test type operation validation."""
    # Arithmetic operations
    assert FLOAT.validate_operation("+", FLOAT) == FLOAT
    assert INT.validate_operation("*", FLOAT) == FLOAT
    assert VEC3.validate_operation("/", FLOAT) == VEC3
    assert MAT4.validate_operation("*", VEC4) == VEC4

    # Boolean operations
    assert BOOL.validate_operation("&&", BOOL) == BOOL
    assert BOOL.validate_operation("||", BOOL) == BOOL

    # Comparison operations
    assert FLOAT.validate_operation("<", FLOAT) == BOOL
    assert INT.validate_operation(">=", INT) == BOOL
    assert VEC3.validate_operation("==", VEC3) == BOOL

    # Invalid operations
    assert VEC2.validate_operation("+", VEC3) is None
    assert MAT3.validate_operation("*", MAT4) is None
    assert BOOL.validate_operation("+", INT) is None


def test_type_compatibility():
    """Test type compatibility checks."""
    assert FLOAT.is_compatible_with(FLOAT)
    assert INT.is_compatible_with(FLOAT)
    assert VEC3.is_compatible_with(VEC3)
    assert VEC3.is_compatible_with(FLOAT)
    assert MAT4.is_compatible_with(MAT4)

    assert not VEC2.is_compatible_with(VEC3)
    assert not MAT3.is_compatible_with(MAT4)
    assert not BOOL.is_compatible_with(INT)


def test_type_conversion():
    """Test type conversion rules."""
    assert INT.can_convert_to(FLOAT)
    assert INT.can_convert_to(INT)
    assert FLOAT.can_convert_to(FLOAT)

    assert not FLOAT.can_convert_to(INT)
    assert not BOOL.can_convert_to(INT)
    assert not VEC2.can_convert_to(VEC3)


def test_common_type():
    """Test common type resolution."""
    assert INT.common_type(INT) == INT
    assert INT.common_type(FLOAT) == FLOAT
    assert FLOAT.common_type(INT) == FLOAT
    assert VEC3.common_type(FLOAT) == VEC3
    assert VEC2.common_type(VEC2) == VEC2

    assert VEC2.common_type(VEC3) is None
    assert BOOL.common_type(INT) is None
    assert MAT3.common_type(MAT4) is None


def test_advanced_type_combinations():
    """Test complex type combinations and edge cases."""
    # Array types
    array_float = GLSLType(TypeKind.FLOAT, array_size=3)
    array_vec3 = GLSLType(TypeKind.VEC3, array_size=4)

    assert str(array_float) == "float[3]"
    assert str(array_vec3) == "vec3[4]"

    # Multiple qualifiers
    complex_type = GLSLType(TypeKind.VEC4, is_uniform=True, is_const=True)
    assert str(complex_type) == "uniform const vec4"

    # Invalid qualifier combinations
    with pytest.raises(GLSLTypeError):
        GLSLType(TypeKind.VEC3, is_uniform=True, is_attribute=True)


def test_advanced_swizzle_operations():
    """Test complex swizzle operations and combinations."""
    # Multiple swizzle operations chain
    vec4_xy = VEC4.validate_swizzle("xy")  # vec4 -> vec2
    assert vec4_xy == VEC2
    vec2_yx = vec4_xy.validate_swizzle("yx")  # vec2 -> vec2 (reversed)
    assert vec2_yx == VEC2

    # Repeated components (valid in GLSL)
    assert VEC4.validate_swizzle("xxx") == VEC3
    assert VEC3.validate_swizzle("xxy") == VEC3
    assert VEC2.validate_swizzle("xx") == VEC2

    # All possible component sets
    assert VEC4.validate_swizzle("wzyx") == VEC4  # Position
    assert VEC4.validate_swizzle("abgr") == VEC4  # Color
    assert VEC4.validate_swizzle("qpts") == VEC4  # Texture

    # Single component access
    assert VEC4.validate_swizzle("x") == FLOAT
    assert VEC3.validate_swizzle("r") == FLOAT
    assert VEC2.validate_swizzle("s") == FLOAT

    # Edge cases and errors
    with pytest.raises(GLSLSwizzleError):
        VEC4.validate_swizzle("")  # Empty swizzle

    with pytest.raises(GLSLSwizzleError):
        VEC4.validate_swizzle("xyzwx")  # Too many components

    with pytest.raises(GLSLSwizzleError):
        VEC4.validate_swizzle("xrgb")  # Mixed sets

    with pytest.raises(GLSLSwizzleError):
        VEC4.validate_swizzle("X")  # Uppercase

    with pytest.raises(GLSLSwizzleError):
        VEC4.validate_swizzle("x y")  # Spaces

    with pytest.raises(GLSLSwizzleError):
        VEC4.validate_swizzle("x,y")  # Invalid characters

    with pytest.raises(GLSLSwizzleError):
        VEC3.validate_swizzle("w")  # Component not in vector

    with pytest.raises(GLSLSwizzleError):
        FLOAT.validate_swizzle("x")  # Non-vector type


def test_advanced_operations():
    """Test complex operation combinations."""
    # Vector-scalar operations
    assert VEC3.validate_operation("*", FLOAT) == VEC3
    assert VEC3.validate_operation("/", INT) == VEC3
    assert FLOAT.validate_operation("*", VEC3) == VEC3

    # Matrix operations
    assert MAT4.validate_operation("*", VEC4) == VEC4  # Matrix-vector mult
    assert MAT3.validate_operation("*", FLOAT) == MAT3  # Matrix-scalar mult
    assert MAT2.validate_operation("*", MAT2) == MAT2  # Matrix-matrix mult

    # Component-wise operations
    assert VEC4.validate_operation("+", VEC4) == VEC4
    assert VEC3.validate_operation("-", VEC3) == VEC3
    assert VEC2.validate_operation("*", VEC2) == VEC2

    # Boolean operations
    assert BOOL.validate_operation("&&", BOOL) == BOOL
    assert BOOL.validate_operation("||", BOOL) == BOOL

    # Comparison operations
    assert INT.validate_operation("<", INT) == BOOL
    assert FLOAT.validate_operation(">=", FLOAT) == BOOL
    assert VEC2.validate_operation("==", VEC2) == BOOL
    assert VEC3.validate_operation("!=", VEC3) == BOOL

    # Invalid operations
    assert VEC2.validate_operation("&&", VEC2) is None  # Non-bool logical op
    assert FLOAT.validate_operation("||", INT) is None  # Non-bool logical op
    assert MAT3.validate_operation("+", VEC3) is None  # Invalid matrix op
    assert VEC4.validate_operation("*", MAT3) is None  # Invalid vector-matrix op
    assert VOID.validate_operation("+", INT) is None  # Void operations


def test_advanced_type_compatibility():
    """Test complex type compatibility scenarios."""
    # Scalar compatibility
    assert INT.is_compatible_with(FLOAT)
    assert not FLOAT.is_compatible_with(BOOL)
    assert not INT.is_compatible_with(VOID)

    # Vector compatibility
    assert VEC3.is_compatible_with(VEC3)
    assert not VEC2.is_compatible_with(VEC3)
    assert VEC4.is_compatible_with(FLOAT)  # Vector-scalar
    assert not VEC4.is_compatible_with(BOOL)  # Vector-bool

    # Matrix compatibility
    assert MAT4.is_compatible_with(MAT4)
    assert not MAT3.is_compatible_with(MAT4)
    assert MAT2.is_compatible_with(FLOAT)  # Matrix-scalar
    assert not MAT3.is_compatible_with(VEC2)  # Invalid matrix-vector

    # Array compatibility
    array_float = GLSLType(TypeKind.FLOAT, array_size=3)
    array_float2 = GLSLType(TypeKind.FLOAT, array_size=3)
    array_float_diff = GLSLType(TypeKind.FLOAT, array_size=4)

    assert array_float.is_compatible_with(array_float2)  # Same size
    assert not array_float.is_compatible_with(array_float_diff)  # Different size


def test_vector_constructor_edge_cases():
    """Test vector constructors with edge cases."""
    # Zero values
    np.testing.assert_array_equal(vec2(0.0), [0.0, 0.0])
    np.testing.assert_array_equal(vec3(0.0), [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(vec4(0.0), [0.0, 0.0, 0.0, 0.0])

    # Integer to float conversion
    np.testing.assert_array_equal(vec2(1), [1.0, 1.0])
    np.testing.assert_array_equal(vec3(1), [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(vec4(1), [1.0, 1.0, 1.0, 1.0])

    # Negative values
    np.testing.assert_array_equal(vec2(-1.0, -2.0), [-1.0, -2.0])
    np.testing.assert_array_equal(vec3(-1.0), [-1.0, -1.0, -1.0])

    # Mixed integer and float
    np.testing.assert_array_equal(vec3(1, 2.0, 3), [1.0, 2.0, 3.0])

    # Vector composition
    v2 = vec2(1.0, 2.0)
    v3 = vec3(v2, 3.0)
    np.testing.assert_array_equal(v3, [1.0, 2.0, 3.0])

    v4 = vec4(v2, 3.0, 4.0)
    np.testing.assert_array_equal(v4, [1.0, 2.0, 3.0, 4.0])

    # Invalid constructions
    with pytest.raises(TypeError):
        vec2(1.0, 2.0, 3.0)  # Too many components

    with pytest.raises(ValueError):
        vec4(vec3(1.0, 2.0, 3.0), 4.0, 5.0)  # Too many components

    with pytest.raises(ValueError):
        vec4(1.0, 2.0)  # Not enough components

    with pytest.raises(ValueError):
        vec4(1.0, 2.0, 3.0)  # Not enough components

    with pytest.raises(ValueError):
        vec4([1.0, 2.0, 3.0, 4.0])  # Invalid input type (raw list)

    with pytest.raises(ValueError):
        vec4(v3, 4.0, 5.0)  # Too many components with vec3


def test_type_conversion_edge_cases():
    """Test type conversion edge cases."""
    # Basic conversions
    assert INT.can_convert_to(FLOAT)
    assert INT.can_convert_to(INT)
    assert FLOAT.can_convert_to(FLOAT)

    # Invalid conversions
    assert not FLOAT.can_convert_to(INT)  # No implicit float to int
    assert not BOOL.can_convert_to(INT)  # No implicit bool to int
    assert not INT.can_convert_to(BOOL)  # No implicit int to bool

    # Vector conversions
    assert not VEC2.can_convert_to(VEC3)  # No implicit vector conversion
    assert not VEC3.can_convert_to(FLOAT)  # No vector to scalar
    assert not FLOAT.can_convert_to(VEC2)  # No scalar to vector

    # Matrix conversions
    assert not MAT3.can_convert_to(MAT4)  # No implicit matrix conversion
    assert not MAT4.can_convert_to(VEC4)  # No matrix to vector
    assert not VEC4.can_convert_to(MAT2)  # No vector to matrix


def test_common_type_edge_cases():
    """Test common type resolution edge cases."""
    # Basic types
    assert INT.common_type(INT) == INT
    assert INT.common_type(FLOAT) == FLOAT
    assert FLOAT.common_type(INT) == FLOAT

    # Vectors with scalars
    assert VEC2.common_type(FLOAT) == VEC2
    assert FLOAT.common_type(VEC3) == VEC3
    assert INT.common_type(VEC4) == VEC4

    # Same-size vectors
    assert VEC2.common_type(VEC2) == VEC2
    assert VEC3.common_type(VEC3) == VEC3
    assert VEC4.common_type(VEC4) == VEC4

    # Different-size vectors
    assert VEC2.common_type(VEC3) is None
    assert VEC3.common_type(VEC4) is None

    # Matrices
    assert MAT2.common_type(MAT2) == MAT2
    assert MAT3.common_type(FLOAT) == MAT3
    assert MAT4.common_type(MAT4) == MAT4

    # Invalid combinations
    assert BOOL.common_type(INT) is None
    assert VOID.common_type(FLOAT) is None
    assert MAT2.common_type(VEC2) is None

    # Arrays
    array_float = GLSLType(TypeKind.FLOAT, array_size=3)
    array_float2 = GLSLType(TypeKind.FLOAT, array_size=3)
    array_float_diff = GLSLType(TypeKind.FLOAT, array_size=4)

    assert array_float.common_type(array_float2) == array_float
    assert array_float.common_type(array_float_diff) is None


def test_type_qualifiers_combinations():
    """Test various combinations of type qualifiers."""
    # Single qualifiers
    uniform_vec3 = GLSLType(TypeKind.VEC3, is_uniform=True)
    assert str(uniform_vec3) == "uniform vec3"

    const_float = GLSLType(TypeKind.FLOAT, is_const=True)
    assert str(const_float) == "const float"

    attribute_vec4 = GLSLType(TypeKind.VEC4, is_attribute=True)
    assert str(attribute_vec4) == "attribute vec4"

    # Combined qualifiers
    uniform_const = GLSLType(TypeKind.VEC2, is_uniform=True, is_const=True)
    assert str(uniform_const) == "uniform const vec2"

    # Array with qualifiers
    uniform_array = GLSLType(TypeKind.FLOAT, is_uniform=True, array_size=4)
    assert str(uniform_array) == "uniform float[4]"

    # Invalid combinations
    with pytest.raises(GLSLTypeError):
        GLSLType(TypeKind.VEC3, is_uniform=True, is_attribute=True)

    with pytest.raises(GLSLTypeError):
        GLSLType(TypeKind.FLOAT, array_size=-1)


def test_integer_vector_types():
    """Test integer vector type properties."""
    assert str(IVEC2) == "ivec2"
    assert str(IVEC3) == "ivec3"
    assert str(IVEC4) == "ivec4"

    assert IVEC2.is_numeric
    assert IVEC3.is_numeric
    assert IVEC4.is_numeric

    assert IVEC2.is_vector
    assert IVEC3.is_vector
    assert IVEC4.is_vector

    assert IVEC2.is_int_vector
    assert IVEC3.is_int_vector
    assert IVEC4.is_int_vector


def test_boolean_vector_types():
    """Test boolean vector type properties."""
    assert str(BVEC2) == "bvec2"
    assert str(BVEC3) == "bvec3"
    assert str(BVEC4) == "bvec4"

    assert not BVEC2.is_numeric
    assert not BVEC3.is_numeric
    assert not BVEC4.is_numeric

    assert BVEC2.is_vector
    assert BVEC3.is_vector
    assert BVEC4.is_vector

    assert BVEC2.is_bool_vector
    assert BVEC3.is_bool_vector
    assert BVEC4.is_bool_vector


def test_integer_vector_constructors():
    """Test integer vector constructor functions."""
    # ivec2
    v2a = ivec2(1, 2)
    assert isinstance(v2a, np.ndarray)
    assert v2a.shape == (2,)
    assert v2a.dtype == np.int32
    np.testing.assert_array_equal(v2a, [1, 2])

    v2b = ivec2(3)
    np.testing.assert_array_equal(v2b, [3, 3])

    # ivec3
    v3a = ivec3(1, 2, 3)
    assert v3a.shape == (3,)
    assert v3a.dtype == np.int32
    np.testing.assert_array_equal(v3a, [1, 2, 3])

    v3b = ivec3(2)
    np.testing.assert_array_equal(v3b, [2, 2, 2])

    v3c = ivec3(ivec2(1, 2), 3)
    np.testing.assert_array_equal(v3c, [1, 2, 3])

    # ivec4
    v4a = ivec4(1, 2, 3, 4)
    assert v4a.shape == (4,)
    assert v4a.dtype == np.int32
    np.testing.assert_array_equal(v4a, [1, 2, 3, 4])

    v4b = ivec4(2)
    np.testing.assert_array_equal(v4b, [2, 2, 2, 2])

    v4c = ivec4(ivec2(1, 2), 3, 4)
    np.testing.assert_array_equal(v4c, [1, 2, 3, 4])

    v4d = ivec4(ivec3(1, 2, 3), 4)
    np.testing.assert_array_equal(v4d, [1, 2, 3, 4])


def test_integer_vector_operations():
    """Test integer vector operations."""
    # Arithmetic operations
    assert IVEC2.validate_operation("+", IVEC2) == IVEC2
    assert IVEC3.validate_operation("*", INT) == IVEC3
    assert IVEC4.validate_operation("/", INT) == IVEC4

    # Comparison operations
    assert IVEC2.validate_operation("==", IVEC2) == BOOL
    assert IVEC3.validate_operation("<", IVEC3) == BOOL
    assert IVEC4.validate_operation(">=", IVEC4) == BOOL

    # Invalid operations
    assert IVEC2.validate_operation("+", VEC3) is None
    assert IVEC3.validate_operation("&&", IVEC3) is None
    assert IVEC4.validate_operation("*", MAT3) is None


def test_vector_type_compatibility():
    """Test vector type compatibility."""
    # Integer vector compatibility
    assert IVEC2.is_compatible_with(IVEC2)
    assert not IVEC2.is_compatible_with(IVEC3)
    assert IVEC3.is_compatible_with(INT)
    assert not IVEC4.is_compatible_with(BOOL)

    # Boolean vector compatibility
    assert BVEC2.is_compatible_with(BVEC2)
    assert not BVEC2.is_compatible_with(BVEC3)
    assert not BVEC3.is_compatible_with(INT)
    assert not BVEC4.is_compatible_with(VEC4)

    # Mixed vector compatibility
    assert not VEC2.is_compatible_with(IVEC2)
    assert not IVEC3.is_compatible_with(VEC3)
    assert not BVEC4.is_compatible_with(IVEC4)


def test_integer_vector_constructor_edge_cases():
    """Test integer vector constructor edge cases."""
    # Zero values
    np.testing.assert_array_equal(ivec2(0), [0, 0])
    np.testing.assert_array_equal(ivec3(0), [0, 0, 0])
    np.testing.assert_array_equal(ivec4(0), [0, 0, 0, 0])

    # Negative values
    np.testing.assert_array_equal(ivec2(-1, -2), [-1, -2])
    np.testing.assert_array_equal(ivec3(-1), [-1, -1, -1])

    # Vector composition
    v2 = ivec2(1, 2)
    v3 = ivec3(v2, 3)
    np.testing.assert_array_equal(v3, [1, 2, 3])

    v4 = ivec4(v2, 3, 4)
    np.testing.assert_array_equal(v4, [1, 2, 3, 4])

    # Invalid constructions
    with pytest.raises(TypeError):
        ivec2(1, 2, 3)  # Too many components

    with pytest.raises(ValueError):
        ivec4(ivec3(1, 2, 3), 4, 5)  # Too many components

    with pytest.raises(ValueError):
        ivec4(1, 2)  # Not enough components

    with pytest.raises(ValueError):
        ivec4([1, 2, 3, 4])  # Invalid input type (raw list)


def test_comprehensive_type_system():
    """Comprehensive test of the entire type system."""
    # All possible types
    ALL_TYPES = [
        VOID,
        BOOL,
        INT,
        FLOAT,
        VEC2,
        VEC3,
        VEC4,
        IVEC2,
        IVEC3,
        IVEC4,
        BVEC2,
        BVEC3,
        BVEC4,
        MAT2,
        MAT3,
        MAT4,
    ]

    NUMERIC_TYPES = [
        INT,
        FLOAT,
        VEC2,
        VEC3,
        VEC4,
        IVEC2,
        IVEC3,
        IVEC4,
        MAT2,
        MAT3,
        MAT4,
    ]
    VECTOR_TYPES = [VEC2, VEC3, VEC4, IVEC2, IVEC3, IVEC4, BVEC2, BVEC3, BVEC4]
    MATRIX_TYPES = [MAT2, MAT3, MAT4]
    SCALAR_TYPES = [VOID, BOOL, INT, FLOAT]

    # Test type properties for all types
    for t in ALL_TYPES:
        # Basic properties
        assert isinstance(str(t), str)
        assert isinstance(t.is_numeric, bool)
        assert isinstance(t.is_vector, bool)
        assert isinstance(t.is_matrix, bool)

        # Array variants
        array_type = GLSLType(t.kind, array_size=3)
        assert "[3]" in str(array_type)

        # Qualifiers - skip void type
        if t.kind != TypeKind.VOID:
            assert "uniform" in str(GLSLType(t.kind, is_uniform=True))
            assert "const" in str(GLSLType(t.kind, is_const=True))
            assert "attribute" in str(GLSLType(t.kind, is_attribute=True))

    # Test all possible type combinations for operations
    OPERATORS = ["+", "-", "*", "/", "%", "&&", "||", "==", "!=", "<", ">", "<=", ">="]

    for t1 in ALL_TYPES:
        for t2 in ALL_TYPES:
            # Test compatibility
            is_compatible = t1.is_compatible_with(t2)
            assert isinstance(is_compatible, bool)

            # Test conversion
            can_convert = t1.can_convert_to(t2)
            assert isinstance(can_convert, bool)

            # Test common type
            common = t1.common_type(t2)
            assert common is None or isinstance(common, GLSLType)

            # Test all operations
            for op in OPERATORS:
                result = t1.validate_operation(op, t2)
                assert result is None or isinstance(result, GLSLType)

                # Verify operation rules
                if op in ("&&", "||"):
                    # Logical operations only work between booleans
                    assert (result == BOOL) == (
                        t1.kind == TypeKind.BOOL and t2.kind == TypeKind.BOOL
                    )

                if op in ("==", "!=", "<", ">", "<=", ">="):
                    # Comparison operations return bool if types are compatible
                    assert (result == BOOL) == t1.is_compatible_with(t2)

    # Test vector swizzling
    SWIZZLE_SETS = [
        "x",
        "xy",
        "xyz",
        "xyzw",
        "r",
        "rg",
        "rgb",
        "rgba",
        "s",
        "st",
        "stp",
        "stpq",
    ]

    for t in VECTOR_TYPES:
        size = t.vector_size()
        assert size in (2, 3, 4)

        for swizzle in SWIZZLE_SETS:
            if len(swizzle) <= size:
                result = t.validate_swizzle(swizzle)
                assert result in (FLOAT, VEC2, VEC3, VEC4)
            else:
                with pytest.raises(GLSLSwizzleError):
                    t.validate_swizzle(swizzle)

    # Test matrix operations
    for m in MATRIX_TYPES:
        size = m.matrix_size()
        assert size in (2, 3, 4)

        # Matrix-matrix multiplication
        assert m.validate_operation("*", m) == m

        # Matrix-scalar multiplication
        assert m.validate_operation("*", FLOAT) == m
        assert m.validate_operation("*", INT) == m

        # Matrix-vector multiplication
        vector = {2: VEC2, 3: VEC3, 4: VEC4}[size]
        assert m.validate_operation("*", vector) == vector

    # Test numeric type properties
    for t in NUMERIC_TYPES:
        assert t.is_numeric
        if t in VECTOR_TYPES:
            assert t.is_vector
            assert not t.is_matrix
        elif t in MATRIX_TYPES:
            assert t.is_matrix
            assert not t.is_vector
        else:
            assert not t.is_vector
            assert not t.is_matrix

    # Test boolean vectors
    for t in [BVEC2, BVEC3, BVEC4]:
        assert t.is_vector
        assert t.is_bool_vector
        assert not t.is_numeric
        assert not t.is_matrix

        # Boolean vectors should only be compatible with themselves
        for other in ALL_TYPES:
            assert t.is_compatible_with(other) == (t == other)

    # Test integer vectors
    for t in [IVEC2, IVEC3, IVEC4]:
        assert t.is_vector
        assert t.is_int_vector
        assert t.is_numeric
        assert not t.is_matrix

        # Integer vectors should be compatible with numeric scalars
        assert t.is_compatible_with(INT)
        assert t.is_compatible_with(FLOAT)
        assert not t.is_compatible_with(BOOL)

    # Test array type edge cases
    with pytest.raises(GLSLTypeError):
        GLSLType(TypeKind.FLOAT, array_size=0)
    with pytest.raises(GLSLTypeError):
        GLSLType(TypeKind.FLOAT, array_size=-1)
    with pytest.raises(GLSLTypeError):
        GLSLType(TypeKind.VEC3, is_uniform=True, is_attribute=True)


def test_color_shader_scenario():
    """Test color manipulation shader scenario."""
    # Input color as vec4
    color = vec4(1.0, 0.5, 0.2, 1.0)
    assert color.dtype == np.float32
    assert color.shape == (4,)

    # Brightness adjustment (scalar multiplication)
    brightness = 0.8
    assert VEC4.validate_operation("*", FLOAT) == VEC4
    dimmed_color = color * brightness
    # Use allclose instead of array_equal for floating point comparison
    np.testing.assert_allclose(dimmed_color, [0.8, 0.4, 0.16, 0.8], rtol=1e-5)

    # Color channel swizzling
    assert VEC4.validate_swizzle("bgr") == VEC3
    assert VEC4.validate_swizzle("bgra") == VEC4


def test_uv_manipulation_scenario():
    """Test UV coordinate manipulation scenario."""
    # UV coordinates as vec2
    uv = vec2(0.5, 0.75)
    assert uv.dtype == np.float32
    assert uv.shape == (2,)

    # Center UV coordinates (uv * 2.0 - 1.0)
    assert VEC2.validate_operation("*", FLOAT) == VEC2
    assert VEC2.validate_operation("-", FLOAT) == VEC2
    centered = uv * 2.0 - 1.0
    np.testing.assert_array_equal(centered, [0.0, 0.5])

    # Tile UV coordinates (fract(uv * 2.0))
    tiled = uv * 2.0
    np.testing.assert_array_equal(tiled % 1.0, [0.0, 0.5])


def test_matrix_transforms_scenario():
    """Test matrix transformation scenario."""
    # Create position vector
    pos = vec4(1.0, 2.0, 3.0, 1.0)
    assert pos.shape == (4,)

    # Test matrix-vector multiplication
    assert MAT4.validate_operation("*", VEC4) == VEC4

    # Test matrix-matrix multiplication
    assert MAT4.validate_operation("*", MAT4) == MAT4

    # Test matrix-scalar multiplication
    assert MAT4.validate_operation("*", FLOAT) == MAT4


def test_conditional_color_scenario():
    """Test conditional color mixing scenario."""
    # Create condition vector
    mask = ivec3(1, 0, 1)
    assert mask.dtype == np.int32

    # Test integer vector operations
    assert IVEC3.validate_operation("*", INT) == IVEC3
    assert IVEC3.validate_operation("+", IVEC3) == IVEC3

    # Test comparison operations
    assert IVEC3.validate_operation("==", IVEC3) == BOOL
    assert IVEC3.validate_operation(">", INT) == BOOL


def test_complex_vector_construction_scenario():
    """Test complex vector construction scenario."""
    # Build vec4 from smaller vectors
    v2 = vec2(1.0, 2.0)
    v3 = vec3(v2, 3.0)
    v4 = vec4(v3, 4.0)
    np.testing.assert_array_equal(v4, [1.0, 2.0, 3.0, 4.0])

    # Build vec4 from vec2 + components
    v4_alt = vec4(v2, 3.0, 4.0)
    np.testing.assert_array_equal(v4_alt, v4)

    # Test integer vector construction
    iv2 = ivec2(1, 2)
    iv3 = ivec3(iv2, 3)
    iv4 = ivec4(iv3, 4)
    np.testing.assert_array_equal(iv4, [1, 2, 3, 4])


def test_array_handling_scenario():
    """Test array handling scenario."""
    # Create array types
    float_array = GLSLType(TypeKind.FLOAT, array_size=4)
    vec2_array = GLSLType(TypeKind.VEC2, array_size=4)

    # Test array compatibility
    assert float_array.is_compatible_with(float_array)
    # Arrays of different base types should not be compatible
    assert not float_array.is_compatible_with(vec2_array)
    assert not vec2_array.is_compatible_with(float_array)

    # Test array operations
    assert float_array.validate_operation("+", float_array) == float_array
    assert vec2_array.validate_operation("*", FLOAT) == vec2_array

    # Test array size mismatches
    float_array_diff = GLSLType(TypeKind.FLOAT, array_size=3)
    assert not float_array.is_compatible_with(float_array_diff)


def test_shader_qualifiers_scenario():
    """Test shader qualifiers scenario."""
    # Create uniform variables
    uniform_time = GLSLType(TypeKind.FLOAT, is_uniform=True)
    uniform_color = GLSLType(TypeKind.VEC4, is_uniform=True)

    # Create vertex attributes
    attribute_position = GLSLType(TypeKind.VEC3, is_attribute=True)
    attribute_normal = GLSLType(TypeKind.VEC3, is_attribute=True)

    # Test qualifier strings
    assert str(uniform_time) == "uniform float"
    assert str(uniform_color) == "uniform vec4"
    assert str(attribute_position) == "attribute vec3"
    assert str(attribute_normal) == "attribute vec3"


def test_shader_composition_scenario():
    """Test shader composition with multiple operations."""
    # Create input variables
    position = vec3(1.0, 2.0, 3.0)
    normal = vec3(0.0, 1.0, 0.0)
    color = vec4(0.8, 0.4, 0.2, 1.0)
    time = 1.0

    # Test vector operations
    assert VEC3.validate_operation("+", VEC3) == VEC3
    assert VEC3.validate_operation("*", FLOAT) == VEC3

    # Test swizzling with operations
    assert VEC4.validate_swizzle("xyz") == VEC3
    assert VEC3.validate_operation("*", VEC3) == VEC3  # Component-wise mult

    # Test type mixing
    assert VEC3.validate_operation("+", FLOAT) == VEC3
    assert VEC4.validate_operation("*", INT) == VEC4


def test_shader_control_flow_scenario():
    """Test shader control flow type checking."""
    # Create condition vectors
    cond1 = ivec2(1, 0)
    cond2 = vec2(0.5, 1.0)

    # Test comparison operations
    assert IVEC2.validate_operation(">", INT) == BOOL
    assert VEC2.validate_operation(">=", FLOAT) == BOOL

    # Test logical operations
    assert BOOL.validate_operation("&&", BOOL) == BOOL
    assert BOOL.validate_operation("||", BOOL) == BOOL

    # Test invalid logical operations
    assert VEC2.validate_operation("&&", VEC2) is None
    assert FLOAT.validate_operation("||", INT) is None


def test_array_edge_cases():
    """Test array type edge cases."""
    # Array size validation
    with pytest.raises(GLSLTypeError):
        GLSLType(TypeKind.FLOAT, array_size=0)  # Zero size
    with pytest.raises(GLSLTypeError):
        GLSLType(TypeKind.FLOAT, array_size=-1)  # Negative size
    with pytest.raises(GLSLTypeError):
        GLSLType(TypeKind.FLOAT, array_size=3.14)  # Float size

    # Array operations
    float_array = GLSLType(TypeKind.FLOAT, array_size=2)
    with pytest.raises(GLSLTypeError):
        # Arrays of arrays not allowed in GLSL
        GLSLType(TypeKind.FLOAT, array_size=float_array.array_size)


def test_type_qualifier_combinations():
    """Test valid and invalid type qualifier combinations."""
    # Invalid combinations
    with pytest.raises(GLSLTypeError):
        GLSLType(TypeKind.VOID, is_uniform=True)  # void cannot be uniform
    with pytest.raises(GLSLTypeError):
        GLSLType(TypeKind.VOID, is_attribute=True)  # void cannot be attribute

    # Valid but tricky combinations
    const_uniform = GLSLType(TypeKind.VEC4, is_const=True, is_uniform=True)
    assert str(const_uniform) == "uniform const vec4"  # Order matters in GLSL


def test_swizzle_edge_cases():
    """Test tricky swizzle operations."""
    # Repeated components
    assert VEC4.validate_swizzle("xxxx") == VEC4
    assert VEC3.validate_swizzle("xxx") == VEC3

    # Mixed valid sets but invalid order
    with pytest.raises(GLSLSwizzleError):
        VEC4.validate_swizzle("xr")  # Cannot mix position and color
    with pytest.raises(GLSLSwizzleError):
        VEC4.validate_swizzle("rs")  # Cannot mix color and texture

    # Case sensitivity
    with pytest.raises(GLSLSwizzleError):
        VEC4.validate_swizzle("RGBA")
    with pytest.raises(GLSLSwizzleError):
        VEC4.validate_swizzle("Rgba")


def test_operation_type_promotion():
    """Test type promotion in operations."""
    # Integer promotion to float
    assert INT.validate_operation("/", INT) == FLOAT  # Division always gives float
    assert INT.validate_operation("*", FLOAT) == FLOAT  # Mixed int/float gives float

    # Vector-scalar operations
    assert VEC3.validate_operation("*", INT) == VEC3  # Should preserve vector type
    assert VEC3.validate_operation("/", FLOAT) == VEC3  # Should preserve vector type


def test_matrix_edge_cases():
    """Test matrix operation edge cases."""
    # Matrix-vector multiplication size mismatch
    assert MAT4.validate_operation("*", VEC3) is None
    assert MAT3.validate_operation("*", VEC4) is None

    # Matrix-matrix multiplication size mismatch
    assert MAT3.validate_operation("*", MAT4) is None
    assert MAT4.validate_operation("*", MAT2) is None

    # Invalid matrix operations
    assert MAT3.validate_operation("%", MAT3) is None  # No modulo for matrices
    assert MAT4.validate_operation("&&", MAT4) is None  # No logical ops for matrices


def test_boolean_vector_operations():
    """Test boolean vector operations."""
    # Component-wise operations
    assert BVEC2.validate_operation("&&", BVEC2) == BVEC2
    assert BVEC3.validate_operation("||", BVEC3) == BVEC3

    # Invalid operations
    assert BVEC4.validate_operation("+", BVEC4) is None  # No arithmetic on bool vectors
    assert (
        BVEC2.validate_operation("*", FLOAT) is None
    )  # No scalar mult for bool vectors


def test_type_conversion_limits():
    """Test type conversion limitations."""
    # Valid conversions
    assert INT.can_convert_to(FLOAT)
    assert VEC2.can_convert_to(VEC2)

    # Invalid conversions
    assert not FLOAT.can_convert_to(INT)  # No implicit float to int
    assert not VEC3.can_convert_to(VEC2)  # No implicit vector size conversion
    assert not MAT3.can_convert_to(MAT4)  # No implicit matrix size conversion
    assert not BOOL.can_convert_to(INT)  # No implicit bool to int


def test_boolean_vector_scalar_compatibility():
    """Test boolean vector compatibility with scalar types."""
    # Boolean vectors should not be compatible with any scalar type
    assert not BVEC2.is_compatible_with(INT)
    assert not BVEC2.is_compatible_with(FLOAT)
    assert not BVEC2.is_compatible_with(BOOL)
    assert not BVEC3.is_compatible_with(INT)
    assert not BVEC4.is_compatible_with(FLOAT)


def test_boolean_vector_vector_compatibility():
    """Test boolean vector compatibility with other vector types."""
    # Boolean vectors should only be compatible with same-size boolean vectors
    assert BVEC2.is_compatible_with(BVEC2)
    assert not BVEC2.is_compatible_with(VEC2)
    assert not BVEC2.is_compatible_with(IVEC2)
    assert not BVEC3.is_compatible_with(VEC3)
    assert not BVEC4.is_compatible_with(IVEC4)


def test_comparison_operations_boolean_vectors():
    """Test comparison operations specifically for boolean vectors."""
    # Boolean vectors should only compare with same-size boolean vectors
    assert BVEC2.validate_operation("==", BVEC2) == BOOL
    assert BVEC2.validate_operation("!=", BVEC2) == BOOL
    assert BVEC2.validate_operation("==", VEC2) is None
    assert BVEC3.validate_operation("!=", INT) is None
    assert BVEC4.validate_operation("==", FLOAT) is None


def test_comparison_operation_type_rules():
    """Test comprehensive comparison operation rules."""
    # Test comparison operations between different type categories
    for op in ["==", "!=", "<", ">", "<=", ">="]:
        # Numeric comparisons
        assert INT.validate_operation(op, INT) == BOOL
        assert FLOAT.validate_operation(op, FLOAT) == BOOL
        assert INT.validate_operation(op, FLOAT) == BOOL

        # Vector comparisons (same size only)
        assert VEC2.validate_operation(op, VEC2) == BOOL
        assert VEC2.validate_operation(op, IVEC2) is None  # Different vector types

        # Invalid comparisons
        assert BOOL.validate_operation(op, INT) is None
        assert VEC2.validate_operation(op, VEC3) is None


def test_array_type_validation():
    """Test array type validation rules."""
    # Invalid array sizes
    with pytest.raises(GLSLTypeError):
        GLSLType(TypeKind.FLOAT, array_size=0)
    with pytest.raises(GLSLTypeError):
        GLSLType(TypeKind.FLOAT, array_size=-1)
    with pytest.raises(GLSLTypeError):
        GLSLType(TypeKind.FLOAT, array_size=3.14)


def test_array_nesting_validation():
    """Test array nesting rules."""
    # Create base array type
    float_array = GLSLType(TypeKind.FLOAT, array_size=2)

    # Test array of arrays (should fail)
    with pytest.raises(GLSLTypeError):
        GLSLType(TypeKind.FLOAT, array_size=float_array)

    # Test array with GLSLType as size (should fail)
    with pytest.raises(GLSLTypeError):
        GLSLType(TypeKind.FLOAT, array_size=FLOAT)


def test_array_operations_validation():
    """Test array operation validation rules."""
    array1 = GLSLType(TypeKind.FLOAT, array_size=2)
    array2 = GLSLType(TypeKind.FLOAT, array_size=2)
    array3 = GLSLType(TypeKind.FLOAT, array_size=3)

    # Same-size arrays should be compatible
    assert array1.is_compatible_with(array2)

    # Different size arrays should not be compatible
    assert not array1.is_compatible_with(array3)

    # Arrays should support basic operations with matching sizes
    assert array1.validate_operation("+", array2) == array1
    assert array1.validate_operation("*", FLOAT) == array1
    assert array1.validate_operation("+", array3) is None


def test_boolean_vector_compatibility_strict():
    """Test strict boolean vector compatibility rules."""
    # Boolean vectors should ONLY be compatible with themselves
    for bvec in [BVEC2, BVEC3, BVEC4]:
        # Test against all scalar types
        assert not bvec.is_compatible_with(BOOL)
        assert not bvec.is_compatible_with(INT)
        assert not bvec.is_compatible_with(FLOAT)
        assert not bvec.is_compatible_with(VOID)

        # Test against all vector types of same size
        size = bvec.vector_size()
        other_vecs = {2: [VEC2, IVEC2], 3: [VEC3, IVEC3], 4: [VEC4, IVEC4]}[size]
        for other_vec in other_vecs:
            assert not bvec.is_compatible_with(other_vec)
            assert not other_vec.is_compatible_with(bvec)


def test_array_nesting_validation_detailed():
    """Test detailed array nesting validation rules."""
    # Test array with array size
    float_array = GLSLType(TypeKind.FLOAT, array_size=2)
    with pytest.raises(GLSLTypeError):
        GLSLType(TypeKind.FLOAT, array_size=float_array)

    # Test array with array size attribute
    with pytest.raises(GLSLTypeError):
        GLSLType(TypeKind.FLOAT, array_size=float_array.array_size)

    # Test array with type as size
    with pytest.raises(GLSLTypeError):
        GLSLType(TypeKind.FLOAT, array_size=FLOAT)

    # Test array with type kind as size
    with pytest.raises(GLSLTypeError):
        GLSLType(TypeKind.FLOAT, array_size=TypeKind.FLOAT)


def test_comparison_operations_detailed():
    """Test detailed comparison operation rules."""
    COMPARISON_OPS = ["==", "!=", "<", ">", "<=", ">="]

    # Test boolean vector comparisons
    for bvec in [BVEC2, BVEC3, BVEC4]:
        # Should only compare with same type
        for op in COMPARISON_OPS:
            assert bvec.validate_operation(op, bvec) == BOOL
            # Should not compare with anything else
            assert bvec.validate_operation(op, INT) is None
            assert bvec.validate_operation(op, FLOAT) is None
            assert bvec.validate_operation(op, BOOL) is None

    # Test numeric vector comparisons
    for vec in [VEC2, VEC3, VEC4]:
        for op in COMPARISON_OPS:
            # Same type comparisons
            assert vec.validate_operation(op, vec) == BOOL
            # Different type comparisons
            assert vec.validate_operation(op, IVEC2) is None
            # Scalar comparisons
            assert vec.validate_operation(op, FLOAT) == BOOL
            assert vec.validate_operation(op, INT) == BOOL


def test_type_compatibility_matrix():
    """Test type compatibility matrix for all types."""
    ALL_TYPES = [
        VOID,
        BOOL,
        INT,
        FLOAT,
        VEC2,
        VEC3,
        VEC4,
        IVEC2,
        IVEC3,
        IVEC4,
        BVEC2,
        BVEC3,
        BVEC4,
        MAT2,
        MAT3,
        MAT4,
    ]

    for t1 in ALL_TYPES:
        for t2 in ALL_TYPES:
            is_compatible = t1.is_compatible_with(t2)
            # Same type should always be compatible
            if t1 == t2:
                assert is_compatible
            # Boolean vectors should only be compatible with themselves
            elif t1.is_bool_vector or t2.is_bool_vector:
                assert not is_compatible


def test_boolean_vector_compatibility_isolation():
    """Test boolean vector compatibility in isolation."""
    # Test each boolean vector type separately
    for bvec in [BVEC2, BVEC3, BVEC4]:
        # Should not be compatible with scalars
        for scalar in [BOOL, INT, FLOAT, VOID]:
            assert not bvec.is_compatible_with(
                scalar
            ), f"{bvec} should not be compatible with {scalar}"
            assert not scalar.is_compatible_with(
                bvec
            ), f"{scalar} should not be compatible with {bvec}"


def test_array_nesting_validation_isolation():
    """Test array nesting validation in isolation."""
    # Test with integer array size (valid)
    valid_array = GLSLType(TypeKind.FLOAT, array_size=2)
    assert valid_array.array_size == 2

    # Test with GLSLType as array size (invalid)
    with pytest.raises(GLSLTypeError, match="Arrays of arrays are not allowed"):
        GLSLType(TypeKind.FLOAT, array_size=valid_array)

    # Test with array size attribute (invalid)
    with pytest.raises(GLSLTypeError, match="Arrays of arrays are not allowed"):
        GLSLType(TypeKind.FLOAT, array_size=valid_array.array_size)


def test_boolean_vector_comparison_isolation():
    """Test boolean vector comparison operations in isolation."""
    for bvec in [BVEC2, BVEC3, BVEC4]:
        # Test equality comparisons (should work)
        assert bvec.validate_operation("==", bvec) == BOOL
        assert bvec.validate_operation("!=", bvec) == BOOL

        # Test ordering comparisons (should not work)
        assert bvec.validate_operation("<", bvec) is None
        assert bvec.validate_operation(">", bvec) is None
        assert bvec.validate_operation("<=", bvec) is None
        assert bvec.validate_operation(">=", bvec) is None


def test_boolean_vector_scalar_compatibility_individual():
    """Test boolean vector compatibility with each scalar type individually."""
    # Test BVEC2
    assert not BVEC2.is_compatible_with(INT), "BVEC2 should not be compatible with INT"
    assert not INT.is_compatible_with(BVEC2), "INT should not be compatible with BVEC2"

    # Test BVEC3
    assert not BVEC3.is_compatible_with(INT), "BVEC3 should not be compatible with INT"
    assert not INT.is_compatible_with(BVEC3), "INT should not be compatible with BVEC3"

    # Test BVEC4
    assert not BVEC4.is_compatible_with(INT), "BVEC4 should not be compatible with INT"
    assert not INT.is_compatible_with(BVEC4), "INT should not be compatible with BVEC4"


def test_array_size_attribute_validation():
    """Test array size attribute validation specifically."""
    float_array = GLSLType(TypeKind.FLOAT, array_size=2)

    # Test with direct array_size access
    size = float_array.array_size
    with pytest.raises(GLSLTypeError, match="Arrays of arrays are not allowed"):
        GLSLType(TypeKind.FLOAT, array_size=size)


def test_boolean_vector_comparison_operators():
    """Test each comparison operator for boolean vectors separately."""
    # Test equality operators (should work)
    assert BVEC2.validate_operation("==", BVEC2) == BOOL, "== should work for BVEC2"
    assert BVEC2.validate_operation("!=", BVEC2) == BOOL, "!= should work for BVEC2"

    # Test relational operators (should not work)
    assert BVEC2.validate_operation("<", BVEC2) is None, "< should not work for BVEC2"
    assert BVEC2.validate_operation(">", BVEC2) is None, "> should not work for BVEC2"
    assert BVEC2.validate_operation("<=", BVEC2) is None, "<= should not work for BVEC2"
    assert BVEC2.validate_operation(">=", BVEC2) is None, ">= should not work for BVEC2"


def test_type_compatibility_boolean_vectors():
    """Test boolean vector compatibility in type system."""
    # Test same-type compatibility
    assert BVEC2.is_compatible_with(BVEC2), "BVEC2 should be compatible with itself"
    assert BVEC3.is_compatible_with(BVEC3), "BVEC3 should be compatible with itself"
    assert BVEC4.is_compatible_with(BVEC4), "BVEC4 should be compatible with itself"

    # Test cross-type incompatibility
    assert not BVEC2.is_compatible_with(
        VEC2
    ), "BVEC2 should not be compatible with VEC2"
    assert not BVEC2.is_compatible_with(
        IVEC2
    ), "BVEC2 should not be compatible with IVEC2"
    assert not BVEC2.is_compatible_with(INT), "BVEC2 should not be compatible with INT"


def test_array_nesting_type_validation():
    """Test array nesting with different type scenarios."""
    # Test with GLSLType instance
    float_array = GLSLType(TypeKind.FLOAT, array_size=2)
    with pytest.raises(GLSLTypeError):
        GLSLType(TypeKind.FLOAT, array_size=float_array)

    # Test with array size attribute
    with pytest.raises(GLSLTypeError):
        GLSLType(TypeKind.FLOAT, array_size=float_array.array_size)

    # Test with TypeKind enum
    with pytest.raises(GLSLTypeError):
        GLSLType(TypeKind.FLOAT, array_size=TypeKind.FLOAT)


def test_boolean_vector_compatibility_core():
    """Test core boolean vector compatibility rules."""
    # Test self-compatibility only
    assert BVEC2.is_compatible_with(BVEC2)
    assert not BVEC2.is_compatible_with(BVEC3)
    assert not BVEC2.is_compatible_with(VEC2)
    assert not BVEC2.is_compatible_with(INT)
    assert not BVEC2.is_compatible_with(BOOL)
    assert not BVEC2.is_compatible_with(FLOAT)


def test_array_nesting_core():
    """Test core array nesting validation."""
    float_array = GLSLType(TypeKind.FLOAT, array_size=2)

    # Test direct array nesting
    with pytest.raises(GLSLTypeError, match="Arrays of arrays are not allowed"):
        GLSLType(TypeKind.FLOAT, array_size=float_array)

    # Test with integer value
    size_value = 2
    GLSLType(TypeKind.FLOAT, array_size=size_value)  # Should work

    # Test with array size attribute (should fail)
    size_attr = float_array.array_size
    with pytest.raises(GLSLTypeError, match="Arrays of arrays are not allowed"):
        GLSLType(TypeKind.FLOAT, array_size=size_attr)


def test_boolean_vector_operations_core():
    """Test core boolean vector operations."""
    # Test equality operations (should work)
    assert BVEC2.validate_operation("==", BVEC2) == BOOL
    assert BVEC2.validate_operation("!=", BVEC2) == BOOL

    # Test relational operations (should not work)
    assert BVEC2.validate_operation("<", BVEC2) is None
    assert BVEC2.validate_operation(">", BVEC2) is None

    # Test logical operations (should work component-wise)
    assert BVEC2.validate_operation("&&", BVEC2) == BVEC2
    assert BVEC2.validate_operation("||", BVEC2) == BVEC2

    # Test arithmetic operations (should not work)
    assert BVEC2.validate_operation("+", BVEC2) is None
    assert BVEC2.validate_operation("*", BVEC2) is None


def test_type_compatibility_rules():
    """Test fundamental type compatibility rules."""
    # Same type compatibility
    assert BVEC2.is_compatible_with(BVEC2)
    assert VEC2.is_compatible_with(VEC2)
    assert INT.is_compatible_with(INT)

    # Cross-type compatibility
    assert not BVEC2.is_compatible_with(VEC2)
    assert not BVEC2.is_compatible_with(INT)
    assert not BVEC2.is_compatible_with(BOOL)

    # Numeric type compatibility
    assert INT.is_compatible_with(FLOAT)
    assert VEC2.is_compatible_with(FLOAT)
    assert not VEC2.is_compatible_with(BOOL)


def test_array_validation_core():
    """Test core array validation rules."""
    # Valid array sizes
    GLSLType(TypeKind.FLOAT, array_size=1)
    GLSLType(TypeKind.FLOAT, array_size=100)

    # Invalid array sizes
    with pytest.raises(GLSLTypeError):
        GLSLType(TypeKind.FLOAT, array_size=0)

    with pytest.raises(GLSLTypeError):
        GLSLType(TypeKind.FLOAT, array_size=-1)

    with pytest.raises(GLSLTypeError):
        GLSLType(TypeKind.FLOAT, array_size=3.14)

    # Array nesting
    float_array = GLSLType(TypeKind.FLOAT, array_size=2)
    with pytest.raises(GLSLTypeError):
        GLSLType(TypeKind.FLOAT, array_size=float_array)
