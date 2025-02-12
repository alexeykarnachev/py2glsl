"""Tests for GLSL type system core functionality."""

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
)


class TestTypeKind:
    """Test TypeKind enum functionality."""

    def test_numeric_types(self):
        """Test numeric type identification."""
        numeric_types = [
            TypeKind.INT,
            TypeKind.FLOAT,
            TypeKind.VEC2,
            TypeKind.VEC3,
            TypeKind.VEC4,
            TypeKind.IVEC2,
            TypeKind.IVEC3,
            TypeKind.IVEC4,
            TypeKind.MAT2,
            TypeKind.MAT3,
            TypeKind.MAT4,
        ]
        non_numeric_types = [TypeKind.VOID, TypeKind.BOOL]

        for type_kind in numeric_types:
            assert type_kind.is_numeric

        for type_kind in non_numeric_types:
            assert not type_kind.is_numeric

    def test_vector_types(self):
        """Test vector type identification."""
        vector_types = [
            TypeKind.VEC2,
            TypeKind.VEC3,
            TypeKind.VEC4,
            TypeKind.IVEC2,
            TypeKind.IVEC3,
            TypeKind.IVEC4,
            TypeKind.BVEC2,
            TypeKind.BVEC3,
            TypeKind.BVEC4,
        ]
        non_vector_types = [
            TypeKind.VOID,
            TypeKind.BOOL,
            TypeKind.INT,
            TypeKind.FLOAT,
            TypeKind.MAT2,
            TypeKind.MAT3,
            TypeKind.MAT4,
        ]

        for type_kind in vector_types:
            assert type_kind.is_vector

        for type_kind in non_vector_types:
            assert not type_kind.is_vector

    def test_matrix_types(self):
        """Test matrix type identification."""
        matrix_types = [TypeKind.MAT2, TypeKind.MAT3, TypeKind.MAT4]
        non_matrix_types = [
            type_kind
            for type_kind in TypeKind
            if type_kind not in [TypeKind.MAT2, TypeKind.MAT3, TypeKind.MAT4]
        ]

        for type_kind in matrix_types:
            assert type_kind.is_matrix

        for type_kind in non_matrix_types:
            assert not type_kind.is_matrix

    def test_vector_sizes(self):
        """Test vector size properties."""
        size_map = {
            TypeKind.VEC2: 2,
            TypeKind.VEC3: 3,
            TypeKind.VEC4: 4,
            TypeKind.IVEC2: 2,
            TypeKind.IVEC3: 3,
            TypeKind.IVEC4: 4,
            TypeKind.BVEC2: 2,
            TypeKind.BVEC3: 3,
            TypeKind.BVEC4: 4,
        }

        for type_kind, expected_size in size_map.items():
            assert type_kind.vector_size == expected_size

        assert TypeKind.FLOAT.vector_size is None
        assert TypeKind.MAT2.vector_size is None

    def test_matrix_sizes(self):
        """Test matrix size properties."""
        size_map = {
            TypeKind.MAT2: 2,
            TypeKind.MAT3: 3,
            TypeKind.MAT4: 4,
        }

        for type_kind, expected_size in size_map.items():
            assert type_kind.matrix_size == expected_size

        assert TypeKind.FLOAT.matrix_size is None
        assert TypeKind.VEC2.matrix_size is None


class TestGLSLType:
    """Test GLSLType class functionality."""

    def test_basic_type_creation(self):
        """Test basic type creation."""
        type_float = GLSLType(TypeKind.FLOAT)
        assert type_float.kind == TypeKind.FLOAT
        assert not type_float.is_uniform
        assert not type_float.is_const
        assert not type_float.is_attribute
        assert type_float.array_size is None

    def test_type_qualifiers(self):
        """Test type qualifiers."""
        uniform_vec3 = GLSLType(TypeKind.VEC3, is_uniform=True)
        assert uniform_vec3.is_uniform
        assert str(uniform_vec3) == "uniform vec3"

        const_float = GLSLType(TypeKind.FLOAT, is_const=True)
        assert const_float.is_const
        assert str(const_float) == "const float"

        attribute_vec4 = GLSLType(TypeKind.VEC4, is_attribute=True)
        assert attribute_vec4.is_attribute
        assert str(attribute_vec4) == "attribute vec4"

    def test_array_types(self):
        """Test array type creation and validation."""
        float_array = GLSLType(TypeKind.FLOAT, array_size=4)
        assert float_array.array_size == 4
        assert str(float_array) == "float[4]"

        with pytest.raises(GLSLTypeError):
            GLSLType(TypeKind.FLOAT, array_size=0)

        with pytest.raises(GLSLTypeError):
            GLSLType(TypeKind.FLOAT, array_size=-1)

    def test_invalid_combinations(self):
        """Test invalid type combinations."""
        # Cannot be both uniform and attribute
        with pytest.raises(GLSLTypeError):
            GLSLType(TypeKind.VEC3, is_uniform=True, is_attribute=True)

        # Void cannot have qualifiers
        with pytest.raises(GLSLTypeError):
            GLSLType(TypeKind.VOID, is_uniform=True)

        with pytest.raises(GLSLTypeError):
            GLSLType(TypeKind.VOID, array_size=3)

    def test_type_properties(self):
        """Test type property checks."""
        assert FLOAT.is_numeric
        assert VEC3.is_numeric
        assert MAT4.is_numeric
        assert not BOOL.is_numeric
        assert not VOID.is_numeric

        assert VEC2.is_vector
        assert IVEC3.is_vector
        assert BVEC4.is_vector
        assert not FLOAT.is_vector
        assert not MAT3.is_vector

        assert MAT2.is_matrix
        assert MAT3.is_matrix
        assert MAT4.is_matrix
        assert not VEC3.is_matrix
        assert not FLOAT.is_matrix

    def test_type_string_representation(self):
        """Test string representation of types."""
        assert str(VOID) == "void"
        assert str(BOOL) == "bool"
        assert str(INT) == "int"
        assert str(FLOAT) == "float"
        assert str(VEC2) == "vec2"
        assert str(VEC3) == "vec3"
        assert str(VEC4) == "vec4"
        assert str(IVEC2) == "ivec2"
        assert str(IVEC3) == "ivec3"
        assert str(IVEC4) == "ivec4"
        assert str(BVEC2) == "bvec2"
        assert str(BVEC3) == "bvec3"
        assert str(BVEC4) == "bvec4"
        assert str(MAT2) == "mat2"
        assert str(MAT3) == "mat3"
        assert str(MAT4) == "mat4"

    def test_combined_qualifiers(self):
        """Test combined type qualifiers."""
        uniform_const = GLSLType(TypeKind.VEC2, is_uniform=True, is_const=True)
        assert str(uniform_const) == "uniform const vec2"

        uniform_array = GLSLType(TypeKind.FLOAT, is_uniform=True, array_size=4)
        assert str(uniform_array) == "uniform float[4]"

    def test_vector_boolean_properties(self):
        """Test vector boolean type properties."""
        assert BVEC2.is_bool_vector
        assert BVEC3.is_bool_vector
        assert BVEC4.is_bool_vector
        assert not VEC2.is_bool_vector
        assert not IVEC2.is_bool_vector

    def test_vector_integer_properties(self):
        """Test vector integer type properties."""
        assert IVEC2.is_int_vector
        assert IVEC3.is_int_vector
        assert IVEC4.is_int_vector
        assert not VEC2.is_int_vector
        assert not BVEC2.is_int_vector


class TestTypeSystemErrors:
    """Test type system error handling."""

    def test_type_error_invalid_array_size(self):
        """Test array size validation errors."""
        with pytest.raises(GLSLTypeError, match="Array size must be positive"):
            GLSLType(TypeKind.FLOAT, array_size=0)

        with pytest.raises(GLSLTypeError, match="Array size must be positive"):
            GLSLType(TypeKind.FLOAT, array_size=-1)

        with pytest.raises(GLSLTypeError, match="Array size must be an integer"):
            GLSLType(TypeKind.FLOAT, array_size=3.14)

    def test_type_error_invalid_qualifiers(self):
        """Test qualifier validation errors."""
        with pytest.raises(
            GLSLTypeError, match="Type cannot be both uniform and attribute"
        ):
            GLSLType(TypeKind.VEC3, is_uniform=True, is_attribute=True)

        with pytest.raises(
            GLSLTypeError, match="Void type cannot have storage qualifiers"
        ):
            GLSLType(TypeKind.VOID, is_uniform=True)

    def test_swizzle_error_invalid_components(self):
        """Test swizzle validation errors."""
        with pytest.raises(GLSLSwizzleError):
            VEC3.validate_swizzle("w")  # w not available in vec3

        with pytest.raises(GLSLSwizzleError):
            VEC4.validate_swizzle("xrs")  # mixed component sets

        with pytest.raises(GLSLSwizzleError):
            FLOAT.validate_swizzle("x")  # non-vector type


class TestTypeSystemSingleton:
    """Test type system singleton types."""

    def test_singleton_types_equality(self):
        """Test singleton type equality."""
        # Same type instances should be equal
        assert VOID == GLSLType(TypeKind.VOID)
        assert FLOAT == GLSLType(TypeKind.FLOAT)
        assert VEC3 == GLSLType(TypeKind.VEC3)
        assert MAT4 == GLSLType(TypeKind.MAT4)

    def test_singleton_types_identity(self):
        """Test singleton type identity."""
        # Same type instances should be identical
        assert VOID is GLSLType(TypeKind.VOID)
        assert FLOAT is GLSLType(TypeKind.FLOAT)
        assert VEC3 is GLSLType(TypeKind.VEC3)
        assert MAT4 is GLSLType(TypeKind.MAT4)

    def test_singleton_types_immutability(self):
        """Test singleton type immutability."""
        with pytest.raises(AttributeError):
            FLOAT.kind = TypeKind.INT

        with pytest.raises(AttributeError):
            VEC3.is_uniform = True

        with pytest.raises(AttributeError):
            MAT4.array_size = 4


class TestTypeSystemComprehensive:
    """Comprehensive tests for type system functionality."""

    def test_type_hierarchy(self):
        """Test type hierarchy relationships."""
        # Numeric type hierarchy
        assert INT.is_numeric
        assert FLOAT.is_numeric
        assert VEC3.is_numeric
        assert MAT4.is_numeric
        assert not BOOL.is_numeric
        assert not VOID.is_numeric

        # Vector type hierarchy
        assert VEC2.is_vector
        assert IVEC3.is_vector
        assert BVEC4.is_vector
        assert not FLOAT.is_vector
        assert not MAT3.is_vector

        # Matrix type hierarchy
        assert MAT2.is_matrix
        assert MAT3.is_matrix
        assert MAT4.is_matrix
        assert not VEC3.is_matrix
        assert not FLOAT.is_matrix

    def test_type_properties_comprehensive(self):
        """Test comprehensive type properties."""
        # Test all vector sizes
        assert VEC2.vector_size() == 2
        assert VEC3.vector_size() == 3
        assert VEC4.vector_size() == 4
        assert IVEC2.vector_size() == 2
        assert BVEC3.vector_size() == 3

        # Test all matrix sizes
        assert MAT2.matrix_size() == 2
        assert MAT3.matrix_size() == 3
        assert MAT4.matrix_size() == 4

        # Test non-vector/matrix types
        assert FLOAT.vector_size() is None
        assert INT.matrix_size() is None

    def test_type_qualifiers_comprehensive(self):
        """Test comprehensive qualifier combinations."""
        # Test all valid qualifier combinations
        uniform_const = GLSLType(TypeKind.VEC4, is_uniform=True, is_const=True)
        assert str(uniform_const) == "uniform const vec4"

        const_array = GLSLType(TypeKind.FLOAT, is_const=True, array_size=4)
        assert str(const_array) == "const float[4]"

        attribute_array = GLSLType(TypeKind.VEC3, is_attribute=True, array_size=2)
        assert str(attribute_array) == "attribute vec3[2]"

    def test_type_validation_comprehensive(self):
        """Test comprehensive type validation."""
        # Test array size validation
        with pytest.raises(GLSLTypeError):
            GLSLType(TypeKind.FLOAT, array_size=0)

        with pytest.raises(GLSLTypeError):
            GLSLType(TypeKind.FLOAT, array_size=-1)

        # Test qualifier validation
        with pytest.raises(GLSLTypeError):
            GLSLType(TypeKind.VOID, is_uniform=True)

        with pytest.raises(GLSLTypeError):
            GLSLType(TypeKind.VOID, array_size=3)

        # Test combined qualifier validation
        with pytest.raises(GLSLTypeError):
            GLSLType(TypeKind.VEC3, is_uniform=True, is_attribute=True)
