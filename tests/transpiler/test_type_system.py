"""Tests for GLSL type system core functionality."""

import pytest

from py2glsl.transpiler.type_system import (
    BOOL,
    FLOAT,
    INT,
    VEC2,
    VEC3,
    VEC4,
    VOID,
    GLSLType,
    GLSLTypeError,
    TypeKind,
)


class TestTypeSystemBasics:
    """Test basic type system functionality."""

    def test_type_creation(self):
        """Test creating types directly."""
        float_type = GLSLType(TypeKind.FLOAT)
        assert str(float_type) == "float"
        assert float_type.is_numeric
        assert not float_type.is_vector
        assert not float_type.is_matrix

        vec3_type = GLSLType(TypeKind.VEC3)
        assert str(vec3_type) == "vec3"
        assert vec3_type.is_numeric
        assert vec3_type.is_vector
        assert not vec3_type.is_matrix

    def test_singleton_types(self):
        """Test predefined singleton types."""
        # Basic properties
        assert str(VOID) == "void"
        assert str(BOOL) == "bool"
        assert str(INT) == "int"
        assert str(FLOAT) == "float"
        assert str(VEC2) == "vec2"
        assert str(VEC3) == "vec3"
        assert str(VEC4) == "vec4"

        # Type properties
        assert not VOID.is_numeric
        assert INT.is_numeric
        assert FLOAT.is_numeric
        assert VEC3.is_numeric

        assert not FLOAT.is_vector
        assert VEC2.is_vector
        assert VEC3.is_vector
        assert VEC4.is_vector

    def test_type_qualifiers(self):
        """Test type qualifiers."""
        # Uniform qualifier
        uniform_float = GLSLType(TypeKind.FLOAT, is_uniform=True)
        assert str(uniform_float) == "uniform float"

        # Const qualifier
        const_vec3 = GLSLType(TypeKind.VEC3, is_const=True)
        assert str(const_vec3) == "const vec3"

        # Attribute qualifier
        attribute_vec4 = GLSLType(TypeKind.VEC4, is_attribute=True)
        assert str(attribute_vec4) == "attribute vec4"

        # Multiple qualifiers
        uniform_const = GLSLType(TypeKind.VEC2, is_uniform=True, is_const=True)
        assert str(uniform_const) == "uniform const vec2"

    def test_array_types(self):
        """Test array type declarations."""
        float_array = GLSLType(TypeKind.FLOAT, array_size=4)
        assert str(float_array) == "float[4]"

        vec2_array = GLSLType(TypeKind.VEC2, array_size=3)
        assert str(vec2_array) == "vec2[3]"

        # Array with qualifiers
        uniform_array = GLSLType(TypeKind.FLOAT, is_uniform=True, array_size=4)
        assert str(uniform_array) == "uniform float[4]"


class TestTypeSystemValidation:
    """Test type system validation rules."""

    def test_invalid_array_size(self):
        """Test array size validation."""
        with pytest.raises(GLSLTypeError):
            GLSLType(TypeKind.FLOAT, array_size=0)

        with pytest.raises(GLSLTypeError):
            GLSLType(TypeKind.FLOAT, array_size=-1)

        with pytest.raises(GLSLTypeError):
            GLSLType(TypeKind.FLOAT, array_size=3.14)

    def test_invalid_qualifiers(self):
        """Test invalid qualifier combinations."""
        # Cannot be both uniform and attribute
        with pytest.raises(GLSLTypeError):
            GLSLType(TypeKind.VEC3, is_uniform=True, is_attribute=True)

        # Void cannot have storage qualifiers
        with pytest.raises(GLSLTypeError):
            GLSLType(TypeKind.VOID, is_uniform=True)

        with pytest.raises(GLSLTypeError):
            GLSLType(TypeKind.VOID, is_attribute=True)


class TestTypeSystemUserPerspective:
    """Test type system from user's perspective."""

    def test_type_comparison(self):
        """Test how types can be compared."""
        # Same types should be equal
        assert GLSLType(TypeKind.FLOAT) == GLSLType(TypeKind.FLOAT)
        assert GLSLType(TypeKind.VEC3) == GLSLType(TypeKind.VEC3)

        # Different types should not be equal
        assert GLSLType(TypeKind.FLOAT) != GLSLType(TypeKind.INT)
        assert GLSLType(TypeKind.VEC2) != GLSLType(TypeKind.VEC3)

        # Arrays of same type and size should be equal
        assert GLSLType(TypeKind.FLOAT, array_size=3) == GLSLType(
            TypeKind.FLOAT, array_size=3
        )

        # Arrays of different sizes should not be equal
        assert GLSLType(TypeKind.FLOAT, array_size=2) != GLSLType(
            TypeKind.FLOAT, array_size=3
        )

    def test_type_properties_usage(self):
        """Test how type properties can be used."""
        # Vector size checks
        assert VEC2.vector_size() == 2
        assert VEC3.vector_size() == 3
        assert VEC4.vector_size() == 4
        assert FLOAT.vector_size() is None

        # Type category checks
        assert VEC3.is_vector
        assert not VEC3.is_matrix
        assert VEC3.is_numeric

        assert FLOAT.is_numeric
        assert not FLOAT.is_vector
        assert not FLOAT.is_matrix

    def test_type_string_representation(self):
        """Test string representation for debugging/display."""
        # Basic types
        assert str(FLOAT) == "float"
        assert str(VEC3) == "vec3"

        # Qualified types
        uniform_vec2 = GLSLType(TypeKind.VEC2, is_uniform=True)
        assert str(uniform_vec2) == "uniform vec2"

        # Array types
        float_array = GLSLType(TypeKind.FLOAT, array_size=3)
        assert str(float_array) == "float[3]"

        # Complex type
        uniform_const_array = GLSLType(
            TypeKind.VEC4, is_uniform=True, is_const=True, array_size=2
        )
        assert str(uniform_const_array) == "uniform const vec4[2]"


class TestTypeSystemEdgeCases:
    """Test edge cases and special scenarios."""

    def test_void_type_restrictions(self):
        """Test special restrictions for void type."""
        # Void type should not have qualifiers
        with pytest.raises(GLSLTypeError):
            GLSLType(TypeKind.VOID, is_uniform=True)

        # Void arrays should not be possible
        with pytest.raises(GLSLTypeError):
            GLSLType(TypeKind.VOID, array_size=3)

        # Void type properties
        assert not VOID.is_numeric
        assert not VOID.is_vector
        assert not VOID.is_matrix
        assert VOID.vector_size() is None
        assert VOID.matrix_size() is None

    def test_type_immutability(self):
        """Test that types are immutable."""
        float_type = GLSLType(TypeKind.FLOAT)

        # Attempt to modify frozen dataclass
        with pytest.raises(Exception):
            float_type.kind = TypeKind.INT

        with pytest.raises(Exception):
            float_type.is_uniform = True

    def test_array_type_edge_cases(self):
        """Test array type edge cases."""
        # Array of size 1 should be valid
        size_one_array = GLSLType(TypeKind.FLOAT, array_size=1)
        assert str(size_one_array) == "float[1]"

        # Large array sizes should be valid
        large_array = GLSLType(TypeKind.FLOAT, array_size=1000)
        assert str(large_array) == "float[1000]"

        # Non-integer array sizes
        with pytest.raises(GLSLTypeError):
            GLSLType(TypeKind.FLOAT, array_size="3")

        with pytest.raises(GLSLTypeError):
            GLSLType(TypeKind.FLOAT, array_size=True)
