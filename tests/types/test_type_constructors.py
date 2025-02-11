"""Tests for GLSL type constructors and runtime implementations."""

import numpy as np
import pytest

from py2glsl.transpiler.type_constructors import (
    bvec2,
    bvec3,
    bvec4,
    ivec2,
    ivec3,
    ivec4,
    vec2,
    vec3,
    vec4,
)


class TestFloatVectorConstruction:
    """Test float vector constructors."""

    def test_vec2_basic(self):
        """Test basic vec2 construction."""
        # Single value expansion
        v = vec2(1.0)
        np.testing.assert_array_equal(v, [1.0, 1.0])
        assert v.dtype == np.float32

        # Two components
        v = vec2(1.0, 2.0)
        np.testing.assert_array_equal(v, [1.0, 2.0])
        assert v.dtype == np.float32

        # Integer to float conversion
        v = vec2(1, 2)
        np.testing.assert_array_equal(v, [1.0, 2.0])
        assert v.dtype == np.float32

    def test_vec3_basic(self):
        """Test basic vec3 construction."""
        # Single value expansion
        v = vec3(1.0)
        np.testing.assert_array_equal(v, [1.0, 1.0, 1.0])
        assert v.dtype == np.float32

        # Three components
        v = vec3(1.0, 2.0, 3.0)
        np.testing.assert_array_equal(v, [1.0, 2.0, 3.0])
        assert v.dtype == np.float32

        # vec2 + scalar
        v2 = vec2(1.0, 2.0)
        v = vec3(v2, 3.0)
        np.testing.assert_array_equal(v, [1.0, 2.0, 3.0])
        assert v.dtype == np.float32

    def test_vec4_basic(self):
        """Test basic vec4 construction."""
        # Single value expansion
        v = vec4(1.0)
        np.testing.assert_array_equal(v, [1.0, 1.0, 1.0, 1.0])
        assert v.dtype == np.float32

        # Four components
        v = vec4(1.0, 2.0, 3.0, 4.0)
        np.testing.assert_array_equal(v, [1.0, 2.0, 3.0, 4.0])
        assert v.dtype == np.float32

        # vec3 + scalar
        v3 = vec3(1.0, 2.0, 3.0)
        v = vec4(v3, 4.0)
        np.testing.assert_array_equal(v, [1.0, 2.0, 3.0, 4.0])
        assert v.dtype == np.float32

        # vec2 + two scalars
        v2 = vec2(1.0, 2.0)
        v = vec4(v2, 3.0, 4.0)
        np.testing.assert_array_equal(v, [1.0, 2.0, 3.0, 4.0])
        assert v.dtype == np.float32


class TestIntegerVectorConstruction:
    """Test integer vector constructors."""

    def test_ivec2_basic(self):
        """Test basic ivec2 construction."""
        # Single value expansion
        v = ivec2(1)
        np.testing.assert_array_equal(v, [1, 1])
        assert v.dtype == np.int32

        # Two components
        v = ivec2(1, 2)
        np.testing.assert_array_equal(v, [1, 2])
        assert v.dtype == np.int32

        # Float to int conversion
        v = ivec2(1.7, 2.3)
        np.testing.assert_array_equal(v, [1, 2])
        assert v.dtype == np.int32

    def test_ivec3_basic(self):
        """Test basic ivec3 construction."""
        # Single value expansion
        v = ivec3(1)
        np.testing.assert_array_equal(v, [1, 1, 1])
        assert v.dtype == np.int32

        # Three components
        v = ivec3(1, 2, 3)
        np.testing.assert_array_equal(v, [1, 2, 3])
        assert v.dtype == np.int32

        # ivec2 + scalar
        v2 = ivec2(1, 2)
        v = ivec3(v2, 3)
        np.testing.assert_array_equal(v, [1, 2, 3])
        assert v.dtype == np.int32

    def test_ivec4_basic(self):
        """Test basic ivec4 construction."""
        # Single value expansion
        v = ivec4(1)
        np.testing.assert_array_equal(v, [1, 1, 1, 1])
        assert v.dtype == np.int32

        # Four components
        v = ivec4(1, 2, 3, 4)
        np.testing.assert_array_equal(v, [1, 2, 3, 4])
        assert v.dtype == np.int32

        # ivec3 + scalar
        v3 = ivec3(1, 2, 3)
        v = ivec4(v3, 4)
        np.testing.assert_array_equal(v, [1, 2, 3, 4])
        assert v.dtype == np.int32


class TestBooleanVectorConstruction:
    """Test boolean vector constructors."""

    def test_bvec2_basic(self):
        """Test basic bvec2 construction."""
        # Single value expansion
        v = bvec2(True)
        np.testing.assert_array_equal(v, [True, True])
        assert v.dtype == bool

        # Two components
        v = bvec2(True, False)
        np.testing.assert_array_equal(v, [True, False])
        assert v.dtype == bool

        # Numeric to bool conversion
        v = bvec2(1, 0)
        np.testing.assert_array_equal(v, [True, False])
        assert v.dtype == bool

    def test_bvec3_basic(self):
        """Test basic bvec3 construction."""
        # Single value expansion
        v = bvec3(True)
        np.testing.assert_array_equal(v, [True, True, True])
        assert v.dtype == bool

        # Three components
        v = bvec3(True, False, True)
        np.testing.assert_array_equal(v, [True, False, True])
        assert v.dtype == bool

        # bvec2 + scalar
        v2 = bvec2(True, False)
        v = bvec3(v2, True)
        np.testing.assert_array_equal(v, [True, False, True])
        assert v.dtype == bool

    def test_bvec4_basic(self):
        """Test basic bvec4 construction."""
        # Single value expansion
        v = bvec4(True)
        np.testing.assert_array_equal(v, [True, True, True, True])
        assert v.dtype == bool

        # Four components
        v = bvec4(True, False, True, False)
        np.testing.assert_array_equal(v, [True, False, True, False])
        assert v.dtype == bool

        # bvec3 + scalar
        v3 = bvec3(True, False, True)
        v = bvec4(v3, False)
        np.testing.assert_array_equal(v, [True, False, True, False])
        assert v.dtype == bool


class TestVectorConstructionErrors:
    """Test vector constructor error cases."""

    def test_vec2_errors(self):
        """Test vec2 constructor errors."""
        # No arguments
        with pytest.raises(ValueError):
            vec2()

        # Too many arguments
        with pytest.raises(TypeError):
            vec2(1.0, 2.0, 3.0)

        # Invalid types
        with pytest.raises(TypeError):
            vec2("1.0", "2.0")

    def test_vec3_errors(self):
        """Test vec3 constructor errors."""
        # No arguments
        with pytest.raises(ValueError):
            vec3()

        # Too many arguments
        with pytest.raises(TypeError):
            vec3(1.0, 2.0, 3.0, 4.0)

        # Invalid vec2 + scalar combination
        with pytest.raises(TypeError):
            vec3(1.0, vec2(2.0, 3.0))

    def test_vec4_errors(self):
        """Test vec4 constructor errors."""
        # No arguments
        with pytest.raises(ValueError):
            vec4()

        # Too many arguments
        with pytest.raises(TypeError):
            vec4(1.0, 2.0, 3.0, 4.0, 5.0)

        # Invalid vec3 + scalar combination
        with pytest.raises(TypeError):
            vec4(1.0, vec3(2.0, 3.0, 4.0))


class TestVectorTypeConversion:
    """Test vector type conversion behavior."""

    def test_float_vector_conversion(self):
        """Test float vector type conversion."""
        # Integer to float vector
        v2 = vec2(ivec2(1, 2))
        np.testing.assert_array_equal(v2, [1.0, 2.0])
        assert v2.dtype == np.float32

        v3 = vec3(ivec3(1, 2, 3))
        np.testing.assert_array_equal(v3, [1.0, 2.0, 3.0])
        assert v3.dtype == np.float32

        # Boolean to float vector
        v2 = vec2(bvec2(True, False))
        np.testing.assert_array_equal(v2, [1.0, 0.0])
        assert v2.dtype == np.float32

    def test_int_vector_conversion(self):
        """Test integer vector type conversion."""
        # Float to int vector (truncation)
        v2 = ivec2(vec2(1.7, 2.3))
        np.testing.assert_array_equal(v2, [1, 2])
        assert v2.dtype == np.int32

        # Boolean to int vector
        v3 = ivec3(bvec3(True, False, True))
        np.testing.assert_array_equal(v3, [1, 0, 1])
        assert v3.dtype == np.int32

    def test_bool_vector_conversion(self):
        """Test boolean vector type conversion."""
        # Float to bool vector
        v2 = bvec2(vec2(0.0, 1.0))
        np.testing.assert_array_equal(v2, [False, True])
        assert v2.dtype == bool

        # Int to bool vector
        v3 = bvec3(ivec3(0, 1, 2))
        np.testing.assert_array_equal(v3, [False, True, True])
        assert v3.dtype == bool


class TestVectorConstructionEdgeCases:
    """Test vector constructor edge cases."""

    def test_zero_values(self):
        """Test construction with zero values."""
        np.testing.assert_array_equal(vec2(0.0), [0.0, 0.0])
        np.testing.assert_array_equal(vec3(0.0), [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(vec4(0.0), [0.0, 0.0, 0.0, 0.0])

    def test_negative_values(self):
        """Test construction with negative values."""
        np.testing.assert_array_equal(vec2(-1.0, -2.0), [-1.0, -2.0])
        np.testing.assert_array_equal(ivec2(-1, -2), [-1, -2])

    def test_mixed_numeric_types(self):
        """Test construction with mixed numeric types."""
        v = vec3(1, 2.0, 3.5)  # int, float, float
        np.testing.assert_array_equal(v, [1.0, 2.0, 3.5])
        assert v.dtype == np.float32

        v = ivec3(1.7, 2, 3.2)  # float, int, float
        np.testing.assert_array_equal(v, [1, 2, 3])
        assert v.dtype == np.int32

    def test_special_float_values(self):
        """Test construction with special float values."""
        # Infinity
        v = vec2(float("inf"), -float("inf"))
        assert np.isinf(v[0]) and v[0] > 0
        assert np.isinf(v[1]) and v[1] < 0

        # NaN
        v = vec2(float("nan"))
        assert np.isnan(v[0]) and np.isnan(v[1])
