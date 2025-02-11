"""Tests for GLSL type validation rules."""

import pytest

from py2glsl.transpiler.type_system import (
    BOOL,
    BVEC2,
    BVEC3,
    FLOAT,
    INT,
    IVEC2,
    IVEC3,
    MAT2,
    MAT3,
    MAT4,
    VEC2,
    VEC3,
    VEC4,
    VOID,
    GLSLSwizzleError,
    GLSLType,
    TypeKind,
)
from py2glsl.transpiler.type_validation import (
    can_convert_to,
    common_type,
    is_compatible_with,
    validate_operation,
    validate_swizzle,
)


class TestSwizzleValidation:
    """Test swizzle validation rules."""

    def test_valid_swizzles(self):
        """Test valid swizzle patterns."""
        # Position components
        assert validate_swizzle(VEC4, "x") == FLOAT
        assert validate_swizzle(VEC4, "xy") == VEC2
        assert validate_swizzle(VEC4, "xyz") == VEC3
        assert validate_swizzle(VEC4, "xyzw") == VEC4

        # Color components
        assert validate_swizzle(VEC4, "r") == FLOAT
        assert validate_swizzle(VEC4, "rg") == VEC2
        assert validate_swizzle(VEC4, "rgb") == VEC3
        assert validate_swizzle(VEC4, "rgba") == VEC4

        # Texture components
        assert validate_swizzle(VEC4, "s") == FLOAT
        assert validate_swizzle(VEC4, "st") == VEC2
        assert validate_swizzle(VEC4, "stp") == VEC3
        assert validate_swizzle(VEC4, "stpq") == VEC4

    def test_invalid_swizzles(self):
        """Test invalid swizzle patterns."""
        # Non-vector type
        with pytest.raises(GLSLSwizzleError):
            validate_swizzle(FLOAT, "x")

        # Empty swizzle
        with pytest.raises(GLSLSwizzleError):
            validate_swizzle(VEC3, "")

        # Mixed component sets
        with pytest.raises(GLSLSwizzleError):
            validate_swizzle(VEC4, "xr")

        # Invalid components
        with pytest.raises(GLSLSwizzleError):
            validate_swizzle(VEC3, "w")  # w not in vec3

        # Too many components
        with pytest.raises(GLSLSwizzleError):
            validate_swizzle(VEC4, "xyzwx")

    def test_repeated_components(self):
        """Test swizzles with repeated components."""
        # All valid
        assert validate_swizzle(VEC4, "xxx") == VEC3
        assert validate_swizzle(VEC3, "xxy") == VEC3
        assert validate_swizzle(VEC2, "xx") == VEC2
        assert validate_swizzle(VEC4, "rrrr") == VEC4


class TestOperationValidation:
    """Test operation validation rules."""

    def test_arithmetic_operations(self):
        """Test arithmetic operation validation."""
        # Numeric scalar operations
        assert validate_operation(FLOAT, "+", FLOAT) == FLOAT
        assert validate_operation(INT, "*", FLOAT) == FLOAT
        assert validate_operation(FLOAT, "/", INT) == FLOAT

        # Vector operations
        assert validate_operation(VEC3, "+", VEC3) == VEC3
        assert validate_operation(VEC3, "*", FLOAT) == VEC3
        assert validate_operation(FLOAT, "*", VEC3) == VEC3

        # Matrix operations
        assert validate_operation(MAT3, "*", MAT3) == MAT3
        assert validate_operation(MAT3, "*", VEC3) == VEC3
        assert validate_operation(MAT3, "*", FLOAT) == MAT3

    def test_logical_operations(self):
        """Test logical operation validation."""
        # Boolean operations
        assert validate_operation(BOOL, "&&", BOOL) == BOOL
        assert validate_operation(BOOL, "||", BOOL) == BOOL

        # Boolean vector operations
        assert validate_operation(BVEC2, "&&", BVEC2) == BVEC2
        assert validate_operation(BVEC3, "||", BVEC3) == BVEC3

        # Invalid logical operations
        assert validate_operation(FLOAT, "&&", FLOAT) is None
        assert validate_operation(VEC3, "||", VEC3) is None

    def test_comparison_operations(self):
        """Test comparison operation validation."""
        # Scalar comparisons
        assert validate_operation(FLOAT, "<", FLOAT) == BOOL
        assert validate_operation(INT, ">=", INT) == BOOL
        assert validate_operation(INT, "==", FLOAT) == BOOL

        # Vector comparisons
        assert validate_operation(VEC3, "==", VEC3) == BOOL
        assert validate_operation(IVEC2, "!=", IVEC2) == BOOL

        # Invalid comparisons
        assert validate_operation(VEC2, "<", VEC3) is None
        assert validate_operation(MAT2, ">", MAT3) is None

    def test_matrix_operations(self):
        """Test matrix operation validation."""
        # Matrix-matrix multiplication
        assert validate_operation(MAT3, "*", MAT3) == MAT3
        assert validate_operation(MAT4, "*", MAT4) == MAT4

        # Matrix-vector multiplication
        assert validate_operation(MAT3, "*", VEC3) == VEC3
        assert validate_operation(MAT4, "*", VEC4) == VEC4

        # Invalid matrix operations
        assert validate_operation(MAT3, "*", VEC4) is None
        assert validate_operation(MAT3, "*", MAT4) is None


class TestTypeCompatibility:
    """Test type compatibility rules."""

    def test_basic_compatibility(self):
        """Test basic type compatibility."""
        # Same type compatibility
        assert is_compatible_with(FLOAT, FLOAT)
        assert is_compatible_with(VEC3, VEC3)
        assert is_compatible_with(MAT4, MAT4)

        # Numeric compatibility
        assert is_compatible_with(INT, FLOAT)
        assert is_compatible_with(FLOAT, INT)

        # Vector-scalar compatibility
        assert is_compatible_with(VEC3, FLOAT)
        assert is_compatible_with(FLOAT, VEC3)

    def test_vector_compatibility(self):
        """Test vector type compatibility."""
        # Same size vectors
        assert is_compatible_with(VEC2, VEC2)
        assert is_compatible_with(VEC3, VEC3)
        assert is_compatible_with(VEC4, VEC4)

        # Different size vectors
        assert not is_compatible_with(VEC2, VEC3)
        assert not is_compatible_with(VEC3, VEC4)

        # Different vector types
        assert not is_compatible_with(VEC2, IVEC2)
        assert not is_compatible_with(VEC3, BVEC3)

    def test_matrix_compatibility(self):
        """Test matrix type compatibility."""
        # Same size matrices
        assert is_compatible_with(MAT2, MAT2)
        assert is_compatible_with(MAT3, MAT3)
        assert is_compatible_with(MAT4, MAT4)

        # Different size matrices
        assert not is_compatible_with(MAT2, MAT3)
        assert not is_compatible_with(MAT3, MAT4)

        # Matrix-scalar compatibility
        assert is_compatible_with(MAT3, FLOAT)
        assert is_compatible_with(FLOAT, MAT3)


class TestTypeConversion:
    """Test type conversion rules."""

    def test_basic_conversions(self):
        """Test basic type conversion rules."""
        # Same type
        assert can_convert_to(FLOAT, FLOAT)
        assert can_convert_to(VEC3, VEC3)
        assert can_convert_to(MAT4, MAT4)

        # Int to float
        assert can_convert_to(INT, FLOAT)
        assert not can_convert_to(FLOAT, INT)

        # Vector conversions
        assert can_convert_to(VEC2, VEC2)
        assert not can_convert_to(VEC2, VEC3)

    def test_vector_conversions(self):
        """Test vector type conversions."""
        # Same size vectors
        assert can_convert_to(IVEC2, VEC2)
        assert can_convert_to(IVEC3, VEC3)
        assert can_convert_to(IVEC4, VEC4)

        # Different size vectors
        assert not can_convert_to(VEC2, VEC3)
        assert not can_convert_to(VEC3, VEC4)

        # Boolean vectors
        assert not can_convert_to(BVEC2, VEC2)
        assert not can_convert_to(VEC3, BVEC3)

    def test_invalid_conversions(self):
        """Test invalid type conversions."""
        # Void type
        assert not can_convert_to(VOID, FLOAT)
        assert not can_convert_to(FLOAT, VOID)

        # Matrix conversions
        assert not can_convert_to(MAT3, MAT4)
        assert not can_convert_to(MAT2, VEC2)

        # Array types
        array_float = GLSLType(TypeKind.FLOAT, array_size=3)
        assert not can_convert_to(array_float, FLOAT)
        assert not can_convert_to(FLOAT, array_float)


class TestCommonTypeResolution:
    """Test common type resolution rules."""

    def test_basic_common_types(self):
        """Test basic common type resolution."""
        # Same type
        assert common_type(FLOAT, FLOAT) == FLOAT
        assert common_type(VEC3, VEC3) == VEC3
        assert common_type(MAT4, MAT4) == MAT4

        # Numeric promotion
        assert common_type(INT, FLOAT) == FLOAT
        assert common_type(FLOAT, INT) == FLOAT

    def test_vector_common_types(self):
        """Test vector common type resolution."""
        # Same size vectors
        assert common_type(VEC2, VEC2) == VEC2
        assert common_type(VEC3, VEC3) == VEC3
        assert common_type(VEC4, VEC4) == VEC4

        # Vector-scalar
        assert common_type(VEC3, FLOAT) == VEC3
        assert common_type(FLOAT, VEC3) == VEC3

        # Different vectors
        assert common_type(VEC2, VEC3) is None
        assert common_type(IVEC2, VEC2) is None

    def test_matrix_common_types(self):
        """Test matrix common type resolution."""
        # Same size matrices
        assert common_type(MAT2, MAT2) == MAT2
        assert common_type(MAT3, MAT3) == MAT3
        assert common_type(MAT4, MAT4) == MAT4

        # Matrix-scalar
        assert common_type(MAT3, FLOAT) == MAT3
        assert common_type(FLOAT, MAT3) == MAT3

        # Different matrices
        assert common_type(MAT2, MAT3) is None
        assert common_type(MAT3, MAT4) is None


class TestValidationEdgeCases:
    """Test validation edge cases."""

    def test_void_type_validation(self):
        """Test void type validation rules."""
        # Void operations
        assert validate_operation(VOID, "+", FLOAT) is None
        assert validate_operation(FLOAT, "*", VOID) is None
        assert validate_operation(VOID, "==", VOID) is None

        # Void compatibility
        assert not is_compatible_with(VOID, FLOAT)
        assert not is_compatible_with(FLOAT, VOID)
        assert not is_compatible_with(VOID, VOID)

        # Void conversion
        assert not can_convert_to(VOID, FLOAT)
        assert not can_convert_to(FLOAT, VOID)

    def test_array_type_validation(self):
        """Test array type validation rules."""
        array1 = GLSLType(TypeKind.FLOAT, array_size=3)
        array2 = GLSLType(TypeKind.FLOAT, array_size=3)
        array3 = GLSLType(TypeKind.FLOAT, array_size=4)

        # Array compatibility
        assert is_compatible_with(array1, array2)
        assert not is_compatible_with(array1, array3)

        # Array operations
        assert validate_operation(array1, "+", array2) == array1
        assert validate_operation(array1, "*", FLOAT) == array1
        assert validate_operation(array1, "+", array3) is None

    def test_mixed_vector_operations(self):
        """Test mixed vector operation validation."""
        # Vector-scalar operations
        assert validate_operation(VEC3, "*", FLOAT) == VEC3
        assert validate_operation(INT, "*", VEC3) == VEC3

        # Invalid vector combinations
        assert validate_operation(VEC2, "+", VEC3) is None
        assert validate_operation(VEC3, "*", IVEC3) is None
        assert validate_operation(VEC4, "/", BVEC4) is None

    def test_boolean_vector_operations(self):
        """Test boolean vector operation validation."""
        # Valid operations
        assert validate_operation(BVEC2, "&&", BVEC2) == BVEC2
        assert validate_operation(BVEC3, "||", BVEC3) == BVEC3

        # Invalid operations
        assert validate_operation(BVEC2, "+", BVEC2) is None
        assert validate_operation(BVEC3, "*", FLOAT) is None
        assert validate_operation(BVEC4, "<", BVEC4) is None

