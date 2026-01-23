"""Tests for type_utils module."""

from py2glsl.transpiler.ir import IRType
from py2glsl.transpiler.type_utils import (
    get_indexable_size,
    infer_literal_type,
    is_swizzle,
    parse_type_string,
    subscript_result_type,
    swizzle_result_type,
)


class TestParseTypeString:
    """Tests for parse_type_string function."""

    def test_simple_type(self) -> None:
        """Test parsing simple types."""
        result = parse_type_string("float")
        assert result.base == "float"
        assert result.array_size is None

    def test_array_type_with_size(self) -> None:
        """Test parsing array types with explicit size."""
        result = parse_type_string("vec3[4]")
        assert result.base == "vec3"
        assert result.array_size == 4

    def test_array_type_without_size(self) -> None:
        """Test parsing array types without size (infer later)."""
        result = parse_type_string("float[]")
        assert result.base == "float"
        assert result.array_size == -1

    def test_vector_type(self) -> None:
        """Test parsing vector types."""
        result = parse_type_string("vec4")
        assert result.base == "vec4"
        assert result.array_size is None

    def test_matrix_type(self) -> None:
        """Test parsing matrix types."""
        result = parse_type_string("mat3")
        assert result.base == "mat3"
        assert result.array_size is None


class TestInferLiteralType:
    """Tests for infer_literal_type function."""

    def test_bool_literal(self) -> None:
        """Test inferring type from bool literal."""
        assert infer_literal_type(True).base == "bool"
        assert infer_literal_type(False).base == "bool"

    def test_int_literal(self) -> None:
        """Test inferring type from int literal."""
        assert infer_literal_type(42).base == "int"
        assert infer_literal_type(0).base == "int"
        assert infer_literal_type(-1).base == "int"

    def test_float_literal(self) -> None:
        """Test inferring type from float literal."""
        assert infer_literal_type(3.14).base == "float"
        assert infer_literal_type(0.0).base == "float"

    def test_other_literal(self) -> None:
        """Test inferring type from other literals defaults to float."""
        assert infer_literal_type("string").base == "float"


class TestIsSwizzle:
    """Tests for is_swizzle function."""

    def test_single_component_xyzw(self) -> None:
        """Test single component swizzles with xyzw."""
        assert is_swizzle("x") is True
        assert is_swizzle("y") is True
        assert is_swizzle("z") is True
        assert is_swizzle("w") is True

    def test_single_component_rgba(self) -> None:
        """Test single component swizzles with rgba."""
        assert is_swizzle("r") is True
        assert is_swizzle("g") is True
        assert is_swizzle("b") is True
        assert is_swizzle("a") is True

    def test_multi_component_swizzle(self) -> None:
        """Test multi-component swizzles."""
        assert is_swizzle("xy") is True
        assert is_swizzle("xyz") is True
        assert is_swizzle("xyzw") is True
        assert is_swizzle("rgb") is True
        assert is_swizzle("rgba") is True

    def test_invalid_swizzle_characters(self) -> None:
        """Test invalid swizzle characters."""
        assert is_swizzle("q") is False
        assert is_swizzle("xq") is False

    def test_too_long_swizzle(self) -> None:
        """Test swizzle longer than 4 components."""
        assert is_swizzle("xyzwx") is False

    def test_empty_swizzle(self) -> None:
        """Test empty swizzle."""
        assert is_swizzle("") is False


class TestSwizzleResultType:
    """Tests for swizzle_result_type function."""

    def test_single_component_returns_float(self) -> None:
        """Test single component swizzle returns float."""
        assert swizzle_result_type("x").base == "float"
        assert swizzle_result_type("r").base == "float"

    def test_two_components_returns_vec2(self) -> None:
        """Test two component swizzle returns vec2."""
        assert swizzle_result_type("xy").base == "vec2"
        assert swizzle_result_type("rg").base == "vec2"

    def test_three_components_returns_vec3(self) -> None:
        """Test three component swizzle returns vec3."""
        assert swizzle_result_type("xyz").base == "vec3"
        assert swizzle_result_type("rgb").base == "vec3"

    def test_four_components_returns_vec4(self) -> None:
        """Test four component swizzle returns vec4."""
        assert swizzle_result_type("xyzw").base == "vec4"
        assert swizzle_result_type("rgba").base == "vec4"


class TestSubscriptResultType:
    """Tests for subscript_result_type function."""

    def test_array_subscript_returns_element_type(self) -> None:
        """Test array subscript returns element type."""
        array_type = IRType(base="vec3", array_size=4)
        assert subscript_result_type(array_type).base == "vec3"

    def test_vec_subscript_returns_float(self) -> None:
        """Test vec subscript returns float."""
        assert subscript_result_type(IRType("vec2")).base == "float"
        assert subscript_result_type(IRType("vec3")).base == "float"
        assert subscript_result_type(IRType("vec4")).base == "float"

    def test_ivec_subscript_returns_int(self) -> None:
        """Test ivec subscript returns int."""
        assert subscript_result_type(IRType("ivec2")).base == "int"
        assert subscript_result_type(IRType("ivec3")).base == "int"
        assert subscript_result_type(IRType("ivec4")).base == "int"

    def test_uvec_subscript_returns_uint(self) -> None:
        """Test uvec subscript returns uint."""
        assert subscript_result_type(IRType("uvec2")).base == "uint"
        assert subscript_result_type(IRType("uvec3")).base == "uint"

    def test_bvec_subscript_returns_bool(self) -> None:
        """Test bvec subscript returns bool."""
        assert subscript_result_type(IRType("bvec2")).base == "bool"
        assert subscript_result_type(IRType("bvec3")).base == "bool"

    def test_mat_subscript_returns_vec(self) -> None:
        """Test matrix subscript returns vector."""
        assert subscript_result_type(IRType("mat2")).base == "vec2"
        assert subscript_result_type(IRType("mat3")).base == "vec3"
        assert subscript_result_type(IRType("mat4")).base == "vec4"

    def test_unknown_type_returns_float(self) -> None:
        """Test unknown type subscript returns float."""
        assert subscript_result_type(IRType("sampler2D")).base == "float"


class TestGetIndexableSize:
    """Tests for get_indexable_size function."""

    def test_array_size(self) -> None:
        """Test getting size from array type."""
        assert get_indexable_size(IRType(base="float", array_size=5)) == 5
        assert get_indexable_size(IRType(base="vec3", array_size=10)) == 10

    def test_vec_sizes(self) -> None:
        """Test getting size from vector types."""
        assert get_indexable_size(IRType("vec2")) == 2
        assert get_indexable_size(IRType("vec3")) == 3
        assert get_indexable_size(IRType("vec4")) == 4

    def test_ivec_sizes(self) -> None:
        """Test getting size from integer vector types."""
        assert get_indexable_size(IRType("ivec2")) == 2
        assert get_indexable_size(IRType("ivec3")) == 3
        assert get_indexable_size(IRType("ivec4")) == 4

    def test_uvec_sizes(self) -> None:
        """Test getting size from unsigned vector types."""
        assert get_indexable_size(IRType("uvec2")) == 2
        assert get_indexable_size(IRType("uvec3")) == 3
        assert get_indexable_size(IRType("uvec4")) == 4

    def test_bvec_sizes(self) -> None:
        """Test getting size from boolean vector types."""
        assert get_indexable_size(IRType("bvec2")) == 2
        assert get_indexable_size(IRType("bvec3")) == 3
        assert get_indexable_size(IRType("bvec4")) == 4

    def test_non_indexable_returns_none(self) -> None:
        """Test non-indexable types return None."""
        assert get_indexable_size(IRType("float")) is None
        assert get_indexable_size(IRType("int")) is None
        assert get_indexable_size(IRType("mat3")) is None
