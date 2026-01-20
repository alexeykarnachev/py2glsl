"""Tests for runtime behavior of builtin types (vec2, vec3, vec4, mat2, mat3, mat4)."""

from py2glsl.builtins import mat2, mat3, mat4, vec2, vec3, vec4


class TestVectorIndexing:
    """Test vector indexing with __getitem__."""

    def test_vec2_indexing(self):
        """Test vec2 indexing."""
        v = vec2(1.0, 2.0)
        assert v[0] == 1.0
        assert v[1] == 2.0

    def test_vec3_indexing(self):
        """Test vec3 indexing."""
        v = vec3(1.0, 2.0, 3.0)
        assert v[0] == 1.0
        assert v[1] == 2.0
        assert v[2] == 3.0

    def test_vec4_indexing(self):
        """Test vec4 indexing."""
        v = vec4(1.0, 2.0, 3.0, 4.0)
        assert v[0] == 1.0
        assert v[1] == 2.0
        assert v[2] == 3.0
        assert v[3] == 4.0


class TestMatrixIndexing:
    """Test matrix indexing with __getitem__."""

    def test_mat2_indexing(self):
        """Test mat2 indexing returns vec2."""
        m = mat2(1.0, 2.0, 3.0, 4.0)
        row0 = m[0]
        row1 = m[1]

        assert isinstance(row0, vec2)
        assert isinstance(row1, vec2)
        assert row0[0] == 1.0
        assert row0[1] == 2.0
        assert row1[0] == 3.0
        assert row1[1] == 4.0

    def test_mat3_indexing(self):
        """Test mat3 indexing returns vec3."""
        m = mat3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
        row0 = m[0]
        row1 = m[1]
        row2 = m[2]

        assert isinstance(row0, vec3)
        assert isinstance(row1, vec3)
        assert isinstance(row2, vec3)
        assert row0[0] == 1.0
        assert row0[1] == 2.0
        assert row0[2] == 3.0
        assert row2[2] == 9.0

    def test_mat4_indexing(self):
        """Test mat4 indexing returns vec4."""
        m = mat4(
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
        )
        row0 = m[0]
        row3 = m[3]

        assert isinstance(row0, vec4)
        assert isinstance(row3, vec4)
        assert row0[0] == 1.0
        assert row0[3] == 4.0
        assert row3[0] == 13.0
        assert row3[3] == 16.0

    def test_nested_matrix_indexing(self):
        """Test nested indexing: matrix[i][j]."""
        m = mat3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)

        # Access individual elements
        assert m[0][0] == 1.0
        assert m[0][1] == 2.0
        assert m[1][0] == 4.0
        assert m[2][2] == 9.0


class TestVectorOperators:
    """Test vector operators (__neg__, __mul__, __add__)."""

    def test_vec3_negation(self):
        """Test vec3 unary negation."""
        v = vec3(1.0, 2.0, 3.0)
        neg_v = -v

        assert isinstance(neg_v, vec3)
        assert neg_v.x == -1.0
        assert neg_v.y == -2.0
        assert neg_v.z == -3.0

    def test_vec3_element_wise_multiplication(self):
        """Test vec3 element-wise multiplication."""
        v1 = vec3(2.0, 3.0, 4.0)
        v2 = vec3(5.0, 6.0, 7.0)
        result = v1 * v2

        assert isinstance(result, vec3)
        assert result.x == 10.0  # 2.0 * 5.0
        assert result.y == 18.0  # 3.0 * 6.0
        assert result.z == 28.0  # 4.0 * 7.0

    def test_vec3_scalar_multiplication(self):
        """Test vec3 scalar multiplication still works."""
        v = vec3(2.0, 3.0, 4.0)
        result = v * 2.0

        assert isinstance(result, vec3)
        assert result.x == 4.0
        assert result.y == 6.0
        assert result.z == 8.0

    def test_vec3_addition(self):
        """Test vec3 addition."""
        v1 = vec3(1.0, 2.0, 3.0)
        v2 = vec3(4.0, 5.0, 6.0)
        result = v1 + v2

        assert isinstance(result, vec3)
        assert result.x == 5.0
        assert result.y == 7.0
        assert result.z == 9.0


class TestMatrixConstructors:
    """Test matrix constructors."""

    def test_mat2_full_constructor(self):
        """Test mat2 with all 4 elements."""
        m = mat2(1.0, 2.0, 3.0, 4.0)
        assert m.data[0][0] == 1.0
        assert m.data[0][1] == 2.0
        assert m.data[1][0] == 3.0
        assert m.data[1][1] == 4.0

    def test_mat2_identity_constructor(self):
        """Test mat2 with single value creates identity-like matrix."""
        m = mat2(1.0)
        # Identity matrix has 1s on diagonal, 0s elsewhere
        assert m.data[0][0] == 1.0
        assert m.data[0][1] == 0.0
        assert m.data[1][0] == 0.0
        assert m.data[1][1] == 1.0

    def test_mat3_full_constructor(self):
        """Test mat3 with all 9 elements."""
        m = mat3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
        assert m.data[0][0] == 1.0
        assert m.data[2][2] == 9.0

    def test_mat3_identity_constructor(self):
        """Test mat3 identity constructor."""
        m = mat3(1.0)
        assert m.data[0][0] == 1.0
        assert m.data[1][1] == 1.0
        assert m.data[2][2] == 1.0
        assert m.data[0][1] == 0.0
        assert m.data[1][0] == 0.0

    def test_mat4_full_constructor(self):
        """Test mat4 with all 16 elements."""
        m = mat4(
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
        )
        assert m.data[0][0] == 1.0
        assert m.data[3][3] == 16.0

    def test_mat4_identity_constructor(self):
        """Test mat4 identity constructor."""
        m = mat4(1.0)
        assert m.data[0][0] == 1.0
        assert m.data[1][1] == 1.0
        assert m.data[2][2] == 1.0
        assert m.data[3][3] == 1.0
        assert m.data[0][1] == 0.0
        assert m.data[1][0] == 0.0
