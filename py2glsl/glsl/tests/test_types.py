import numpy as np
import pytest

from py2glsl.glsl.types import mat3, mat4, vec2, vec3, vec4


@pytest.mark.parametrize("vec_type,components", [(vec2, 2), (vec3, 3), (vec4, 4)])
class TestVectorTypes:
    def test_initialization(self, vec_type, components):
        # Test scalar initialization
        v_scalar = vec_type(2.0)
        assert np.allclose(v_scalar.data, np.full(components, 2.0))

        # Test list initialization
        v_list = vec_type([i + 1 for i in range(components)])
        assert np.allclose(v_list.data, np.arange(1, components + 1))

        # Test tuple initialization
        v_tuple = vec_type(*(i + 1 for i in range(components)))
        assert np.allclose(v_tuple.data, np.arange(1, components + 1))

    def test_arithmetic(self, vec_type, components):
        a = vec_type(2.0)
        b = vec_type(3.0)

        # Vector-vector operations
        assert a + b == vec_type(5.0)
        assert b - a == vec_type(1.0)
        assert a * b == vec_type(6.0)
        assert b / a == vec_type(1.5)

        # Vector-scalar operations
        assert a + 1.0 == vec_type(3.0)
        assert 2.0 * b == vec_type(6.0)
        assert b - 0.5 == vec_type(2.5)
        assert a / 2.0 == vec_type(1.0)

    def test_swizzling(self, vec_type, components):
        values = [float(i + 1) for i in range(components)]
        v = vec_type(*values)

        # Component access
        assert v.x == 1.0
        if components >= 2:
            assert v.y == 2.0
        if components >= 3:
            assert v.z == 3.0
        if components >= 4:
            assert v.w == 4.0

        # Swizzle combinations
        if components >= 2:
            assert v.xy == vec2(1.0, 2.0)
        if components >= 3:
            assert v.zyx == vec3(3.0, 2.0, 1.0)
        if components >= 4:
            assert v.wzyx == vec4(4.0, 3.0, 2.0, 1.0)

    def test_swizzle_errors(self, vec_type, components):
        v = vec_type(*[1.0] * components)

        # Check component access errors based on vector size
        if components < 4:
            with pytest.raises(AttributeError):
                _ = v.w
        if components < 3:
            with pytest.raises(AttributeError):
                _ = v.z

        # Check invalid swizzle patterns
        if components < 3:
            with pytest.raises(AttributeError):
                _ = v.xyzx  # Contains invalid 'z' component
        elif components == 3:
            with pytest.raises(AttributeError):
                _ = v.xyzw  # Contains invalid 'w' component

        # Invalid component name should always fail
        with pytest.raises(AttributeError):
            _ = v.foo

    def test_type_preservation(self, vec_type, components):
        v = vec_type(2.0)
        result = v * 3.0 + vec_type(1.0)
        assert isinstance(result, vec_type)
        assert np.allclose(result.data, np.full(components, 7.0))

    def test_comparison(self, vec_type, components):
        a = vec_type(2.0)
        b = vec_type(2.0)
        c = vec_type(3.0)

        assert a == b
        assert a != c


class TestMatrixTypes:
    @pytest.mark.parametrize("mat_type,dim", [(mat3, 3), (mat4, 4)])
    def test_matrix_operations(self, mat_type, dim):
        # Test identity matrix
        eye = mat_type(np.eye(dim))
        assert np.allclose(eye.data, np.eye(dim))

        # Test matrix multiplication
        m = mat_type(np.arange(dim * dim, dtype=np.float32).reshape(dim, dim))
        result = m @ eye
        assert np.allclose(result.data, m.data)

        # Test transpose
        assert np.allclose(m.transpose().data, m.data.T)

    def test_matrix_vector_mult(self):
        # mat4 x vec4
        m = mat4(np.eye(4))
        v = vec4(1.0, 2.0, 3.0, 4.0)
        result = m @ v
        assert isinstance(result, vec4)
        assert np.allclose(result.data, [1.0, 2.0, 3.0, 4.0])

        # mat3 x vec3
        m = mat3(np.eye(3))
        v = vec3(1.0, 2.0, 3.0)
        result = m @ v
        assert isinstance(result, vec3)
        assert np.allclose(result.data, [1.0, 2.0, 3.0])

    def test_matrix_errors(self):
        with pytest.raises(ValueError):
            mat3([1, 2])  # Not enough elements

        with pytest.raises(TypeError):
            m = mat4(np.eye(4))
            v = vec3(1.0, 2.0, 3.0)
            _ = m @ v  # Dimension mismatch


def test_repr():
    # Test vector representation
    v = vec3(1.5, 2.0, 3.5)
    assert repr(v) == "vec3(1.500, 2.000, 3.500)"

    # Test matrix representation
    m = mat4(np.eye(4))
    assert "mat4" in repr(m)
    assert "1.000" in repr(m)
    assert "0.000" in repr(m)


def test_numpy_interop():
    # Test conversion from/to numpy arrays
    arr = np.array([1.0, 2.0, 3.0])
    v = vec3(arr)
    assert np.allclose(v.data, arr)

    arr_back = v.to_array()
    assert isinstance(arr_back, np.ndarray)
    assert np.allclose(arr_back, arr)
