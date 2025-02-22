import ast
from textwrap import dedent

import pytest

from py2glsl.transpiler.type_system import GLSLTypeError, TypeInferer, TypeInfo


def parse(source: str) -> ast.Module:
    return ast.parse(dedent(source.strip()))


def test_scalar_types():
    code = parse(
        """
    def test(a: float, b: int) -> float:
        return a + b
    """
    )

    inferer = TypeInferer()
    inferer.visit(code)

    assert inferer.symbols["a"] == TypeInfo.FLOAT
    assert inferer.symbols["b"] == TypeInfo.FLOAT
    assert inferer.current_return == TypeInfo.FLOAT


def test_vector_ops():
    code = parse(
        """
    def test(v: vec3) -> vec3:
        return v * 2.0
    """
    )

    inferer = TypeInferer()
    inferer.visit(code)

    assert inferer.current_return == TypeInfo.VEC3


def test_matrix_mult():
    code = parse(
        """
    def test(m1: mat3, m2: mat3) -> mat3:
        return m1 @ m2
    """
    )

    inferer = TypeInferer()
    inferer.visit(code)

    assert inferer.current_return == TypeInfo.MAT3


def test_type_mismatch():
    code = parse(
        """
    def test():
        a: vec2 = vec2(1.0)
        return a * vec3(1.0)
    """
    )

    inferer = TypeInferer()
    with pytest.raises(GLSLTypeError) as exc:
        inferer.visit(code)

    # Update assertion to match new error message
    assert "Vector size mismatch for component-wise multiply: vec2 vs vec3" in str(
        exc.value
    )


def test_vector_constructor():
    code = parse(
        """
    def test() -> vec3:
        return vec3(1.0, 0.0, 0.5)
    """
    )

    inferer = TypeInferer()
    inferer.visit(code)

    assert inferer.current_return == TypeInfo.VEC3


def test_swizzle_ops():
    code = parse(
        """
    def test(v: vec4) -> vec3:
        return v.xyz
    """
    )

    inferer = TypeInferer()
    inferer.visit(code)

    assert inferer.current_return == TypeInfo.VEC3


def test_bool_ops():
    code = parse(
        """
    def test(a: bool) -> bool:
        return a and (1.0 < 2.0)
    """
    )

    inferer = TypeInferer()
    inferer.visit(code)

    assert inferer.current_return == TypeInfo.BOOL


def test_scalar_promotion():
    code = parse(
        """
    def test() -> float:
        return 5 * 3.14
    """
    )

    inferer = TypeInferer()
    inferer.visit(code)

    assert inferer.current_return == TypeInfo.FLOAT


def test_invalid_matrix():
    code = parse(
        """
    def test(m: mat4) -> mat3:
        return m
    """
    )

    inferer = TypeInferer()
    with pytest.raises(GLSLTypeError) as exc:
        inferer.visit(code)

    assert "Return type mismatch: mat4 vs mat3" in str(exc.value)


def test_mixed_vector_ops():
    code = parse(
        """
    def test(v: vec2) -> vec2:
        return 2.0 * v + vec2(0.5, 1.0)
    """
    )

    inferer = TypeInferer()
    inferer.visit(code)

    assert inferer.current_return == TypeInfo.VEC2


def test_nested_swizzle():
    code = parse(
        """
    def test(v: vec4) -> vec2:
        return v.xyz.yx
    """
    )

    inferer = TypeInferer()
    inferer.visit(code)

    assert inferer.current_return == TypeInfo.VEC2


def test_invalid_swizzle():
    code = parse(
        """
    def test(v: vec3) -> vec4:
        return v.xyzw
    """
    )

    inferer = TypeInferer()
    with pytest.raises(GLSLTypeError) as exc:
        inferer.visit(code)

    assert "Swizzle component 'w' out of bounds for vec3" in str(exc.value)


def test_bool_comparison():
    code = parse(
        """
    def test(a: float) -> bool:
        return (a > 0.5) and (vec2(a) == vec2(1.0))
    """
    )
    inferer = TypeInferer()
    inferer.visit(code)
    assert inferer.current_return == TypeInfo.BOOL


def test_matrix_vector_mult():
    code = parse(
        """
    def test(m: mat3, v: vec3) -> vec3:
        return m @ v
    """
    )

    inferer = TypeInferer()
    inferer.visit(code)

    assert inferer.current_return == TypeInfo.VEC3


def test_invalid_matrix_mult():
    code = parse(
        """
    def test(m: mat3, v: vec4) -> vec3:
        return m @ v
    """
    )

    inferer = TypeInferer()
    with pytest.raises(GLSLTypeError) as exc:
        inferer.visit(code)

    assert "Matrix(cols=3) and vector(dim=4) dimension mismatch" in str(exc.value)


def test_unary_ops():
    code = parse(
        """
    def test(a: float) -> float:
        return -a
    """
    )

    inferer = TypeInferer()
    inferer.visit(code)

    assert inferer.current_return == TypeInfo.FLOAT


def test_conditional_expr():
    code = parse(
        """
    def test(a: float) -> vec3:
        return vec3(a) if a > 0.5 else vec3(0.0)
    """
    )

    inferer = TypeInferer()
    inferer.visit(code)

    assert inferer.current_return == TypeInfo.VEC3


def test_array_subscript():
    code = parse(
        """
    def test(v: vec4) -> float:
        return v[0]
    """
    )

    inferer = TypeInferer()
    inferer.visit(code)

    assert inferer.current_return == TypeInfo.FLOAT


def test_invalid_subscript():
    code = parse(
        """
    def test(v: float) -> float:
        return v[0]
    """
    )

    inferer = TypeInferer()
    with pytest.raises(GLSLTypeError) as exc:
        inferer.visit(code)

    assert "Subscript on non-vector/matrix type" in str(exc.value)


def test_function_parameters():
    code = parse(
        """
    def test(vs_uv: vec2, u_time: float, u_res: vec2) -> vec4:
        return vec4(vs_uv * u_time * u_res, 0.0, 1.0)
    """
    )

    inferer = TypeInferer()
    inferer.visit(code)

    assert inferer.symbols["vs_uv"] == TypeInfo.VEC2
    assert inferer.symbols["u_time"] == TypeInfo.FLOAT
    assert inferer.symbols["u_res"] == TypeInfo.VEC2
    assert inferer.current_return == TypeInfo.VEC4


class TestMatrixVectorOperations:
    """Tests for matrix and vector type interactions"""

    @pytest.mark.parametrize("size", [3, 4])
    def test_valid_matrix_matrix_mult(self, size):
        code = parse(
            f"""
        def test(m1: mat{size}, m2: mat{size}) -> mat{size}:
            return m1 @ m2
        """
        )

        inferer = TypeInferer()
        inferer.visit(code)
        assert inferer.current_return == getattr(TypeInfo, f"MAT{size}")

    @pytest.mark.parametrize("size", [3, 4])
    def test_valid_matrix_vector_mult(self, size):
        code = parse(
            f"""
        def test(m: mat{size}, v: vec{size}) -> vec{size}:
            return m @ v
        """
        )

        inferer = TypeInferer()
        inferer.visit(code)
        assert inferer.current_return == getattr(TypeInfo, f"VEC{size}")

    def test_invalid_matrix_vector_dimension(self):
        code = parse(
            """
        def test(m: mat3, v: vec4) -> vec3:
            return m @ v
        """
        )

        inferer = TypeInferer()
        with pytest.raises(GLSLTypeError) as exc:
            inferer.visit(code)

        assert "Matrix(cols=3) and vector(dim=4) dimension mismatch" in str(exc.value)

    @pytest.mark.parametrize("op", ["*", "+", "-", "/"])
    def test_vector_component_wise_ops(self, op):
        code = parse(
            f"""
        def test(a: vec3, b: vec3) -> vec3:
            return a {op} b
        """
        )

        inferer = TypeInferer()
        inferer.visit(code)
        assert inferer.current_return == TypeInfo.VEC3

    def test_vector_scalar_mult(self):
        code = parse(
            """
        def test(v: vec2, s: float) -> vec2:
            return v * s
        """
        )

        inferer = TypeInferer()
        inferer.visit(code)
        assert inferer.current_return == TypeInfo.VEC2

    def test_scalar_matrix_mult(self):
        code = parse(
            """
        def test(s: float, m: mat4) -> mat4:
            return s * m
        """
        )

        inferer = TypeInferer()
        inferer.visit(code)
        assert inferer.current_return == TypeInfo.MAT4

    def test_int_float_promotion(self):
        code = parse(
            """
        def test(a: int, b: float) -> float:
            return a * b
        """
        )

        inferer = TypeInferer()
        inferer.visit(code)
        assert inferer.current_return == TypeInfo.FLOAT

    @pytest.mark.parametrize(
        "vec_type,other_type", [("vec2", "vec3"), ("vec3", "vec4"), ("vec4", "vec2")]
    )
    def test_invalid_component_wise_mismatch(self, vec_type, other_type):
        code = parse(
            f"""
        def test(a: {vec_type}, b: {other_type}) -> {vec_type}:
            return a * b
        """
        )

        inferer = TypeInferer()
        with pytest.raises(GLSLTypeError) as exc:
            inferer.visit(code)

        assert "Vector size mismatch for component-wise multiply" in str(exc.value)

    def test_matrix_matrix_dimension_mismatch(self):
        code = parse(
            """
        def test(m1: mat3, m2: mat4) -> mat3:
            return m1 @ m2
        """
        )

        inferer = TypeInferer()
        with pytest.raises(GLSLTypeError) as exc:
            inferer.visit(code)

        assert "Matrix multiplication dimension mismatch" in str(exc.value)

    def test_vector_matrix_mult_error(self):
        code = parse(
            """
        def test(v: vec4, m: mat4) -> vec4:
            return v @ m
        """
        )

        inferer = TypeInferer()
        with pytest.raises(GLSLTypeError) as exc:
            inferer.visit(code)

        assert "Vector-matrix multiplication is not supported" in str(exc.value)
