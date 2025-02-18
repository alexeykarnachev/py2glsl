import ast
from textwrap import dedent

import pytest

from py2glsl.transpiler.type_system import GlslTypeError, TypeInferer, TypeInfo


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
    with pytest.raises(GlslTypeError) as exc:
        inferer.visit(code)

    assert "Type mismatch: vec2 vs vec3" in str(exc.value)


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
    with pytest.raises(GlslTypeError) as exc:
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
    with pytest.raises(GlslTypeError) as exc:
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
    with pytest.raises(GlslTypeError) as exc:
        inferer.visit(code)

    assert "Matrix-vector dimension mismatch" in str(exc.value)


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
    with pytest.raises(GlslTypeError) as exc:
        inferer.visit(code)

    assert "Subscript on non-vector/matrix type" in str(exc.value)
