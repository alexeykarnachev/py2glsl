import ast
from textwrap import dedent

import pytest

from py2glsl.transpiler.type_system import GlslTypeError, TypeInferer, TypeInfo


def parse_code(source: str) -> ast.Module:
    return ast.parse(dedent(source.strip()))


def test_basic_types():
    code = parse_code(
        """
    def test() -> float:
        a: float = 3.14
        b = 2
        return a + b
    """
    )

    inferer = TypeInferer()
    inferer.visit(code)

    assert inferer.symbols["a"] == TypeInfo("float")
    assert inferer.symbols["b"] == TypeInfo("float")
    assert inferer.current_fn_return == TypeInfo("float")


def test_vector_annotation():
    code = parse_code(
        """
    def test(v: Vector[float, float]) -> vec2:
        return v
    """
    )

    inferer = TypeInferer()
    inferer.visit(code)

    assert inferer.symbols["v"] == TypeInfo("vec2", vector_size=2)
    assert inferer.current_fn_return == TypeInfo("vec2", vector_size=2)


def test_type_mismatch():
    code = parse_code(
        """
    def test():
        a: float = 3.14
        b: bool = True  # Now using incompatible bool type
        return a + b  # This should fail
    """
    )

    inferer = TypeInferer()
    with pytest.raises(GlslTypeError) as exc:
        inferer.visit(code)

    assert "Type mismatch in operation: float vs bool" in str(exc.value)


def test_matrix_operations():
    code = parse_code(
        """
    def transform(m: mat4, v: vec4) -> vec4:
        return m * v
    """
    )

    inferer = TypeInferer()
    inferer.visit(code)

    assert inferer.symbols["m"] == TypeInfo("mat4", matrix_dim=(4, 4))
    assert inferer.symbols["v"] == TypeInfo("vec4", vector_size=4)
    assert inferer.current_fn_return == TypeInfo("vec4", vector_size=4)


def test_missing_annotation():
    code = parse_code(
        """
    def test(a):
        return a * 2.0
    """
    )

    inferer = TypeInferer()
    with pytest.raises(GlslTypeError) as exc:
        inferer.visit(code)

    assert "requires type annotation" in str(exc.value)


def test_binary_promotion():
    code = parse_code(
        """
    def test() -> float:
        a = 3    # int literal
        b = 2.5  # float
        return a * b
    """
    )

    inferer = TypeInferer()
    inferer.visit(code)

    assert inferer.symbols["a"] == TypeInfo("float")
    assert inferer.symbols["b"] == TypeInfo("float")
    assert inferer.current_fn_return == TypeInfo("float")
