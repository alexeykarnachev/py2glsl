import pytest

from py2glsl.types.base import (
    BOOL,
    FLOAT,
    INT,
    MAT4,
    VEC2,
    VEC3,
    VEC4,
    FunctionSignature,
    GLSLType,
    TypeContext,
    TypeKind,
    TypePromotionRule,
)


class TestCoreTypes:
    def test_type_properties(self):
        assert FLOAT.is_numeric
        assert not FLOAT.is_vector
        assert VEC3.is_vector
        assert VEC3.size == 3
        assert VEC3.component_type == FLOAT

    def test_type_equality(self):
        assert FLOAT == GLSLType(TypeKind.FLOAT)
        assert VEC2 != VEC3
        assert VEC4.component_type == FLOAT


class TestTypePromotion:
    @pytest.fixture
    def promotion_rules(self):
        return [
            TypePromotionRule(INT, FLOAT, True),
            TypePromotionRule(FLOAT, VEC2, False),
        ]

    def test_implicit_promotion(self, promotion_rules):
        int_to_float = next(r for r in promotion_rules if r.source == INT)
        assert int_to_float.implicit
        assert int_to_float.target == FLOAT

    def test_explicit_promotion(self, promotion_rules):
        float_to_vec2 = next(r for r in promotion_rules if r.target == VEC2)
        assert not float_to_vec2.implicit


class TestFunctionResolution:
    @pytest.fixture
    def mix_overloads(self):
        return [
            FunctionSignature([FLOAT, FLOAT, FLOAT], FLOAT),
            FunctionSignature([VEC3, VEC3, FLOAT], VEC3),
            FunctionSignature([VEC2, VEC2, VEC2], VEC2),
        ]

    def test_resolve_scalar_mix(self, mix_overloads):
        args = [FLOAT, FLOAT, FLOAT]
        match = next(f for f in mix_overloads if f.parameters == args)
        assert match.return_type == FLOAT

    def test_resolve_vector_mix(self, mix_overloads):
        args = [VEC3, VEC3, FLOAT]
        match = next(f for f in mix_overloads if f.parameters == args)
        assert match.return_type == VEC3


class TestTypeContext:
    @pytest.fixture
    def ctx(self):
        return TypeContext(
            variables={"pos": VEC3},
            functions={"abs": [FunctionSignature([FLOAT], FLOAT)]},
            structs={},
        )

    def test_variable_shadowing(self, ctx):
        child = ctx.child_context()
        child.variables["pos"] = VEC4  # Shadow parent variable
        assert child.variables["pos"] == VEC4
        assert ctx.variables["pos"] == VEC3  # Parent unchanged

    def test_function_overloads(self, ctx):
        assert len(ctx.functions["abs"]) == 1
        ctx.functions["abs"].append(FunctionSignature([VEC2], VEC2))
        assert len(ctx.functions["abs"]) == 2


class TestErrorConditions:
    def test_invalid_assignment(self):
        ctx = TypeContext(variables={}, functions={}, structs={})
        ctx.variables["x"] = FLOAT
        with pytest.raises(TypeError):
            ctx.variables["x"] = INT  # Reassign different type

    def test_undefined_variable(self):
        ctx = TypeContext(variables={}, functions={}, structs={})
        with pytest.raises(KeyError):
            _ = ctx.variables["missing"]


class TestStructTypes:
    def test_struct_equality(self):
        point_type = GLSLType(TypeKind.STRUCT, members={"x": FLOAT, "y": FLOAT})
        point_type2 = GLSLType(TypeKind.STRUCT, members={"x": FLOAT, "y": FLOAT})
        color_type = GLSLType(
            TypeKind.STRUCT, members={"r": FLOAT, "g": FLOAT, "b": FLOAT}
        )

        assert point_type != color_type
        assert point_type == point_type2  # Same structure
