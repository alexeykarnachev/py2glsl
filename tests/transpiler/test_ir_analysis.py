"""Tests for ir_analysis module."""

from py2glsl.transpiler.ir import (
    IRBinOp,
    IRCall,
    IRConstruct,
    IRDeclare,
    IRExprStmt,
    IRFieldAccess,
    IRFor,
    IRFunction,
    IRIf,
    IRLiteral,
    IRName,
    IRReturn,
    IRSubscript,
    IRSwizzle,
    IRTernary,
    IRType,
    IRUnaryOp,
    IRVariable,
    IRWhile,
)
from py2glsl.transpiler.ir_analysis import find_called_functions, topological_sort


class TestFindCalledFunctions:
    """Tests for find_called_functions."""

    def test_no_calls(self) -> None:
        """Test function with no calls returns empty set."""
        func = IRFunction(
            name="foo",
            params=[],
            return_type=IRType("float"),
            body=[IRReturn(value=IRLiteral(result_type=IRType("float"), value=1.0))],
        )
        result = find_called_functions(func, {"bar", "baz"})
        assert result == set()

    def test_single_call(self) -> None:
        """Test function with single call to user function."""
        func = IRFunction(
            name="foo",
            params=[],
            return_type=IRType("float"),
            body=[
                IRReturn(value=IRCall(result_type=IRType("float"), func="bar", args=[]))
            ],
        )
        result = find_called_functions(func, {"bar", "baz"})
        assert result == {"bar"}

    def test_builtin_call_not_included(self) -> None:
        """Test that builtin calls are not included."""
        func = IRFunction(
            name="foo",
            params=[],
            return_type=IRType("float"),
            body=[
                IRReturn(
                    value=IRCall(
                        result_type=IRType("float"),
                        func="sin",
                        args=[IRLiteral(result_type=IRType("float"), value=1.0)],
                    )
                )
            ],
        )
        result = find_called_functions(func, {"bar"})
        assert result == set()

    def test_nested_calls(self) -> None:
        """Test function with nested calls."""
        func = IRFunction(
            name="foo",
            params=[],
            return_type=IRType("float"),
            body=[
                IRReturn(
                    value=IRCall(
                        result_type=IRType("float"),
                        func="bar",
                        args=[IRCall(result_type=IRType("float"), func="baz", args=[])],
                    )
                )
            ],
        )
        result = find_called_functions(func, {"bar", "baz"})
        assert result == {"bar", "baz"}

    def test_call_in_binop(self) -> None:
        """Test call inside binary operation."""
        func = IRFunction(
            name="foo",
            params=[],
            return_type=IRType("float"),
            body=[
                IRReturn(
                    value=IRBinOp(
                        result_type=IRType("float"),
                        op="+",
                        left=IRCall(result_type=IRType("float"), func="bar", args=[]),
                        right=IRLiteral(result_type=IRType("float"), value=1.0),
                    )
                )
            ],
        )
        result = find_called_functions(func, {"bar"})
        assert result == {"bar"}

    def test_call_in_if_statement(self) -> None:
        """Test calls in if statement branches."""
        func = IRFunction(
            name="foo",
            params=[],
            return_type=None,
            body=[
                IRIf(
                    condition=IRCall(result_type=IRType("bool"), func="check", args=[]),
                    then_body=[
                        IRExprStmt(
                            expr=IRCall(
                                result_type=IRType("float"), func="bar", args=[]
                            )
                        )
                    ],
                    else_body=[
                        IRExprStmt(
                            expr=IRCall(
                                result_type=IRType("float"), func="baz", args=[]
                            )
                        )
                    ],
                )
            ],
        )
        result = find_called_functions(func, {"check", "bar", "baz"})
        assert result == {"check", "bar", "baz"}

    def test_call_in_for_loop(self) -> None:
        """Test calls in for loop."""
        func = IRFunction(
            name="foo",
            params=[],
            return_type=None,
            body=[
                IRFor(
                    init=IRDeclare(
                        var=IRVariable(name="i", type=IRType("int")),
                        init=IRLiteral(result_type=IRType("int"), value=0),
                    ),
                    condition=IRBinOp(
                        result_type=IRType("bool"),
                        op="<",
                        left=IRName(result_type=IRType("int"), name="i"),
                        right=IRCall(
                            result_type=IRType("int"), func="get_count", args=[]
                        ),
                    ),
                    update=None,
                    body=[
                        IRExprStmt(
                            expr=IRCall(
                                result_type=IRType("float"), func="process", args=[]
                            )
                        )
                    ],
                )
            ],
        )
        result = find_called_functions(func, {"get_count", "process"})
        assert result == {"get_count", "process"}

    def test_call_in_while_loop(self) -> None:
        """Test calls in while loop."""
        func = IRFunction(
            name="foo",
            params=[],
            return_type=None,
            body=[
                IRWhile(
                    condition=IRCall(
                        result_type=IRType("bool"), func="should_continue", args=[]
                    ),
                    body=[
                        IRExprStmt(
                            expr=IRCall(
                                result_type=IRType("float"), func="work", args=[]
                            )
                        )
                    ],
                )
            ],
        )
        result = find_called_functions(func, {"should_continue", "work"})
        assert result == {"should_continue", "work"}

    def test_call_in_unary_op(self) -> None:
        """Test call inside unary operation."""
        func = IRFunction(
            name="foo",
            params=[],
            return_type=IRType("float"),
            body=[
                IRReturn(
                    value=IRUnaryOp(
                        result_type=IRType("float"),
                        op="-",
                        operand=IRCall(
                            result_type=IRType("float"), func="bar", args=[]
                        ),
                    )
                )
            ],
        )
        result = find_called_functions(func, {"bar"})
        assert result == {"bar"}

    def test_call_in_construct(self) -> None:
        """Test call inside constructor."""
        func = IRFunction(
            name="foo",
            params=[],
            return_type=IRType("vec2"),
            body=[
                IRReturn(
                    value=IRConstruct(
                        result_type=IRType("vec2"),
                        args=[
                            IRCall(result_type=IRType("float"), func="get_x", args=[]),
                            IRCall(result_type=IRType("float"), func="get_y", args=[]),
                        ],
                    )
                )
            ],
        )
        result = find_called_functions(func, {"get_x", "get_y"})
        assert result == {"get_x", "get_y"}

    def test_call_in_swizzle(self) -> None:
        """Test call as base of swizzle."""
        func = IRFunction(
            name="foo",
            params=[],
            return_type=IRType("float"),
            body=[
                IRReturn(
                    value=IRSwizzle(
                        result_type=IRType("float"),
                        base=IRCall(
                            result_type=IRType("vec2"), func="get_vec", args=[]
                        ),
                        components="x",
                    )
                )
            ],
        )
        result = find_called_functions(func, {"get_vec"})
        assert result == {"get_vec"}

    def test_call_in_field_access(self) -> None:
        """Test call as base of field access."""
        func = IRFunction(
            name="foo",
            params=[],
            return_type=IRType("float"),
            body=[
                IRReturn(
                    value=IRFieldAccess(
                        result_type=IRType("float"),
                        base=IRCall(
                            result_type=IRType("MyStruct"), func="get_struct", args=[]
                        ),
                        field="value",
                    )
                )
            ],
        )
        result = find_called_functions(func, {"get_struct"})
        assert result == {"get_struct"}

    def test_call_in_subscript(self) -> None:
        """Test calls in subscript base and index."""
        func = IRFunction(
            name="foo",
            params=[],
            return_type=IRType("float"),
            body=[
                IRReturn(
                    value=IRSubscript(
                        result_type=IRType("float"),
                        base=IRCall(
                            result_type=IRType("float[4]"), func="get_arr", args=[]
                        ),
                        index=IRCall(
                            result_type=IRType("int"), func="get_idx", args=[]
                        ),
                    )
                )
            ],
        )
        result = find_called_functions(func, {"get_arr", "get_idx"})
        assert result == {"get_arr", "get_idx"}

    def test_call_in_ternary(self) -> None:
        """Test calls in ternary expression."""
        func = IRFunction(
            name="foo",
            params=[],
            return_type=IRType("float"),
            body=[
                IRReturn(
                    value=IRTernary(
                        result_type=IRType("float"),
                        condition=IRCall(
                            result_type=IRType("bool"), func="check", args=[]
                        ),
                        true_expr=IRCall(
                            result_type=IRType("float"), func="get_a", args=[]
                        ),
                        false_expr=IRCall(
                            result_type=IRType("float"), func="get_b", args=[]
                        ),
                    )
                )
            ],
        )
        result = find_called_functions(func, {"check", "get_a", "get_b"})
        assert result == {"check", "get_a", "get_b"}


class TestTopologicalSort:
    """Tests for topological_sort."""

    def test_empty_list(self) -> None:
        """Test empty function list."""
        result = topological_sort([])
        assert result == []

    def test_single_function(self) -> None:
        """Test single function."""
        func = IRFunction(
            name="foo",
            params=[],
            return_type=IRType("float"),
            body=[IRReturn(value=IRLiteral(result_type=IRType("float"), value=1.0))],
        )
        result = topological_sort([func])
        assert result == [func]

    def test_independent_functions(self) -> None:
        """Test functions with no dependencies."""
        func_a = IRFunction(
            name="a",
            params=[],
            return_type=IRType("float"),
            body=[IRReturn(value=IRLiteral(result_type=IRType("float"), value=1.0))],
        )
        func_b = IRFunction(
            name="b",
            params=[],
            return_type=IRType("float"),
            body=[IRReturn(value=IRLiteral(result_type=IRType("float"), value=2.0))],
        )
        result = topological_sort([func_a, func_b])
        # Both orderings are valid for independent functions
        assert {f.name for f in result} == {"a", "b"}

    def test_simple_dependency(self) -> None:
        """Test simple A calls B dependency."""
        func_b = IRFunction(
            name="b",
            params=[],
            return_type=IRType("float"),
            body=[IRReturn(value=IRLiteral(result_type=IRType("float"), value=1.0))],
        )
        func_a = IRFunction(
            name="a",
            params=[],
            return_type=IRType("float"),
            body=[
                IRReturn(value=IRCall(result_type=IRType("float"), func="b", args=[]))
            ],
        )
        # Input in caller-first order
        result = topological_sort([func_a, func_b])
        # Callee (b) should come before caller (a)
        names = [f.name for f in result]
        assert names.index("b") < names.index("a")

    def test_chain_dependency(self) -> None:
        """Test chain A -> B -> C."""
        func_c = IRFunction(
            name="c",
            params=[],
            return_type=IRType("float"),
            body=[IRReturn(value=IRLiteral(result_type=IRType("float"), value=1.0))],
        )
        func_b = IRFunction(
            name="b",
            params=[],
            return_type=IRType("float"),
            body=[
                IRReturn(value=IRCall(result_type=IRType("float"), func="c", args=[]))
            ],
        )
        func_a = IRFunction(
            name="a",
            params=[],
            return_type=IRType("float"),
            body=[
                IRReturn(value=IRCall(result_type=IRType("float"), func="b", args=[]))
            ],
        )
        result = topological_sort([func_a, func_b, func_c])
        names = [f.name for f in result]
        assert names.index("c") < names.index("b")
        assert names.index("b") < names.index("a")

    def test_diamond_dependency(self) -> None:
        """Test diamond: A -> B, A -> C, B -> D, C -> D."""
        func_d = IRFunction(
            name="d",
            params=[],
            return_type=IRType("float"),
            body=[IRReturn(value=IRLiteral(result_type=IRType("float"), value=1.0))],
        )
        func_b = IRFunction(
            name="b",
            params=[],
            return_type=IRType("float"),
            body=[
                IRReturn(value=IRCall(result_type=IRType("float"), func="d", args=[]))
            ],
        )
        func_c = IRFunction(
            name="c",
            params=[],
            return_type=IRType("float"),
            body=[
                IRReturn(value=IRCall(result_type=IRType("float"), func="d", args=[]))
            ],
        )
        func_a = IRFunction(
            name="a",
            params=[],
            return_type=IRType("float"),
            body=[
                IRReturn(
                    value=IRBinOp(
                        result_type=IRType("float"),
                        op="+",
                        left=IRCall(result_type=IRType("float"), func="b", args=[]),
                        right=IRCall(result_type=IRType("float"), func="c", args=[]),
                    )
                )
            ],
        )
        result = topological_sort([func_a, func_b, func_c, func_d])
        names = [f.name for f in result]
        # D must come before B and C
        assert names.index("d") < names.index("b")
        assert names.index("d") < names.index("c")
        # B and C must come before A
        assert names.index("b") < names.index("a")
        assert names.index("c") < names.index("a")
