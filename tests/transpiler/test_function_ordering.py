"""Tests for proper function dependency ordering in code generation."""

import ast

from py2glsl.transpiler.backends import create_backend
from py2glsl.transpiler.backends.models import BackendType
from py2glsl.transpiler.models import CollectedInfo, FunctionInfo


def test_function_dependency_order() -> None:
    """Test that functions are generated in proper dependency order."""
    # Arrange - Create a test case with nested function dependencies
    info = CollectedInfo()

    # Create three functions with dependencies:
    # main depends on helper1, helper1 depends on helper2

    # Helper2 function (no dependencies)
    helper2_node = ast.FunctionDef(
        name="helper2",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="x", annotation=ast.Constant(value="float"))],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[
            ast.Return(
                value=ast.BinOp(
                    left=ast.Name(id="x", ctx=ast.Load()),
                    op=ast.Mult(),
                    right=ast.Constant(value=2.0),
                )
            )
        ],
        decorator_list=[],
        returns=ast.Constant(value="float"),
    )

    # Helper1 function (depends on helper2)
    helper1_node = ast.FunctionDef(
        name="helper1",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="x", annotation=ast.Constant(value="float"))],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[
            ast.Return(
                value=ast.Call(
                    func=ast.Name(id="helper2", ctx=ast.Load()),
                    args=[ast.Name(id="x", ctx=ast.Load())],
                    keywords=[],
                )
            )
        ],
        decorator_list=[],
        returns=ast.Constant(value="float"),
    )

    # Main function (depends on helper1)
    main_node = ast.FunctionDef(
        name="main_func",
        args=ast.arguments(
            posonlyargs=[],
            args=[
                ast.arg(arg="vs_uv", annotation=ast.Constant(value="vec2")),
                ast.arg(arg="u_time", annotation=ast.Constant(value="float")),
            ],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[
            ast.Return(
                value=ast.Call(
                    func=ast.Name(id="vec4", ctx=ast.Load()),
                    args=[
                        ast.Call(
                            func=ast.Name(id="helper1", ctx=ast.Load()),
                            args=[ast.Name(id="u_time", ctx=ast.Load())],
                            keywords=[],
                        ),
                        ast.Constant(value=0.0),
                        ast.Constant(value=0.0),
                        ast.Constant(value=1.0),
                    ],
                    keywords=[],
                )
            )
        ],
        decorator_list=[],
        returns=ast.Constant(value="vec4"),
    )

    # Add functions to collected info
    info.functions["helper2"] = FunctionInfo(
        name="helper2",
        return_type="float",
        param_types=["float"],
        node=helper2_node,
    )

    info.functions["helper1"] = FunctionInfo(
        name="helper1",
        return_type="float",
        param_types=["float"],
        node=helper1_node,
    )

    info.functions["main_func"] = FunctionInfo(
        name="main_func",
        return_type="vec4",
        param_types=["vec2", "float"],
        node=main_node,
    )

    # Act
    backend = create_backend(BackendType.STANDARD)
    glsl_code, _ = backend.generate_code(info, "main_func")

    # Assert - Check that functions are in correct dependency order
    # helper2 should appear before helper1, which should appear before main_func
    helper2_pos = glsl_code.find("float helper2(float x)")
    helper1_pos = glsl_code.find("float helper1(float x)")
    main_pos = glsl_code.find("vec4 main_func(vec2 vs_uv, float u_time)")

    assert helper2_pos < helper1_pos, "helper2 should be defined before helper1"
    assert helper1_pos < main_pos, "helper1 should be defined before main_func"
