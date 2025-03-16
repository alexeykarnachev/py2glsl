"""Tests for the transpiler code generation functionality."""

import ast

import pytest

from py2glsl.transpiler.backends import create_backend
from py2glsl.transpiler.backends.models import BackendType
from py2glsl.transpiler.errors import TranspilerError
from py2glsl.transpiler.models import (
    CollectedInfo,
    FunctionInfo,
    StructDefinition,
    StructField,
)


@pytest.fixture
def basic_collected_info() -> CollectedInfo:
    """Fixture providing a basic collected info for testing."""
    info = CollectedInfo()

    # Create a simple shader function
    shader_node = ast.FunctionDef(
        name="shader",
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
                        ast.Constant(value=1.0),
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

    info.functions["shader"] = FunctionInfo(
        name="shader",
        return_type="vec4",
        param_types=["vec2", "float"],
        node=shader_node,
    )

    return info


@pytest.fixture
def complex_collected_info() -> CollectedInfo:
    """Fixture providing a more complex collected info for testing."""
    info = CollectedInfo()

    # Add a global constant
    info.globals["PI"] = ("float", "3.14159")

    # Add a struct
    info.structs["Material"] = StructDefinition(
        name="Material",
        fields=[
            StructField(name="color", type_name="vec3"),
            StructField(name="shininess", type_name="float"),
        ],
    )

    # Create a helper function
    helper_node = ast.FunctionDef(
        name="calculate_lighting",
        args=ast.arguments(
            posonlyargs=[],
            args=[
                ast.arg(arg="pos", annotation=ast.Constant(value="vec3")),
                ast.arg(arg="normal", annotation=ast.Constant(value="vec3")),
                ast.arg(arg="material", annotation=ast.Constant(value="Material")),
            ],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[
            ast.Return(
                value=ast.BinOp(
                    left=ast.Attribute(
                        value=ast.Name(id="material", ctx=ast.Load()),
                        attr="color",
                        ctx=ast.Load(),
                    ),
                    op=ast.Mult(),
                    right=ast.Constant(value=0.5),
                )
            )
        ],
        decorator_list=[],
        returns=ast.Constant(value="vec3"),
    )

    info.functions["calculate_lighting"] = FunctionInfo(
        name="calculate_lighting",
        return_type="vec3",
        param_types=["vec3", "vec3", "Material"],
        node=helper_node,
    )

    # Create a main shader function
    shader_node = ast.FunctionDef(
        name="main_shader",
        args=ast.arguments(
            posonlyargs=[],
            args=[
                ast.arg(arg="vs_uv", annotation=ast.Constant(value="vec2")),
                ast.arg(arg="u_time", annotation=ast.Constant(value="float")),
                ast.arg(arg="u_material", annotation=ast.Constant(value="Material")),
            ],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[
            ast.Assign(
                targets=[ast.Name(id="pos", ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id="vec3", ctx=ast.Load()),
                    args=[
                        ast.Name(id="vs_uv", ctx=ast.Load()),
                        ast.Constant(value=0.0),
                    ],
                    keywords=[],
                ),
            ),
            ast.Assign(
                targets=[ast.Name(id="normal", ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id="vec3", ctx=ast.Load()),
                    args=[
                        ast.Constant(value=0.0),
                        ast.Constant(value=0.0),
                        ast.Constant(value=1.0),
                    ],
                    keywords=[],
                ),
            ),
            ast.Assign(
                targets=[ast.Name(id="color", ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id="calculate_lighting", ctx=ast.Load()),
                    args=[
                        ast.Name(id="pos", ctx=ast.Load()),
                        ast.Name(id="normal", ctx=ast.Load()),
                        ast.Name(id="u_material", ctx=ast.Load()),
                    ],
                    keywords=[],
                ),
            ),
            ast.Return(
                value=ast.Call(
                    func=ast.Name(id="vec4", ctx=ast.Load()),
                    args=[
                        ast.Name(id="color", ctx=ast.Load()),
                        ast.Constant(value=1.0),
                    ],
                    keywords=[],
                )
            ),
        ],
        decorator_list=[],
        returns=ast.Constant(value="vec4"),
    )

    info.functions["main_shader"] = FunctionInfo(
        name="main_shader",
        return_type="vec4",
        param_types=["vec2", "float", "Material"],
        node=shader_node,
    )

    return info


class TestGenerateGLSL:
    """Tests for the generate_glsl function."""

    def test_generate_simple_shader(self, basic_collected_info: CollectedInfo) -> None:
        """Test generating GLSL code for a simple shader."""
        # Act
        backend = create_backend(BackendType.STANDARD)
        glsl_code, used_uniforms = backend.generate_code(basic_collected_info, "shader")

        # Assert
        assert "#version 460 core" in glsl_code
        assert "uniform float u_time;" in glsl_code
        assert "vec4 shader(vec2 vs_uv, float u_time)" in glsl_code
        assert "return vec4(1.0, 0.0, 0.0, 1.0);" in glsl_code
        assert "void main() {" in glsl_code
        assert "fragColor = shader(vs_uv, u_time);" in glsl_code
        assert used_uniforms == {"u_time"}

    def test_generate_complex_shader(
        self, complex_collected_info: CollectedInfo
    ) -> None:
        """Test generating GLSL code for a complex shader."""
        # Act
        backend = create_backend(BackendType.STANDARD)
        glsl_code, used_uniforms = backend.generate_code(
            complex_collected_info, "main_shader"
        )

        # Assert
        assert "#version 460 core" in glsl_code

        # Check uniforms
        assert "uniform float u_time;" in glsl_code
        assert "uniform Material u_material;" in glsl_code

        # Check globals
        assert "const float PI = 3.14159;" in glsl_code

        # Check struct definition
        assert "struct Material {" in glsl_code
        assert "vec3 color;" in glsl_code
        assert "float shininess;" in glsl_code

        # Check helper function
        assert (
            "vec3 calculate_lighting(vec3 pos, vec3 normal, Material material)"
            in glsl_code
        )
        assert "return material.color * 0.5;" in glsl_code

        # Check main function
        assert (
            "vec4 main_shader(vec2 vs_uv, float u_time, Material u_material)"
            in glsl_code
        )
        assert "vec3 pos = vec3(vs_uv, 0.0);" in glsl_code
        assert "vec3 normal = vec3(0.0, 0.0, 1.0);" in glsl_code
        assert "vec3 color = calculate_lighting(pos, normal, u_material);" in glsl_code
        assert "return vec4(color, 1.0);" in glsl_code

        # Check main entry point
        assert "void main() {" in glsl_code
        assert "fragColor = main_shader(vs_uv, u_time, u_material);" in glsl_code

        # Check used uniforms
        assert used_uniforms == {"u_time", "u_material"}

    def test_missing_helper_return_type(
        self, basic_collected_info: CollectedInfo
    ) -> None:
        """Test that missing return type on helper function raises an error."""
        # Arrange
        helper_node = ast.FunctionDef(
            name="helper",
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
            returns=None,  # No return type annotation
        )

        basic_collected_info.functions["helper"] = FunctionInfo(
            name="helper",
            return_type=None,  # No return type
            param_types=["float"],
            node=helper_node,
        )

        # Act & Assert
        backend = create_backend(BackendType.STANDARD)
        with pytest.raises(
            TranspilerError,
            match="Helper function 'helper' lacks return type annotation",
        ):
            backend.generate_code(basic_collected_info, "shader")

    def test_empty_function_body(self, basic_collected_info: CollectedInfo) -> None:
        """Test that empty function body raises an error."""
        # Arrange
        empty_body_node = ast.FunctionDef(
            name="empty_func",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="vs_uv", annotation=ast.Constant(value="vec2"))],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=[],  # Empty body
            decorator_list=[],
            returns=ast.Constant(value="vec4"),
        )

        basic_collected_info.functions["empty_func"] = FunctionInfo(
            name="empty_func",
            return_type="vec4",
            param_types=["vec2"],
            node=empty_body_node,
        )

        # Act & Assert
        backend = create_backend(BackendType.STANDARD)
        with pytest.raises(TranspilerError, match="Empty function body not supported"):
            backend.generate_code(basic_collected_info, "empty_func")

    def test_version_directive_first_line(
        self, basic_collected_info: CollectedInfo
    ) -> None:
        """Test that version directive is the first line of generated code."""
        # Act
        backend = create_backend(BackendType.STANDARD)
        glsl_code, _ = backend.generate_code(basic_collected_info, "shader")

        # Assert
        lines = glsl_code.split("\n")
        assert lines[0] == "#version 460 core"

    def test_function_dependency_order(self) -> None:
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

        assert helper2_pos < helper1_pos < main_pos, (
            "Functions not in correct dependency order"
        )
