"""Tests for the transpiler models module."""

import ast

from py2glsl.transpiler.models import (
    CollectedInfo,
    FunctionInfo,
    StructDefinition,
    StructField,
)


class TestStructField:
    """Tests for the StructField dataclass."""

    def test_struct_field_creation(self):
        """Test creating a StructField instance."""
        # Arrange & Act
        field = StructField(name="position", type_name="vec3")

        # Assert
        assert field.name == "position"
        assert field.type_name == "vec3"
        assert field.default_value is None

    def test_struct_field_with_default(self):
        """Test creating a StructField with a default value."""
        # Arrange & Act
        field = StructField(
            name="color", type_name="vec4", default_value="vec4(1.0, 1.0, 1.0, 1.0)"
        )

        # Assert
        assert field.name == "color"
        assert field.type_name == "vec4"
        assert field.default_value == "vec4(1.0, 1.0, 1.0, 1.0)"


class TestStructDefinition:
    """Tests for the StructDefinition dataclass."""

    def test_struct_definition_creation(self):
        """Test creating a StructDefinition instance."""
        # Arrange
        fields = [
            StructField(name="position", type_name="vec3"),
            StructField(name="color", type_name="vec4"),
        ]

        # Act
        struct_def = StructDefinition(name="Vertex", fields=fields)

        # Assert
        assert struct_def.name == "Vertex"
        assert len(struct_def.fields) == 2
        assert struct_def.fields[0].name == "position"
        assert struct_def.fields[1].name == "color"


class TestFunctionInfo:
    """Tests for the FunctionInfo dataclass."""

    def test_function_info_creation(self):
        """Test creating a FunctionInfo instance."""
        # Arrange
        dummy_node = ast.FunctionDef(
            name="test_func",
            args=ast.arguments(
                args=[], posonlyargs=[], kwonlyargs=[], kw_defaults=[], defaults=[]
            ),
            body=[],
            decorator_list=[],
        )

        # Act
        func_info = FunctionInfo(
            name="test_func",
            return_type="vec4",
            param_types=["vec2", "float"],
            node=dummy_node,
        )

        # Assert
        assert func_info.name == "test_func"
        assert func_info.return_type == "vec4"
        assert func_info.param_types == ["vec2", "float"]
        assert func_info.node is dummy_node


class TestCollectedInfo:
    """Tests for the CollectedInfo dataclass."""

    def test_collected_info_default_values(self):
        """Test that CollectedInfo initializes empty dictionaries by default."""
        # Act
        info = CollectedInfo()

        # Assert
        assert isinstance(info.functions, dict)
        assert len(info.functions) == 0
        assert isinstance(info.structs, dict)
        assert len(info.structs) == 0
        assert isinstance(info.globals, dict)
        assert len(info.globals) == 0

    def test_collected_info_add_function(self):
        """Test adding a function to CollectedInfo."""
        # Arrange
        info = CollectedInfo()
        dummy_node = ast.FunctionDef(
            name="test_func",
            args=ast.arguments(
                args=[], posonlyargs=[], kwonlyargs=[], kw_defaults=[], defaults=[]
            ),
            body=[],
            decorator_list=[],
        )
        func_info = FunctionInfo(
            name="test_func",
            return_type="vec4",
            param_types=["vec2", "float"],
            node=dummy_node,
        )

        # Act
        info.functions["test_func"] = func_info

        # Assert
        assert "test_func" in info.functions
        assert info.functions["test_func"] is func_info

    def test_collected_info_add_struct(self):
        """Test adding a struct to CollectedInfo."""
        # Arrange
        info = CollectedInfo()
        struct_def = StructDefinition(
            name="Vertex", fields=[StructField(name="position", type_name="vec3")]
        )

        # Act
        info.structs["Vertex"] = struct_def

        # Assert
        assert "Vertex" in info.structs
        assert info.structs["Vertex"] is struct_def

    def test_collected_info_add_global(self):
        """Test adding a global constant to CollectedInfo."""
        # Arrange
        info = CollectedInfo()

        # Act
        info.globals["PI"] = ("float", "3.14159")

        # Assert
        assert "PI" in info.globals
        assert info.globals["PI"] == ("float", "3.14159")
