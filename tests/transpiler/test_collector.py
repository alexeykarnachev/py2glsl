"""Tests for the transpiler collector module."""

import ast

from py2glsl.transpiler.collector import collect_info


class TestCollectInfo:
    """Tests for the collect_info function."""

    def test_collect_empty(self):
        """Test collecting info from an empty AST."""
        # Arrange
        tree = ast.parse("")

        # Act
        collected = collect_info(tree)

        # Assert
        assert len(collected.functions) == 0
        assert len(collected.structs) == 0
        assert len(collected.globals) == 0

    def test_collect_function(self):
        """Test collecting function information."""
        # Arrange
        code = """
    def shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
        return vec4(1.0, 0.0, 0.0, 1.0)
    """
        tree = ast.parse(code)

        # Act
        collected = collect_info(tree)

        # Assert
        assert "shader" in collected.functions
        func_info = collected.functions["shader"]
        assert func_info.name == "shader"
        assert func_info.return_type == "vec4"
        assert func_info.param_types == ["vec2", "float"]
        assert isinstance(func_info.node, ast.FunctionDef)

    def test_collect_struct(self):
        """Test collecting struct information from dataclasses."""
        # Arrange
        code = """
        from dataclasses import dataclass
        
        @dataclass
        class Material:
            color: 'vec3'
            shininess: 'float' = 32.0
        """
        tree = ast.parse(code)

        # Act
        collected = collect_info(tree)

        # Assert
        assert "Material" in collected.structs
        struct_def = collected.structs["Material"]
        assert struct_def.name == "Material"
        assert len(struct_def.fields) == 2

        # Check first field
        assert struct_def.fields[0].name == "color"
        assert struct_def.fields[0].type_name == "vec3"
        assert struct_def.fields[0].default_value is None

        # Check second field with default value
        assert struct_def.fields[1].name == "shininess"
        assert struct_def.fields[1].type_name == "float"
        assert struct_def.fields[1].default_value == "32.0"

    def test_collect_global(self):
        """Test collecting global variables."""
        # Arrange
        code = """
        PI: 'float' = 3.14159
        MAX_STEPS: 'int' = 100
        """
        tree = ast.parse(code)

        # Act
        collected = collect_info(tree)

        # Assert
        assert "PI" in collected.globals
        assert collected.globals["PI"] == ("float", "3.14159")

        assert "MAX_STEPS" in collected.globals
        assert collected.globals["MAX_STEPS"] == ("int", "100")

    def test_collect_complex_setup(self):
        """Test collecting information from a complex setup with functions, structs, and globals."""
        # Arrange
        code = """
        from dataclasses import dataclass
        
        PI: 'float' = 3.14159
        
        @dataclass
        class Light:
            position: 'vec3'
            color: 'vec3'
            intensity: 'float' = 1.0
        
        def calculate_lighting(pos: 'vec3', normal: 'vec3', light: 'Light') -> 'vec3':
            direction = normalize(light.position - pos)
            return light.color * light.intensity * max(0.0, dot(normal, direction))
        
        def shader(vs_uv: 'vec2', u_time: 'float') -> 'vec4':
            pos = vec3(vs_uv * 2.0 - 1.0, 0.0)
            normal = vec3(0.0, 0.0, 1.0)
            light = Light(
                position=vec3(sin(u_time), cos(u_time), 1.0),
                color=vec3(1.0, 1.0, 1.0)
            )
            color = calculate_lighting(pos, normal, light)
            return vec4(color, 1.0)
        """
        tree = ast.parse(code)

        # Act
        collected = collect_info(tree)

        # Assert
        # Check globals
        assert "PI" in collected.globals
        assert collected.globals["PI"] == ("float", "3.14159")

        # Check structs
        assert "Light" in collected.structs
        light_struct = collected.structs["Light"]
        assert len(light_struct.fields) == 3

        # Check functions
        assert "calculate_lighting" in collected.functions
        assert "shader" in collected.functions

        shader_func = collected.functions["shader"]
        assert shader_func.return_type == "vec4"
        assert shader_func.param_types == ["vec2", "float"]

        lighting_func = collected.functions["calculate_lighting"]
        assert lighting_func.return_type == "vec3"
        assert lighting_func.param_types == ["vec3", "vec3", "Light"]

    def test_ignore_non_dataclass(self):
        """Test that non-dataclass classes are ignored."""
        # Arrange
        code = """
        class RegularClass:
            def __init__(self):
                self.x = 0
                self.y = 0
        """
        tree = ast.parse(code)

        # Act
        collected = collect_info(tree)

        # Assert
        assert "RegularClass" not in collected.structs
        assert len(collected.structs) == 0

    def test_collect_complex_global_skipped(self):
        """Test that complex global expressions are skipped."""
        # Arrange
        code = """
        SIMPLE: 'float' = 3.14
        COMPLEX: 'float' = 2.0 + 3.0  # This should be skipped as it's not a simple expression
        """
        tree = ast.parse(code)

        # Act
        collected = collect_info(tree)

        # Assert
        assert "SIMPLE" in collected.globals
        assert "COMPLEX" not in collected.globals
