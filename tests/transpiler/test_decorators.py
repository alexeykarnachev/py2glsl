"""Tests for shader stage decorators."""

from typing import Any

from py2glsl.builtins import vec2, vec3, vec4
from py2glsl.decorators import compute, fragment, vertex
from py2glsl.transpiler.ir import ShaderStage


class TestVertexDecorator:
    """Test @vertex decorator."""

    def test_vertex_sets_stage(self):
        """Test that @vertex sets shader stage to VERTEX."""

        @vertex
        def my_vertex(position: vec3) -> vec4:
            return vec4(1.0)

        func: Any = my_vertex
        assert hasattr(func, "_shader_stage")
        assert func._shader_stage == ShaderStage.VERTEX

    def test_vertex_sets_entry_point(self):
        """Test that @vertex marks function as entry point."""

        @vertex
        def my_vertex(position: vec3) -> vec4:
            return vec4(1.0)

        func: Any = my_vertex
        assert hasattr(func, "_is_entry_point")
        assert func._is_entry_point is True

    def test_vertex_preserves_function(self):
        """Test that @vertex preserves the original function."""

        @vertex
        def my_vertex(x: float) -> vec4:
            return vec4(x, x, x, 1.0)

        # Function should still be callable
        result = my_vertex(1.0)
        assert result is not None


class TestFragmentDecorator:
    """Test @fragment decorator."""

    def test_fragment_sets_stage(self):
        """Test that @fragment sets shader stage to FRAGMENT."""

        @fragment
        def my_fragment(uv: vec2) -> vec4:
            return vec4(1.0)

        func: Any = my_fragment
        assert hasattr(func, "_shader_stage")
        assert func._shader_stage == ShaderStage.FRAGMENT

    def test_fragment_sets_entry_point(self):
        """Test that @fragment marks function as entry point."""

        @fragment
        def my_fragment(uv: vec2) -> vec4:
            return vec4(1.0)

        func: Any = my_fragment
        assert hasattr(func, "_is_entry_point")
        assert func._is_entry_point is True

    def test_fragment_preserves_function(self):
        """Test that @fragment preserves the original function."""

        @fragment
        def my_fragment(x: float) -> vec4:
            return vec4(x, x, x, 1.0)

        result = my_fragment(0.5)
        assert result is not None


class TestComputeDecorator:
    """Test @compute decorator."""

    def test_compute_sets_stage(self):
        """Test that @compute sets shader stage to COMPUTE."""

        @compute()
        def my_compute() -> None:
            pass

        func: Any = my_compute
        assert hasattr(func, "_shader_stage")
        assert func._shader_stage == ShaderStage.COMPUTE

    def test_compute_sets_entry_point(self):
        """Test that @compute marks function as entry point."""

        @compute()
        def my_compute() -> None:
            pass

        func: Any = my_compute
        assert hasattr(func, "_is_entry_point")
        assert func._is_entry_point is True

    def test_compute_default_workgroup_size(self):
        """Test that @compute has default workgroup size."""

        @compute()
        def my_compute() -> None:
            pass

        func: Any = my_compute
        assert hasattr(func, "_workgroup_size")
        assert func._workgroup_size == (1, 1, 1)

    def test_compute_custom_workgroup_size(self):
        """Test that @compute accepts custom workgroup size."""

        @compute(workgroup_size=(8, 8, 1))
        def my_compute() -> None:
            pass

        func: Any = my_compute
        assert hasattr(func, "_workgroup_size")
        assert func._workgroup_size == (8, 8, 1)

    def test_compute_preserves_function(self):
        """Test that @compute preserves the original function."""

        @compute()
        def my_compute() -> None:
            return None

        result = my_compute()
        assert result is None


class TestDecoratorCombinations:
    """Test decorator combinations and edge cases."""

    def test_multiple_decorated_functions(self):
        """Test that multiple decorated functions don't interfere."""

        @vertex
        def vert(pos: float) -> vec4:
            return vec4(pos)

        @fragment
        def frag(uv: float) -> vec4:
            return vec4(uv)

        vert_func: Any = vert
        frag_func: Any = frag
        assert vert_func._shader_stage == ShaderStage.VERTEX
        assert frag_func._shader_stage == ShaderStage.FRAGMENT

    def test_decorated_function_with_multiple_params(self):
        """Test decorated function with multiple parameters."""

        @fragment
        def shader(a: float, b: float, c: float) -> vec4:
            return vec4(a, b, c, 1.0)

        func: Any = shader
        assert func._is_entry_point is True
        result = shader(0.1, 0.2, 0.3)
        assert result is not None

    def test_decorated_function_no_params(self):
        """Test decorated function with no parameters."""

        @compute()
        def empty_compute() -> None:
            pass

        func: Any = empty_compute
        assert func._workgroup_size == (1, 1, 1)
