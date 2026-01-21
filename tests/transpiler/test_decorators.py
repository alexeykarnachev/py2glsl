"""Tests for shader stage decorators."""

from py2glsl.builtins import vec4
from py2glsl.decorators import compute, fragment, vertex
from py2glsl.transpiler.ir import ShaderStage


class TestVertexDecorator:
    """Test @vertex decorator."""

    def test_vertex_sets_stage(self):
        """Test that @vertex sets shader stage to VERTEX."""

        @vertex
        def my_vertex(position: "vec3") -> vec4:
            return vec4(1.0)

        assert hasattr(my_vertex, "_shader_stage")
        assert my_vertex._shader_stage == ShaderStage.VERTEX

    def test_vertex_sets_entry_point(self):
        """Test that @vertex marks function as entry point."""

        @vertex
        def my_vertex(position: "vec3") -> vec4:
            return vec4(1.0)

        assert hasattr(my_vertex, "_is_entry_point")
        assert my_vertex._is_entry_point is True

    def test_vertex_preserves_function(self):
        """Test that @vertex preserves the original function."""

        @vertex
        def my_vertex(x: "float") -> vec4:
            return vec4(x, x, x, 1.0)

        # Function should still be callable
        result = my_vertex(1.0)
        assert result is not None


class TestFragmentDecorator:
    """Test @fragment decorator."""

    def test_fragment_sets_stage(self):
        """Test that @fragment sets shader stage to FRAGMENT."""

        @fragment
        def my_fragment(uv: "vec2") -> vec4:
            return vec4(1.0)

        assert hasattr(my_fragment, "_shader_stage")
        assert my_fragment._shader_stage == ShaderStage.FRAGMENT

    def test_fragment_sets_entry_point(self):
        """Test that @fragment marks function as entry point."""

        @fragment
        def my_fragment(uv: "vec2") -> vec4:
            return vec4(1.0)

        assert hasattr(my_fragment, "_is_entry_point")
        assert my_fragment._is_entry_point is True

    def test_fragment_preserves_function(self):
        """Test that @fragment preserves the original function."""

        @fragment
        def my_fragment(x: "float") -> vec4:
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

        assert hasattr(my_compute, "_shader_stage")
        assert my_compute._shader_stage == ShaderStage.COMPUTE

    def test_compute_sets_entry_point(self):
        """Test that @compute marks function as entry point."""

        @compute()
        def my_compute() -> None:
            pass

        assert hasattr(my_compute, "_is_entry_point")
        assert my_compute._is_entry_point is True

    def test_compute_default_workgroup_size(self):
        """Test that @compute has default workgroup size."""

        @compute()
        def my_compute() -> None:
            pass

        assert hasattr(my_compute, "_workgroup_size")
        assert my_compute._workgroup_size == (1, 1, 1)

    def test_compute_custom_workgroup_size(self):
        """Test that @compute accepts custom workgroup size."""

        @compute(workgroup_size=(8, 8, 1))
        def my_compute() -> None:
            pass

        assert my_compute._workgroup_size == (8, 8, 1)

    def test_compute_preserves_function(self):
        """Test that @compute preserves the original function."""
        call_count = [0]

        @compute()
        def my_compute() -> None:
            call_count[0] += 1

        my_compute()
        assert call_count[0] == 1


class TestDecoratorStacking:
    """Test that decorators work correctly."""

    def test_multiple_decorated_functions(self):
        """Test multiple functions can be decorated independently."""

        @vertex
        def vs_main() -> vec4:
            return vec4(1.0)

        @fragment
        def fs_main() -> vec4:
            return vec4(1.0)

        assert vs_main._shader_stage == ShaderStage.VERTEX
        assert fs_main._shader_stage == ShaderStage.FRAGMENT

    def test_decorated_function_name_preserved(self):
        """Test that decorated functions preserve their names."""

        @vertex
        def my_custom_vertex() -> vec4:
            return vec4(1.0)

        @fragment
        def my_custom_fragment() -> vec4:
            return vec4(1.0)

        @compute()
        def my_custom_compute() -> None:
            pass

        assert my_custom_vertex.__name__ == "my_custom_vertex"
        assert my_custom_fragment.__name__ == "my_custom_fragment"
        assert my_custom_compute.__name__ == "my_custom_compute"
