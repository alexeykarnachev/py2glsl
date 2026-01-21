"""Tests for IR builder module."""

import pytest

from py2glsl import ShaderContext
from py2glsl.builtins import vec3, vec4
from py2glsl.transpiler import transpile
from py2glsl.transpiler.models import TranspilerError


class TestForLoops:
    """Test for loop transpilation."""

    def test_for_range_single_arg(self):
        """Test for loop with range(n)."""

        def shader(ctx: ShaderContext) -> vec4:
            total: float = 0.0
            for _i in range(5):
                total = total + 1.0
            return vec4(total, 0.0, 0.0, 1.0)

        code, _ = transpile(shader)
        expected = """\
#version 460 core

in vec2 vs_uv;
uniform float u_time;
uniform vec2 u_resolution;
uniform float u_aspect;
uniform vec2 u_mouse_pos;
uniform vec2 u_mouse_uv;
out vec4 fragColor;

vec4 shader() {
    float total = 0.0;
    for (int _i = 0; (_i < 5); _i += 1) {
        total = (total + 1.0);
    }
    return vec4(total, 0.0, 0.0, 1.0);
}

void main() {
    fragColor = shader();
}"""
        assert code == expected

    def test_for_range_two_args(self):
        """Test for loop with range(start, end)."""

        def shader(ctx: ShaderContext) -> vec4:
            total: float = 0.0
            for _i in range(2, 8):
                total = total + 1.0
            return vec4(total, 0.0, 0.0, 1.0)

        code, _ = transpile(shader)
        expected = """\
#version 460 core

in vec2 vs_uv;
uniform float u_time;
uniform vec2 u_resolution;
uniform float u_aspect;
uniform vec2 u_mouse_pos;
uniform vec2 u_mouse_uv;
out vec4 fragColor;

vec4 shader() {
    float total = 0.0;
    for (int _i = 2; (_i < 8); _i += 1) {
        total = (total + 1.0);
    }
    return vec4(total, 0.0, 0.0, 1.0);
}

void main() {
    fragColor = shader();
}"""
        assert code == expected

    def test_for_range_three_args(self):
        """Test for loop with range(start, end, step)."""

        def shader(ctx: ShaderContext) -> vec4:
            total: float = 0.0
            for _i in range(0, 10, 2):
                total = total + 1.0
            return vec4(total, 0.0, 0.0, 1.0)

        code, _ = transpile(shader)
        expected = """\
#version 460 core

in vec2 vs_uv;
uniform float u_time;
uniform vec2 u_resolution;
uniform float u_aspect;
uniform vec2 u_mouse_pos;
uniform vec2 u_mouse_uv;
out vec4 fragColor;

vec4 shader() {
    float total = 0.0;
    for (int _i = 0; (_i < 10); _i += 2) {
        total = (total + 1.0);
    }
    return vec4(total, 0.0, 0.0, 1.0);
}

void main() {
    fragColor = shader();
}"""
        assert code == expected


class TestWhileLoops:
    """Test while loop transpilation."""

    def test_while_simple(self):
        """Test simple while loop."""

        def shader(ctx: ShaderContext) -> vec4:
            x: float = 0.0
            while x < 10.0:
                x = x + 1.0
            return vec4(x, 0.0, 0.0, 1.0)

        code, _ = transpile(shader)
        expected = """\
#version 460 core

in vec2 vs_uv;
uniform float u_time;
uniform vec2 u_resolution;
uniform float u_aspect;
uniform vec2 u_mouse_pos;
uniform vec2 u_mouse_uv;
out vec4 fragColor;

vec4 shader() {
    float x = 0.0;
    while ((x < 10.0)) {
        x = (x + 1.0);
    }
    return vec4(x, 0.0, 0.0, 1.0);
}

void main() {
    fragColor = shader();
}"""
        assert code == expected


class TestIfStatements:
    """Test if/elif/else transpilation."""

    def test_if_simple(self):
        """Test simple if statement."""

        def shader(ctx: ShaderContext) -> vec4:
            x: float = 0.0
            if ctx.u_time > 1.0:
                x = 1.0
            return vec4(x, 0.0, 0.0, 1.0)

        code, _ = transpile(shader)
        expected = """\
#version 460 core

in vec2 vs_uv;
uniform float u_time;
uniform vec2 u_resolution;
uniform float u_aspect;
uniform vec2 u_mouse_pos;
uniform vec2 u_mouse_uv;
out vec4 fragColor;

vec4 shader() {
    float x = 0.0;
    if ((u_time > 1.0)) {
        x = 1.0;
    }
    return vec4(x, 0.0, 0.0, 1.0);
}

void main() {
    fragColor = shader();
}"""
        assert code == expected

    def test_if_else(self):
        """Test if-else statement."""

        def shader(ctx: ShaderContext) -> vec4:
            x: float = 0.0
            if ctx.u_time > 1.0:  # noqa: SIM108 - testing if-else, not ternary
                x = 1.0
            else:
                x = 0.5
            return vec4(x, 0.0, 0.0, 1.0)

        code, _ = transpile(shader)
        expected = """\
#version 460 core

in vec2 vs_uv;
uniform float u_time;
uniform vec2 u_resolution;
uniform float u_aspect;
uniform vec2 u_mouse_pos;
uniform vec2 u_mouse_uv;
out vec4 fragColor;

vec4 shader() {
    float x = 0.0;
    if ((u_time > 1.0)) {
        x = 1.0;
    } else {
        x = 0.5;
    }
    return vec4(x, 0.0, 0.0, 1.0);
}

void main() {
    fragColor = shader();
}"""
        assert code == expected

    def test_if_variable_hoisting(self):
        """Test that variables assigned in if branches are hoisted."""

        def shader(ctx: ShaderContext) -> vec4:
            if ctx.u_time > 1.0:  # noqa: SIM108 - testing variable hoisting
                color = vec3(1.0, 0.0, 0.0)
            else:
                color = vec3(0.0, 1.0, 0.0)
            return vec4(color.x, color.y, color.z, 1.0)

        code, _ = transpile(shader)
        expected = """\
#version 460 core

in vec2 vs_uv;
uniform float u_time;
uniform vec2 u_resolution;
uniform float u_aspect;
uniform vec2 u_mouse_pos;
uniform vec2 u_mouse_uv;
out vec4 fragColor;

vec4 shader() {
    vec3 color;
    if ((u_time > 1.0)) {
        color = vec3(1.0, 0.0, 0.0);
    } else {
        color = vec3(0.0, 1.0, 0.0);
    }
    return vec4(color.x, color.y, color.z, 1.0);
}

void main() {
    fragColor = shader();
}"""
        assert code == expected


class TestAugmentedAssignment:
    """Test augmented assignment operators."""

    def test_plus_equals(self):
        """Test += operator."""

        def shader(ctx: ShaderContext) -> vec4:
            x: float = 1.0
            x += 2.0
            return vec4(x, 0.0, 0.0, 1.0)

        code, _ = transpile(shader)
        expected = """\
#version 460 core

in vec2 vs_uv;
uniform float u_time;
uniform vec2 u_resolution;
uniform float u_aspect;
uniform vec2 u_mouse_pos;
uniform vec2 u_mouse_uv;
out vec4 fragColor;

vec4 shader() {
    float x = 1.0;
    x += 2.0;
    return vec4(x, 0.0, 0.0, 1.0);
}

void main() {
    fragColor = shader();
}"""
        assert code == expected


class TestTernaryExpressions:
    """Test ternary/conditional expressions."""

    def test_ternary_simple(self):
        """Test simple ternary expression."""

        def shader(ctx: ShaderContext) -> vec4:
            x = 1.0 if ctx.u_time > 0.5 else 0.0
            return vec4(x, 0.0, 0.0, 1.0)

        code, _ = transpile(shader)
        expected = """\
#version 460 core

in vec2 vs_uv;
uniform float u_time;
uniform vec2 u_resolution;
uniform float u_aspect;
uniform vec2 u_mouse_pos;
uniform vec2 u_mouse_uv;
out vec4 fragColor;

vec4 shader() {
    float x = ((u_time > 0.5) ? 1.0 : 0.0);
    return vec4(x, 0.0, 0.0, 1.0);
}

void main() {
    fragColor = shader();
}"""
        assert code == expected


class TestBooleanOperations:
    """Test boolean operations."""

    def test_and_operator(self):
        """Test 'and' operator."""

        def shader(ctx: ShaderContext) -> vec4:
            x: float = 0.0
            if ctx.u_time > 0.0 and ctx.u_time < 1.0:
                x = 1.0
            return vec4(x, 0.0, 0.0, 1.0)

        code, _ = transpile(shader)
        expected = """\
#version 460 core

in vec2 vs_uv;
uniform float u_time;
uniform vec2 u_resolution;
uniform float u_aspect;
uniform vec2 u_mouse_pos;
uniform vec2 u_mouse_uv;
out vec4 fragColor;

vec4 shader() {
    float x = 0.0;
    if (((u_time > 0.0) && (u_time < 1.0))) {
        x = 1.0;
    }
    return vec4(x, 0.0, 0.0, 1.0);
}

void main() {
    fragColor = shader();
}"""
        assert code == expected

    def test_or_operator(self):
        """Test 'or' operator."""

        def shader(ctx: ShaderContext) -> vec4:
            x: float = 0.0
            if ctx.u_time < 0.0 or ctx.u_time > 1.0:
                x = 1.0
            return vec4(x, 0.0, 0.0, 1.0)

        code, _ = transpile(shader)
        expected = """\
#version 460 core

in vec2 vs_uv;
uniform float u_time;
uniform vec2 u_resolution;
uniform float u_aspect;
uniform vec2 u_mouse_pos;
uniform vec2 u_mouse_uv;
out vec4 fragColor;

vec4 shader() {
    float x = 0.0;
    if (((u_time < 0.0) || (u_time > 1.0))) {
        x = 1.0;
    }
    return vec4(x, 0.0, 0.0, 1.0);
}

void main() {
    fragColor = shader();
}"""
        assert code == expected

    def test_not_operator(self):
        """Test 'not' operator."""

        def shader(ctx: ShaderContext) -> vec4:
            flag: bool = True
            x: float = 0.0
            if not flag:
                x = 1.0
            return vec4(x, 0.0, 0.0, 1.0)

        code, _ = transpile(shader)
        expected = """\
#version 460 core

in vec2 vs_uv;
uniform float u_time;
uniform vec2 u_resolution;
uniform float u_aspect;
uniform vec2 u_mouse_pos;
uniform vec2 u_mouse_uv;
out vec4 fragColor;

vec4 shader() {
    bool flag = true;
    float x = 0.0;
    if (!flag) {
        x = 1.0;
    }
    return vec4(x, 0.0, 0.0, 1.0);
}

void main() {
    fragColor = shader();
}"""
        assert code == expected


class TestBreakContinue:
    """Test break and continue statements."""

    def test_break_in_for(self):
        """Test break statement in for loop."""

        def shader(ctx: ShaderContext) -> vec4:
            x: float = 0.0
            for i in range(10):
                if i > 5:
                    break
                x = x + 1.0
            return vec4(x, 0.0, 0.0, 1.0)

        code, _ = transpile(shader)
        expected = """\
#version 460 core

in vec2 vs_uv;
uniform float u_time;
uniform vec2 u_resolution;
uniform float u_aspect;
uniform vec2 u_mouse_pos;
uniform vec2 u_mouse_uv;
out vec4 fragColor;

vec4 shader() {
    float x = 0.0;
    for (int i = 0; (i < 10); i += 1) {
        if ((i > 5)) {
            break;
        }
        x = (x + 1.0);
    }
    return vec4(x, 0.0, 0.0, 1.0);
}

void main() {
    fragColor = shader();
}"""
        assert code == expected

    def test_continue_in_for(self):
        """Test continue statement in for loop."""

        def shader(ctx: ShaderContext) -> vec4:
            x: float = 0.0
            for i in range(10):
                if i < 5:
                    continue
                x = x + 1.0
            return vec4(x, 0.0, 0.0, 1.0)

        code, _ = transpile(shader)
        expected = """\
#version 460 core

in vec2 vs_uv;
uniform float u_time;
uniform vec2 u_resolution;
uniform float u_aspect;
uniform vec2 u_mouse_pos;
uniform vec2 u_mouse_uv;
out vec4 fragColor;

vec4 shader() {
    float x = 0.0;
    for (int i = 0; (i < 10); i += 1) {
        if ((i < 5)) {
            continue;
        }
        x = (x + 1.0);
    }
    return vec4(x, 0.0, 0.0, 1.0);
}

void main() {
    fragColor = shader();
}"""
        assert code == expected


class TestSwizzling:
    """Test vector swizzling."""

    def test_swizzle_xy(self):
        """Test xy swizzle."""

        def shader(ctx: ShaderContext) -> vec4:
            v = vec3(1.0, 2.0, 3.0)
            xy = v.xy
            return vec4(xy.x, xy.y, 0.0, 1.0)

        code, _ = transpile(shader)
        expected = """\
#version 460 core

in vec2 vs_uv;
uniform float u_time;
uniform vec2 u_resolution;
uniform float u_aspect;
uniform vec2 u_mouse_pos;
uniform vec2 u_mouse_uv;
out vec4 fragColor;

vec4 shader() {
    vec3 v = vec3(1.0, 2.0, 3.0);
    vec2 xy = v.xy;
    return vec4(xy.x, xy.y, 0.0, 1.0);
}

void main() {
    fragColor = shader();
}"""
        assert code == expected


class TestErrors:
    """Test error handling in IR builder."""

    def test_for_non_range_error(self):
        """Test that for loop without range() raises error."""
        code = """
def shader(ctx):
    for i in [1, 2, 3]:
        pass
    return vec4(1.0)
"""
        with pytest.raises(TranspilerError, match="range"):
            transpile(code, main_func="shader")

    def test_for_non_name_target_error(self):
        """Test that for loop with non-name target raises error."""
        code = """
def shader(ctx):
    for a, b in range(10):
        pass
    return vec4(1.0)
"""
        with pytest.raises(TranspilerError, match="simple name"):
            transpile(code, main_func="shader")
