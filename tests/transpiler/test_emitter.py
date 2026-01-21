"""Tests for the code emitter module."""

from dataclasses import dataclass

from py2glsl import ShaderContext, TargetType
from py2glsl.builtins import vec3, vec4
from py2glsl.transpiler import transpile


class TestFunctionOrdering:
    """Test that functions are emitted in correct dependency order."""

    def test_callee_before_caller(self):
        """Test that called functions appear before callers in output."""

        def helper_a(x: "float") -> "float":
            return x * 2.0

        def helper_b(x: "float") -> "float":
            return helper_a(x) + 1.0

        def shader(ctx: ShaderContext) -> vec4:
            result = helper_b(ctx.vs_uv.x)
            return vec4(result, 0.0, 0.0, 1.0)

        code, _ = transpile(helper_a, helper_b, shader, main_func="shader")
        expected = """\
#version 460 core

in vec2 vs_uv;
uniform float u_time;
uniform vec2 u_resolution;
uniform float u_aspect;
uniform vec2 u_mouse_pos;
uniform vec2 u_mouse_uv;
out vec4 fragColor;

float helper_a(float x) {
    return x * 2.0;
}

float helper_b(float x) {
    return helper_a(x) + 1.0;
}

vec4 shader() {
    float result = helper_b(vs_uv.x);
    return vec4(result, 0.0, 0.0, 1.0);
}

void main() {
    fragColor = shader();
}"""
        assert code == expected


class TestStructEmission:
    """Test struct code emission."""

    def test_struct_definition(self):
        """Test that struct is properly emitted."""

        @dataclass
        class Material:
            color: "vec3"
            roughness: "float"

        def shader(ctx: ShaderContext) -> vec4:
            mat = Material(vec3(1.0, 0.0, 0.0), 0.5)
            return vec4(mat.color.x, mat.color.y, mat.color.z, 1.0)

        code, _ = transpile(Material, shader, main_func="shader")
        expected = """\
#version 460 core

struct Material {
    vec3 color;
    float roughness;
};

in vec2 vs_uv;
uniform float u_time;
uniform vec2 u_resolution;
uniform float u_aspect;
uniform vec2 u_mouse_pos;
uniform vec2 u_mouse_uv;
out vec4 fragColor;

vec4 shader() {
    Material mat = Material(vec3(1.0, 0.0, 0.0), 0.5);
    return vec4(mat.color.x, mat.color.y, mat.color.z, 1.0);
}

void main() {
    fragColor = shader();
}"""
        assert code == expected


class TestConstEmission:
    """Test const variable emission."""

    def test_const_variables(self):
        """Test const variable emission."""

        def shader(ctx: ShaderContext) -> vec4:
            return vec4(1.0, 0.0, 0.0, 1.0)

        code, _ = transpile(shader, MAX_DIST=100.0, PI=3.14159)
        expected = """\
#version 460 core

const float MAX_DIST = 100.0;
const float PI = 3.14159;
in vec2 vs_uv;
uniform float u_time;
uniform vec2 u_resolution;
uniform float u_aspect;
uniform vec2 u_mouse_pos;
uniform vec2 u_mouse_uv;
out vec4 fragColor;

vec4 shader() {
    return vec4(1.0, 0.0, 0.0, 1.0);
}

void main() {
    fragColor = shader();
}"""
        assert code == expected


class TestTargetSpecificEmission:
    """Test target-specific code emission."""

    def test_opengl46_target(self):
        """Test OpenGL 4.6 target."""

        def shader(ctx: ShaderContext) -> vec4:
            return vec4(ctx.vs_uv.x, ctx.vs_uv.y, 0.0, 1.0)

        code, _ = transpile(shader, target=TargetType.OPENGL46)
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
    return vec4(vs_uv.x, vs_uv.y, 0.0, 1.0);
}

void main() {
    fragColor = shader();
}"""
        assert code == expected

    def test_opengl33_target(self):
        """Test OpenGL 3.3 target."""

        def shader(ctx: ShaderContext) -> vec4:
            return vec4(ctx.vs_uv.x, ctx.vs_uv.y, 0.0, 1.0)

        code, _ = transpile(shader, target=TargetType.OPENGL33)
        expected = """\
#version 330 core

in vec2 vs_uv;
uniform float u_time;
uniform vec2 u_resolution;
uniform float u_aspect;
uniform vec2 u_mouse_pos;
uniform vec2 u_mouse_uv;
out vec4 fragColor;

vec4 shader() {
    return vec4(vs_uv.x, vs_uv.y, 0.0, 1.0);
}

void main() {
    fragColor = shader();
}"""
        assert code == expected

    def test_shadertoy_target(self):
        """Test Shadertoy target outputs paste-able code."""

        def shader(ctx: ShaderContext) -> vec4:
            return vec4(ctx.vs_uv.x, ctx.vs_uv.y, 0.0, 1.0)

        code, _ = transpile(shader, target=TargetType.SHADERTOY)
        # Shadertoy target outputs clean, paste-able code with no version/uniforms
        expected = """\
vec4 shader() {
    return vec4((fragCoord / iResolution.xy).x, (fragCoord / iResolution.xy).y, 0.0, 1.0);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    fragColor = shader();
}"""
        assert code == expected

    def test_webgl2_target(self):
        """Test WebGL 2 target."""

        def shader(ctx: ShaderContext) -> vec4:
            return vec4(ctx.vs_uv.x, ctx.vs_uv.y, 0.0, 1.0)

        code, _ = transpile(shader, target=TargetType.WEBGL2)
        expected = """\
#version 300 es
precision highp float;
precision highp int;

in vec2 vs_uv;
uniform float u_time;
uniform vec2 u_resolution;
uniform float u_aspect;
uniform vec2 u_mouse_pos;
uniform vec2 u_mouse_uv;
out vec4 fragColor;

vec4 shader() {
    return vec4(vs_uv.x, vs_uv.y, 0.0, 1.0);
}

void main() {
    fragColor = shader();
}"""
        assert code == expected


class TestPowerOperator:
    """Test power operator emission."""

    def test_power_becomes_pow(self):
        """Test power operator becomes pow() call."""

        def shader(ctx: ShaderContext) -> vec4:
            x = 2.0**3.0
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
    float x = pow(2.0, 3.0);
    return vec4(x, 0.0, 0.0, 1.0);
}

void main() {
    fragColor = shader();
}"""
        assert code == expected
