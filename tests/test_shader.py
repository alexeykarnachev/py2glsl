from typing import Any

import pytest

from py2glsl import py2glsl, vec2, vec3, vec4


def test_minimal_valid_shader() -> None:
    def minimal_shader(vs_uv: vec2) -> vec4:
        return vec4(1.0, 0.0, 0.0, 1.0)

    result = py2glsl(minimal_shader)
    assert "vec4 shader(vec2 vs_uv)" in result.fragment_source
    assert "return vec4(1.0, 0.0, 0.0, 1.0)" in result.fragment_source


def test_uniforms() -> None:
    def uniform_shader(vs_uv: vec2, *, color: float) -> vec4:
        return vec4(color, 0.0, 0.0, 1.0)

    result = py2glsl(uniform_shader)
    assert "uniform float color;" in result.fragment_source


def test_variables() -> None:
    def var_shader(vs_uv: vec2) -> vec4:
        x = 1.0
        return vec4(x, 0.0, 0.0, 1.0)

    result = py2glsl(var_shader)
    assert "float x = 1.0;" in result.fragment_source


def test_arithmetic() -> None:
    def math_shader(vs_uv: vec2) -> vec4:
        x = 1.0 + 2.0
        y = 3.0 * 4.0
        return vec4(x, y, 0.0, 1.0)

    result = py2glsl(math_shader)
    assert "float x = 1.0 + 2.0;" in result.fragment_source
    assert "float y = 3.0 * 4.0;" in result.fragment_source


def test_vector_operations() -> None:
    def vec_shader(vs_uv: vec2) -> vec4:
        v = vs_uv * 2.0
        return vec4(v, 0.0, 1.0)

    result = py2glsl(vec_shader)
    assert "vec2 v = vs_uv * 2.0;" in result.fragment_source


def test_control_flow() -> None:
    def if_shader(vs_uv: vec2, *, threshold: float) -> vec4:
        if threshold > 0.5:
            return vec4(1.0, 0.0, 0.0, 1.0)
        return vec4(0.0, 1.0, 0.0, 1.0)

    result = py2glsl(if_shader)
    assert "if (threshold > 0.5)" in result.fragment_source


def test_nested_functions() -> None:
    def func_shader(vs_uv: vec2) -> vec4:
        def double(x: float) -> float:
            return x * 2.0

        val = double(0.5)
        return vec4(val, 0.0, 0.0, 1.0)

    result = py2glsl(func_shader)
    assert "float double(float x)" in result.fragment_source
    assert "return x * 2.0;" in result.fragment_source


def test_type_inference() -> None:
    def type_shader(vs_uv: vec2) -> vec4:
        a = 1.0
        b = vs_uv
        c = vec4(1.0, 2.0, 3.0, 4.0)
        return c

    result = py2glsl(type_shader)
    assert "float a = 1.0;" in result.fragment_source
    assert "vec2 b = vs_uv;" in result.fragment_source
    assert "vec4 c = vec4(1.0, 2.0, 3.0, 4.0);" in result.fragment_source


def test_swizzling() -> None:
    def swizzle_shader(vs_uv: vec2) -> vec4:
        xy = vs_uv.xy
        yx = vs_uv.yx
        return vec4(xy.x, xy.y, yx.x, yx.y)

    result = py2glsl(swizzle_shader)
    assert "vec2 xy = vs_uv.xy;" in result.fragment_source
    assert "vec2 yx = vs_uv.yx;" in result.fragment_source


def test_builtin_functions() -> None:
    def builtin_shader(vs_uv: vec2) -> vec4:
        l = length(vs_uv)
        n = normalize(vs_uv)
        return vec4(n, l, 1.0)

    result = py2glsl(
        builtin_shader
    )  # Fixed: using builtin_shader instead of result_shader
    assert "float l = length(vs_uv);" in result.fragment_source
    assert "vec2 n = normalize(vs_uv);" in result.fragment_source


def test_compound_assignments() -> None:
    def shader(vs_uv: vec2) -> vec4:
        x = 1.0
        x += 2.0
        x *= 3.0
        return vec4(x, 0.0, 0.0, 1.0)

    result = py2glsl(shader)
    assert "x += 2.0;" in result.fragment_source
    assert "x *= 3.0;" in result.fragment_source


def test_error_handling() -> None:
    with pytest.raises(TypeError, match="First argument must be vs_uv"):

        def invalid_shader1(pos: float, *, u_time: float) -> vec4:
            return vec4(1.0, 0.0, 0.0, 1.0)

        py2glsl(invalid_shader1)

    with pytest.raises(TypeError, match="Shader must return vec4"):

        def invalid_shader2(vs_uv: vec2) -> float:
            return 1.0

        py2glsl(invalid_shader2)

    with pytest.raises(TypeError, match="All arguments except vs_uv must be uniforms"):

        def invalid_shader3(vs_uv: vec2, time: float) -> vec4:
            return vec4(1.0)

        py2glsl(invalid_shader3)


def test_complex_shader() -> None:
    def shader(vs_uv: vec2, *, u_time: float) -> vec4:
        def sdf_circle(p: vec2, r: float) -> float:
            return length(p) - r

        p = vs_uv * 2.0 - 1.0
        d = sdf_circle(p, 0.5 + sin(u_time) * 0.3)
        color = vec3(1.0 - smoothstep(0.0, 0.01, d))

        return vec4(color, 1.0)

    result = py2glsl(shader)
    assert "float sdf_circle(vec2 p, float r)" in result.fragment_source
    assert "vec2 p = (vs_uv * 2.0) - 1.0;" in result.fragment_source
    assert (
        "vec3 color = vec3(1.0 - smoothstep(0.0, 0.01, d));" in result.fragment_source
    )


def test_sdf_operations() -> None:
    def shader(vs_uv: vec2, *, u_time: float) -> vec4:
        def sdf_circle(p: vec2, r: float) -> float:
            return length(p) - r

        def sdf_box(p: vec2, b: vec2) -> float:
            d = abs(p) - b
            return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0)

        def smooth_min(a: float, b: float, k: float) -> float:
            h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
            return mix(b, a, h) - k * h * (1.0 - h)

        p = vs_uv * 2.0 - 1.0
        circle_d = sdf_circle(p, 0.5)
        box_d = sdf_box(p, vec2(0.3, 0.3))
        d = smooth_min(circle_d, box_d, 0.2)

        return vec4(vec3(1.0 - smoothstep(0.0, 0.01, d)), 1.0)

    result = py2glsl(shader)
    assert "float sdf_circle(vec2 p, float r)" in result.fragment_source
    assert "float sdf_box(vec2 p, vec2 b)" in result.fragment_source
    assert "float smooth_min(float a, float b, float k)" in result.fragment_source
    assert "vec2 d = abs(p) - b;" in result.fragment_source


def test_complex_swizzling() -> None:
    def shader(vs_uv: vec2, *, u_color: vec4) -> vec4:
        def rotate(p: vec2, a: float) -> vec2:
            c = cos(a)
            s = sin(a)
            return vec2(p.x * c - p.y * s, p.x * s + p.y * c)

        p = vs_uv * 2.0 - 1.0
        p = rotate(p, 0.3)

        color = u_color
        color.yx += p.xy * 0.5

        return color.zyxw

    result = py2glsl(shader)
    assert "vec2 rotate(vec2 p, float a)" in result.fragment_source
    assert "color.yx += p.xy * 0.5;" in result.fragment_source
    assert "return color.zyxw;" in result.fragment_source


def test_nested_control_flow() -> None:
    def shader(vs_uv: vec2, *, u_params: vec4) -> vec4:
        result = vec4(0.0)
        p = vs_uv * 2.0 - 1.0

        if length(p) < u_params.x:
            if p.x > 0.0:
                if p.y > 0.0:
                    result = vec4(1.0, 0.0, 0.0, 1.0)
                else:
                    result = vec4(0.0, 1.0, 0.0, 1.0)
            else:
                result = vec4(0.0, 0.0, 1.0, 1.0)
        else:
            result = mix(vec4(0.5), vec4(u_params.y, u_params.z, u_params.w, 1.0), 0.5)

        return result

    result = py2glsl(shader)
    assert "if (length(p) < u_params.x)" in result.fragment_source
    assert "if (p.x > 0.0)" in result.fragment_source
    assert "if (p.y > 0.0)" in result.fragment_source
    assert (
        "mix(vec4(0.5), vec4(u_params.y, u_params.z, u_params.w, 1.0), 0.5)"
        in result.fragment_source
    )


def test_complex_math() -> None:
    def shader(vs_uv: vec2, *, u_time: float) -> vec4:
        def fbm(p: vec2) -> float:
            def noise(p: vec2) -> float:
                i = floor(p)
                f = p - i
                u = f * f * (3.0 - 2.0 * f)
                return mix(
                    mix(
                        dot(sin(i), vec2(12.9898, 78.233)),
                        dot(sin(i + vec2(1.0, 0.0)), vec2(12.9898, 78.233)),
                        u.x,
                    ),
                    mix(
                        dot(sin(i + vec2(0.0, 1.0)), vec2(12.9898, 78.233)),
                        dot(sin(i + vec2(1.0, 1.0)), vec2(12.9898, 78.233)),
                        u.x,
                    ),
                    u.y,
                )

            value = 0.0
            amplitude = 0.5
            frequency = 1.0

            for i in range(6):
                value += amplitude * noise(p * frequency)
                frequency *= 2.0
                amplitude *= 0.5

            return value

        p = vs_uv * 8.0
        p += u_time * 0.5

        value = fbm(p)
        color = mix(vec3(0.2, 0.3, 0.4), vec3(0.8, 0.7, 0.6), value)

        return vec4(color, 1.0)

    result = py2glsl(shader)
    assert "float fbm(vec2 p)" in result.fragment_source
    assert "float noise(vec2 p)" in result.fragment_source
    assert "for (int i = 0; i < 6; i++)" in result.fragment_source


def test_complex_vector_operations() -> None:
    def shader(vs_uv: vec2, *, u_params: vec4) -> vec4:
        def polar(p: vec2) -> vec2:
            return vec2(length(p), atan(p.y, p.x))

        def cartesian(p: vec2) -> vec2:
            return vec2(p.x * cos(p.y), p.x * sin(p.y))

        p = vs_uv * 2.0 - 1.0
        pol = polar(p)
        pol.y += u_params.z * pol.x + u_params.w
        p = cartesian(pol)

        return vec4(normalize(p), length(p), 1.0)

    result = py2glsl(shader)
    assert "vec2 polar(vec2 p)" in result.fragment_source
    assert "vec2 cartesian(vec2 p)" in result.fragment_source
    assert "pol.y += (u_params.z * pol.x) + u_params.w;" in result.fragment_source


def test_ray_marching() -> None:
    def shader(vs_uv: vec2, *, u_camera: vec4) -> vec4:
        def sdf_scene(p: vec2) -> float:
            def repeat(p: vec2, size: float) -> vec2:
                return p - size * floor(p / size)

            cell = repeat(p, 1.0)
            d1 = length(cell) - 0.3
            d2 = length(cell - 0.5) - 0.1

            return min(d1, d2)

        def ray_march(ro: vec2, rd: vec2) -> float:
            t = 0.0
            for i in range(64):
                p = ro + rd * t
                d = sdf_scene(p)
                t += d
                if d < 0.001:
                    break
                if t > 10.0:
                    break
            return t

        p = vs_uv * 2.0 - 1.0
        ro = vec2(u_camera.x, u_camera.y)
        rd = normalize(p - ro)

        d = ray_march(ro, rd)
        fog = 1.0 / (1.0 + d * d * 0.1)

        return vec4(vec3(fog), 1.0)

    result = py2glsl(shader)
    assert "float sdf_scene(vec2 p)" in result.fragment_source
    assert "vec2 repeat(vec2 p, float size)" in result.fragment_source
    assert "float ray_march(vec2 ro, vec2 rd)" in result.fragment_source
    assert "for (int i = 0; i < 64; i++)" in result.fragment_source
    assert "if (d < 0.001)" in result.fragment_source


def test_complex_uniforms() -> None:
    def shader(
        vs_uv: vec2,
        *,
        u_time: float,
        u_resolution: vec2,
        u_mouse: vec2,
        u_color1: vec4,
        u_color2: vec4,
        u_params: vec4,
    ) -> vec4:
        aspect = u_resolution.x / u_resolution.y
        p = vs_uv * 2.0 - 1.0
        p.x *= aspect

        mouse = u_mouse * 2.0 - 1.0
        mouse.x *= aspect

        d = length(p - mouse)
        glow = 0.02 / d

        color = mix(
            u_color1,
            u_color2,
            smoothstep(
                u_params.x, u_params.y, sin(d * u_params.z + u_time * u_params.w)
            ),
        )

        return color * glow

    result = py2glsl(shader)
    assert "uniform float u_time;" in result.fragment_source
    assert "uniform vec2 u_resolution;" in result.fragment_source
    assert "uniform vec2 u_mouse;" in result.fragment_source
    assert "uniform vec4 u_color1;" in result.fragment_source
    assert "uniform vec4 u_color2;" in result.fragment_source
    assert "uniform vec4 u_params;" in result.fragment_source
    assert "float aspect = u_resolution.x / u_resolution.y;" in result.fragment_source
