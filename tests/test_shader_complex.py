import asyncio

import glfw
import imageio.v3 as iio
import numpy as np
import pytest
from PIL import Image

from py2glsl import (
    animate,
    py2glsl,
    render_array,
    render_gif,
    render_image,
    render_video,
    vec2,
    vec4,
)
from py2glsl.builtins import length, normalize, sin, smoothstep
from py2glsl.types import Vec2, Vec3, Vec4, vec2, vec3, vec4


def test_complex_shader_0() -> None:
    def shader(vs_uv: vec2, *, u_time: float) -> vec4:
        def sdf_circle(p: vec2, r: float) -> float:
            return length(p) - r

        p = vs_uv * 2.0 - 1.0
        d = sdf_circle(p, 0.5 + sin(u_time) * 0.3)
        color = vec3(1.0 - smoothstep(0.0, 0.01, d))

        return vec4(color, 1.0)

    result = py2glsl(shader)
    assert "float sdf_circle(vec2 p, float r)" in result.fragment_source
    # Check declarations and assignments separately
    assert "vec2 p;" in result.fragment_source
    assert "p = (vs_uv * 2.0) - 1.0;" in result.fragment_source
    assert "vec3 color;" in result.fragment_source
    assert "color = vec3(1.0 - smoothstep(0.0, 0.01, d));" in result.fragment_source


def test_complex_shader_1():
    """Test more complex shader with multiple features"""

    def complex_shader(vs_uv: vec2, *, u_time: float = 0.0) -> vec4:
        # Center UV
        uv = vs_uv * 2.0 - 1.0

        # Create animated circle
        d = length(uv)
        radius = 0.3 + sin(u_time * 2.0) * 0.1
        circle = smoothstep(radius, radius - 0.01, d)

        # Animate color
        r = 0.5 + 0.5 * sin(u_time)
        g = 0.5 + 0.5 * sin(u_time + 2.094)
        b = 0.5 + 0.5 * sin(u_time + 4.189)

        return vec4(circle * r, circle * g, circle * b, circle)

    # Test static render
    arr = render_array(complex_shader, size=(64, 64))
    assert not np.any(np.isnan(arr))
    assert np.all(arr >= 0.0) and np.all(arr <= 1.0)

    # Test animation
    frames = []
    for i in range(5):
        frame = render_array(complex_shader, size=(64, 64), u_time=i * 0.1)
        frames.append(frame)

    # Verify frames are different
    assert not np.array_equal(frames[0], frames[-1])


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

    # Check declarations and assignments separately in sdf_box function
    assert "vec2 d;" in result.fragment_source
    assert "d = abs(p) - b;" in result.fragment_source


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


def test_complex_expressions():
    """Test complex nested expressions"""

    def shader(vs_uv: vec2) -> vec4:
        x = (1.0 + 2.0) * (3.0 - 4.0) / 2.0
        return vec4(x, 0.0, 0.0, 1.0)

    result = py2glsl(shader)
    # Check declaration and assignment separately
    assert "float x;" in result.fragment_source
    assert any(
        expr in result.fragment_source
        for expr in [
            "x = (1.0 + 2.0) * (3.0 - 4.0) / 2.0;",
            "x = ((1.0 + 2.0) * (3.0 - 4.0)) / 2.0;",
        ]
    )
