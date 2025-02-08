import pytest

from py2glsl import (
    abs,
    atan,
    clamp,
    cos,
    dot,
    length,
    max,
    min,
    mix,
    normalize,
    py2glsl,
    sin,
    smoothstep,
    vec2,
    vec3,
    vec4,
)


def test_shader_validation():
    # Invalid: wrong first argument type
    def invalid_shader1(pos: float, *, u_time: float) -> vec4:
        return vec4(1.0, 0.0, 0.0, 1.0)

    # Invalid: wrong return type
    def invalid_shader2(vs_uv: vec2, *, u_time: float) -> float:
        return 1.0

    # Invalid: no uniforms marked with *
    def invalid_shader3(vs_uv: vec2, u_time: float) -> vec4:
        return vec4(1.0, 0.0, 0.0, 1.0)

    # Invalid: missing type hints
    def invalid_shader4(vs_uv, *, u_time) -> vec4:
        return vec4(1.0, 0.0, 0.0, 1.0)

    with pytest.raises(TypeError, match="First argument must be vec2"):
        py2glsl(invalid_shader1)

    with pytest.raises(TypeError, match="Shader must return vec4"):
        py2glsl(invalid_shader2)

    with pytest.raises(TypeError, match="All arguments except vs_uv must be uniforms"):
        py2glsl(invalid_shader3)

    with pytest.raises(TypeError, match="All arguments must have type hints"):
        py2glsl(invalid_shader4)


def test_vector_operations():
    def vector_ops_shader(vs_uv: vec2, *, u_scale: float) -> vec4:
        # Vector construction
        v2 = vec2(1.0, 2.0)
        v3 = vec3(v2, 3.0)
        v4 = vec4(v3, 1.0)

        # Vector swizzling
        xy = v3.xy
        rgb = v4.rgb

        # Vector math
        scaled = vs_uv * u_scale
        added = v2 + scaled
        normalized = normalize(added)

        return vec4(normalized, 0.0, 1.0)

    result = py2glsl(vector_ops_shader)

    expected = """
    #version 460
    
    in vec2 vs_uv;
    out vec4 fs_color;
    uniform float u_scale;

    vec4 shader(vec2 vs_uv) {
        vec2 v2 = vec2(1.0, 2.0);
        vec3 v3 = vec3(v2, 3.0);
        vec4 v4 = vec4(v3, 1.0);
        
        vec2 xy = v3.xy;
        vec3 rgb = v4.rgb;
        
        vec2 scaled = vs_uv * u_scale;
        vec2 added = v2 + scaled;
        vec2 normalized = normalize(added);
        
        return vec4(normalized, 0.0, 1.0);
    }

    void main() {
        fs_color = shader(vs_uv);
    }
    """.strip()

    assert result.fragment_source.strip() == expected


def test_dot_product():
    def dot_shader(vs_uv: vec2, *, u_light_dir: vec2) -> vec4:
        def get_normal(p: vec2) -> vec2:
            return normalize(p)

        p = vs_uv * 2.0 - 1.0
        n = get_normal(p)
        light_dir = normalize(u_light_dir)

        # Basic diffuse lighting using dot product
        diffuse = max(dot(n, light_dir), 0.0)

        # Specular using dot for reflection
        reflected = normalize(2.0 * dot(n, light_dir) * n - light_dir)
        specular = pow(max(dot(reflected, vec2(0.0, 1.0)), 0.0), 32.0)

        return vec4(vec3(diffuse * 0.7 + specular * 0.3), 1.0)

    result = py2glsl(dot_shader)

    expected = """
    #version 460
    
    in vec2 vs_uv;
    out vec4 fs_color;
    uniform vec2 u_light_dir;

    vec2 get_normal(vec2 p) {
        return normalize(p);
    }

    vec4 shader(vec2 vs_uv) {
        vec2 p = vs_uv * 2.0 - 1.0;
        vec2 n = get_normal(p);
        vec2 light_dir = normalize(u_light_dir);
        
        float diffuse = max(dot(n, light_dir), 0.0);
        
        vec2 reflected = normalize(2.0 * dot(n, light_dir) * n - light_dir);
        float specular = pow(max(dot(reflected, vec2(0.0, 1.0)), 0.0), 32.0);
        
        return vec4(vec3(diffuse * 0.7 + specular * 0.3), 1.0);
    }

    void main() {
        fs_color = shader(vs_uv);
    }
    """.strip()

    assert result.fragment_source.strip() == expected


def test_math_functions():
    def math_shader(vs_uv: vec2, *, u_time: float) -> vec4:
        p = vs_uv * 2.0 - 1.0

        # Basic math
        d = length(p)
        angle = atan(p.y, p.x)

        # Trig functions
        s = sin(angle + u_time)
        c = cos(u_time * 2.0)

        # Mix and clamp
        color = mix(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), s)
        intensity = clamp(d, 0.0, 1.0)

        return vec4(color * intensity, 1.0)

    result = py2glsl(math_shader)

    expected = """
    #version 460
    
    in vec2 vs_uv;
    out vec4 fs_color;
    uniform float u_time;

    vec4 shader(vec2 vs_uv) {
        vec2 p = vs_uv * 2.0 - 1.0;
        
        float d = length(p);
        float angle = atan(p.y, p.x);
        
        float s = sin(angle + u_time);
        float c = cos(u_time * 2.0);
        
        vec3 color = mix(vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), s);
        float intensity = clamp(d, 0.0, 1.0);
        
        return vec4(color * intensity, 1.0);
    }

    void main() {
        fs_color = shader(vs_uv);
    }
    """.strip()

    assert result.fragment_source.strip() == expected


def test_nested_functions():
    def nested_shader(vs_uv: vec2, *, u_params: vec4) -> vec4:
        def sdf_circle(p: vec2, r: float) -> float:
            return length(p) - r

        def sdf_box(p: vec2, b: vec2) -> float:
            d = abs(p) - b
            return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0)

        def smooth_min(a: float, b: float, k: float) -> float:
            h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
            return mix(b, a, h) - k * h * (1.0 - h)

        p = vs_uv * 2.0 - 1.0
        circle_d = sdf_circle(p, u_params.x)
        box_d = sdf_box(p, vec2(u_params.y, u_params.z))
        d = smooth_min(circle_d, box_d, u_params.w)

        return vec4(vec3(1.0 - smoothstep(0.0, 0.01, d)), 1.0)

    result = py2glsl(nested_shader)

    expected = """
    #version 460
    
    in vec2 vs_uv;
    out vec4 fs_color;
    uniform vec4 u_params;

    float sdf_circle(vec2 p, float r) {
        return length(p) - r;
    }

    float sdf_box(vec2 p, vec2 b) {
        vec2 d = abs(p) - b;
        return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
    }

    float smooth_min(float a, float b, float k) {
        float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
        return mix(b, a, h) - k * h * (1.0 - h);
    }

    vec4 shader(vec2 vs_uv) {
        vec2 p = vs_uv * 2.0 - 1.0;
        float circle_d = sdf_circle(p, u_params.x);
        float box_d = sdf_box(p, vec2(u_params.y, u_params.z));
        float d = smooth_min(circle_d, box_d, u_params.w);
        
        return vec4(vec3(1.0 - smoothstep(0.0, 0.01, d)), 1.0);
    }

    void main() {
        fs_color = shader(vs_uv);
    }
    """.strip()

    assert result.fragment_source.strip() == expected


def test_control_flow():
    def control_flow_shader(vs_uv: vec2, *, u_mode: float) -> vec4:
        if u_mode < 0.5:
            return vec4(1.0, 0.0, 0.0, 1.0)
        elif u_mode < 1.5:
            return vec4(0.0, 1.0, 0.0, 1.0)
        else:
            return vec4(0.0, 0.0, 1.0, 1.0)

    result = py2glsl(control_flow_shader)

    expected = """
    #version 460
    
    in vec2 vs_uv;
    out vec4 fs_color;
    uniform float u_mode;

    vec4 shader(vec2 vs_uv) {
        if (u_mode < 0.5) {
            return vec4(1.0, 0.0, 0.0, 1.0);
        }
        else if (u_mode < 1.5) {
            return vec4(0.0, 1.0, 0.0, 1.0);
        }
        else {
            return vec4(0.0, 0.0, 1.0, 1.0);
        }
    }

    void main() {
        fs_color = shader(vs_uv);
    }
    """.strip()

    assert result.fragment_source.strip() == expected


def test_loops():
    def loop_shader(vs_uv: vec2, *, u_iterations: float) -> vec4:
        p = vs_uv * 2.0 - 1.0
        z = vec2(0.0)

        for i in range(int(u_iterations)):
            z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + p
            if length(z) > 2.0:
                break

        return vec4(length(z) * 0.5, 0.0, 0.0, 1.0)

    result = py2glsl(loop_shader)

    expected = """
    #version 460
    
    in vec2 vs_uv;
    out vec4 fs_color;
    uniform float u_iterations;

    vec4 shader(vec2 vs_uv) {
        vec2 p = vs_uv * 2.0 - 1.0;
        vec2 z = vec2(0.0);
        
        for (int i = 0; i < int(u_iterations); i++) {
            z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + p;
            if (length(z) > 2.0) {
                break;
            }
        }
        
        return vec4(length(z) * 0.5, 0.0, 0.0, 1.0);
    }

    void main() {
        fs_color = shader(vs_uv);
    }
    """.strip()

    assert result.fragment_source.strip() == expected
