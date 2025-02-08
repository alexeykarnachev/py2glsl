import pytest

from py2glsl import py2glsl, vec2, vec4


def test_minimal_valid_shader():
    """Test absolute minimal valid shader"""

    def minimal_shader(vs_uv: vec2) -> vec4:
        return vec4(1.0, 0.0, 0.0, 1.0)

    result = py2glsl(minimal_shader)
    assert "vec4 shader(vec2 vs_uv)" in result.fragment_source
    assert "return vec4(1.0, 0.0, 0.0, 1.0)" in result.fragment_source


def test_simple_uniform():
    """Test shader with single uniform"""

    def uniform_shader(vs_uv: vec2, *, color: float) -> vec4:
        return vec4(color, 0.0, 0.0, 1.0)

    result = py2glsl(uniform_shader)
    assert "uniform float color;" in result.fragment_source


def test_simple_variable():
    """Test local variable declaration"""

    def var_shader(vs_uv: vec2) -> vec4:
        x = 1.0
        return vec4(x, 0.0, 0.0, 1.0)

    result = py2glsl(var_shader)
    assert "float x = 1.0;" in result.fragment_source


def test_simple_arithmetic():
    """Test basic arithmetic operations"""

    def math_shader(vs_uv: vec2) -> vec4:
        x = 1.0 + 2.0
        y = 3.0 * 4.0
        return vec4(x, y, 0.0, 1.0)

    result = py2glsl(math_shader)
    assert "float x = 1.0 + 2.0;" in result.fragment_source
    assert "float y = 3.0 * 4.0;" in result.fragment_source


def test_simple_vector_ops():
    """Test basic vector operations"""

    def vec_shader(vs_uv: vec2) -> vec4:
        v = vs_uv * 2.0
        return vec4(v, 0.0, 1.0)

    result = py2glsl(vec_shader)
    assert "vec2 v = vs_uv * 2.0;" in result.fragment_source


def test_simple_if():
    """Test simple if statement"""

    def if_shader(vs_uv: vec2, *, threshold: float) -> vec4:
        if threshold > 0.5:
            return vec4(1.0, 0.0, 0.0, 1.0)
        return vec4(0.0, 1.0, 0.0, 1.0)

    result = py2glsl(if_shader)
    assert "if (threshold > 0.5)" in result.fragment_source


def test_simple_function():
    """Test simple nested function"""

    def func_shader(vs_uv: vec2) -> vec4:
        def double(x: float) -> float:
            return x * 2.0

        val = double(0.5)
        return vec4(val, 0.0, 0.0, 1.0)

    result = py2glsl(func_shader)
    assert "float double(float x)" in result.fragment_source
    assert "return x * 2.0;" in result.fragment_source


def test_type_inference_simple():
    """Test basic type inference"""

    def type_shader(vs_uv: vec2) -> vec4:
        a = 1.0  # float
        b = vs_uv  # vec2
        c = vec4(1.0, 2.0, 3.0, 4.0)  # vec4
        return c

    result = py2glsl(type_shader)
    assert "float a = 1.0;" in result.fragment_source
    assert "vec2 b = vs_uv;" in result.fragment_source
    assert "vec4 c = vec4(1.0, 2.0, 3.0, 4.0);" in result.fragment_source


def test_swizzle_simple():
    """Test basic swizzling operations"""

    def swizzle_shader(vs_uv: vec2) -> vec4:
        xy = vs_uv.xy
        yx = vs_uv.yx
        return vec4(xy.x, xy.y, yx.x, yx.y)

    result = py2glsl(swizzle_shader)
    assert "vec2 xy = vs_uv.xy;" in result.fragment_source
    assert "vec2 yx = vs_uv.yx;" in result.fragment_source


def test_builtin_simple():
    """Test basic built-in function usage"""

    def builtin_shader(vs_uv: vec2) -> vec4:
        l = length(vs_uv)
        n = normalize(vs_uv)
        return vec4(n, l, 1.0)

    result = py2glsl(builtin_shader)
    assert "float l = length(vs_uv);" in result.fragment_source
    assert "vec2 n = normalize(vs_uv);" in result.fragment_source


def test_multiple_uniforms():
    """Test multiple uniforms with different types"""

    def shader(vs_uv: vec2, *, scale: float, offset: vec2, color: vec4) -> vec4:
        return color * vec4(vs_uv * scale + offset, 0.0, 1.0)

    result = py2glsl(shader)
    assert "uniform float scale;" in result.fragment_source
    assert "uniform vec2 offset;" in result.fragment_source
    assert "uniform vec4 color;" in result.fragment_source


def test_compound_assignment():
    """Test compound assignments (+=, -=, etc)"""

    def shader(vs_uv: vec2) -> vec4:
        x = 1.0
        x += 2.0
        x *= 3.0
        return vec4(x, 0.0, 0.0, 1.0)

    result = py2glsl(shader)
    assert "x += 2.0;" in result.fragment_source
    assert "x *= 3.0;" in result.fragment_source


def test_nested_if_else():
    """Test nested if-else structures"""

    def shader(vs_uv: vec2, *, x: float, y: float) -> vec4:
        if x > 0.0:
            if y > 0.0:
                return vec4(1.0, 0.0, 0.0, 1.0)
            else:
                return vec4(0.0, 1.0, 0.0, 1.0)
        return vec4(0.0, 0.0, 1.0, 1.0)

    result = py2glsl(shader)
    shader_body = result.fragment_source[result.fragment_source.find("shader(") :]
    assert shader_body.count("if") == 2
    assert shader_body.count("else") == 1


def test_chained_assignments():
    """Test chained assignments"""

    def shader(vs_uv: vec2) -> vec4:
        x = y = z = 1.0
        return vec4(x, y, z, 1.0)

    result = py2glsl(shader)
    assert "float x = 1.0;" in result.fragment_source
    assert "float y = 1.0;" in result.fragment_source
    assert "float z = 1.0;" in result.fragment_source


def test_complex_expressions():
    """Test complex nested expressions"""

    def shader(vs_uv: vec2) -> vec4:
        x = (1.0 + 2.0) * (3.0 - 4.0) / 2.0
        return vec4(x, 0.0, 0.0, 1.0)

    result = py2glsl(shader)
    # Accept both forms of parentheses
    assert any(
        expr in result.fragment_source
        for expr in [
            "(1.0 + 2.0) * (3.0 - 4.0) / 2.0",
            "((1.0 + 2.0) * (3.0 - 4.0)) / 2.0",
        ]
    )


def test_function_calls_chain():
    """Test chained function calls"""

    def shader(vs_uv: vec2) -> vec4:
        return vec4(normalize(abs(vs_uv * 2.0 - 1.0)), 0.0, 1.0)

    result = py2glsl(shader)
    # Accept both forms of parentheses
    assert any(
        expr in result.fragment_source
        for expr in [
            "normalize(abs(vs_uv * 2.0 - 1.0))",
            "normalize(abs((vs_uv * 2.0) - 1.0))",
        ]
    )


def test_multiple_nested_functions():
    """Test multiple nested function definitions"""

    def shader(vs_uv: vec2) -> vec4:
        def f1(x: float) -> float:
            return x * 2.0

        def f2(x: float) -> float:
            return f1(x) + 1.0

        val = f2(0.5)
        return vec4(val, 0.0, 0.0, 1.0)

    result = py2glsl(shader)
    assert "float f1(float x)" in result.fragment_source
    assert "float f2(float x)" in result.fragment_source
    assert "return f1(x) + 1.0;" in result.fragment_source


def test_vector_constructors():
    """Test different vector construction patterns"""

    def shader(vs_uv: vec2) -> vec4:
        v2a = vec2(1.0, 2.0)
        v2b = vec2(v2a.x)  # same value
        v3a = vec3(v2a, 3.0)
        v3b = vec3(1.0)  # all same
        v4a = vec4(v3a, 1.0)
        v4b = vec4(v2a, v2b)
        return v4a + v4b

    result = py2glsl(shader)
    assert "vec2 v2a = vec2(1.0, 2.0);" in result.fragment_source
    assert "vec2 v2b = vec2(v2a.x);" in result.fragment_source
    assert "vec3 v3a = vec3(v2a, 3.0);" in result.fragment_source
    assert "vec3 v3b = vec3(1.0);" in result.fragment_source
    assert "vec4 v4a = vec4(v3a, 1.0);" in result.fragment_source
    assert "vec4 v4b = vec4(v2a, v2b);" in result.fragment_source


def test_complex_swizzling():
    """Test complex swizzling operations"""

    def shader(vs_uv: vec2) -> vec4:
        v4 = vec4(vs_uv, 0.0, 1.0)
        xyz = v4.xyz
        zyx = v4.zyx
        return vec4(xyz.xy, zyx.xy)

    result = py2glsl(shader)
    assert "vec4 v4 = vec4(vs_uv, 0.0, 1.0);" in result.fragment_source
    assert "vec3 xyz = v4.xyz;" in result.fragment_source
    assert "vec3 zyx = v4.zyx;" in result.fragment_source


def test_type_inference_complex():
    """Test complex type inference scenarios"""

    def shader(vs_uv: vec2, *, scale: float) -> vec4:
        # Type should be inferred from operations
        a = vs_uv * scale  # vec2
        b = vs_uv.x * scale  # float
        c = normalize(a)  # vec2
        d = length(a)  # float
        return vec4(c, b, d)

    result = py2glsl(shader)
    assert "vec2 a = vs_uv * scale;" in result.fragment_source
    assert "float b = vs_uv.x * scale;" in result.fragment_source
    assert "vec2 c = normalize(a);" in result.fragment_source
    assert "float d = length(a);" in result.fragment_source


def test_builtin_functions_chain():
    """Test chained built-in function calls"""

    def shader(vs_uv: vec2) -> vec4:
        v = normalize(abs(sin(vs_uv * 6.28318530718)))
        l = length(clamp(v, 0.0, 1.0))
        return vec4(v, l, 1.0)

    result = py2glsl(shader)
    assert "normalize(abs(sin(vs_uv * 6.28318530718)))" in result.fragment_source
    assert "length(clamp(v, 0.0, 1.0))" in result.fragment_source


def test_error_first_argument_name():
    """Test error when first argument is not named vs_uv"""
    with pytest.raises(TypeError, match="First argument must be vs_uv"):

        def shader(pos: vec2, *, u_time: float) -> vec4:
            return vec4(1.0, 0.0, 0.0, 1.0)

        py2glsl(shader)


def test_vector_swizzle_type():
    """Test vector swizzling with correct type inference"""

    def shader(vs_uv: vec2) -> vec4:
        v4 = vec4(1.0, 2.0, 3.0, 4.0)
        rgb = v4.rgb  # Should be vec3
        xy = v4.xy  # Should be vec2
        x = v4.x  # Should be float
        return vec4(x, xy, 1.0)

    result = py2glsl(shader)
    assert "vec3 rgb = v4.rgb;" in result.fragment_source
    assert "vec2 xy = v4.xy;" in result.fragment_source
    assert "float x = v4.x;" in result.fragment_source


def test_parentheses_consistency():
    """Test consistent parentheses in expressions"""

    def shader(vs_uv: vec2) -> vec4:
        x = (1.0 + 2.0) * (3.0 - 4.0)
        y = vs_uv * 2.0 - 1.0
        return vec4(x, y, 0.0)

    result = py2glsl(shader)
    assert "(" in result.fragment_source and ")" in result.fragment_source
    assert "vs_uv * 2.0" in result.fragment_source


def test_function_return_type():
    """Test function return type preservation"""

    def shader(vs_uv: vec2) -> vec4:
        def get_normal(p: vec2) -> vec2:
            return normalize(p)

        n = get_normal(vs_uv)
        return vec4(n, 0.0, 1.0)

    result = py2glsl(shader)
    assert "vec2 get_normal(vec2 p)" in result.fragment_source
    assert "vec2 n = get_normal(vs_uv);" in result.fragment_source


def test_builtin_function_types():
    """Test built-in function type inference"""

    def shader(vs_uv: vec2) -> vec4:
        d = length(vs_uv)  # float
        n = normalize(vs_uv)  # vec2
        m = mix(vec3(1), vec3(0), 0.5)  # vec3
        return vec4(d, n, 1.0)

    result = py2glsl(shader)
    assert "float d = length(vs_uv);" in result.fragment_source
    assert "vec2 n = normalize(vs_uv);" in result.fragment_source
    assert "vec3 m = mix(vec3(1.0), vec3(0.0), 0.5);" in result.fragment_source


def test_code_formatting():
    """Test consistent code formatting"""

    def shader(vs_uv: vec2) -> vec4:
        if length(vs_uv) > 1.0:
            return vec4(1.0)
        return vec4(0.0)

    result = py2glsl(shader)
    # Check for consistent bracing style
    assert "{\n" in result.fragment_source
    assert "if (length(vs_uv) > 1.0)\n" in result.fragment_source


def test_vector_operations_types():
    """Test vector operation type preservation"""

    def shader(vs_uv: vec2) -> vec4:
        v2 = vec2(1.0, 2.0)
        v3 = vec3(v2, 3.0)
        v4 = vec4(v3, 1.0)
        scaled = v2 * 2.0  # Should stay vec2
        added = v2 + v2  # Should stay vec2
        return v4

    result = py2glsl(shader)
    assert "vec2 scaled = v2 * 2.0;" in result.fragment_source
    assert "vec2 added = v2 + v2;" in result.fragment_source


def test_math_function_types():
    """Test math function type preservation"""

    def shader(vs_uv: vec2) -> vec4:
        angle = atan(vs_uv.y, vs_uv.x)  # float
        s = sin(angle)  # float
        c = cos(angle)  # float
        return vec4(s, c, 0.0, 1.0)

    result = py2glsl(shader)
    assert "float angle = atan(vs_uv.y, vs_uv.x);" in result.fragment_source
    assert "float s = sin(angle);" in result.fragment_source
    assert "float c = cos(angle);" in result.fragment_source


def test_nested_function_types():
    """Test nested function type preservation"""

    def shader(vs_uv: vec2) -> vec4:
        def circle_sdf(p: vec2, r: float) -> float:
            return length(p) - r

        def smooth_min(a: float, b: float, k: float) -> float:
            h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
            return mix(b, a, h) - k * h * (1.0 - h)

        d = circle_sdf(vs_uv, 0.5)
        s = smooth_min(d, 0.0, 0.1)
        return vec4(s)

    result = py2glsl(shader)
    assert "float circle_sdf(vec2 p, float r)" in result.fragment_source
    assert "float smooth_min(float a, float b, float k)" in result.fragment_source
    assert "float d = circle_sdf(vs_uv, 0.5);" in result.fragment_source


def test_error_messages():
    """Test specific error messages"""

    def shader1(pos: float) -> vec4:
        return vec4(1.0)

    def shader2(vs_uv: vec2) -> float:
        return 1.0

    def shader3(vs_uv: vec2, time: float) -> vec4:
        return vec4(1.0)

    with pytest.raises(TypeError, match="First argument must be vs_uv"):
        py2glsl(shader1)

    with pytest.raises(TypeError, match="Shader must return vec4"):
        py2glsl(shader2)

    with pytest.raises(TypeError, match="All arguments except vs_uv must be uniforms"):
        py2glsl(shader3)


def test_code_formatting_style():
    """Test GLSL code formatting rules"""

    def shader(vs_uv: vec2, *, u_val: float) -> vec4:
        if u_val > 0.0:
            return vec4(1.0)
        return vec4(0.0)

    result = py2glsl(shader)

    # Check only version, in/out declarations, and uniforms have no indentation
    for line in result.fragment_source.split("\n"):
        if line and not line.isspace():
            if any(
                line.startswith(prefix)
                for prefix in ["#version", "in ", "out ", "uniform "]
            ):
                assert not line.startswith("    ")


def test_expression_grouping():
    """Test expression parentheses rules"""

    def shader(vs_uv: vec2) -> vec4:
        x = vs_uv.x * 2.0 - 1.0
        y = (vs_uv.y * 2.0) - 1.0
        return vec4(x, y, 0.0, 1.0)

    result = py2glsl(shader)

    # Both forms should produce same output
    assert "(vs_uv.x * 2.0) - 1.0" in result.fragment_source
    assert "(vs_uv.y * 2.0) - 1.0" in result.fragment_source


def test_type_inference_consistency():
    """Test consistent type inference"""

    def shader(vs_uv: vec2, *, u_dir: vec2) -> vec4:
        n = normalize(u_dir)  # Should stay vec2
        d = dot(n, vs_uv)  # Should be float
        return vec4(d)

    result = py2glsl(shader)

    assert "vec2 n = normalize(u_dir)" in result.fragment_source
    assert "float d = dot(n, vs_uv)" in result.fragment_source


def test_function_formatting():
    """Test function declaration formatting"""

    def shader(vs_uv: vec2) -> vec4:
        def helper(x: float) -> float:
            return x * 2.0

        return vec4(helper(vs_uv.x))

    result = py2glsl(shader)

    assert "float helper(float x)\n{" in result.fragment_source
    assert "vec4 shader(vec2 vs_uv)\n{" in result.fragment_source


def test_indentation_consistency():
    """Test consistent indentation rules"""

    def shader(vs_uv: vec2, *, u_val: float) -> vec4:
        if u_val > 0.0:
            x = 1.0
            if u_val > 1.0:
                x = 2.0
        return vec4(x)

    result = py2glsl(shader)

    lines = result.fragment_source.split("\n")
    indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    assert len(set(indents)) <= 3  # Should only have 0, 4, 8 spaces


def test_vector_swizzle_formatting():
    """Test vector swizzle formatting"""

    def shader(vs_uv: vec2) -> vec4:
        v = vec4(vs_uv.xy, 0.0, 1.0)
        return v.rgba

    result = py2glsl(shader)

    assert "vs_uv.xy" in result.fragment_source
    assert "v.rgba" in result.fragment_source


def test_simple_for_loop():
    """Test basic for loop with integer bounds"""

    def shader(vs_uv: vec2) -> vec4:
        x = 0.0
        for i in range(5):
            x += 1.0
        return vec4(x)

    result = py2glsl(shader)
    assert "for (int i = 0; i < 5; i++)" in result.fragment_source


def test_nested_for_loops():
    """Test nested for loops with integer bounds"""

    def shader(vs_uv: vec2) -> vec4:
        x = 0.0
        for i in range(3):
            for j in range(2):
                x += 1.0
        return vec4(x)

    result = py2glsl(shader)
    assert "for (int i = 0; i < 3; i++)" in result.fragment_source
    assert "for (int j = 0; j < 2; j++)" in result.fragment_source


def test_for_loop_with_range_start():
    """Test for loop with start and end range"""

    def shader(vs_uv: vec2) -> vec4:
        x = 0.0
        for i in range(1, 4):
            x += 1.0
        return vec4(x)

    result = py2glsl(shader)
    assert "for (int i = 1; i < 4; i++)" in result.fragment_source


def test_loop_bounds_integer():
    """Test that loop bounds remain integers"""

    def shader(vs_uv: vec2) -> vec4:
        x = 0.0
        for i in range(5):  # Simple integer bound
            x += 1.0
        return vec4(x)

    result = py2glsl(shader)
    assert "for (int i = 0; i < 5; i++)" in result.fragment_source


def test_loop_bounds_float_error():
    """Test that float loop bounds raise error"""

    def shader(vs_uv: vec2) -> vec4:
        x = 0.0
        for i in range(5.0):  # Should raise error
            x += 1.0
        return vec4(x)

    with pytest.raises(ValueError, match="Loop bounds must be integers"):
        py2glsl(shader)


def test_loop_bounds_expression():
    """Test loop bounds with expressions"""

    def shader(vs_uv: vec2) -> vec4:
        x = 0.0
        count = 3
        for i in range(count + 2):  # Expression that evaluates to integer
            x += 1.0
        return vec4(x)

    result = py2glsl(shader)
    assert "for (int i = 0; i < count + 2; i++)" in result.fragment_source


def test_vertex_shader_interface():
    """Test vertex shader interface generation"""

    def shader(vs_uv: vec2) -> vec4:
        return vec4(1.0, 0.0, 0.0, 1.0)

    result = py2glsl(shader)
    assert "in vec2 vs_uv;" in result.fragment_source
    assert "out vec4 fs_color;" in result.fragment_source


def test_vertex_shader_uv_usage():
    """Test proper vs_uv usage in shader"""

    def shader(vs_uv: vec2) -> vec4:
        return vec4(vs_uv.x, vs_uv.y, 0.0, 1.0)

    result = py2glsl(shader)
    assert "vs_uv.x" in result.fragment_source
    assert "vs_uv.y" in result.fragment_source


def test_vertex_shader_swizzle():
    """Test vs_uv swizzling operations"""

    def shader(vs_uv: vec2) -> vec4:
        xy = vs_uv.xy
        yx = vs_uv.yx
        return vec4(xy.x, xy.y, yx.x, yx.y)

    result = py2glsl(shader)
    assert "vec2 xy = vs_uv.xy;" in result.fragment_source
    assert "vec2 yx = vs_uv.yx;" in result.fragment_source


def test_vertex_shader_precision():
    """Test precision handling with vs_uv coordinates"""

    def shader(vs_uv: vec2) -> vec4:
        uv = vs_uv * 2.0 - 1.0  # Convert [0,1] to [-1,1]
        return vec4(uv, 0.0, 1.0)

    result = py2glsl(shader)
    assert "vec2 uv = (vs_uv * 2.0) - 1.0;" in result.fragment_source


def test_vertex_shader_with_uniforms():
    """Test vs_uv interaction with uniforms"""

    def shader(vs_uv: vec2, *, u_scale: float) -> vec4:
        pos = vs_uv * u_scale
        return vec4(pos, 0.0, 1.0)

    result = py2glsl(shader)
    assert "uniform float u_scale;" in result.fragment_source
    assert "vec2 pos = vs_uv * u_scale;" in result.fragment_source


def test_vertex_shader_function_params():
    """Test vs_uv usage in function parameters"""

    def shader(vs_uv: vec2) -> vec4:
        def transform(p: vec2) -> vec2:
            return p * 2.0 - 1.0

        pos = transform(vs_uv)
        return vec4(pos, 0.0, 1.0)

    result = py2glsl(shader)
    assert "vec2 transform(vec2 p)" in result.fragment_source
    assert "return (p * 2.0) - 1.0;" in result.fragment_source  # Fixed parentheses


def test_vertex_shader_complex_usage():
    """Test complex vs_uv manipulations"""

    def shader(vs_uv: vec2) -> vec4:
        def polar(p: vec2) -> vec2:
            return vec2(length(p), atan(p.y, p.x))

        center = vs_uv * 2.0 - 1.0
        polar_coords = polar(center)
        return vec4(polar_coords, 0.0, 1.0)

    result = py2glsl(shader)
    assert "vec2 center = (vs_uv * 2.0) - 1.0;" in result.fragment_source
    assert "vec2 polar_coords = polar(center);" in result.fragment_source


def test_vertex_shader_resolution():
    """Test vs_uv with resolution uniform"""

    def shader(vs_uv: vec2, *, u_resolution: vec2) -> vec4:
        aspect = u_resolution.x / u_resolution.y
        pos = vs_uv
        pos.x *= aspect
        return vec4(pos, 0.0, 1.0)

    result = py2glsl(shader)
    assert "uniform vec2 u_resolution;" in result.fragment_source
    assert "float aspect = u_resolution.x / u_resolution.y;" in result.fragment_source


def test_vertex_shader_time():
    """Test vs_uv with time-based animation"""

    def shader(vs_uv: vec2, *, u_time: float) -> vec4:
        pos = vs_uv * 2.0 - 1.0
        angle = u_time
        x = pos.x * cos(angle) - pos.y * sin(angle)
        y = pos.x * sin(angle) + pos.y * cos(angle)
        return vec4(x, y, 0.0, 1.0)

    result = py2glsl(shader)
    assert "uniform float u_time;" in result.fragment_source
    assert "vec2 pos = (vs_uv * 2.0) - 1.0;" in result.fragment_source


def test_vertex_shader_mouse():
    """Test vs_uv with mouse interaction"""

    def shader(vs_uv: vec2, *, u_mouse: vec2) -> vec4:
        dist = length(vs_uv - u_mouse)
        glow = 0.1 / (dist + 0.1)
        return vec4(glow, glow, glow, 1.0)

    result = py2glsl(shader)
    assert "uniform vec2 u_mouse;" in result.fragment_source
    assert "float dist = length(vs_uv - u_mouse);" in result.fragment_source


def test_vertex_shader_input_attributes():
    """Test vertex shader input attribute declarations"""

    def shader(vs_uv: vec2) -> vec4:
        return vec4(1.0)

    result = py2glsl(shader)
    # Verify vertex shader has proper input declarations
    assert "layout(location = 0) in vec2 in_pos;" in result.vertex_source
    assert "layout(location = 1) in vec2 in_uv;" in result.vertex_source
    assert "out vec2 vs_uv;" in result.vertex_source


def test_uniform_declaration_and_usage():
    """Test uniform declaration and usage in shaders"""

    def shader(vs_uv: vec2, *, u_time: float, u_resolution: vec2) -> vec4:
        return vec4(u_time, u_resolution.x, u_resolution.y, 1.0)

    result = py2glsl(shader)
    # Verify uniform declarations
    assert "uniform float u_time;" in result.fragment_source
    assert "uniform vec2 u_resolution;" in result.fragment_source


def test_precision_handling():
    """Test numerical precision handling in shaders"""

    def shader(vs_uv: vec2) -> vec4:
        x = 0.0  # Should be exactly 0.0
        y = 1.0  # Should be exactly 1.0
        return vec4(x, y, 0.0, 1.0)

    result = py2glsl(shader)
    # Verify exact float values
    assert "0.0" in result.fragment_source
    assert "1.0" in result.fragment_source


def test_glsl_syntax_validation():
    """Test GLSL syntax validation"""

    def shader(vs_uv: vec2) -> vec4:
        render_called = True  # This should be converted to float/bool
        return vec4(1.0)

    result = py2glsl(shader)
    # Verify Python bool is converted to GLSL bool/float
    assert (
        "bool render_called = true;" in result.fragment_source
        or "float render_called = 1.0;" in result.fragment_source
    )


def test_vertex_shader_coordinate_mapping():
    """Test vertex shader coordinate mapping"""

    def shader(vs_uv: vec2) -> vec4:
        # Map [0,1] to [-1,1]
        pos = vs_uv * 2.0 - 1.0
        return vec4(pos, 0.0, 1.0)

    result = py2glsl(shader)
    # Verify coordinate transformation
    assert "vec2 pos = (vs_uv * 2.0) - 1.0;" in result.fragment_source


def test_shader_interface_validation():
    """Test shader interface validation"""

    def shader(vs_uv: vec2) -> vec4:
        return vec4(vs_uv, 0.0, 1.0)

    result = py2glsl(shader)
    # Verify shader interface
    assert "in vec2 vs_uv;" in result.fragment_source
    assert "out vec4 fs_color;" in result.fragment_source
    assert "void main()" in result.fragment_source
    assert "fs_color = shader(vs_uv);" in result.fragment_source


def test_uniform_type_validation():
    """Test uniform type validation"""

    def shader(vs_uv: vec2, *, u_float: float, u_vec2: vec2, u_vec4: vec4) -> vec4:
        return u_vec4

    result = py2glsl(shader)
    # Verify uniform type declarations
    assert "uniform float u_float;" in result.fragment_source
    assert "uniform vec2 u_vec2;" in result.fragment_source
    assert "uniform vec4 u_vec4;" in result.fragment_source


def test_shader_main_function():
    """Test shader main function generation"""

    def shader(vs_uv: vec2) -> vec4:
        return vec4(1.0)

    result = py2glsl(shader)
    expected_main = """
void main()
{
    fs_color = shader(vs_uv);
}
""".strip()
    assert expected_main in result.fragment_source


def test_bool_conversion():
    """Test basic boolean literal conversion"""

    def shader(vs_uv: vec2) -> vec4:
        x = True
        y = False
        return vec4(1.0)

    result = py2glsl(shader)
    assert "bool x = true;" in result.fragment_source
    assert "bool y = false;" in result.fragment_source
