"""Gold file based tests for shader transpilation.

Generate/update gold outputs and validate GLSL compiles:
    make gold-generate

Gold files are organized by theme in tests/data/gold/:
    - basic.yaml: Simple returns, arithmetic, swizzling
    - control_flow.yaml: Loops, conditionals, break/continue
    - functions_structs.yaml: Helper functions, structs, constants
    - targets.yaml: Different GLSL targets (OpenGL, Shadertoy, WebGL)

Test case format:
    - name: test_name
      target: opengl46  # optional, default: opengl46
      main_func: shader  # optional, default: shader
      python: |
        PI = 3.14159  # auto-detected as const

        def shader(ctx: ShaderContext) -> vec4:
            return vec4(1.0, 0.0, 0.0, 1.0)
      expected: |
        #version 460 core
        ...
"""

from pathlib import Path

import pytest
import yaml

from py2glsl import TargetType, transpile

# Standard vertex shader for fragment shader compilation
VERTEX_SHADER = """
#version 460 core
in vec2 in_position;
out vec2 vs_uv;
void main() {
    vs_uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

# Version mapping for different targets
VERSION_TO_VERTEX = {
    "#version 460 core": """
#version 460 core
in vec2 in_position;
out vec2 vs_uv;
void main() {
    vs_uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
""",
    "#version 330 core": """
#version 330 core
in vec2 in_position;
out vec2 vs_uv;
void main() {
    vs_uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
""",
    "#version 300 es": """
#version 300 es
precision highp float;
in vec2 in_position;
out vec2 vs_uv;
void main() {
    vs_uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
""",
}

GOLD_DIR = Path(__file__).parent / "data" / "gold"

TARGET_MAP = {
    "opengl46": TargetType.OPENGL46,
    "opengl33": TargetType.OPENGL33,
    "shadertoy": TargetType.SHADERTOY,
    "webgl2": TargetType.WEBGL2,
}


def load_gold_file(filepath: Path) -> list[dict]:
    """Load test cases from a gold file."""
    with open(filepath) as f:
        return yaml.safe_load(f) or []


def load_all_gold_cases() -> list[tuple[str, dict, Path]]:
    """Load all test cases from all gold files."""
    cases = []
    for gold_file in sorted(GOLD_DIR.glob("*.yaml")):
        file_cases = load_gold_file(gold_file)
        for case in file_cases:
            cases.append((case["name"], case, gold_file))
    return cases


def transpile_case(case: dict) -> str:
    """Transpile a test case and return the GLSL code."""
    python_code = case["python"]
    target = TARGET_MAP.get(case.get("target", "opengl46"), TargetType.OPENGL46)
    main_func = case.get("main_func", "shader")

    code, _ = transpile(python_code, target=target, main_func=main_func)
    return code


def get_test_params() -> list[tuple[str, dict, Path]]:
    """Get test parameters for pytest parametrization."""
    return load_all_gold_cases()


def get_test_ids() -> list[str]:
    """Get test IDs for pytest parametrization."""
    return [name for name, _, _ in load_all_gold_cases()]


class TestGoldShaders:
    """Test shader transpilation against gold outputs."""

    @pytest.mark.parametrize(
        "name,case,gold_file", get_test_params(), ids=get_test_ids()
    )
    def test_shader(self, name, case, gold_file):
        """Test that transpiled output matches expected gold output."""
        actual = transpile_case(case)
        expected = case["expected"].rstrip("\n")

        if actual != expected:
            import difflib

            diff = difflib.unified_diff(
                expected.splitlines(keepends=True),
                actual.splitlines(keepends=True),
                fromfile="expected",
                tofile="actual",
            )
            diff_str = "".join(diff)
            pytest.fail(
                f"Output mismatch for '{name}' in {gold_file.name}:\n{diff_str}"
            )


def generate_gold_outputs():
    """Generate gold outputs for all test cases."""
    print("\n" + "=" * 60)
    print("GENERATING GOLD OUTPUTS")
    print("=" * 60)

    total_updated = 0
    total_failed = []

    for gold_file in sorted(GOLD_DIR.glob("*.yaml")):
        cases = load_gold_file(gold_file)
        updated = 0
        failed = []

        print(f"\n{gold_file.name}:")

        for case in cases:
            name = case["name"]
            try:
                actual = transpile_case(case)
                old_expected = case.get("expected", "").rstrip("\n")

                if actual != old_expected:
                    case["expected"] = actual + "\n"
                    updated += 1
                    print(f"  UPDATED: {name}")
                else:
                    print(f"  OK: {name}")
            except Exception as e:
                failed.append((name, str(e)))
                print(f"  FAILED: {name} - {e}")

        # Write back to YAML with literal block style
        class LiteralStr(str):
            pass

        def literal_str_representer(dumper, data):
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")

        yaml.add_representer(LiteralStr, literal_str_representer)

        for case in cases:
            if "\n" in case.get("python", ""):
                case["python"] = LiteralStr(case["python"])
            if "\n" in case.get("expected", ""):
                case["expected"] = LiteralStr(case["expected"])

        with open(gold_file, "w") as f:
            yaml.dump(
                cases,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                width=1000,
            )

        total_updated += updated
        total_failed.extend([(gold_file.name, n, e) for n, e in failed])

    print("\n" + "=" * 60)
    print(f"Total updated: {total_updated}, Total failed: {len(total_failed)}")
    if total_failed:
        print("\nFailed cases:")
        for file, name, error in total_failed:
            print(f"  - {file}/{name}: {error}")
    print("=" * 60)


def get_vertex_shader(fragment_shader: str) -> str:
    """Get appropriate vertex shader for the fragment shader version."""
    first_line = fragment_shader.strip().split("\n")[0]
    return VERSION_TO_VERTEX.get(first_line, VERTEX_SHADER)


def is_shadertoy_format(glsl: str) -> bool:
    """Check if GLSL is in Shadertoy format (no version directive)."""
    first_line = glsl.strip().split("\n")[0]
    return not first_line.startswith("#version")


def wrap_shadertoy(glsl: str) -> str:
    """Wrap Shadertoy format GLSL for compilation."""
    return f"""#version 460 core
in vec2 vs_uv;
out vec4 fragColor;
uniform float iTime;
uniform vec2 iResolution;
vec2 fragCoord;

{glsl}

void main() {{
    fragCoord = vs_uv * iResolution;
    mainImage(fragColor, fragCoord);
}}
"""


def validate_gold_glsl():
    """Validate that all expected GLSL in gold files compiles successfully."""
    import moderngl

    print("\n" + "=" * 60)
    print("VALIDATING GOLD GLSL COMPILATION")
    print("=" * 60)

    ctx = moderngl.create_standalone_context()
    total = 0
    failed = []

    for gold_file in sorted(GOLD_DIR.glob("*.yaml")):
        cases = load_gold_file(gold_file)
        print(f"\n{gold_file.name}:")

        for case in cases:
            name = case["name"]
            expected = case.get("expected", "")
            if not expected:
                continue

            total += 1

            if is_shadertoy_format(expected):
                fragment_shader = wrap_shadertoy(expected)
                vertex_shader = VERTEX_SHADER
            else:
                fragment_shader = expected
                vertex_shader = get_vertex_shader(expected)

            try:
                program = ctx.program(
                    vertex_shader=vertex_shader,
                    fragment_shader=fragment_shader,
                )
                program.release()
                print(f"  OK: {name}")
            except Exception as e:
                failed.append((gold_file.name, name, str(e)))
                print(f"  FAILED: {name}")
                print(f"    Error: {e}")

    ctx.release()

    print("\n" + "=" * 60)
    print(f"Total: {total}, Passed: {total - len(failed)}, Failed: {len(failed)}")
    if failed:
        print("\nFailed shaders:")
        for file, name, error in failed:
            print(f"  - {file}/{name}: {error[:80]}...")
    print("=" * 60)

    return len(failed) == 0


CLI_IMPORTS = """from dataclasses import dataclass
from py2glsl import ShaderContext
from py2glsl.builtins import (
    vec2, vec3, vec4, mat2, mat3, mat4,
    sin, cos, tan, asin, acos, atan,
    sinh, cosh, tanh, asinh, acosh, atanh,
    abs, floor, ceil, fract, sqrt, pow, exp, log, exp2, log2,
    mod, min, max, clamp, mix, step, smoothstep,
    length, distance, dot, cross, normalize, reflect, refract, faceforward,
    sign, radians, degrees, inverse_sqrt, round, trunc,
)
"""


class TestCLICodePath:
    """Test the CLI's file-loading code path using gold test cases.

    This ensures the CLI's _get_transpiled_shader function produces
    the same output as the direct transpile() call used in gold tests.
    """

    @pytest.mark.parametrize(
        "name,case,gold_file", get_test_params(), ids=get_test_ids()
    )
    def test_cli_code_path(self, name, case, gold_file, tmp_path):
        """Test that CLI code path produces same output as direct transpile."""
        # Import CLI internals
        from py2glsl.main import _get_transpiled_shader

        python_code = case["python"]
        target_str = case.get("target", "opengl46")
        main_func = case.get("main_func", "shader")

        # Add imports so the file can be loaded as a module
        # The CLI needs to import the file to find the main function
        full_code = CLI_IMPORTS + python_code

        # Write the Python code to a temp file
        temp_file = tmp_path / f"{name}.py"
        temp_file.write_text(full_code)

        # Get expected output from direct transpile (the gold standard)
        expected = case["expected"].rstrip("\n")

        # Get actual output through CLI code path
        try:
            actual, _ = _get_transpiled_shader(str(temp_file), target_str, main_func)
        except SystemExit:
            pytest.fail(f"CLI code path raised SystemExit for '{name}'")

        if actual != expected:
            import difflib

            diff = difflib.unified_diff(
                expected.splitlines(keepends=True),
                actual.splitlines(keepends=True),
                fromfile="expected (direct transpile)",
                tofile="actual (CLI code path)",
            )
            diff_str = "".join(diff)
            pytest.fail(
                f"CLI code path mismatch for '{name}' in {gold_file.name}:\n{diff_str}"
            )
