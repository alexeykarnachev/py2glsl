"""Shader stage decorators.

Mark functions as shader entry points:

    from py2glsl import ShaderContext
    from py2glsl.builtins import vec4
    from py2glsl.decorators import fragment, vertex, compute

    @vertex
    def vs_main(ctx: ShaderContext) -> vec4:
        return vec4(ctx.vs_uv, 0.0, 1.0)

    @fragment
    def fs_main(ctx: ShaderContext) -> vec4:
        return vec4(ctx.vs_uv, 0.0, 1.0)

    @compute(workgroup_size=(8, 8, 1))
    def cs_main(ctx: ShaderContext) -> None:
        pass
"""

from collections.abc import Callable
from typing import TypeVar

from py2glsl.transpiler.ir import ShaderStage

F = TypeVar("F", bound=Callable[..., object])


def vertex(func: F) -> F:
    """Mark function as vertex shader entry point."""
    func._shader_stage = ShaderStage.VERTEX  # type: ignore[attr-defined]
    func._is_entry_point = True  # type: ignore[attr-defined]
    return func


def fragment(func: F) -> F:
    """Mark function as fragment shader entry point."""
    func._shader_stage = ShaderStage.FRAGMENT  # type: ignore[attr-defined]
    func._is_entry_point = True  # type: ignore[attr-defined]
    return func


def compute(
    workgroup_size: tuple[int, int, int] = (1, 1, 1),
) -> Callable[[F], F]:
    """Mark function as compute shader entry point."""

    def decorator(func: F) -> F:
        func._shader_stage = ShaderStage.COMPUTE  # type: ignore[attr-defined]
        func._is_entry_point = True  # type: ignore[attr-defined]
        func._workgroup_size = workgroup_size  # type: ignore[attr-defined]
        return func

    return decorator
