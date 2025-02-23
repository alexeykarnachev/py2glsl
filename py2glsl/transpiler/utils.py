import ast
import inspect
from textwrap import dedent
from typing import Callable

from py2glsl.transpiler.type_system import TypeInferer, TypeInfo


def extract_function_body(func_body_ast: list, inferer: TypeInferer) -> list[str]:
    """Convert Python function AST to GLSL code using provided type information"""
    processed = []
    for node in func_body_ast:
        inferer.visit(node)

        if isinstance(node, ast.AnnAssign):
            target_type = inferer.symbols[node.target.id].glsl_name
            processed.append(
                f"{target_type} {ast.unparse(node.target)} = {ast.unparse(node.value)};"
            )
        elif isinstance(node, ast.Return):
            processed.append(f"return {ast.unparse(node.value)};")
        else:
            processed.append(ast.unparse(node) + ";")
    return processed
