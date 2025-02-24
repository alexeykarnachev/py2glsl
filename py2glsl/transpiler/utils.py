import ast
import inspect
import re
from textwrap import dedent
from typing import Callable

from py2glsl.transpiler.type_system import TypeInferer, TypeInfo


def extract_function_body(
    func_def: ast.FunctionDef,
    type_inferer: TypeInferer,
) -> list[str]:
    """Convert Python function AST to GLSL code with type annotations"""
    body = []
    for stmt in func_def.body:  # Now accessing .body of FunctionDef
        type_inferer.visit(stmt)

        if isinstance(stmt, ast.AnnAssign):
            target = ast.unparse(stmt.target)
            value = ast.unparse(stmt.value)
            target_type = type_inferer.symbols[stmt.target.id].glsl_name
            body.append(f"{target_type} {target} = {value};")

        elif isinstance(stmt, ast.Return):
            value = ast.unparse(stmt.value)
            body.append(f"return {value};")

        else:
            line = ast.unparse(stmt).replace("\n", " ")
            if not line.endswith("}"):
                line += ";"
            body.append(line)

    return body
