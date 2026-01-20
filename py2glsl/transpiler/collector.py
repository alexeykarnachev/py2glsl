"""AST collector for shader transpilation."""

import ast

from loguru import logger

from py2glsl.transpiler.ast_parser import generate_simple_expr, get_annotation_type
from py2glsl.transpiler.errors import TranspilerError
from py2glsl.transpiler.models import (
    CollectedInfo,
    FunctionInfo,
    StructDefinition,
    StructField,
)


def _collect_function(node: ast.FunctionDef, collected: CollectedInfo) -> None:
    """Collect function info from a FunctionDef node."""
    param_types = [get_annotation_type(arg.annotation) for arg in node.args.args]
    return_type = get_annotation_type(node.returns)
    collected.functions[node.name] = FunctionInfo(
        name=node.name,
        return_type=return_type,
        param_types=param_types,
        node=node,
    )
    logger.debug(f"Collected function: {node.name} -> {return_type}")


def _collect_struct(node: ast.ClassDef, collected: CollectedInfo) -> None:
    """Collect struct info from a dataclass ClassDef node."""
    is_dataclass = any(
        isinstance(d, ast.Name) and d.id == "dataclass" for d in node.decorator_list
    )
    if not is_dataclass:
        return

    fields = []
    for stmt in node.body:
        if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Name):
            continue

        field_type = get_annotation_type(stmt.annotation)
        if field_type is None:
            raise TranspilerError(
                f"Missing type annotation for struct field {stmt.target.id}"
            )

        default_value = generate_simple_expr(stmt.value) if stmt.value else None
        fields.append(
            StructField(
                name=stmt.target.id,
                type_name=field_type,
                default_value=default_value,
            )
        )

    collected.structs[node.name] = StructDefinition(name=node.name, fields=fields)
    logger.debug(f"Collected struct: {node.name} with {len(fields)} fields")


def _collect_global(node: ast.AnnAssign, collected: CollectedInfo) -> None:
    """Collect global variable from an annotated assignment."""
    if not isinstance(node.target, ast.Name) or not node.value:
        return

    expr_type = get_annotation_type(node.annotation)
    try:
        value = generate_simple_expr(node.value)
        collected.globals[node.target.id] = (expr_type or "float", value)
        logger.debug(f"Collected global: {node.target.id}: {expr_type}")
    except TranspilerError:
        pass  # Skip complex expressions


class _Collector(ast.NodeVisitor):
    """AST visitor that collects functions, structs, and globals."""

    def __init__(self, collected: CollectedInfo) -> None:
        self.collected = collected

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        _collect_function(node, self.collected)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        _collect_struct(node, self.collected)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        _collect_global(node, self.collected)
        self.generic_visit(node)


def collect_info(tree: ast.AST) -> CollectedInfo:
    """Collect functions, structs, and globals from an AST."""
    collected = CollectedInfo()
    _Collector(collected).visit(tree)
    return collected
