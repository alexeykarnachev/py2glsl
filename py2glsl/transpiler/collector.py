"""
AST collector for the GLSL shader transpiler.

This module collects information about functions, structs, and global variables
from Python AST nodes to prepare for GLSL code generation.
"""

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


def collect_info(tree: ast.AST) -> CollectedInfo:
    """Collect information about functions, structs, and globals from the AST.

    Args:
        tree: AST of the Python code to be transpiled

    Returns:
        CollectedInfo containing functions, structs, and global variables
    """
    collected = CollectedInfo()

    class Visitor(ast.NodeVisitor):
        """AST visitor collecting function, struct, and global information."""

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            """Visit function definition nodes and collect information about them."""
            param_types = [
                get_annotation_type(arg.annotation) for arg in node.args.args
            ]
            return_type = get_annotation_type(node.returns)
            collected.functions[node.name] = FunctionInfo(
                name=node.name,
                return_type=return_type,
                param_types=param_types,
                node=node,
            )
            logger.debug(
                f"Collected function: {node.name}, return_type: {return_type}, "
                f"params: {param_types}"
            )
            self.generic_visit(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
            """Visit class definition nodes for struct information.

            Processes classes marked with @dataclass and extracts their fields.
            """
            is_dataclass = any(
                (isinstance(d, ast.Name) and d.id == "dataclass") or
                (isinstance(d, ast.Attribute) and
                 isinstance(d.value, ast.Name) and
                 d.value.id == "dataclasses" and
                 d.attr == "dataclass")
                for d in node.decorator_list
            )
            if is_dataclass:
                fields = []
                for stmt in node.body:
                    if isinstance(stmt, ast.AnnAssign) and isinstance(
                        stmt.target, ast.Name
                    ):
                        field_type = get_annotation_type(stmt.annotation)
                        default_value = None
                        if stmt.value:
                            default_value = generate_simple_expr(stmt.value)
                        # Make sure field_type is not None before creating a StructField
                        if field_type is not None:
                            fields.append(
                                StructField(
                                    name=stmt.target.id,
                                    type_name=field_type,
                                    default_value=default_value,
                                )
                            )
                        else:
                            raise TranspilerError(
                                f"Missing type annotation for struct field "
                                f"{stmt.target.id}"
                            )
                collected.structs[node.name] = StructDefinition(
                    name=node.name, fields=fields
                )
                logger.debug(
                    f"Collected struct: {node.name}, "
                    f"fields: {[(f.name, f.type_name) for f in fields]}"
                )
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
            """Visit annotated assignment nodes to collect global variables."""
            if isinstance(node.target, ast.Name) and node.value:
                expr_type = get_annotation_type(node.annotation)
                try:
                    value = generate_simple_expr(node.value)
                    # Make sure expr_type is not None
                    if expr_type is not None:
                        collected.globals[node.target.id] = (expr_type, value)
                    else:
                        # Default to float if no type annotation
                        collected.globals[node.target.id] = ("float", value)
                    logger.debug(
                        f"Collected global: {node.target.id}, "
                        f"type: {expr_type}, value: {value}"
                    )
                except Exception:
                    # Skip if we can't generate a simple expression
                    # (likely a complex expression that needs further processing)
                    pass
            self.generic_visit(node)

    visitor = Visitor()
    visitor.visit(tree)
    return collected
