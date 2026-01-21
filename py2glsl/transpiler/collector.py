"""AST collector for shader transpilation."""

import ast

from loguru import logger

from py2glsl.transpiler.ast_parser import generate_simple_expr, get_annotation_type
from py2glsl.transpiler.models import (
    CollectedInfo,
    FunctionInfo,
    StructDefinition,
    StructField,
    TranspilerError,
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


def _infer_literal_type(value: ast.Constant) -> tuple[str, str] | None:
    """Infer GLSL type and value string from a Python literal."""
    if isinstance(value.value, bool):
        return ("bool", "true" if value.value else "false")
    elif isinstance(value.value, float):
        return ("float", str(value.value))
    elif isinstance(value.value, int):
        return ("int", str(value.value))
    return None


def _collect_constant(
    node: ast.Assign | ast.AnnAssign, collected: CollectedInfo
) -> None:
    """Collect module-level constant from assignment.

    Handles both annotated assignments (PI: float = 3.14) and
    simple assignments (PI = 3.14) with type inference.
    """
    if isinstance(node, ast.AnnAssign):
        if not isinstance(node.target, ast.Name) or not node.value:
            return
        name = node.target.id
        type_str = get_annotation_type(node.annotation)
        try:
            value_str = generate_simple_expr(node.value)
            collected.globals[name] = (type_str or "float", value_str)
            logger.debug(f"Collected constant: {name}: {type_str} = {value_str}")
        except TranspilerError:
            pass  # Skip complex expressions

    elif isinstance(node, ast.Assign):
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return
        if not isinstance(node.value, ast.Constant):
            return

        name = node.targets[0].id
        inferred = _infer_literal_type(node.value)
        if inferred:
            type_str, value_str = inferred
            collected.globals[name] = (type_str, value_str)
            logger.debug(f"Collected constant: {name} = {value_str} ({type_str})")


class _Collector(ast.NodeVisitor):
    """AST visitor that collects functions, structs, and globals at module level."""

    def __init__(self, collected: CollectedInfo) -> None:
        self.collected = collected
        self._in_function = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        _collect_function(node, self.collected)
        # Don't visit children - we don't want to collect locals as globals

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        _collect_struct(node, self.collected)
        # Don't visit children

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        _collect_constant(node, self.collected)

    def visit_Assign(self, node: ast.Assign) -> None:
        _collect_constant(node, self.collected)


class _MutationDetector(ast.NodeVisitor):
    """Detect assignments to global variables inside functions."""

    def __init__(self, global_names: set[str], mutable_globals: set[str]) -> None:
        self.global_names = global_names
        self.mutable_globals = mutable_globals
        self.local_names: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Track function parameters as local
        for arg in node.args.args:
            self.local_names.add(arg.arg)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                name = target.id
                if name in self.global_names and name not in self.local_names:
                    self.mutable_globals.add(name)
                # After first assignment, it's local (shadowing)
                self.local_names.add(name)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # Annotated assignment always declares a new local variable (shadowing)
        # so it's NOT a mutation of the global
        if isinstance(node.target, ast.Name):
            self.local_names.add(node.target.id)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name):
            name = node.target.id
            if name in self.global_names and name not in self.local_names:
                self.mutable_globals.add(name)
        self.generic_visit(node)


def collect_info(tree: ast.AST) -> CollectedInfo:
    """Collect functions, structs, and globals from an AST."""
    collected = CollectedInfo()
    collector = _Collector(collected)
    # Only visit top-level nodes in the module
    if isinstance(tree, ast.Module):
        for node in tree.body:
            collector.visit(node)
    else:
        collector.visit(tree)

    # Detect which globals are mutated inside functions
    global_names = set(collected.globals.keys())
    for func_info in collected.functions.values():
        detector = _MutationDetector(global_names, collected.mutable_globals)
        detector.visit(func_info.node)

    if collected.mutable_globals:
        logger.debug(f"Mutable globals detected: {collected.mutable_globals}")

    return collected
