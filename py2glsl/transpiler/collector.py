"""AST collector for shader transpilation."""

import ast

from loguru import logger

from py2glsl.transpiler.ast_parser import generate_simple_expr, get_annotation_type
from py2glsl.transpiler.ast_utils import eval_const_expr, infer_literal_glsl_type
from py2glsl.transpiler.models import (
    CollectedInfo,
    FunctionInfo,
    MethodInfo,
    StructDefinition,
    StructField,
    TranspilerError,
)


def _collect_function(node: ast.FunctionDef, collected: CollectedInfo) -> None:
    """Collect function info from a FunctionDef node."""
    param_types = [get_annotation_type(arg.annotation) for arg in node.args.args]
    return_type = get_annotation_type(node.returns)

    # Collect default values for parameters
    param_defaults: list[str] = []
    for default in node.args.defaults:
        try:
            param_defaults.append(generate_simple_expr(default))
        except TranspilerError:
            param_defaults.append("")  # Can't transpile complex defaults

    collected.functions[node.name] = FunctionInfo(
        name=node.name,
        return_type=return_type,
        param_types=param_types,
        node=node,
        param_defaults=param_defaults,
    )
    logger.debug(f"Collected function: {node.name} -> {return_type}")


def _collect_struct(node: ast.ClassDef, collected: CollectedInfo) -> None:
    """Collect struct info from a dataclass or regular class.

    Supports:
    - @dataclass decorated classes (fields from annotations)
    - Regular classes with __init__ (fields from self.x = y assignments)
    - Instance methods (converted to functions with struct as first param)
    """
    is_dataclass = any(
        isinstance(d, ast.Name) and d.id == "dataclass" for d in node.decorator_list
    )

    fields: list[StructField] = []
    methods: dict[str, MethodInfo] = {}
    has_custom_init = False
    init_param_types: dict[str, str] = {}

    # First pass: find __init__ and extract parameter types for regular classes
    if not is_dataclass:
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__":
                has_custom_init = True
                # Extract parameter types from __init__ (skip self)
                for arg in stmt.args.args[1:]:
                    param_type = get_annotation_type(arg.annotation)
                    if param_type:
                        init_param_types[arg.arg] = param_type
                break

    # Collect fields
    for stmt in node.body:
        # Dataclass style: annotated class attributes
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
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

        # Regular class: extract fields from __init__ body
        elif (
            isinstance(stmt, ast.FunctionDef)
            and stmt.name == "__init__"
            and not is_dataclass
        ):
            fields.extend(_extract_fields_from_init(stmt, init_param_types))

        # Collect instance methods (not __init__, not dunder methods)
        elif isinstance(stmt, ast.FunctionDef) and not stmt.name.startswith("_"):
            method_info = _collect_method(stmt, node.name)
            if method_info:
                methods[stmt.name] = method_info

    # Skip classes with no fields (not a struct)
    if not fields and not is_dataclass:
        return

    collected.structs[node.name] = StructDefinition(
        name=node.name,
        fields=fields,
        methods=methods,
        has_custom_init=has_custom_init,
    )
    logger.debug(
        f"Collected struct: {node.name} with {len(fields)} fields, "
        f"{len(methods)} methods"
    )


def _extract_fields_from_init(
    init_node: ast.FunctionDef, param_types: dict[str, str]
) -> list[StructField]:
    """Extract struct fields from __init__ body by finding self.x = y patterns."""
    fields: list[StructField] = []
    seen_fields: set[str] = set()

    for stmt in init_node.body:
        if not isinstance(stmt, ast.Assign):
            continue
        for target in stmt.targets:
            # Look for self.field_name = ...
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                field_name = target.attr
                if field_name in seen_fields:
                    continue
                seen_fields.add(field_name)

                # Try to get type from parameter with same name
                field_type = None
                if isinstance(stmt.value, ast.Name) and stmt.value.id in param_types:
                    field_type = param_types[stmt.value.id]

                # Skip fields without type annotations (class won't be usable as struct)
                if field_type is None:
                    continue

                fields.append(StructField(name=field_name, type_name=field_type))

    return fields


def _collect_method(node: ast.FunctionDef, struct_name: str) -> MethodInfo | None:
    """Collect method info from a method definition."""
    # Skip if no self parameter
    if not node.args.args or node.args.args[0].arg != "self":
        return None

    # Get parameter types (skip self)
    param_types = [get_annotation_type(arg.annotation) for arg in node.args.args[1:]]
    param_names = [arg.arg for arg in node.args.args[1:]]
    return_type = get_annotation_type(node.returns)

    # Collect default values
    param_defaults: list[str] = []
    for default in node.args.defaults:
        try:
            param_defaults.append(generate_simple_expr(default))
        except TranspilerError:
            param_defaults.append("")

    return MethodInfo(
        name=node.name,
        struct_name=struct_name,
        return_type=return_type,
        param_types=param_types,
        param_names=param_names,
        node=node,
        param_defaults=param_defaults,
    )


def _collect_constant(
    node: ast.Assign | ast.AnnAssign, collected: CollectedInfo
) -> None:
    """Collect module-level constant from assignment.

    Handles both annotated assignments (PI: float = 3.14) and
    simple assignments (PI = 3.14) with type inference.
    Also evaluates constant expressions like TAU = PI * 2.0.
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

        name = node.targets[0].id

        # Try simple literal first
        if isinstance(node.value, ast.Constant):
            inferred = infer_literal_glsl_type(node.value)
            if inferred:
                type_str, value_str = inferred
                collected.globals[name] = (type_str, value_str)
                logger.debug(f"Collected constant: {name} = {value_str} ({type_str})")
            return

        # Try to evaluate constant expression
        result = eval_const_expr(node.value, collected.globals)
        if result:
            type_str, value = result
            # Format the value appropriately
            if type_str == "float":
                value_str = str(value)
            elif type_str == "int":
                value_str = str(int(value))
            elif type_str == "bool":
                value_str = "true" if value else "false"
            else:
                value_str = str(value)
            collected.globals[name] = (type_str, value_str)
            logger.debug(f"Collected constant expr: {name} = {value_str} ({type_str})")


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
