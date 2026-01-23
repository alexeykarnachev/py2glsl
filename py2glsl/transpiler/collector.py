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


def _evaluate_const_expr(
    node: ast.expr, globals_dict: dict[str, tuple[str, str]]
) -> tuple[str, float | int | bool] | None:
    """Evaluate a constant expression using already-collected globals.

    Returns (type_str, value) or None if not evaluable.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return ("bool", node.value)
        elif isinstance(node.value, float):
            return ("float", node.value)
        elif isinstance(node.value, int):
            return ("int", node.value)
        return None

    if isinstance(node, ast.Name):
        # Reference to another constant
        if node.id in globals_dict:
            type_str, value_str = globals_dict[node.id]
            try:
                if type_str == "int":
                    return (type_str, int(value_str))
                elif type_str == "float":
                    return (type_str, float(value_str))
                elif type_str == "bool":
                    return (type_str, value_str == "true")
            except ValueError:
                pass
        return None

    if isinstance(node, ast.BinOp):
        left_result = _evaluate_const_expr(node.left, globals_dict)
        right_result = _evaluate_const_expr(node.right, globals_dict)
        if left_result is None or right_result is None:
            return None

        left_type, left_val = left_result
        right_type, right_val = right_result

        # Determine result type (float if either is float)
        result_type = "float" if "float" in (left_type, right_type) else left_type

        # Evaluate the operation
        op = node.op
        if isinstance(op, ast.Add):
            result_val = left_val + right_val
        elif isinstance(op, ast.Sub):
            result_val = left_val - right_val
        elif isinstance(op, ast.Mult):
            result_val = left_val * right_val
        elif isinstance(op, ast.Div):
            result_val = left_val / right_val
        elif isinstance(op, ast.FloorDiv):
            result_val = left_val // right_val
        elif isinstance(op, ast.Mod):
            result_val = left_val % right_val
        elif isinstance(op, ast.Pow):
            result_val = left_val**right_val
        else:
            return None

        return (result_type, result_val)

    if isinstance(node, ast.UnaryOp):
        operand_result = _evaluate_const_expr(node.operand, globals_dict)
        if operand_result is None:
            return None

        operand_type, operand_val = operand_result
        if isinstance(node.op, ast.USub):
            return (operand_type, -operand_val)
        elif isinstance(node.op, ast.UAdd):
            return (operand_type, operand_val)

    return None


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
            inferred = _infer_literal_type(node.value)
            if inferred:
                type_str, value_str = inferred
                collected.globals[name] = (type_str, value_str)
                logger.debug(f"Collected constant: {name} = {value_str} ({type_str})")
            return

        # Try to evaluate constant expression
        result = _evaluate_const_expr(node.value, collected.globals)
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
