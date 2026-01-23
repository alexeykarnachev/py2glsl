"""IR analysis utilities for dependency tracking and ordering.

Provides functions for analyzing IR structures, finding dependencies between
functions, and topological sorting.
"""

from collections import defaultdict

from py2glsl.transpiler.ir import (
    IRAssign,
    IRAugmentedAssign,
    IRBinOp,
    IRCall,
    IRConstruct,
    IRDeclare,
    IRExpr,
    IRExprStmt,
    IRFieldAccess,
    IRFor,
    IRFunction,
    IRIf,
    IRReturn,
    IRStmt,
    IRSubscript,
    IRSwizzle,
    IRTernary,
    IRUnaryOp,
    IRWhile,
)


def topological_sort(functions: list[IRFunction]) -> list[IRFunction]:
    """Sort functions so that callees come before callers.

    This ensures that helper functions are defined before the functions
    that call them, which is required by GLSL.

    Args:
        functions: List of IR functions to sort.

    Returns:
        Sorted list with callees before callers.
    """
    func_map = {f.name: f for f in functions}
    func_names = set(func_map.keys())

    # Build dependency graph: func -> set of functions it calls
    dependencies: dict[str, set[str]] = defaultdict(set)
    for func in functions:
        dependencies[func.name] = find_called_functions(func, func_names)

    # Count dependencies for each function
    in_degree: dict[str, int] = {name: len(deps) for name, deps in dependencies.items()}

    # Start with functions that don't call any other user functions
    queue = [name for name, degree in in_degree.items() if degree == 0]
    result: list[IRFunction] = []
    visited: set[str] = set()

    while queue:
        name = queue.pop(0)
        if name in visited:
            continue
        visited.add(name)

        # Check if all dependencies are processed
        if all(dep in visited for dep in dependencies[name]):
            result.append(func_map[name])
            # Add functions that depend on this one
            for other_name, deps in dependencies.items():
                if (
                    name in deps
                    and other_name not in visited
                    and all(d in visited for d in dependencies[other_name])
                ):
                    queue.append(other_name)
        else:
            # Dependencies not ready, process them first
            for dep in dependencies[name]:
                if dep not in visited:
                    queue.insert(0, dep)
            queue.append(name)

    # Add any remaining functions (in case of cycles or disconnected)
    for func in functions:
        if func not in result:
            result.append(func)

    return result


def find_called_functions(func: IRFunction, all_func_names: set[str]) -> set[str]:
    """Find all user-defined functions called by this function.

    Args:
        func: The function to analyze.
        all_func_names: Set of all user-defined function names.

    Returns:
        Set of function names called by the given function.
    """
    called: set[str] = set()
    _visit_stmts(func.body, called, all_func_names)
    return called


def _visit_stmts(stmts: list[IRStmt], called: set[str], func_names: set[str]) -> None:
    """Visit statements to find function calls."""
    for stmt in stmts:
        _visit_stmt(stmt, called, func_names)


def _visit_stmt(stmt: IRStmt, called: set[str], func_names: set[str]) -> None:
    """Visit a statement to find function calls."""
    match stmt:
        case IRDeclare(init=init):
            if init:
                _visit_expr(init, called, func_names)
        case IRAssign(value=value):
            _visit_expr(value, called, func_names)
        case IRAugmentedAssign(value=value):
            _visit_expr(value, called, func_names)
        case IRReturn(value=value):
            if value:
                _visit_expr(value, called, func_names)
        case IRIf(condition=cond, then_body=then_b, else_body=else_b):
            _visit_expr(cond, called, func_names)
            _visit_stmts(then_b, called, func_names)
            _visit_stmts(else_b, called, func_names)
        case IRFor(init=init, condition=cond, update=update, body=body):
            if init:
                _visit_stmt(init, called, func_names)
            if cond:
                _visit_expr(cond, called, func_names)
            if update:
                _visit_stmt(update, called, func_names)
            _visit_stmts(body, called, func_names)
        case IRWhile(condition=cond, body=body):
            _visit_expr(cond, called, func_names)
            _visit_stmts(body, called, func_names)
        case IRExprStmt(expr=expr):
            _visit_expr(expr, called, func_names)


def _visit_expr(expr: IRExpr, called: set[str], func_names: set[str]) -> None:
    """Visit an expression to find function calls."""
    match expr:
        case IRCall(func=func_name, args=args):
            if func_name in func_names:
                called.add(func_name)
            for arg in args:
                _visit_expr(arg, called, func_names)
        case IRBinOp(left=left, right=right):
            _visit_expr(left, called, func_names)
            _visit_expr(right, called, func_names)
        case IRUnaryOp(operand=operand):
            _visit_expr(operand, called, func_names)
        case IRConstruct(args=args):
            for arg in args:
                _visit_expr(arg, called, func_names)
        case IRSwizzle(base=base):
            _visit_expr(base, called, func_names)
        case IRFieldAccess(base=base):
            _visit_expr(base, called, func_names)
        case IRSubscript(base=base, index=index):
            _visit_expr(base, called, func_names)
            _visit_expr(index, called, func_names)
        case IRTernary(condition=cond, true_expr=t, false_expr=f):
            _visit_expr(cond, called, func_names)
            _visit_expr(t, called, func_names)
            _visit_expr(f, called, func_names)
