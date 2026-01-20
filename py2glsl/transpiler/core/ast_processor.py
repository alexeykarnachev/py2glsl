"""Language-agnostic AST processing for shader transpilation."""

import ast
from typing import Any, TypeVar

from py2glsl.transpiler.models import CollectedInfo

T = TypeVar("T")


class DependencyResolver:
    """Resolves function dependencies for correct ordering."""

    def __init__(self, collected: CollectedInfo) -> None:
        self.collected = collected
        self.dependencies: dict[str, set[str]] = {}
        self._build_dependency_graph()

    def _build_dependency_graph(self) -> None:
        """Build dependency graph for all functions."""
        for func_name, func_info in self.collected.functions.items():
            self.dependencies[func_name] = self._find_function_calls(func_info.node)

    def _find_function_calls(self, ast_node: ast.AST) -> set[str]:
        """Find all user-defined function calls within an AST node."""
        called_functions: set[str] = set()
        collected_functions = self.collected.functions

        class CallVisitor(ast.NodeVisitor):
            def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
                if isinstance(node.func, ast.Name):
                    if node.func.id in collected_functions:
                        called_functions.add(node.func.id)
                self.generic_visit(node)

        CallVisitor().visit(ast_node)
        return called_functions

    def get_ordered_functions(self, main_func: str) -> list[str]:
        """Get functions ordered by dependencies (dependencies first)."""
        ordered: list[str] = []
        visited: set[str] = set()

        def visit(func_name: str) -> None:
            if func_name in visited:
                return
            for dep in self.dependencies.get(func_name, set()):
                visit(dep)
            ordered.append(func_name)
            visited.add(func_name)

        visit(main_func)
        for func_name in self.collected.functions:
            visit(func_name)

        return ordered


class SymbolTable:
    """Symbol table for tracking variable types during code generation."""

    def __init__(self) -> None:
        self.scopes: list[dict[str, Any]] = [{}]

    def enter_scope(self) -> None:
        """Enter a new scope."""
        self.scopes.append({})

    def exit_scope(self) -> None:
        """Exit current scope."""
        if len(self.scopes) > 1:
            self.scopes.pop()

    def add_symbol(self, name: str, value: T) -> None:
        """Add symbol to current scope."""
        self.scopes[-1][name] = value

    def lookup(self, name: str) -> Any | None:
        """Look up symbol from inner to outer scope."""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        return None

    def current_scope(self) -> dict[str, Any]:
        """Get current scope."""
        return self.scopes[-1]
