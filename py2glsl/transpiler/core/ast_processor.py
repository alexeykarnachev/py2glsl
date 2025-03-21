"""Language-agnostic AST processing for shader transpilation.

This module provides classes and functions for processing Python AST in a way
that is independent of the target shader language.
"""

import ast
from collections import deque
from typing import Any, Dict, Set, TypeVar

from loguru import logger

from py2glsl.transpiler.models import CollectedInfo, FunctionInfo


class DependencyResolver:
    """Resolves function dependencies for correct ordering."""

    def __init__(self, collected: CollectedInfo) -> None:
        """Initialize with the collected info.

        Args:
            collected: Information about functions, structs, and globals
        """
        self.collected = collected
        self.dependencies: Dict[str, Set[str]] = {}
        self._build_dependency_graph()

    def _build_dependency_graph(self) -> None:
        """Build the dependency graph for all functions."""
        for func_name, func_info in self.collected.functions.items():
            self.dependencies[func_name] = self._find_function_calls(func_info.node)

    def _find_function_calls(self, ast_node: ast.AST) -> Set[str]:
        """Find all function calls within an AST node.

        Args:
            ast_node: The AST node to search

        Returns:
            Set of function names that are called within the node
        """
        called_functions = set()
        collected_functions = self.collected.functions

        class FunctionCallVisitor(ast.NodeVisitor):
            def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
                if (
                    isinstance(node.func, ast.Name)
                    and node.func.id in collected_functions
                ):
                    called_functions.add(node.func.id)
                self.generic_visit(node)

        FunctionCallVisitor().visit(ast_node)
        return called_functions

    def get_ordered_functions(self, main_func: str) -> list[str]:
        """Get functions ordered by dependencies, starting from main.

        Args:
            main_func: The name of the main function

        Returns:
            List of function names in dependency order
        """
        ordered: list[str] = []
        visited: Set[str] = set()

        def visit(func_name: str) -> None:
            """Visit a function and its dependencies."""
            if func_name in visited:
                return
            
            # Visit dependencies first
            for dep in self.dependencies.get(func_name, set()):
                visit(dep)
            
            ordered.append(func_name)
            visited.add(func_name)

        # Start with main function
        visit(main_func)

        # Add any remaining functions that weren't dependencies of main
        for func_name in self.collected.functions:
            visit(func_name)

        return ordered


T = TypeVar("T")


class SymbolTable:
    """Symbol table for tracking variable types during code generation."""

    def __init__(self) -> None:
        """Initialize an empty symbol table."""
        self.scopes: list[Dict[str, Any]] = [{}]

    def enter_scope(self) -> None:
        """Enter a new scope."""
        self.scopes.append({})

    def exit_scope(self) -> None:
        """Exit the current scope."""
        if len(self.scopes) > 1:
            self.scopes.pop()

    def add_symbol(self, name: str, value: T) -> None:
        """Add a symbol to the current scope.

        Args:
            name: Symbol name
            value: Symbol value
        """
        self.scopes[-1][name] = value

    def lookup(self, name: str) -> T | None:
        """Look up a symbol in all scopes, from inner to outer.

        Args:
            name: Symbol name

        Returns:
            Symbol value or None if not found
        """
        # Search from inner to outer scope
        for scope in reversed(self.scopes):
            if name in scope:
                value: T = scope[name]
                return value
        return None

    def current_scope(self) -> Dict[str, T]:
        """Get the current scope.

        Returns:
            The current scope dictionary
        """
        return self.scopes[-1]