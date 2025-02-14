"""Function call generation and validation."""

import ast
from dataclasses import dataclass
from typing import Callable, List

from py2glsl.types import GLSLType, GLSLTypeError, can_convert_to

from .type_mappings import (
    BUILTIN_FUNCTIONS,
    BUILTIN_FUNCTIONS_ARGS,
    MATRIX_CONSTRUCTORS,
    TYPE_CONSTRUCTORS,
    VALID_VECTOR_COMBINATIONS,
    VECTOR_CONSTRUCTORS,
)


@dataclass
class FunctionCallGenerator:
    """Handles function call generation and validation."""

    get_type: Callable[[ast.AST], GLSLType]
    generate_expression: Callable[[ast.AST], str]

    def generate(self, node: ast.Call) -> str:
        """Generate function call with validation."""
        if not isinstance(node.func, ast.Name):
            raise GLSLTypeError("Only simple function calls are supported")

        func_name = node.func.id.lower()

        # Vector constructors
        if func_name in VECTOR_CONSTRUCTORS:
            return self._generate_vector_constructor(func_name, node.args)

        # Type conversions
        if func_name in TYPE_CONSTRUCTORS:
            return self._generate_type_conversion(func_name, node.args)

        # Built-in functions
        if func_name in BUILTIN_FUNCTIONS:
            return self._generate_builtin_function(func_name, node.args)

        # Matrix constructors
        if func_name in MATRIX_CONSTRUCTORS:
            return self._generate_matrix_constructor(func_name, node.args)

        raise GLSLTypeError(f"Unknown function: {func_name}")

    def _generate_vector_constructor(self, func_name: str, args: List[ast.AST]) -> str:
        """Generate vector constructor call."""
        size = VECTOR_CONSTRUCTORS[func_name]

        # Special case: single scalar argument
        if len(args) == 1:
            arg_type = self.get_type(args[0])
            if not arg_type.is_vector:
                return f"{func_name}({self.generate_expression(args[0])})"
            if arg_type.vector_size() != size:
                raise GLSLTypeError(f"Cannot construct {func_name} from {arg_type}")

        # Get component sizes
        arg_sizes = []
        for arg in args:
            arg_type = self.get_type(arg)
            if arg_type.is_vector:
                arg_sizes.append(arg_type.vector_size())
            else:
                arg_sizes.append(1)

        # Validate combination
        if tuple(arg_sizes) not in VALID_VECTOR_COMBINATIONS[size]:
            raise GLSLTypeError(
                f"Invalid arguments for {func_name} constructor: "
                f"cannot construct from components {arg_sizes}"
            )

        # Generate call
        args_str = [self.generate_expression(arg) for arg in args]
        return f"{func_name}({', '.join(args_str)})"

    def _generate_type_conversion(self, func_name: str, args: List[ast.AST]) -> str:
        """Generate type conversion call."""
        if len(args) != 1:
            raise GLSLTypeError(f"{func_name} requires exactly one argument")

        arg_type = self.get_type(args[0])
        if not can_convert_to(arg_type, TYPE_CONSTRUCTORS[func_name]):
            raise GLSLTypeError(
                f"Cannot convert type {arg_type} to {TYPE_CONSTRUCTORS[func_name]}"
            )

        return f"{func_name}({self.generate_expression(args[0])})"

    def _generate_builtin_function(self, func_name: str, args: List[ast.AST]) -> str:
        """Generate built-in function call."""
        args_str = [self.generate_expression(arg) for arg in args]

        # Validate argument count
        if func_name in BUILTIN_FUNCTIONS_ARGS:
            expected_args = BUILTIN_FUNCTIONS_ARGS[func_name]
            if len(args) != expected_args:
                raise GLSLTypeError(
                    f"{func_name} requires exactly {expected_args} arguments"
                )

        return f"{func_name}({', '.join(args_str)})"

    def _generate_matrix_constructor(self, func_name: str, args: List[ast.AST]) -> str:
        """Generate matrix constructor call."""
        size = MATRIX_CONSTRUCTORS[func_name]
        args_str = [self.generate_expression(arg) for arg in args]

        # Count total components
        total_components = 0
        for arg in args:
            arg_type = self.get_type(arg)
            if arg_type.is_matrix:
                total_components += arg_type.matrix_size() * arg_type.matrix_size()
            elif arg_type.is_vector:
                total_components += arg_type.vector_size()
            else:
                total_components += 1

        if total_components != size:
            raise GLSLTypeError(
                f"Invalid number of components for {func_name} constructor: "
                f"expected {size}, got {total_components}"
            )

        return f"{func_name}({', '.join(args_str)})"
