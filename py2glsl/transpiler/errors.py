"""
Exceptions and error handling for the GLSL shader transpiler.

This module defines custom exceptions that are raised during the transpilation process.
"""


class TranspilerError(Exception):
    """Exception raised for errors during shader code transpilation.

    This is the main exception class used throughout the transpiler to report errors
    in a user-friendly way.

    Examples:
        >>> raise TranspilerError("Unknown function: my_func")
        TranspilerError: Unknown function: my_func
    """

    pass
