"""
Exceptions and error handling for the GLSL shader transpiler.

This module defines custom exceptions that are raised during the transpilation process.
"""

import os
import inspect
import traceback
from typing import Optional, Any


class TranspilerError(Exception):
    """Exception raised for errors during shader code transpilation.

    This is the main exception class used throughout the transpiler to report errors
    in a user-friendly way.

    The class includes functionality to track the source file and line number where
    the error originated, both in the transpiler code and in the user's shader code.

    Examples:
        >>> raise TranspilerError("Unknown function: my_func")
        TranspilerError: Unknown function: my_func
    """

    def __init__(self, message: str, node: Optional[Any] = None):
        """Initialize the exception with a message and optional AST node.

        Args:
            message: The error message
            node: Optional AST node where the error occurred
        """
        self.message = message
        self.node = node
        self.transpiler_frame = None
        self.file_path = None
        self.lineno = None
            
        # Try to get the original file and line info from traceback
        stack = traceback.extract_stack()
        
        # First check if node has the actual file line 
        if node and hasattr(node, 'actual_file_line'):
            self.lineno = getattr(node, 'actual_file_line')
            if os.environ.get("PY2GLSL_CURRENT_FILE"):
                self.file_path = os.environ.get("PY2GLSL_CURRENT_FILE")
        # Next check if we have source file information directly on the node
        elif node and hasattr(node, 'source_file'):
            self.file_path = getattr(node, 'source_file')
            if hasattr(node, 'lineno'):
                self.lineno = getattr(node, 'lineno')
        # Otherwise, check if we have source file info from environment
        elif os.environ.get("PY2GLSL_CURRENT_FILE"):
            self.file_path = os.environ.get("PY2GLSL_CURRENT_FILE")
            if node and hasattr(node, 'lineno'):
                # Get the AST line number 
                ast_lineno = getattr(node, 'lineno')
                
                # Get line offset from environment if it exists
                line_offset = 0
                if os.environ.get("PY2GLSL_LINE_OFFSET"):
                    try:
                        line_offset = int(os.environ.get("PY2GLSL_LINE_OFFSET"))
                    except ValueError:
                        pass
                        
                # Calculate actual file line number by adding offset
                self.lineno = ast_lineno + line_offset
        # Finally, fall back to extracting info from the stack
        elif stack:
            # Skip the current frame and look for the first frame outside errors.py
            for frame in reversed(stack[:-1]):
                file_path, lineno, func, _ = frame
                if os.path.basename(file_path) != "errors.py":
                    self.transpiler_frame = frame
                    self.file_path = file_path
                    self.lineno = lineno
                    break
        
        # Format the message with location info if available
        location_info = ""
        if self.file_path:
            # Try to extract the original filename without the full path
            filename = os.path.basename(self.file_path)
            location_info = f" in {filename}"
            if self.lineno:
                location_info += f" at line {self.lineno}"
        
        # Call the parent constructor with our formatted message
        super().__init__(f"{message}{location_info}")
    
    def with_node(self, node: Any) -> 'TranspilerError':
        """Create a new TranspilerError with the same message but a different node.
        
        Args:
            node: AST node to associate with the error
            
        Returns:
            A new TranspilerError instance with the updated node
        """
        return TranspilerError(self.message, node)
