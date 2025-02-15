"""GLSL Type System - Type Context Management"""

from typing import Dict, List, Optional

from .base import GLSLType
from .errors import TypeContextError


class TypeContext:
    """Manages type information during analysis with scope awareness"""
    
    def __init__(self):
        # Stack of scopes (global scope is always present)
        self.scopes: List[Dict[str, GLSLType]] = [{}]
        
        # Current function context
        self.current_function: Optional[GLSLType] = None
        
        # Uniform variables
        self.uniforms: Dict[str, GLSLType]] = {}
        
        # Attributes
        self.attributes: Dict[str, GLSLType]] = {}

    def enter_scope(self) -> None:
        """Enter a new scope"""
        self.scopes.append({})

    def exit_scope(self) -> None:
        """Exit current scope"""
        if len(self.scopes) > 1:
            self.scopes.pop()

    def add_variable(
        self,
        name: str,
        type_: GLSLType,
        is_uniform: bool = False,
        is_attribute: bool = False
    ) -> None:
        """Add variable to current scope"""
        if is_uniform:
            if name in self.uniforms:
                raise TypeContextError(f"Uniform '{name}' already defined")
            self.uniforms[name] = type_
        elif is_attribute:
            if name in self.attributes:
                raise TypeContextError(f"Attribute '{name}' already defined")
            self.attributes[name] = type_
        else:
            if name in self.scopes[-1]:
                raise TypeContextError(f"Variable '{name}' already defined in this scope")
            self.scopes[-1][name] = type_

    def get_type(self, name: str) -> Optional[GLSLType]:
        """Get type of variable by name, searching through scopes"""
        # Search current scope first
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
                
        # Check uniforms
        if name in self.uniforms:
            return self.uniforms[name]
            
        # Check attributes
        if name in self.attributes:
            return self.attributes[name]
            
        return None

    def set_function_context(self, func_type: GLSLType) -> None:
        """Set current function context"""
        self.current_function = func_type

    def clear_function_context(self) -> None:
        """Clear current function context"""
        self.current_function = None

    def get_return_type(self) -> Optional[GLSLType]:
        """Get return type of current function"""
        return self.current_function.return_type if self.current_function else None

    def __enter__(self):
        """Context manager support"""
        self.enter_scope()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support"""
        self.exit_scope()

__all__ = ["TypeContext"]
