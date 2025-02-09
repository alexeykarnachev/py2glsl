"""Shader analysis for GLSL code generation."""

import ast
import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Union

from loguru import logger

from .constants import FLOAT_TYPE, INT_TYPE, VEC2_TYPE, VEC3_TYPE, VEC4_TYPE
from .types import GLSLType, TypeInference


@dataclass
class ShaderAnalysis:
    """Result of shader analysis."""

    uniforms: Dict[str, GLSLType]
    functions: List[ast.FunctionDef]
    main_function: ast.FunctionDef
    type_info: TypeInference
    hoisted_vars: Dict[str, Set[str]]  # Scope -> Variables


class ShaderAnalyzer:
    def __init__(self) -> None:
        """Initialize analyzer."""
        self.uniforms: Dict[str, GLSLType] = {}
        self.functions: List[ast.FunctionDef] = []
        self.main_function: Optional[ast.FunctionDef] = None
        self.type_inference = TypeInference()
        self.hoisted_vars: Dict[str, Set[str]] = {}
        self.current_scope: str = "global"

    def _analyze_arguments(self, func_def: ast.FunctionDef) -> None:
        """Analyze function arguments.

        Args:
            func_def: Function definition AST node
        """
        logger.debug(f"Analyzing arguments for function: {func_def.name}")

        for arg in func_def.args.args:
            if arg.arg == "vs_uv":
                if not self._is_valid_vs_uv_type(arg.annotation):
                    raise TypeError("First argument must be vs_uv: vec2")
            elif not arg.arg.startswith("_"):  # Skip underscore prefixed args
                if not arg.annotation:
                    raise TypeError("All arguments must have type hints")
                glsl_type = self._analyze_type_annotation(arg.annotation)
                glsl_type.is_uniform = True
                self.uniforms[arg.arg] = glsl_type

    def _analyze_body(self, func_def: ast.FunctionDef) -> None:
        """Analyze function body."""
        logger.debug(f"Analyzing body for function: {func_def.name}")

        # Enter function scope
        self._enter_scope(func_def.name)

        # Initialize scope variables if needed
        if self.current_scope not in self.hoisted_vars:
            self.hoisted_vars[self.current_scope] = set()

        # Collect nested functions
        for item in func_def.body:
            if isinstance(item, ast.FunctionDef):
                self.functions.append(item)
                self._analyze_body(item)
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        self.hoisted_vars[self.current_scope].add(target.id)

        # Exit function scope
        self._exit_scope()

    def analyze(self, func_or_tree: Union[Any, ast.Module]) -> ShaderAnalysis:
        """Analyze shader function or AST."""
        try:
            if isinstance(func_or_tree, ast.Module):
                tree = func_or_tree
                logger.debug("Analyzing provided AST")
            else:
                source = inspect.getsource(func_or_tree)
                logger.debug(f"Got source from function:\n{source}")
                tree = ast.parse(source)
                logger.debug("Parsed source to AST")

            # Get the function definition
            if not isinstance(tree.body[0], ast.FunctionDef):
                raise TypeError("Expected a function definition")

            func_def = tree.body[0]
            logger.debug(f"Found function definition: {func_def.name}")

            # Initialize analysis state
            self.uniforms = {}
            self.functions = []
            self.main_function = func_def
            self.type_inference = TypeInference()
            self.hoisted_vars = {}

            # Analyze function arguments
            self._analyze_arguments(func_def)

            # Analyze function body
            self._analyze_body(func_def)

            return ShaderAnalysis(
                uniforms=self.uniforms,
                functions=self.functions,
                main_function=self.main_function,
                type_info=self.type_inference,
                hoisted_vars=self.hoisted_vars,
            )

        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            raise

    def _enter_scope(self, name: str) -> None:
        """Enter a new scope."""
        self.current_scope = name
        if name not in self.hoisted_vars:
            self.hoisted_vars[name] = set()

    def _exit_scope(self) -> None:
        """Exit current scope."""
        self.current_scope = "global"

    def _is_valid_vs_uv_type(self, annotation: Optional[ast.AST]) -> bool:
        """Check if type annotation is valid for vs_uv."""
        if isinstance(annotation, ast.Name):
            return annotation.id in ("vec2", "Vec2")
        return False

    def _analyze_type_annotation(self, annotation: ast.AST) -> GLSLType:
        """Convert Python type annotation to GLSL type."""
        if isinstance(annotation, ast.Name):
            type_map = {
                "float": GLSLType(FLOAT_TYPE),
                "int": GLSLType(INT_TYPE),
                "vec2": GLSLType(VEC2_TYPE),
                "Vec2": GLSLType(VEC2_TYPE),
                "vec3": GLSLType(VEC3_TYPE),
                "Vec3": GLSLType(VEC3_TYPE),
                "vec4": GLSLType(VEC4_TYPE),
                "Vec4": GLSLType(VEC4_TYPE),
            }
            return type_map.get(annotation.id, GLSLType(FLOAT_TYPE))
        return GLSLType(FLOAT_TYPE)

    def _analyze_variables(self, node: ast.AST) -> None:
        """Analyze variable declarations and usage."""
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        glsl_type = self.type_inference.get_type(child.value)
                        self.type_inference.register_type(target.id, glsl_type)
                        self.hoisted_vars[self.current_scope].add(target.id)
