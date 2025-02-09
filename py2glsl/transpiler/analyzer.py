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
        self.hoisted_vars = {"global": set()}
        self.current_scope = "global"
        self.scope_stack = []

        self.uniforms: Dict[str, GLSLType] = {}
        self.functions: List[ast.FunctionDef] = []
        self.main_function: Optional[ast.FunctionDef] = None
        self.type_inference = TypeInference()

    def _analyze_arguments(self, func_def: ast.FunctionDef) -> None:
        """Analyze function arguments and collect uniforms.

        Args:
            func_def: Function definition AST node
        """
        logger.debug(f"Analyzing arguments for function: {func_def.name}")

        # First check vs_uv argument
        if func_def.args.args:
            first_arg = func_def.args.args[0]
            if first_arg.arg != "vs_uv":
                raise TypeError("First argument must be vs_uv: vec2")
            if not self._is_valid_vs_uv_type(first_arg.annotation):
                raise TypeError("First argument must be vs_uv: vec2")

        # Process keyword-only arguments as uniforms
        for arg in func_def.args.kwonlyargs:
            if not arg.annotation:
                raise TypeError("All uniform arguments must have type annotations")

            # Skip internal uniforms (prefixed with _)
            if arg.arg.startswith("_"):
                continue

            # Convert Python type annotation to GLSL type
            glsl_type = self._analyze_type_annotation(arg.annotation)
            glsl_type.is_uniform = True

            # Register uniform
            self.uniforms[arg.arg] = glsl_type
            logger.debug(f"Registered uniform: {arg.arg}: {glsl_type}")

            # Add to hoisted variables for the current scope
            self.hoisted_vars[self.current_scope].add(arg.arg)

    def _analyze_body(self, func_def: ast.FunctionDef) -> None:
        """Analyze function body."""
        logger.debug(f"Analyzing body for function: {func_def.name}")

        # Enter function scope
        self._enter_scope(func_def.name)

        try:
            # Collect all variables that need to be hoisted in this scope
            self._collect_hoisted_vars(func_def)

            # Collect nested functions
            for item in func_def.body:
                if isinstance(item, ast.FunctionDef):
                    self.functions.append(item)
                    self._analyze_body(item)

        finally:
            # Exit function scope
            self._exit_scope()

    def _collect_hoisted_vars(self, node: ast.AST) -> None:
        """Collect all variables that need to be hoisted in current scope.

        This method walks the AST and identifies all variables that need declaration
        at the start of their scope.

        Args:
            node: AST node to analyze
        """
        # Track variables we've seen to avoid duplicates
        seen_vars: Set[str] = set()

        for child in ast.walk(node):
            # Handle variable assignments
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        if var_name not in seen_vars:
                            self.hoisted_vars[self.current_scope].add(var_name)
                            seen_vars.add(var_name)

                            # Infer and register type
                            var_type = self.type_inference.get_type(child.value)
                            self.type_inference.register_type(var_name, var_type)
                            logger.debug(
                                f"Hoisted variable {var_name}: {var_type} in scope {self.current_scope}"
                            )

            # Handle for loop variables
            elif isinstance(child, ast.For):
                if isinstance(child.target, ast.Name):
                    var_name = child.target.id
                    if var_name not in seen_vars:
                        self.hoisted_vars[self.current_scope].add(var_name)
                        seen_vars.add(var_name)
                        # Register as int type for loop counters
                        self.type_inference.register_type(var_name, GLSLType("int"))
                        logger.debug(
                            f"Hoisted loop variable {var_name} in scope {self.current_scope}"
                        )

            # Handle function parameters
            elif isinstance(child, ast.FunctionDef):
                # Only process if this is a nested function in current scope
                if child != node:
                    for arg in child.args.args:
                        var_name = arg.arg
                        if var_name not in seen_vars:
                            self.hoisted_vars[self.current_scope].add(var_name)
                            seen_vars.add(var_name)
                            # Infer type from annotation
                            if arg.annotation:
                                var_type = self._analyze_type_annotation(arg.annotation)
                                self.type_inference.register_type(var_name, var_type)
                                logger.debug(
                                    f"Hoisted parameter {var_name}: {var_type} in scope {self.current_scope}"
                                )

            # Handle compound assignments (+=, -=, etc.)
            elif isinstance(child, ast.AugAssign):
                if isinstance(child.target, ast.Name):
                    var_name = child.target.id
                    if var_name not in seen_vars:
                        self.hoisted_vars[self.current_scope].add(var_name)
                        seen_vars.add(var_name)
                        # Infer type from operation
                        var_type = self.type_inference.get_type(child.value)
                        self.type_inference.register_type(var_name, var_type)
                        logger.debug(
                            f"Hoisted augmented variable {var_name}: {var_type} in scope {self.current_scope}"
                        )

        logger.debug(
            f"Collected hoisted variables in scope {self.current_scope}: {self.hoisted_vars[self.current_scope]}"
        )

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
        self.scope_stack.append(self.current_scope)
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
