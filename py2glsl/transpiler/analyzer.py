"""Shader analysis for GLSL code generation."""

import ast
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from loguru import logger

from py2glsl.transpiler.constants import FLOAT_TYPE, INT_TYPE, VEC2_TYPE, VEC4_TYPE
from py2glsl.transpiler.types import GLSLContext, GLSLType, TypeInference


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
        self.hoisted_vars: Dict[str, Set[str]] = {}
        self.current_scope = "global"
        self.scope_stack: List[str] = []
        self.uniforms: Dict[str, GLSLType] = {}
        self.functions: List[ast.FunctionDef] = []
        self.main_function: Optional[ast.FunctionDef] = None
        self.type_inference = TypeInference()
        self._init_scope("global")

    def _init_scope(self, scope_name: str) -> None:
        """Initialize a new scope."""
        if scope_name not in self.hoisted_vars:
            self.hoisted_vars[scope_name] = set()

    def _analyze_arguments(self, func_def: ast.FunctionDef) -> None:
        """Analyze function arguments and collect uniforms."""
        logger.debug(f"Analyzing arguments for function: {func_def.name}")
        self._init_scope(func_def.name)

        # Register vs_uv type
        if func_def.args.args:
            first_arg = func_def.args.args[0]
            if first_arg.arg != "vs_uv":
                raise TypeError("First argument must be vs_uv: vec2")

            # Always register vs_uv as vec2
            self.type_inference.register_type(first_arg.arg, GLSLType(VEC2_TYPE))

        # Process keyword-only arguments as uniforms
        for arg in func_def.args.kwonlyargs:
            if not arg.annotation:
                raise TypeError("All uniform arguments must have type annotations")

            if arg.arg.startswith("_"):
                continue

            # Convert Python type annotation to GLSL type
            glsl_type = self._analyze_type_annotation(arg.annotation)
            glsl_type.is_uniform = True

            # Register uniform and type
            self.uniforms[arg.arg] = glsl_type
            self.type_inference.register_type(arg.arg, glsl_type)
            self.hoisted_vars[func_def.name].add(arg.arg)

    def _analyze_body(self, func_def: ast.FunctionDef) -> None:
        """Analyze function body."""
        self._enter_scope(func_def.name)

        try:
            # Register return type
            if func_def.returns:
                return_type = self._analyze_type_annotation(func_def.returns)
                self.type_inference.register_type("return", return_type)

            # Collect hoisted variables
            self._collect_hoisted_vars(func_def)

            # Collect nested functions
            for node in func_def.body:
                if isinstance(node, ast.FunctionDef):
                    self.functions.append(node)
                    self._analyze_body(node)

        finally:
            self._exit_scope()

    def _collect_hoisted_vars(self, node: ast.AST) -> None:
        """Collect variables that need to be hoisted in current scope."""
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        if target.id not in self.hoisted_vars[self.current_scope]:
                            var_type = self.type_inference.get_type(child.value)
                            self.type_inference.register_type(target.id, var_type)
                            self.hoisted_vars[self.current_scope].add(target.id)

            elif isinstance(child, ast.For):
                if isinstance(child.target, ast.Name):
                    if child.target.id not in self.hoisted_vars[self.current_scope]:
                        self.type_inference.register_type(
                            child.target.id, GLSLType(INT_TYPE)
                        )
                        self.hoisted_vars[self.current_scope].add(child.target.id)

    def _analyze_type_annotation(self, node: ast.AST) -> GLSLType:
        """Convert Python type annotation to GLSL type."""
        if isinstance(node, ast.Name):
            if node.id in ("vec2", "Vec2"):
                return GLSLType(VEC2_TYPE)
            elif node.id in ("vec3", "Vec3"):
                return GLSLType(VEC3_TYPE)
            elif node.id in ("vec4", "Vec4"):
                return GLSLType(VEC4_TYPE)
            elif node.id == "float":
                return GLSLType(FLOAT_TYPE)
            elif node.id == "int":
                return GLSLType(INT_TYPE)

        # Default to vec4 for shader return type
        if self.current_scope == "shader":
            return GLSLType(VEC4_TYPE)

        return GLSLType(FLOAT_TYPE)

    def _enter_scope(self, name: str) -> None:
        """Enter a new scope."""
        self._init_scope(name)
        self.scope_stack.append(self.current_scope)
        self.current_scope = name

    def _exit_scope(self) -> None:
        """Exit current scope."""
        if self.scope_stack:
            self.current_scope = self.scope_stack.pop()
        else:
            self.current_scope = "global"

    def analyze(self, tree: ast.Module) -> ShaderAnalysis:
        """Analyze shader function or AST."""
        if not isinstance(tree.body[0], ast.FunctionDef):
            raise TypeError("Expected a function definition")

        func_def = tree.body[0]
        logger.debug(f"Found function definition: {func_def.name}")

        # Initialize analysis state
        self.uniforms = {}
        self.functions = []
        self.main_function = func_def
        self.type_inference = TypeInference()
        self.hoisted_vars = {"global": set()}
        self.current_scope = "global"
        self.scope_stack = []

        # Analyze function
        self._analyze_arguments(func_def)
        self._analyze_body(func_def)

        # Ensure shader returns vec4
        if not isinstance(func_def.returns, ast.Name) or func_def.returns.id not in (
            "vec4",
            "Vec4",
        ):
            raise TypeError("Shader must return vec4")

        return ShaderAnalysis(
            uniforms=self.uniforms,
            functions=self.functions,
            main_function=self.main_function,
            type_info=self.type_inference,
            hoisted_vars=self.hoisted_vars,
        )
