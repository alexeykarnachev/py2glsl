"""Shader analysis for GLSL code generation."""

import ast
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Union, cast

from loguru import logger

from py2glsl.transpiler.types import (
    BOOL,
    FLOAT,
    INT,
    VEC2,
    VEC3,
    VEC4,
    GLSLType,
    TypeKind,
    TypeRegistry,
)


class GLSLContext(Enum):
    """Context for shader analysis."""

    DEFAULT = auto()  # Default context
    LOOP = auto()  # Inside loop (for integer inference)
    FUNCTION = auto()  # Inside function definition
    VECTOR_INIT = auto()  # Vector initialization
    EXPRESSION = auto()  # Inside expression


class ShaderAnalysis:
    """Result of shader analysis."""

    def __init__(self):
        self.uniforms: Dict[str, GLSLType] = {}
        self.functions: List[ast.FunctionDef] = []
        self.main_function: Optional[ast.FunctionDef] = None
        self.type_registry = TypeRegistry()
        self.hoisted_vars: Dict[str, Set[str]] = {"global": set()}
        self.var_types: Dict[str, Dict[str, GLSLType]] = {"global": {}}
        self.current_scope = "global"
        self.scope_stack: List[str] = []


class ShaderAnalyzer:
    """Analyzes Python shader code for GLSL generation."""

    def __init__(self):
        self.analysis = ShaderAnalysis()
        self.current_context = GLSLContext.DEFAULT
        self.context_stack: List[GLSLContext] = []
        self.loop_vars: Set[str] = set()
        self.type_constraints: Dict[str, GLSLType] = {}

    def push_context(self, ctx: GLSLContext) -> None:
        """Push new analysis context."""
        self.context_stack.append(self.current_context)
        self.current_context = ctx

    def pop_context(self) -> None:
        """Pop analysis context."""
        if self.context_stack:
            self.current_context = self.context_stack.pop()
        else:
            self.current_context = GLSLContext.DEFAULT

    def enter_scope(self, name: str) -> None:
        """Enter a new variable scope."""
        if name not in self.analysis.hoisted_vars:
            self.analysis.hoisted_vars[name] = set()
            self.analysis.var_types[name] = {}
        self.analysis.scope_stack.append(self.analysis.current_scope)
        self.analysis.current_scope = name

    def exit_scope(self) -> None:
        """Exit current scope."""
        if self.analysis.scope_stack:
            self.analysis.current_scope = self.analysis.scope_stack.pop()
        else:
            self.analysis.current_scope = "global"

    def get_variable_type(self, name: str) -> Optional[GLSLType]:
        """Get type of variable from current or parent scope."""
        scope = self.analysis.current_scope
        while True:
            if name in self.analysis.var_types[scope]:
                return self.analysis.var_types[scope][name]
            if scope == "global":
                break
            scope = self.analysis.scope_stack[-1]
        return None

    def infer_type(self, node: ast.AST) -> GLSLType:
        """Infer GLSL type from AST node."""
        if isinstance(node, ast.Name):
            # Handle vs_uv special case
            if node.id == "vs_uv":
                return VEC2
            # Look up variable type
            var_type = self.get_variable_type(node.id)
            if var_type:
                return var_type
            # Check uniforms
            if node.id in self.analysis.uniforms:
                return self.analysis.uniforms[node.id]

        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return BOOL  # Fixed: Return BOOL type for boolean literals
            elif isinstance(node.value, (int, float)):
                # Integer literals in loop context stay int
                if self.current_context == GLSLContext.LOOP and isinstance(
                    node.value, int
                ):
                    return INT
                return FLOAT

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                # First check if it's a user-defined function
                for func in self.analysis.functions:
                    if func.name == node.func.id:
                        return self.get_type_from_annotation(func.returns)

                # Type conversion functions (existing code)
                if node.func.id == "float":
                    return FLOAT
                elif node.func.id == "bool":
                    return BOOL
                elif node.func.id == "int":
                    return INT
                # Vector constructors (existing code)
                elif node.func.id in ("vec2", "Vec2"):
                    return VEC2
                elif node.func.id in ("vec3", "Vec3"):
                    return VEC3
                elif node.func.id in ("vec4", "Vec4"):
                    return VEC4
                # Built-in functions (existing code)
                return self.infer_builtin_return_type(node.func.id, node.args)

        elif isinstance(node, ast.BinOp):
            left_type = self.infer_type(node.left)
            right_type = self.infer_type(node.right)
            return self.infer_binary_op_type(left_type, right_type, type(node.op))

        elif isinstance(node, ast.Attribute):
            value_type = self.infer_type(node.value)
            if value_type.is_vector():
                # Handle swizzling
                if len(node.attr) == 1:
                    return FLOAT
                elif len(node.attr) == 2:
                    return VEC2
                elif len(node.attr) == 3:
                    return VEC3
                elif len(node.attr) == 4:
                    return VEC4

        return FLOAT  # Default to float for unknown expressions

    def analyze_statement(self, node: ast.AST) -> None:
        """Analyze statement node."""
        if isinstance(node, ast.For):
            self.push_context(GLSLContext.LOOP)

            # Pre-analyze range arguments to ensure integer types
            if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
                if node.iter.func.id == "range":
                    # Register loop variable as integer
                    if isinstance(node.target, ast.Name):
                        logger.debug(
                            f"Registering loop variable {node.target.id} as INT"
                        )
                        self.register_variable(node.target.id, INT)

                    # Analyze range arguments
                    for arg in node.iter.args:
                        if isinstance(arg, ast.Name):
                            self.register_variable(arg.id, INT)
                        elif isinstance(arg, ast.BinOp):
                            if isinstance(arg.left, ast.Name):
                                self.register_variable(arg.left.id, INT)
                            if isinstance(arg.right, ast.Name):
                                self.register_variable(arg.right.id, INT)
                        elif isinstance(arg, ast.Constant):
                            value = arg.value if isinstance(arg, ast.Constant) else None
                            if isinstance(value, float):
                                raise ValueError("Loop bounds must be integers")

                # Process loop body
                for stmt in node.body:
                    # For assignments in loop body, ensure variables are hoisted
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if isinstance(target, ast.Name):
                                value_type = self.infer_type(stmt.value)
                                logger.debug(
                                    f"Hoisting loop variable {target.id} as {value_type}"
                                )
                                self.register_variable(target.id, value_type)
                    self.analyze_statement(stmt)

            self.pop_context()

        elif isinstance(node, ast.Assign):
            value_type = self.infer_type(node.value)
            logger.debug(f"Analyzing assignment with inferred type: {value_type}")

            for target in node.targets:
                if isinstance(target, ast.Name):
                    # Handle boolean literals
                    if isinstance(node.value, ast.Constant) and isinstance(
                        node.value.value, bool
                    ):
                        logger.debug(f"Registering boolean variable {target.id}")
                        self.register_variable(target.id, BOOL)
                        continue

                    # Handle type conversion functions
                    if isinstance(node.value, ast.Call) and isinstance(
                        node.value.func, ast.Name
                    ):
                        func_name = node.value.func.id
                        if func_name == "float":
                            logger.debug(f"Registering float conversion {target.id}")
                            self.register_variable(target.id, FLOAT)
                            continue
                        elif func_name == "bool":
                            logger.debug(f"Registering bool conversion {target.id}")
                            self.register_variable(target.id, BOOL)
                            continue
                        elif func_name == "int":
                            logger.debug(f"Registering int conversion {target.id}")
                            self.register_variable(target.id, INT)
                            continue

                    # Default case
                    logger.debug(f"Registering variable {target.id} as {value_type}")
                    self.register_variable(target.id, value_type)

        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name):
                value_type = self.infer_type(node.value)
                self.register_variable(node.target.id, value_type)

        elif isinstance(node, ast.If):
            for stmt in node.body:
                self.analyze_statement(stmt)
            for stmt in node.orelse:
                self.analyze_statement(stmt)

        elif isinstance(node, ast.FunctionDef):
            self.analyze_function_def(node)

    def infer_builtin_return_type(
        self, func_name: str, args: List[ast.AST]
    ) -> GLSLType:
        """Infer return type of built-in function."""
        arg_types = [self.infer_type(arg) for arg in args]

        # Special cases for specific functions
        if func_name == "mix":
            # mix returns same type as first argument
            return arg_types[0] if arg_types else FLOAT

        # Scalar return
        if func_name in {"length", "dot"}:
            return FLOAT

        # Same as input vector
        if func_name in {"normalize", "abs"}:
            if arg_types and arg_types[0].is_vector():
                return arg_types[0]
            return FLOAT

        # Component-wise operations
        if func_name in {"sin", "cos", "floor", "ceil"}:
            return arg_types[0] if arg_types else FLOAT

        return FLOAT

    def infer_binary_op_type(
        self, left: GLSLType, right: GLSLType, op: type
    ) -> GLSLType:
        """Infer result type of binary operation."""
        # Vector operations take precedence
        if left.is_vector() or right.is_vector():
            return left if left.is_vector() else right

        # Integer operations in loop context
        if self.current_context == GLSLContext.LOOP:
            if left.kind == right.kind == INT:
                return INT

        # Default to float for numeric operations
        return FLOAT

    def analyze_function_def(self, node: ast.FunctionDef) -> None:
        """Analyze function definition."""
        self.enter_scope(node.name)

        # For nested functions, analyze arguments normally
        if len(self.analysis.scope_stack) > 1:
            for arg in node.args.args:
                if arg.annotation:
                    arg_type = self.get_type_from_annotation(arg.annotation)
                    self.register_variable(arg.arg, arg_type)
            self.analysis.functions.append(node)
        else:
            # Only check vs_uv requirement for top-level functions
            # Validate return type
            if not isinstance(node.returns, ast.Name) or node.returns.id != "vec4":
                raise TypeError("Shader must return vec4")

            # Analyze arguments
            if not node.args.args or node.args.args[0].arg != "vs_uv":
                raise TypeError("First argument must be vs_uv: vec2")
            self.register_variable("vs_uv", VEC2)

            # Validate all other arguments are uniforms
            if len(node.args.args) > 1:
                raise TypeError("All arguments except vs_uv must be uniforms")

            # Process uniforms (keyword-only arguments)
            for arg in node.args.kwonlyargs:
                if arg.annotation is None:
                    raise TypeError(f"Uniform {arg.arg} must have type annotation")
                base_type = self.get_type_from_annotation(arg.annotation)
                uniform_type = GLSLType(
                    kind=base_type.kind,
                    is_uniform=True,
                    is_const=base_type.is_const,
                    is_attribute=base_type.is_attribute,
                    array_size=base_type.array_size,
                )
                self.analysis.uniforms[arg.arg] = uniform_type

            # Store as main function
            self.analysis.main_function = node

        # Analyze body
        for stmt in node.body:
            self.analyze_statement(stmt)

        self.exit_scope()

    def get_type_from_annotation(self, annotation: ast.AST) -> GLSLType:
        """Convert Python type annotation to GLSL type."""
        if isinstance(annotation, ast.Name):
            if annotation.id in ("vec2", "Vec2"):
                return VEC2
            elif annotation.id in ("vec3", "Vec3"):
                return VEC3
            elif annotation.id in ("vec4", "Vec4"):
                return VEC4
            elif annotation.id == "float":
                return FLOAT
            elif annotation.id == "int":
                return INT
            elif annotation.id == "bool":
                return BOOL
        raise TypeError(f"Unsupported type annotation: {annotation}")

    def analyze(self, tree: ast.Module) -> ShaderAnalysis:
        """Two-phase analysis of shader AST."""
        self.analysis = ShaderAnalysis()
        self.type_constraints = {}

        # Phase 1: Collect type constraints
        self._collect_type_constraints(tree)

        # Phase 2: Apply type constraints to analysis
        self._apply_type_constraints()

        # Phase 3: Main analysis
        self._analyze_shader(tree)

        return self.analysis

    def _collect_type_constraints(self, tree: ast.Module) -> None:
        """Collect type constraints from AST."""
        for node in ast.walk(tree):
            # Handle loop-related constraints
            if isinstance(node, ast.For):
                if isinstance(node.iter, ast.Call) and isinstance(
                    node.iter.func, ast.Name
                ):
                    if node.iter.func.id == "range":
                        # Register loop variable
                        if isinstance(node.target, ast.Name):
                            self.type_constraints[node.target.id] = INT

                        # Register variables used in range bounds
                        for arg in node.iter.args:
                            if isinstance(arg, ast.Name):
                                self.type_constraints[arg.id] = INT
                            elif isinstance(arg, ast.BinOp):
                                if isinstance(arg.left, ast.Name):
                                    self.type_constraints[arg.left.id] = INT
                                if isinstance(arg.right, ast.Name):
                                    self.type_constraints[arg.right.id] = INT

            # Handle assignments that affect loop variables
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id in self.type_constraints:
                            continue  # Skip if already constrained

                        # Infer type from value
                        if isinstance(node.value, ast.Constant):
                            if isinstance(node.value.value, int):
                                self.type_constraints[target.id] = INT

    def _apply_type_constraints(self) -> None:
        """Apply collected type constraints to all scopes."""
        # Apply constraints to global scope first
        for var_name, var_type in self.type_constraints.items():
            if var_name not in self.analysis.var_types["global"]:
                self.analysis.var_types["global"][var_name] = var_type
                self.analysis.hoisted_vars["global"].add(var_name)

    def register_variable(self, name: str, glsl_type: GLSLType) -> None:
        """Register variable in current scope."""
        scope = self.analysis.current_scope
        logger.debug(
            f"Registering variable {name} of type {glsl_type} in scope {scope}"
        )

        # Important: Don't override existing type if already registered
        if name in self.analysis.var_types[scope]:
            existing_type = self.analysis.var_types[scope][name]
            logger.debug(
                f"Variable {name} already registered with type {existing_type}"
            )
            return

        # Check if there's a type constraint first
        if name in self.type_constraints:
            constrained_type = self.type_constraints[name]
            logger.debug(f"Found type constraint for {name}: {constrained_type}")
            # Only apply constraint if it doesn't conflict with special types
            if not (
                glsl_type.kind
                in (TypeKind.BOOL, TypeKind.VEC2, TypeKind.VEC3, TypeKind.VEC4)
            ):
                glsl_type = constrained_type

        # Register in current scope
        self.analysis.hoisted_vars[scope].add(name)
        self.analysis.var_types[scope][name] = glsl_type
        logger.debug(f"Added {name} of type {glsl_type} to hoisted vars in {scope}")

    def _analyze_shader(self, tree: ast.Module) -> None:
        """Main analysis phase."""
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                self.analysis.main_function = node
                self.analyze_function_def(node)
