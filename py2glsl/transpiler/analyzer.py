"""Shader analysis for GLSL code generation."""

import ast
from enum import Enum, auto

from loguru import logger

from py2glsl.transpiler.type_system import GLSLType, TypeKind


class GLSLContext(Enum):
    """Context for shader analysis."""

    DEFAULT = auto()
    LOOP = auto()
    FUNCTION = auto()
    VECTOR_INIT = auto()
    EXPRESSION = auto()


class ShaderAnalysis:
    """Result of shader analysis."""

    def __init__(self):
        self.uniforms: dict[str, GLSLType] = {}
        self.functions: list[ast.FunctionDef] = []
        self.main_function: ast.FunctionDef | None = None
        self.hoisted_vars: dict[str, set[str]] = {"global": set()}
        self.var_types: dict[str, dict[str, GLSLType]] = {"global": {}}
        self.current_scope = "global"
        self.scope_stack: list[str] = []


class ShaderAnalyzer:
    """Analyzes Python shader code for GLSL generation."""

    def __init__(self):
        """Initialize analyzer."""
        self.analysis = ShaderAnalysis()
        self.current_context = GLSLContext.DEFAULT
        self.context_stack: list[GLSLContext] = []
        self.current_scope = "global"
        self.scope_stack: list[str] = []
        self.type_constraints: dict[str, GLSLType] = {}

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
        """Enter a new scope."""
        if name not in self.analysis.hoisted_vars:
            self.analysis.hoisted_vars[name] = set()
            self.analysis.var_types[name] = {}
        self.scope_stack.append(self.current_scope)
        self.current_scope = name

    def exit_scope(self) -> None:
        """Exit current scope."""
        if self.scope_stack:
            self.current_scope = self.scope_stack.pop()
        else:
            self.current_scope = "global"

    def analyze(self, tree: ast.Module) -> ShaderAnalysis:
        """Two-phase analysis of shader AST."""
        self.analysis = ShaderAnalysis()
        self.type_constraints = {}
        self.current_scope = "global"
        self.scope_stack = []

        # Phase 1: Collect type constraints
        self._collect_type_constraints(tree)

        # Phase 2: Apply type constraints
        self._apply_type_constraints()

        # Phase 3: Main analysis
        self._analyze_shader(tree)

        return self.analysis

    def get_variable_type(self, name: str) -> GLSLType | None:
        """Get type of variable from current or parent scope."""
        scope = self.analysis.current_scope
        while True:
            if name in self.analysis.var_types[scope]:
                return self.analysis.var_types[scope][name]
            if scope == "global":
                break
            scope = self.analysis.scope_stack[-1]
        return None

    def get_type_from_annotation(self, annotation: ast.AST) -> GLSLType:
        """Convert Python type annotation to GLSL type."""
        if isinstance(annotation, ast.Name):
            type_map = {
                "vec2": TypeKind.VEC2,
                "Vec2": TypeKind.VEC2,
                "vec3": TypeKind.VEC3,
                "Vec3": TypeKind.VEC3,
                "vec4": TypeKind.VEC4,
                "Vec4": TypeKind.VEC4,
                "float": TypeKind.FLOAT,
                "int": TypeKind.INT,
                "bool": TypeKind.BOOL,
            }
            if annotation.id in type_map:
                return GLSLType(type_map[annotation.id])
        raise TypeError(f"Unsupported type annotation: {annotation}")

    def infer_type(self, node: ast.AST) -> GLSLType:
        """Infer GLSL type from AST node."""
        if isinstance(node, ast.Name):
            # Handle vs_uv special case
            if node.id == "vs_uv":
                return GLSLType(TypeKind.VEC2)
            # Look up variable type
            var_type = self.get_variable_type(node.id)
            if var_type:
                return var_type
            # Check uniforms
            if node.id in self.analysis.uniforms:
                return self.analysis.uniforms[node.id]

        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return GLSLType(TypeKind.BOOL)
            elif isinstance(node.value, int):
                if self.current_context == GLSLContext.LOOP:
                    return GLSLType(TypeKind.INT)
                return GLSLType(TypeKind.FLOAT)
            elif isinstance(node.value, float):
                return GLSLType(TypeKind.FLOAT)

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                # Vector constructors
                if node.func.id.lower() in {"vec2", "vec3", "vec4"}:
                    return GLSLType(TypeKind[node.func.id.upper()])

                # Built-in functions
                return self.infer_builtin_return_type(node.func.id, node.args)

        elif isinstance(node, ast.BinOp):
            left_type = self.infer_type(node.left)
            right_type = self.infer_type(node.right)

            # For operations with vectors, preserve vector type
            if left_type.kind.is_vector() and right_type.kind.is_vector():
                # Both vectors - use highest dimension
                return (
                    left_type
                    if left_type.kind.value >= right_type.kind.value
                    else right_type
                )
            elif left_type.kind.is_vector():
                return left_type
            elif right_type.kind.is_vector():
                return right_type

            return GLSLType(TypeKind.FLOAT)

        elif isinstance(node, ast.Attribute):
            value_type = self.infer_type(node.value)
            if value_type.kind.is_vector():
                # Single component access returns float
                if len(node.attr) == 1:
                    return GLSLType(TypeKind.FLOAT)
                # Multi-component access returns appropriate vector
                return GLSLType(TypeKind[f"VEC{len(node.attr)}"])

        return GLSLType(TypeKind.FLOAT)  # Default to float

    def infer_builtin_return_type(
        self, func_name: str, args: list[ast.AST]
    ) -> GLSLType:
        """Infer return type of built-in function."""
        arg_types = [self.infer_type(arg) for arg in args]

        # Vector-preserving functions (keep input type)
        if func_name in {"abs", "normalize", "max", "min"}:
            if arg_types:
                return arg_types[0]

        # Scalar return functions
        if func_name in {"length", "dot"}:
            return GLSLType(TypeKind.FLOAT)

        # Component-wise operations
        if func_name in {"sin", "cos", "floor", "ceil"}:
            return arg_types[0] if arg_types else GLSLType(TypeKind.FLOAT)

        # Mix returns same type as first argument
        if func_name == "mix":
            return arg_types[0] if arg_types else GLSLType(TypeKind.FLOAT)

        # Vector constructors
        if func_name.lower() in {"vec2", "vec3", "vec4"}:
            return GLSLType(TypeKind[func_name.upper()])

        return GLSLType(TypeKind.FLOAT)

    def analyze_statement(self, node: ast.AST) -> None:
        """Analyze statement node."""
        match node:
            case ast.Assign():
                value_type = self.infer_type(node.value)
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        # Only mark as uniform if it's a keyword-only argument
                        is_uniform = self.analysis.main_function and target.id in {
                            arg.arg
                            for arg in self.analysis.main_function.args.kwonlyargs
                        }
                        var_type = GLSLType(value_type.kind, is_uniform=is_uniform)
                        self.register_variable(target.id, var_type)

            case ast.AugAssign():
                if isinstance(node.target, ast.Name):
                    value_type = self.infer_type(node.value)
                    self.register_variable(node.target.id, value_type)

            case ast.For():
                self.push_context(GLSLContext.LOOP)
                if isinstance(node.iter, ast.Call) and isinstance(
                    node.iter.func, ast.Name
                ):
                    if node.iter.func.id == "range":
                        if isinstance(node.target, ast.Name):
                            self.register_variable(
                                node.target.id, GLSLType(TypeKind.INT)
                            )
                        for arg in node.iter.args:
                            if isinstance(arg, ast.Name):
                                self.register_variable(arg.id, GLSLType(TypeKind.INT))
                            elif isinstance(arg, ast.BinOp):
                                if isinstance(arg.left, ast.Name):
                                    self.register_variable(
                                        arg.left.id, GLSLType(TypeKind.INT)
                                    )
                                if isinstance(arg.right, ast.Name):
                                    self.register_variable(
                                        arg.right.id, GLSLType(TypeKind.INT)
                                    )
                for stmt in node.body:
                    self.analyze_statement(stmt)
                self.pop_context()

            case ast.If():
                self.analyze_statement(node.test)
                for stmt in node.body:
                    self.analyze_statement(stmt)
                for stmt in node.orelse:
                    self.analyze_statement(stmt)

            case ast.Compare():
                self.analyze_statement(node.left)
                for comparator in node.comparators:
                    self.analyze_statement(comparator)

            case ast.BinOp():
                self.analyze_statement(node.left)
                self.analyze_statement(node.right)

            case ast.UnaryOp():
                self.analyze_statement(node.operand)

            case ast.Call():
                for arg in node.args:
                    self.analyze_statement(arg)
                if isinstance(node.func, ast.Name):
                    # Handle built-in functions
                    if node.func.id in {"vec2", "vec3", "vec4"}:
                        return_type = GLSLType(TypeKind[node.func.id.upper()])
                        self.register_variable(node.func.id, return_type)

            case ast.Attribute():
                self.analyze_statement(node.value)

            case ast.Return():
                if node.value:
                    self.analyze_statement(node.value)

            case ast.Break() | ast.Continue():
                pass

            case ast.FunctionDef():
                self.analyze_function_def(node)

            case ast.Name() | ast.Constant():
                pass  # These are handled by infer_type

            case ast.Expr():
                self.analyze_statement(node.value)

            case _:
                raise ValueError(f"Unsupported statement type: {type(node)}")

    def analyze_function_def(self, node: ast.FunctionDef) -> None:
        """Analyze function definition."""
        self.enter_scope(node.name)

        try:
            # Process arguments
            for arg in node.args.args:
                if arg.annotation:
                    arg_type = self.get_type_from_annotation(arg.annotation)
                    self.register_variable(arg.arg, arg_type)

            # Process keyword-only arguments as uniforms
            for arg in node.args.kwonlyargs:
                if arg.annotation:
                    base_type = self.get_type_from_annotation(arg.annotation)
                    uniform_type = GLSLType(base_type.kind, is_uniform=True)
                    self.analysis.uniforms[arg.arg] = uniform_type

            # Store main function reference or add to functions list
            if len(self.scope_stack) == 0:  # Top-level function
                self.analysis.main_function = node
            else:
                # Add nested function to functions list
                self.analysis.functions.append(node)

            # Extract nested functions first
            nested_functions = []
            body = []
            for stmt in node.body:
                if isinstance(stmt, ast.FunctionDef):
                    nested_functions.append(stmt)
                else:
                    body.append(stmt)

            # Analyze nested functions
            for func in nested_functions:
                self.analyze_function_def(func)

            # Replace body without nested functions
            node.body = body

            # Analyze remaining body
            for stmt in node.body:
                self.analyze_statement(stmt)

        finally:
            self.exit_scope()

    def register_variable(self, name: str, glsl_type: GLSLType) -> None:
        """Register variable in current scope."""
        scope = self.current_scope
        logger.debug(
            f"Registering variable {name} of type {glsl_type} in scope {scope}"
        )

        # Don't override existing type if already registered
        if name in self.analysis.var_types[scope]:
            existing_type = self.analysis.var_types[scope][name]
            logger.debug(
                f"Variable {name} already registered with type {existing_type}"
            )
            return

        # Check if there's a type constraint
        if name in self.type_constraints:
            constrained_type = self.type_constraints[name]
            logger.debug(f"Found type constraint for {name}: {constrained_type}")
            if TypeConverter.can_convert(glsl_type, constrained_type):
                glsl_type = constrained_type

        # Register in current scope
        self.analysis.hoisted_vars[scope].add(name)
        self.analysis.var_types[scope][name] = glsl_type
        logger.debug(f"Added {name} of type {glsl_type} to hoisted vars in {scope}")

    def _collect_type_constraints(self, tree: ast.Module) -> None:
        """Collect type constraints from AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                if isinstance(node.iter, ast.Call) and isinstance(
                    node.iter.func, ast.Name
                ):
                    if node.iter.func.id == "range":
                        # Register loop variable
                        if isinstance(node.target, ast.Name):
                            self.type_constraints[node.target.id] = GLSLType(
                                TypeKind.INT
                            )

                        # Register variables used in range bounds
                        for arg in node.iter.args:
                            if isinstance(arg, ast.Name):
                                self.type_constraints[arg.id] = GLSLType(TypeKind.INT)
                            elif isinstance(arg, ast.BinOp):
                                if isinstance(arg.left, ast.Name):
                                    self.type_constraints[arg.left.id] = GLSLType(
                                        TypeKind.INT
                                    )
                                if isinstance(arg.right, ast.Name):
                                    self.type_constraints[arg.right.id] = GLSLType(
                                        TypeKind.INT
                                    )

    def _apply_type_constraints(self) -> None:
        """Apply collected type constraints to all scopes."""
        for var_name, var_type in self.type_constraints.items():
            if var_name not in self.analysis.var_types["global"]:
                self.analysis.var_types["global"][var_name] = var_type
                self.analysis.hoisted_vars["global"].add(var_name)

    def _analyze_shader(self, tree: ast.Module) -> None:
        """Main analysis phase."""
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                self.analysis.main_function = node
                self.analyze_function_def(node)
