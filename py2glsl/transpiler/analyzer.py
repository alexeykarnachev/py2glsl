"""Shader analysis for GLSL code generation."""

import ast
import inspect
from enum import Enum, auto
from textwrap import dedent

from loguru import logger

from py2glsl.transpiler.operators import AUGASSIGN_OPERATORS
from py2glsl.transpiler.type_mappings import (
    MATRIX_CONSTRUCTORS,
    TYPE_CONSTRUCTORS,
    VECTOR_TYPES,
    ArgumentBehavior,
    get_builtin_info,
)
from py2glsl.types.base import TypeKind

from ..types import (
    BOOL,
    BVEC2,
    BVEC3,
    BVEC4,
    FLOAT,
    INT,
    IVEC2,
    IVEC3,
    IVEC4,
    MAT2,
    MAT3,
    MAT4,
    VEC2,
    VEC3,
    VEC4,
    VOID,
    GLSLSwizzleError,
    GLSLType,
    GLSLTypeError,
    can_convert_to,
    is_compatible_with,
    validate_operation,
)
from .constants import BUILTIN_UNIFORMS


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
        """Initialize analysis results."""
        self.uniforms = BUILTIN_UNIFORMS.copy()
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
        self.current_return_type: GLSLType | None = None

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

    def get_type_from_annotation(self, annotation: ast.AST) -> GLSLType:
        """Convert Python type annotation to GLSL type."""
        if isinstance(annotation, ast.Name):
            type_map = {
                # Float vectors
                "vec2": VEC2,
                "Vec2": VEC2,
                "vec3": VEC3,
                "Vec3": VEC3,
                "vec4": VEC4,
                "Vec4": VEC4,
                # Integer vectors
                "ivec2": IVEC2,
                "IVec2": IVEC2,
                "ivec3": IVEC3,
                "IVec3": IVEC3,
                "ivec4": IVEC4,
                "IVec4": IVEC4,
                # Boolean vectors
                "bvec2": BVEC2,
                "BVec2": BVEC2,
                "bvec3": BVEC3,
                "BVec3": BVEC3,
                "bvec4": BVEC4,
                "BVec4": BVEC4,
                # Matrices
                "mat2": MAT2,
                "Mat2": MAT2,
                "mat3": MAT3,
                "Mat3": MAT3,
                "mat4": MAT4,
                "Mat4": MAT4,
                # Basic types
                "float": FLOAT,
                "Float": FLOAT,
                "int": INT,
                "Int": INT,
                "bool": BOOL,
                "Bool": BOOL,
                "void": VOID,
                "Void": VOID,
            }
            if annotation.id in type_map:
                return type_map[annotation.id]
        raise GLSLTypeError(f"Unsupported type annotation: {annotation}")

    def get_variable_type(self, name: str) -> GLSLType | None:
        """Get type of variable from current or parent scope."""
        scope = self.current_scope
        while True:
            if name in self.analysis.var_types[scope]:
                return self.analysis.var_types[scope][name]
            if scope == "global":
                break
            scope = self.scope_stack[-1]
        return None

    def infer_type(self, node: ast.AST) -> GLSLType:
        """Infer GLSL type from AST node with validation."""
        logger.debug(f"Inferring type for node: {ast.dump(node)}")

        if self.current_context == GLSLContext.LOOP:
            if isinstance(node, ast.Constant):
                # Use numeric detection here for loop variables
                if isinstance(node.value, float):
                    return FLOAT
                return INT

        match node:
            case ast.Name():
                # Special case for vertex shader UV coordinates
                if node.id == "vs_uv":
                    return VEC2
                # Look up variable type
                var_type = self.get_variable_type(node.id)
                if var_type:
                    return var_type
                # Check uniforms
                if node.id in self.analysis.uniforms:
                    return self.analysis.uniforms[node.id]
                raise GLSLTypeError(f"Undefined variable: {node.id}")

            case ast.Constant():
                if isinstance(node.value, bool):
                    return BOOL
                elif isinstance(node.value, int):
                    return INT if self.current_context == GLSLContext.LOOP else FLOAT
                elif isinstance(node.value, float):
                    return FLOAT
                raise GLSLTypeError(f"Unsupported constant type: {type(node.value)}")

            case ast.Call():
                # Handle math module functions
                if isinstance(node.func, ast.Attribute) and isinstance(
                    node.func.value, ast.Name
                ):
                    if node.func.value.id == "math":
                        # Map Python math functions to GLSL functions
                        math_funcs = {
                            "sqrt": FLOAT,
                            "sin": FLOAT,
                            "cos": FLOAT,
                            "tan": FLOAT,
                            "asin": FLOAT,
                            "acos": FLOAT,
                            "atan": FLOAT,
                            "abs": None,  # Returns same type as input
                        }
                        if node.func.attr in math_funcs:
                            return_type = math_funcs[node.func.attr]
                            if return_type is None:
                                # Function returns same type as input
                                return self.infer_type(node.args[0])
                            return return_type
                        raise GLSLTypeError(
                            f"Unsupported math function: {node.func.attr}"
                        )

                # Handle regular function calls
                if isinstance(node.func, ast.Name):
                    return self._infer_call_type(node)

                raise GLSLTypeError(f"Invalid function call: {ast.dump(node)}")

            case ast.BoolOp():
                # For 'and' operations in GLSL
                if isinstance(node.op, ast.And):
                    # Check all values are boolean
                    for value in node.values:
                        value_type = self.infer_type(value)
                        if value_type != BOOL:
                            raise GLSLTypeError(
                                f"Boolean operation requires bool operands, got {value_type}"
                            )
                    return BOOL
                raise GLSLTypeError("Only 'and' operations are supported")

            case ast.BinOp():
                return self._infer_binary_operation(node)

            case ast.Compare():
                return self._infer_comparison(node)

            case ast.UnaryOp():
                return self._infer_unary_operation(node)

            case ast.Attribute():
                return self._infer_attribute_type(node)

            case _:
                raise GLSLTypeError(f"Unsupported expression type: {type(node)}")

    def _infer_call_type(self, node: ast.Call) -> GLSLType:
        """Infer return type of function call."""
        if not isinstance(node.func, ast.Name):
            raise GLSLTypeError("Invalid function call")

        func_name = node.func.id
        arg_types = [self.infer_type(arg) for arg in node.args]

        # Check built-in function
        if builtin := get_builtin_info(func_name):
            if not (builtin.min_args <= len(arg_types) <= builtin.max_args):
                raise GLSLTypeError(
                    f"{func_name} requires {builtin.min_args} to {builtin.max_args} arguments"
                )

            match builtin.arg_behavior:
                case ArgumentBehavior.EXACT:
                    return builtin.return_type if builtin.return_type else arg_types[0]
                case ArgumentBehavior.PRESERVE:
                    return arg_types[0]
                case ArgumentBehavior.FLEXIBLE:
                    if func_name == "mix":
                        if not is_compatible_with(arg_types[0], arg_types[1]):
                            raise GLSLTypeError(
                                "mix requires compatible x and y arguments"
                            )
                        return arg_types[0]
                    return arg_types[0]

        # Type conversion functions
        if func_name in TYPE_CONSTRUCTORS:
            if len(arg_types) != 1:
                raise GLSLTypeError(f"{func_name} requires exactly one argument")
            return TYPE_CONSTRUCTORS[func_name]

        # Vector constructors
        if func_name in VECTOR_TYPES:
            target_type, size = VECTOR_TYPES[func_name]
            if len(arg_types) == 1 and not arg_types[0].is_vector:
                return target_type
            total_components = sum(
                t.vector_size() if t.is_vector else 1 for t in arg_types
            )
            if total_components != size:
                raise GLSLTypeError(
                    f"Invalid number of components for {func_name} construction: "
                    f"expected {size}, got {total_components}"
                )
            return target_type

        # Matrix constructors
        if func_name in MATRIX_CONSTRUCTORS:
            size = MATRIX_CONSTRUCTORS[func_name]
            total_components = sum(
                (
                    t.matrix_size() * t.matrix_size()
                    if t.is_matrix
                    else t.vector_size() if t.is_vector else 1
                )
                for t in arg_types
            )
            if total_components != size:
                raise GLSLTypeError(
                    f"Invalid number of components for {func_name} constructor: "
                    f"expected {size}, got {total_components}"
                )
            return GLSLType(kind=TypeKind.from_str(func_name))

        # User-defined functions
        if func_name in self.analysis.var_types["global"]:
            return self.analysis.var_types["global"][func_name]

        raise GLSLTypeError(f"Unknown function: {func_name}")

    def _infer_binary_operation(self, node: ast.BinOp) -> GLSLType:
        """Infer type of binary operation."""
        logger.debug(f"Inferring binary operation: {ast.dump(node)}")

        left_type = self.infer_type(node.left)
        right_type = self.infer_type(node.right)

        # Map AST operators to GLSL operators
        op_map = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.Mod: "%",
            ast.BitAnd: "&",
            ast.BitOr: "|",
            ast.BitXor: "^",
            ast.LShift: "<<",
            ast.RShift: ">>",
        }

        if type(node.op) not in op_map:
            raise GLSLTypeError(f"Unsupported binary operator: {type(node.op)}")

        op = op_map[type(node.op)]

        # Handle vector operations
        if left_type.is_vector or right_type.is_vector:
            # Vector-vector operations
            if left_type.is_vector and right_type.is_vector:
                if left_type.vector_size() != right_type.vector_size():
                    raise GLSLTypeError(
                        f"Cannot operate on vectors of different sizes: {left_type} and {right_type}"
                    )
                if left_type.kind != right_type.kind:
                    raise GLSLTypeError(
                        f"Cannot operate on vectors of different types: {left_type} and {right_type}"
                    )
                return left_type

            # Vector-scalar operations
            if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                vector_type = left_type if left_type.is_vector else right_type
                scalar_type = right_type if left_type.is_vector else left_type

                # Allow float and int scalars with vectors
                if scalar_type.kind in (TypeKind.FLOAT, TypeKind.INT):
                    return vector_type

                raise GLSLTypeError(
                    f"Invalid scalar type {scalar_type} for vector operation"
                )

            raise GLSLTypeError(
                f"Operator {op} not supported between {left_type} and {right_type}"
            )

        # Handle matrix operations
        if left_type.is_matrix or right_type.is_matrix:
            # Matrix-matrix operations
            if left_type.is_matrix and right_type.is_matrix:
                if left_type.matrix_size() != right_type.matrix_size():
                    raise GLSLTypeError(
                        f"Cannot operate on matrices of different sizes: {left_type} and {right_type}"
                    )
                # Only multiplication and division are allowed
                if isinstance(node.op, (ast.Mult, ast.Div)):
                    return left_type
                raise GLSLTypeError(f"Operator {op} not supported between matrices")

            # Matrix-scalar operations
            if isinstance(node.op, (ast.Mult, ast.Div)):
                matrix_type = left_type if left_type.is_matrix else right_type
                scalar_type = right_type if left_type.is_matrix else left_type

                if scalar_type.kind in (TypeKind.FLOAT, TypeKind.INT):
                    return matrix_type

                raise GLSLTypeError(
                    f"Invalid scalar type {scalar_type} for matrix operation"
                )

            raise GLSLTypeError(
                f"Operator {op} not supported between {left_type} and {right_type}"
            )

        # Handle scalar operations
        if left_type.kind == TypeKind.BOOL or right_type.kind == TypeKind.BOOL:
            # Boolean operations are only allowed with certain operators
            if isinstance(node.op, (ast.BitAnd, ast.BitOr, ast.BitXor)):
                if left_type.kind == TypeKind.BOOL and right_type.kind == TypeKind.BOOL:
                    return BOOL
            raise GLSLTypeError(f"Invalid operation {op} with boolean type")

        # Numeric scalar operations
        if not (left_type.is_numeric and right_type.is_numeric):
            raise GLSLTypeError(
                f"Non-numeric operation between {left_type} and {right_type}"
            )

        # Use type validation system for final result type
        result_type = validate_operation(left_type, op, right_type)
        if result_type is None:
            raise GLSLTypeError(
                f"Invalid operation {op} between {left_type} and {right_type}"
            )

        logger.debug(f"Binary operation result type: {result_type}")
        return result_type

    def _infer_comparison(self, node: ast.Compare) -> GLSLType:
        """Infer type of comparison operation."""
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise GLSLTypeError("Only simple comparisons are supported")

        left_type = self.infer_type(node.left)
        right_type = self.infer_type(node.comparators[0])

        op_map = {
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
        }

        if type(node.ops[0]) not in op_map:
            raise GLSLTypeError(f"Unsupported comparison operator: {type(node.ops[0])}")

        result_type = validate_operation(
            left_type, op_map[type(node.ops[0])], right_type
        )
        if result_type is None:
            raise GLSLTypeError(
                f"Invalid comparison between {left_type} and {right_type}"
            )
        return result_type

    def _infer_unary_operation(self, node: ast.UnaryOp) -> GLSLType:
        """Infer type of unary operation."""
        operand_type = self.infer_type(node.operand)

        op_map = {
            ast.UAdd: "+",
            ast.USub: "-",
            ast.Not: "!",
            ast.Invert: "~",
        }

        if type(node.op) not in op_map:
            raise GLSLTypeError(f"Unsupported unary operator: {type(node.op)}")

        if isinstance(node.op, ast.Not):
            if operand_type == BOOL or operand_type.is_bool_vector:
                return operand_type
            raise GLSLTypeError(f"Cannot apply logical not to type {operand_type}")

        if not operand_type.is_numeric:
            raise GLSLTypeError(
                f"Cannot apply unary {op_map[type(node.op)]} to {operand_type}"
            )

        return operand_type

    def _infer_attribute_type(self, node: ast.Attribute) -> GLSLType:
        """Infer type of attribute access (swizzling)."""
        value_type = self.infer_type(node.value)

        # Handle vector swizzling
        if value_type.is_vector:
            try:
                return value_type.validate_swizzle(node.attr)
            except GLSLSwizzleError as e:
                raise GLSLTypeError(f"Invalid swizzle operation: {e}")

        raise GLSLTypeError(f"Cannot access attribute of type {value_type}")

    def _infer_builtin_return_type(
        self, func_name: str, args: list[ast.AST]
    ) -> GLSLType:
        """Infer return type of built-in function."""
        arg_types = [self.infer_type(arg) for arg in args]

        # Scalar return functions
        scalar_funcs = {
            "length",
            "dot",
            "distance",
            "noise",
            "abs",
            "sign",
            "floor",
            "ceil",
            "fract",
            "sin",
            "cos",
            "tan",
            "asin",
            "acos",
            "atan",
            "radians",
            "degrees",
        }
        if func_name in scalar_funcs:
            return FLOAT

        # Vector-preserving functions
        preserve_funcs = {
            "normalize",
            "reflect",
            "refract",
            "abs",
            "sign",
            "floor",
            "ceil",
            "fract",
            "sin",
            "cos",
            "tan",
        }
        if func_name in preserve_funcs and arg_types:
            if arg_types[0].is_vector:
                return arg_types[0]
            return FLOAT

        # Mix function
        if func_name == "mix" and len(arg_types) == 3:
            if not is_compatible_with(arg_types[0], arg_types[1]):
                raise GLSLTypeError("Mix requires compatible x and y arguments")
            return arg_types[0]

        raise GLSLTypeError(f"Unknown or unsupported built-in function: {func_name}")

    def analyze(self, shader_func) -> ShaderAnalysis:
        """Analyze shader AST or function."""
        logger.debug("Starting shader analysis")

        self.analysis = ShaderAnalysis()
        self.type_constraints = {}
        self.current_scope = "global"
        self.scope_stack = []

        # Handle both AST and function inputs
        if isinstance(shader_func, ast.Module):
            tree = shader_func
        else:
            # Get function source code and create AST
            source = dedent(inspect.getsource(shader_func))
            tree = ast.parse(source)
            logger.debug(f"Parsed AST: {ast.dump(tree, indent=2)}")

        # Phase 1: Collect type constraints
        logger.debug("Phase 1: Collecting type constraints")
        self._collect_type_constraints(tree)
        logger.debug(f"Collected constraints: {self.type_constraints}")

        # Phase 2: Apply type constraints
        logger.debug("Phase 2: Applying type constraints")
        self._apply_type_constraints()
        logger.debug(f"Applied constraints: {self.analysis.var_types}")

        # Phase 3: Analyze all nodes
        logger.debug("Phase 3: Analyzing nodes")
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                logger.debug(f"Analyzing function: {node.name}")
                if not self.analysis.main_function:
                    self.analysis.main_function = node
                self.analyze_function_def(node)

        return self.analysis

    def _register_function(self, node: ast.FunctionDef) -> None:
        """Register function for later reference."""
        if not node.returns:
            raise GLSLTypeError(f"Function {node.name} missing return type annotation")

        return_type = self.get_type_from_annotation(node.returns)

        # Register function in global scope
        self.analysis.functions.append(node)
        self.analysis.var_types["global"][node.name] = return_type

    def analyze_statement(self, node: ast.AST) -> None:
        """Analyze statement with type validation."""
        logger.debug(f"Analyzing statement: {ast.dump(node)}")

        match node:
            case ast.Assign():
                self._analyze_assignment(node)

            case ast.AnnAssign():
                self._analyze_annotated_assignment(node)

            case ast.AugAssign():
                self._analyze_aug_assignment(node)

            case ast.If():
                self._analyze_if_statement(node)

            case ast.For():
                self._analyze_for_loop(node)

            case ast.While():
                self._analyze_while_loop(node)

            case ast.Return():
                self._analyze_return(node)

            case ast.Break() | ast.Continue():
                if self.current_context != GLSLContext.LOOP:
                    raise GLSLTypeError("Break/continue outside of loop")

            case ast.Expr():
                self.infer_type(node.value)  # Validate expression type

            case ast.FunctionDef():
                self.analyze_function_def(node)

            case _:
                raise GLSLTypeError(f"Unsupported statement type: {type(node)}")

    def _analyze_annotated_assignment(self, node: ast.AnnAssign) -> None:
        """Analyze annotated assignment statement."""
        if not isinstance(node.target, ast.Name):
            raise GLSLTypeError("Only simple assignments are supported")

        # Get type from annotation
        target_type = self.get_type_from_annotation(node.annotation)

        # Register variable with annotated type
        self.register_variable(node.target.id, target_type)

        # If there's a value, validate it
        if node.value:
            value_type = self.infer_type(node.value)
            if not can_convert_to(value_type, target_type):
                raise GLSLTypeError(
                    f"Cannot assign value of type {value_type} to variable of type {target_type}"
                )

    def _analyze_assignment(self, node: ast.Assign) -> None:
        """Analyze assignment and register variable."""
        try:
            # Get type of the value being assigned
            value_type = self.infer_type(node.value)

            for target in node.targets:
                if not isinstance(target, ast.Name):
                    raise GLSLTypeError("Only simple assignments are supported")

                # Check if variable already exists in current scope
                existing_type = self.get_variable_type(target.id)
                if existing_type:
                    # Validate assignment compatibility
                    if not can_convert_to(value_type, existing_type):
                        raise GLSLTypeError(
                            f"Cannot assign value of type {value_type} to variable of type {existing_type}"
                        )
                else:
                    # Register new variable in current scope
                    self.analysis.var_types[self.current_scope][target.id] = value_type
                    # Add to hoisted vars for declaration
                    self.analysis.hoisted_vars[self.current_scope].add(target.id)

                logger.debug(f"Registered variable {target.id} of type {value_type}")

        except Exception as e:
            raise GLSLTypeError(f"Error analyzing assignment: {e!s}") from e

    def _analyze_aug_assignment(self, node: ast.AugAssign) -> None:
        """Analyze augmented assignment statement with vector component support."""
        logger.debug(f"Analyzing augmented assignment: {ast.dump(node)}")

        try:
            # Handle vector component assignments (e.g., vec.x *= scalar)
            if isinstance(node.target, ast.Attribute) and isinstance(
                node.target.value, ast.Name
            ):
                vector_name = node.target.value.id
                component = node.target.attr

                logger.debug(f"Vector component assignment: {vector_name}.{component}")

                # Get vector type
                vector_type = self.get_variable_type(vector_name)
                if not vector_type:
                    raise GLSLTypeError(f"Undefined vector: {vector_name}")
                if not vector_type.is_vector:
                    raise GLSLTypeError(
                        f"Cannot access component of non-vector type: {vector_name}"
                    )

                # Validate component access
                try:
                    component_type = vector_type.validate_swizzle(component)
                except GLSLSwizzleError as e:
                    raise GLSLTypeError(f"Invalid vector component access: {e}")

                # Get value type
                value_type = self.infer_type(node.value)
                logger.debug(
                    f"Component type: {component_type}, Value type: {value_type}"
                )

                # Validate operator
                if type(node.op) not in AUGASSIGN_OPERATORS:
                    raise GLSLTypeError(
                        f"Unsupported augmented assignment operator: {type(node.op)}"
                    )

                op = AUGASSIGN_OPERATORS[type(node.op)]
                base_op = op[0]  # Get the basic operator without '='

                # Validate operation using type system
                result_type = validate_operation(component_type, base_op, value_type)
                if result_type is None:
                    raise GLSLTypeError(
                        f"Invalid operation {base_op} between types {component_type} and {value_type}"
                    )
                if not can_convert_to(result_type, component_type):
                    raise GLSLTypeError(
                        f"Cannot assign result of type {result_type} to component of type {component_type}"
                    )

                logger.debug(
                    f"Vector component assignment validated: {vector_name}.{component} {op}"
                )
                return

            # Handle regular augmented assignments
            if not isinstance(node.target, ast.Name):
                raise GLSLTypeError("Only simple augmented assignments are supported")

            target_name = node.target.id
            target_type = self.get_variable_type(target_name)
            if not target_type:
                raise GLSLTypeError(f"Undefined variable: {target_name}")

            value_type = self.infer_type(node.value)
            logger.debug(
                f"Regular assignment - Target type: {target_type}, Value type: {value_type}"
            )

            if type(node.op) not in AUGASSIGN_OPERATORS:
                raise GLSLTypeError(
                    f"Unsupported augmented assignment operator: {type(node.op)}"
                )

            op = AUGASSIGN_OPERATORS[type(node.op)]
            base_op = op[0]  # Get the basic operator without '='

            # Validate operation using type system
            result_type = validate_operation(target_type, base_op, value_type)
            if result_type is None:
                raise GLSLTypeError(
                    f"Invalid operation {base_op} between types {target_type} and {value_type}"
                )
            if not can_convert_to(result_type, target_type):
                raise GLSLTypeError(
                    f"Cannot assign result of type {result_type} to variable of type {target_type}"
                )

            logger.debug(f"Regular augmented assignment validated: {target_name} {op}")

        except GLSLTypeError:
            raise
        except Exception as e:
            raise GLSLTypeError(f"Error in augmented assignment analysis: {e}") from e

    def _analyze_if_statement(self, node: ast.If) -> None:
        """Analyze if statement."""
        condition_type = self.infer_type(node.test)
        if condition_type != BOOL:
            raise GLSLTypeError(f"If condition must be boolean, got {condition_type}")

        for stmt in node.body:
            self.analyze_statement(stmt)
        for stmt in node.orelse:
            self.analyze_statement(stmt)

    def _analyze_for_loop(self, node: ast.For) -> None:
        """Analyze for loop."""

        self.push_context(GLSLContext.LOOP)

        # Range-loops analysis
        if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
            if node.iter.func.id == "range":
                # Check first argument type to determine loop type
                if node.iter.args:
                    arg_type = self.infer_type(node.iter.args[0])
                    # This will affect how loop variable is typed

        try:
            if not (
                isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"
            ):
                raise GLSLTypeError("Only range-based for loops are supported")

            if not isinstance(node.target, ast.Name):
                raise GLSLTypeError("Only simple loop variables are supported")

            # Register loop variable as int
            self.register_variable(node.target.id, INT)

            # Analyze range arguments
            for arg in node.iter.args:
                arg_type = self.infer_type(arg)
                if not can_convert_to(arg_type, INT):
                    raise GLSLTypeError(
                        f"Range argument must be integer, got {arg_type}"
                    )

            for stmt in node.body:
                self.analyze_statement(stmt)

        finally:
            self.pop_context()

    def _analyze_while_loop(self, node: ast.While) -> None:
        """Analyze while loop."""
        self.push_context(GLSLContext.LOOP)

        try:
            condition_type = self.infer_type(node.test)
            if condition_type != BOOL:
                raise GLSLTypeError(
                    f"While condition must be boolean, got {condition_type}"
                )

            for stmt in node.body:
                self.analyze_statement(stmt)

        finally:
            self.pop_context()

    def _analyze_return(self, node: ast.Return) -> None:
        """Analyze return statement."""
        if self.current_return_type is None:
            raise GLSLTypeError("Return statement outside function")

        if node.value is None:
            if self.current_return_type != VOID:
                raise GLSLTypeError(f"Function must return {self.current_return_type}")
            return

        return_type = self.infer_type(node.value)
        if not can_convert_to(return_type, self.current_return_type):
            raise GLSLTypeError(
                f"Cannot return {return_type} from function returning {self.current_return_type}"
            )

    def analyze_function_def(self, node: ast.FunctionDef) -> None:
        """Analyze function definition with type validation."""
        logger.debug(f"Analyzing function: {node.name}")

        self.enter_scope(node.name)
        prev_return_type = self.current_return_type

        try:
            # Get return type
            if node.returns:
                self.current_return_type = self.get_type_from_annotation(node.returns)
            else:
                self.current_return_type = VOID

            # Process arguments
            for arg in node.args.args:
                if not arg.annotation:
                    raise GLSLTypeError(
                        f"Missing type annotation for argument {arg.arg}"
                    )
                arg_type = self.get_type_from_annotation(arg.annotation)
                self.register_variable(arg.arg, arg_type)

            # Process keyword-only arguments as uniforms
            for arg in node.args.kwonlyargs:
                if not arg.annotation:
                    raise GLSLTypeError(
                        f"Missing type annotation for uniform {arg.arg}"
                    )
                base_type = self.get_type_from_annotation(arg.annotation)
                uniform_type = GLSLType(
                    kind=base_type.kind,
                    is_uniform=True,
                    array_size=base_type.array_size,
                )
                self.analysis.uniforms[arg.arg] = uniform_type

            # Add built-in uniforms if not already present
            for name, type_ in BUILTIN_UNIFORMS.items():
                if name not in self.analysis.uniforms:
                    self.analysis.uniforms[name] = GLSLType(
                        kind=type_.kind,
                        is_uniform=True,
                        array_size=type_.array_size,
                    )

            # Register function type in global scope
            self.analysis.var_types["global"][node.name] = self.current_return_type

            # Process body
            for stmt in node.body:
                if isinstance(stmt, ast.FunctionDef):
                    logger.debug(f"Found nested function: {stmt.name}")
                    # Only add nested functions to functions list if not in global scope
                    if self.current_scope != "global":
                        if stmt not in self.analysis.functions:  # Add this check
                            self.analysis.functions.append(stmt)
                    self.analyze_function_def(stmt)
                else:
                    logger.debug(f"Analyzing statement: {type(stmt)}")
                    self.analyze_statement(stmt)

            # Validate return type
            if self.current_return_type != VOID:
                has_return = any(
                    isinstance(stmt, ast.Return) and stmt.value is not None
                    for stmt in node.body
                )
                if not has_return:
                    raise GLSLTypeError(
                        f"Function {node.name} must return a value of type {self.current_return_type}"
                    )

        finally:
            self.current_return_type = prev_return_type
            self.exit_scope()

    def register_variable(self, name: str, glsl_type: GLSLType) -> None:
        """Register variable with type validation."""
        logger.debug(f"Registering variable {name} of type {glsl_type}")

        scope = self.current_scope

        # Check existing type compatibility
        if name in self.analysis.var_types[scope]:
            existing_type = self.analysis.var_types[scope][name]
            if not can_convert_to(glsl_type, existing_type):
                raise GLSLTypeError(
                    f"Cannot assign type {glsl_type} to variable {name} of type {existing_type}"
                )
            return

        # Apply type constraints
        if name in self.type_constraints:
            constrained_type = self.type_constraints[name]
            if not can_convert_to(glsl_type, constrained_type):
                raise GLSLTypeError(
                    f"Type {glsl_type} does not satisfy constraint {constrained_type} for {name}"
                )
            glsl_type = constrained_type

        self.analysis.hoisted_vars[scope].add(name)
        self.analysis.var_types[scope][name] = glsl_type

    def _collect_type_constraints(self, tree: ast.Module) -> None:
        """Collect type constraints from AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                if isinstance(node.iter, ast.Call) and isinstance(
                    node.iter.func, ast.Name
                ):
                    if node.iter.func.id == "range":
                        if isinstance(node.target, ast.Name):
                            self.type_constraints[node.target.id] = INT
                        for arg in node.iter.args:
                            if isinstance(arg, ast.Name):
                                self.type_constraints[arg.id] = INT

    def _apply_type_constraints(self) -> None:
        """Apply collected type constraints."""
        for var_name, var_type in self.type_constraints.items():
            if var_name not in self.analysis.var_types["global"]:
                self.analysis.var_types["global"][var_name] = var_type
                self.analysis.hoisted_vars["global"].add(var_name)

    def _analyze_shader(self, tree: ast.Module) -> None:
        """Main analysis phase."""
        # Reset state
        self.analysis.functions = []
        self.analysis.main_function = None

        # Process all top-level functions
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                # Set as main function
                self.analysis.main_function = node
                # Analyze function (this will collect nested functions)
                self.analyze_function_def(node)
