"""Build IR from CollectedInfo."""

import ast

from py2glsl.context import CONTEXT_BUILTINS
from py2glsl.transpiler.ast_parser import get_annotation_type_with_default
from py2glsl.transpiler.ast_utils import eval_constant, substitute_name
from py2glsl.transpiler.constants import (
    BOOL_RESULT_FUNCTIONS,
    PRESERVE_TYPE_FUNCTIONS,
    SCALAR_RESULT_FUNCTIONS,
    TYPE_CONSTRUCTORS,
    binop_to_str,
    cmpop_to_str,
    unaryop_to_str,
)
from py2glsl.transpiler.ir import (
    IRAssign,
    IRAugmentedAssign,
    IRBinOp,
    IRBreak,
    IRCall,
    IRConstruct,
    IRContinue,
    IRDeclare,
    IRExpr,
    IRExprStmt,
    IRFieldAccess,
    IRFor,
    IRFunction,
    IRIf,
    IRLiteral,
    IRName,
    IRParameter,
    IRReturn,
    IRStmt,
    IRStruct,
    IRSubscript,
    IRSwizzle,
    IRTernary,
    IRType,
    IRUnaryOp,
    IRVariable,
    IRWhile,
    ShaderIR,
    ShaderStage,
    StorageClass,
)
from py2glsl.transpiler.models import (
    CollectedInfo,
    FunctionInfo,
    MethodInfo,
    TranspilerError,
)
from py2glsl.transpiler.type_checker import get_expr_type, infer_binop_result_type
from py2glsl.transpiler.type_utils import (
    get_indexable_size,
    infer_literal_type,
    is_swizzle,
    parse_type_string,
    subscript_result_type,
    swizzle_result_type,
)


class IRBuilder:
    """Builds ShaderIR from CollectedInfo."""

    def __init__(self, collected: CollectedInfo, entry_point: str):
        self.collected = collected
        self.entry_point_name = entry_point
        self.symbols: dict[str, IRType] = {}
        self._context_param_name: str | None = None
        self._tmp_var_counter: int = 0
        self._pending_stmts: list[IRStmt] = []
        self._self_param_name: str | None = None  # For method 'self' parameter
        self._self_type: IRType | None = None  # Type of 'self' in methods

    def build(self, stage: ShaderStage = ShaderStage.FRAGMENT) -> ShaderIR:
        """Build complete ShaderIR."""
        structs = self._build_structs()
        variables = self._build_variables()
        functions = self._build_functions()

        return ShaderIR(
            stage=stage,
            structs=structs,
            variables=variables,
            functions=functions,
            entry_point=self.entry_point_name,
        )

    def _build_structs(self) -> list[IRStruct]:
        """Convert StructDefinitions to IRStructs."""
        result = []
        for struct_def in self.collected.structs.values():
            fields = [(f.name, IRType(f.type_name)) for f in struct_def.fields]
            result.append(IRStruct(name=struct_def.name, fields=fields))
        return result

    def _build_variables(self) -> list[IRVariable]:
        """Build variable declarations from entry point parameters."""
        variables: list[IRVariable] = []

        # Add globals (as const if immutable, local if mutable)
        for name, (type_name, value) in self.collected.globals.items():
            is_mutable = name in self.collected.mutable_globals
            variables.append(
                IRVariable(
                    name=name,
                    type=IRType(type_name),
                    storage=StorageClass.LOCAL if is_mutable else StorageClass.CONST,
                    init_value=value,
                )
            )

        # Extract variables from entry point parameter annotations
        entry_func = self.collected.functions.get(self.entry_point_name)
        if entry_func:
            # Add all context builtins as variables when using ShaderContext
            if self._uses_shader_context(entry_func):
                for name, (glsl_type, storage_str) in CONTEXT_BUILTINS.items():
                    storage = (
                        StorageClass.INPUT
                        if storage_str == "input"
                        else StorageClass.UNIFORM
                    )
                    variables.append(
                        IRVariable(name=name, type=IRType(glsl_type), storage=storage)
                    )

            # Add output variable based on return type (for fragment shaders)
            if entry_func.return_type:
                variables.append(
                    IRVariable(
                        name="fragColor",
                        type=IRType(entry_func.return_type),
                        storage=StorageClass.OUTPUT,
                    )
                )

        return variables

    def _uses_shader_context(self, func_info: FunctionInfo) -> bool:
        """Check if function uses ShaderContext parameter."""
        for arg in func_info.node.args.args:
            if (
                arg.annotation
                and isinstance(arg.annotation, ast.Name)
                and arg.annotation.id == "ShaderContext"
            ):
                return True
        return False

    def _build_functions(self) -> list[IRFunction]:
        """Convert FunctionInfos and MethodInfos to IRFunctions."""
        result = []
        # Build regular functions
        for func_name, func_info in self.collected.functions.items():
            is_entry = func_name == self.entry_point_name
            ir_func = self._build_function(func_info, is_entry)
            result.append(ir_func)

        # Build methods from all structs
        for struct_def in self.collected.structs.values():
            for method_info in struct_def.methods.values():
                ir_func = self._build_method(method_info)
                result.append(ir_func)

        return result

    def _build_function(self, func_info: FunctionInfo, is_entry: bool) -> IRFunction:
        """Build a single IRFunction from FunctionInfo."""
        params: list[IRParameter] = []
        self.symbols = {}
        self._context_param_name = None
        self._global_constants: set[str] = set()

        uses_context = self._uses_shader_context(func_info)

        for i, arg in enumerate(func_info.node.args.args):
            # Skip ShaderContext parameter - it's not a real GLSL parameter
            if uses_context and self._is_context_param(arg):
                self._context_param_name = arg.arg
                # Add all context builtins to symbols
                for name, (glsl_type, _) in CONTEXT_BUILTINS.items():
                    self.symbols[name] = IRType(glsl_type)
                continue

            param_type = func_info.param_types[i] or "float"
            ir_type = IRType(param_type)
            params.append(IRParameter(name=arg.arg, type=ir_type))
            self.symbols[arg.arg] = ir_type

        # Add globals to symbols (only if not already defined as parameter)
        # Only immutable ones are tracked as constants
        for name, (type_name, _) in self.collected.globals.items():
            if name not in self.symbols:  # Don't overwrite parameters
                self.symbols[name] = IRType(type_name)
                if name not in self.collected.mutable_globals:
                    self._global_constants.add(name)

        # Build return type
        return_type = None
        if func_info.return_type:
            return_type = IRType(func_info.return_type)

        # Build body statements, skipping docstrings
        body: list[IRStmt] = []
        for i, stmt in enumerate(func_info.node.body):
            # Skip docstring (string constant as first statement)
            if i == 0 and self._is_docstring(stmt):
                continue
            body.extend(self._build_stmt(stmt))

        return IRFunction(
            name=func_info.name,
            params=params,
            return_type=return_type,
            body=body,
            is_entry_point=is_entry,
        )

    def _build_method(self, method_info: MethodInfo) -> IRFunction:
        """Build an IRFunction from a MethodInfo (instance method).

        Transforms: def method(self, x: float) -> float
        Into GLSL:  float StructName_method(StructName self, float x)
        """
        struct_type = IRType(method_info.struct_name)
        func_name = f"{method_info.struct_name}_{method_info.name}"

        # Reset state
        self.symbols = {}
        self._context_param_name = None
        self._global_constants = set()
        self._self_param_name = "self"
        self._self_type = struct_type

        # First parameter is always self (the struct instance)
        params: list[IRParameter] = [IRParameter(name="self", type=struct_type)]
        self.symbols["self"] = struct_type

        # Add remaining parameters
        for i, param_name in enumerate(method_info.param_names):
            param_type = method_info.param_types[i] or "float"
            ir_type = IRType(param_type)
            params.append(IRParameter(name=param_name, type=ir_type))
            self.symbols[param_name] = ir_type

        # Add globals to symbols
        for name, (type_name, _) in self.collected.globals.items():
            if name not in self.symbols:
                self.symbols[name] = IRType(type_name)
                if name not in self.collected.mutable_globals:
                    self._global_constants.add(name)

        # Build return type
        return_type = None
        if method_info.return_type:
            return_type = IRType(method_info.return_type)

        # Build body statements
        body: list[IRStmt] = []
        for i, stmt in enumerate(method_info.node.body):
            if i == 0 and self._is_docstring(stmt):
                continue
            body.extend(self._build_stmt(stmt))

        # Reset method state
        self._self_param_name = None
        self._self_type = None

        return IRFunction(
            name=func_name,
            params=params,
            return_type=return_type,
            body=body,
            is_entry_point=False,
        )

    def _is_context_param(self, arg: ast.arg) -> bool:
        """Check if argument is a ShaderContext parameter."""
        if arg.annotation and isinstance(arg.annotation, ast.Name):
            return arg.annotation.id == "ShaderContext"
        return False

    def _filter_context_args(
        self, func_name: str, args: list[ast.expr]
    ) -> list[ast.expr]:
        """Filter out ShaderContext arguments when calling a user function."""
        func_info = self.collected.functions.get(func_name)
        if not func_info:
            return args  # Builtin function, keep all args

        # Check which parameter positions are ShaderContext
        func_args = func_info.node.args.args
        filtered = []
        for i, arg in enumerate(args):
            if i < len(func_args) and self._is_context_param(func_args[i]):
                continue  # Skip ShaderContext argument
            filtered.append(arg)
        return filtered

    def _is_docstring(self, stmt: ast.stmt) -> bool:
        """Check if statement is a docstring."""
        if not isinstance(stmt, ast.Expr):
            return False
        if not isinstance(stmt.value, ast.Constant):
            return False
        return isinstance(stmt.value.value, str)

    def _fill_method_default_args(
        self, method_info: MethodInfo, arg_exprs: list[IRExpr]
    ) -> list[IRExpr]:
        """Fill in default arguments for method calls.

        arg_exprs[0] is 'self', remaining are method arguments.
        """
        if not method_info.param_defaults:
            return arg_exprs

        # Number of params (excluding self)
        num_params = len(method_info.param_names)
        # Number of provided args (excluding self)
        num_provided = len(arg_exprs) - 1
        num_defaults = len(method_info.param_defaults)

        if num_provided >= num_params:
            return arg_exprs

        # Build the required defaults
        result = list(arg_exprs)
        for i in range(num_provided, num_params):
            default_idx = i - (num_params - num_defaults)
            if default_idx >= 0 and default_idx < num_defaults:
                default_val = method_info.param_defaults[default_idx]
                if default_val:
                    param_type = method_info.param_types[i] or "float"
                    result.append(
                        IRLiteral(result_type=IRType(param_type), value=default_val)
                    )

        return result

    def _hoist_if_variables(self, if_node: ast.If) -> list[IRStmt]:
        """Find variables assigned in if/elif/else branches and hoist declarations.

        In Python, variables assigned in any branch are accessible after the if.
        In GLSL, we need to declare them before the if statement.
        """
        # Collect all variable names assigned in any branch
        assigned_vars: dict[str, IRType] = {}
        self._collect_assigned_vars(if_node, assigned_vars)

        # Create declarations for variables not already in symbols
        hoisted: list[IRStmt] = []
        for name, ir_type in assigned_vars.items():
            if name not in self.symbols:
                var = IRVariable(name=name, type=ir_type)
                self.symbols[name] = ir_type
                hoisted.append(IRDeclare(var=var, init=None))

        return hoisted

    def _collect_assigned_vars(
        self, node: ast.AST, assigned: dict[str, IRType]
    ) -> None:
        """Recursively collect variables assigned in a node."""
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id not in self.symbols:
                    # Infer type from the value
                    ir_type = self._infer_expr_type(node.value)
                    assigned[target.id] = ir_type
        elif isinstance(node, ast.If):
            for stmt in node.body:
                self._collect_assigned_vars(stmt, assigned)
            for stmt in node.orelse:
                self._collect_assigned_vars(stmt, assigned)
        elif isinstance(node, ast.For | ast.While):
            for stmt in node.body:
                self._collect_assigned_vars(stmt, assigned)

    def _infer_expr_type(self, node: ast.expr) -> IRType:
        """Infer the IR type of an expression without building it.

        Delegates to type_checker.get_expr_type() and wraps result in IRType.
        """
        # Build symbols dict with string types for type_checker
        # Use str(v) to preserve array size info (e.g., "vec3[2]" not just "vec3")
        str_symbols: dict[str, str | None] = {
            k: str(v) for k, v in self.symbols.items()
        }
        try:
            type_str = get_expr_type(node, str_symbols, self.collected)
            return parse_type_string(type_str)
        except TranspilerError:
            # Fallback for cases not handled by type_checker
            return IRType("float")

    def _flush_pending_stmts(self) -> list[IRStmt]:
        """Flush and return any pending statements from walrus operator expressions."""
        if not self._pending_stmts:
            return []
        stmts = self._pending_stmts
        self._pending_stmts = []
        return stmts

    def _build_stmt(self, node: ast.stmt) -> list[IRStmt]:
        """Build IR statement(s) from AST statement."""
        match node:
            case ast.Return(value=value):
                if value:
                    # Handle tuple return (e.g., return a, b)
                    # Convert to vector construction
                    if isinstance(value, ast.Tuple):
                        elts = value.elts
                        if 2 <= len(elts) <= 4:
                            args = [self._build_expr(e) for e in elts]
                            pending = self._flush_pending_stmts()
                            vec_type = IRType(f"vec{len(elts)}")
                            vec_construct = IRConstruct(result_type=vec_type, args=args)
                            return [*pending, IRReturn(value=vec_construct)]
                    expr = self._build_expr(value)
                    pending = self._flush_pending_stmts()
                    return [*pending, IRReturn(value=expr)]
                return [IRReturn()]

            case ast.Assign(targets=targets, value=value):
                result: list[IRStmt] = []

                # Handle tuple unpacking: a, b = func()
                if len(targets) == 1 and isinstance(targets[0], ast.Tuple):
                    return self._build_tuple_unpack(targets[0], value)

                value_expr = self._build_expr(value)
                pending = self._flush_pending_stmts()
                result.extend(pending)
                for target in targets:
                    is_new_var = isinstance(target, ast.Name) and (
                        target.id not in self.symbols
                        or target.id in self._global_constants
                    )
                    if is_new_var and isinstance(target, ast.Name):
                        # New variable or shadowing a global constant
                        ir_type = value_expr.result_type
                        var = IRVariable(name=target.id, type=ir_type)
                        self.symbols[target.id] = ir_type
                        self._global_constants.discard(target.id)  # Now local
                        result.append(IRDeclare(var=var, init=value_expr))
                    else:
                        target_expr = self._build_expr(target)
                        result.append(IRAssign(target=target_expr, value=value_expr))
                return result

            case ast.AnnAssign(target=target, annotation=ann, value=value):
                type_name = get_annotation_type_with_default(ann)
                ir_type = parse_type_string(type_name)
                if isinstance(target, ast.Name):
                    init = None
                    if value:
                        init = self._build_expr_with_type_hint(value, ir_type)
                        # If size was inferred (-1), use the actual type from init
                        if ir_type.array_size == -1 and init is not None:
                            ir_type = init.result_type
                    var = IRVariable(name=target.id, type=ir_type)
                    self.symbols[target.id] = ir_type
                    pending = self._flush_pending_stmts()
                    return [*pending, IRDeclare(var=var, init=init)]
                return []

            case ast.AugAssign(target=target, op=op, value=value):
                op_str = binop_to_str(type(op).__name__)
                target_expr = self._build_expr(target)
                value_expr = self._build_expr(value)
                pending = self._flush_pending_stmts()
                return [
                    *pending,
                    IRAugmentedAssign(target=target_expr, op=op_str, value=value_expr),
                ]

            case ast.If(test=test, body=body, orelse=orelse):
                # Pre-scan to hoist variable declarations assigned in branches
                hoisted = self._hoist_if_variables(node)
                cond = self._build_expr(test)
                pending = self._flush_pending_stmts()
                then_body = []
                for s in body:
                    then_body.extend(self._build_stmt(s))
                else_body = []
                for s in orelse:
                    else_body.extend(self._build_stmt(s))
                return [
                    *hoisted,
                    *pending,
                    IRIf(condition=cond, then_body=then_body, else_body=else_body),
                ]

            case ast.For(target=target, iter=iter_expr, body=body):
                return self._build_for_loop(target, iter_expr, body)

            case ast.While(test=test, body=body):
                cond = self._build_expr(test)
                pending = self._flush_pending_stmts()
                while_body = []
                for s in body:
                    while_body.extend(self._build_stmt(s))
                return [*pending, IRWhile(condition=cond, body=while_body)]

            case ast.Expr(value=value):
                expr = self._build_expr(value)
                pending = self._flush_pending_stmts()
                return [*pending, IRExprStmt(expr=expr)]

            case ast.Break():
                return [IRBreak()]

            case ast.Continue():
                return [IRContinue()]

            case ast.Pass():
                return []

        return []

    def _build_for_loop(
        self, target: ast.expr, iter_expr: ast.expr, body: list[ast.stmt]
    ) -> list[IRStmt]:
        """Build IR for a for loop.

        Supports:
        - range() iteration
        - for item in array (direct array iteration)
        - for i, item in enumerate(array)
        """
        # Check for enumerate(array) pattern
        if (
            isinstance(iter_expr, ast.Call)
            and isinstance(iter_expr.func, ast.Name)
            and iter_expr.func.id == "enumerate"
            and len(iter_expr.args) == 1
            and isinstance(target, ast.Tuple)
            and len(target.elts) == 2
            and all(isinstance(e, ast.Name) for e in target.elts)
        ):
            return self._build_enumerate_loop(target, iter_expr.args[0], body)

        # Check for direct array iteration: for item in array_name
        if isinstance(target, ast.Name) and isinstance(iter_expr, ast.Name):
            array_type = self.symbols.get(iter_expr.id)
            if array_type and array_type.array_size is not None:
                return self._build_array_iteration_loop(
                    target.id, iter_expr.id, array_type, body
                )

        if not isinstance(target, ast.Name):
            raise TranspilerError("For loop target must be a simple name")

        if not isinstance(iter_expr, ast.Call):
            raise TranspilerError("For loop must iterate over range()")

        if not isinstance(iter_expr.func, ast.Name) or iter_expr.func.id != "range":
            raise TranspilerError("For loop must iterate over range()")

        args = iter_expr.args
        start: IRExpr
        end: IRExpr
        step: IRExpr
        if len(args) == 1:
            start = IRLiteral(result_type=IRType("int"), value=0)
            end = self._build_expr(args[0])
            step = IRLiteral(result_type=IRType("int"), value=1)
        elif len(args) == 2:
            start = self._build_expr(args[0])
            end = self._build_expr(args[1])
            step = IRLiteral(result_type=IRType("int"), value=1)
        elif len(args) == 3:
            start = self._build_expr(args[0])
            end = self._build_expr(args[1])
            step = self._build_expr(args[2])
        else:
            raise TranspilerError("range() requires 1-3 arguments")

        loop_var = IRVariable(name=target.id, type=IRType("int"))
        self.symbols[target.id] = IRType("int")

        init = IRDeclare(var=loop_var, init=start)
        condition = IRBinOp(
            result_type=IRType("bool"),
            op="<",
            left=IRName(result_type=IRType("int"), name=target.id),
            right=end,
        )
        update = IRAugmentedAssign(
            target=IRName(result_type=IRType("int"), name=target.id),
            op="+",
            value=step,
        )

        for_body = []
        for s in body:
            for_body.extend(self._build_stmt(s))

        return [IRFor(init=init, condition=condition, update=update, body=for_body)]

    def _build_array_iteration_loop(
        self,
        item_name: str,
        array_name: str,
        array_type: IRType,
        body: list[ast.stmt],
    ) -> list[IRStmt]:
        """Build IR for: for item in array."""
        size = array_type.array_size
        elem_type = IRType(array_type.base)
        counter_name = f"_i{self._tmp_var_counter}"
        self._tmp_var_counter += 1

        loop_var = IRVariable(name=counter_name, type=IRType("int"))
        self.symbols[counter_name] = IRType("int")
        self.symbols[item_name] = elem_type

        zero = IRLiteral(result_type=IRType("int"), value=0)
        init = IRDeclare(var=loop_var, init=zero)
        condition = IRBinOp(
            result_type=IRType("bool"),
            op="<",
            left=IRName(result_type=IRType("int"), name=counter_name),
            right=IRLiteral(result_type=IRType("int"), value=size),
        )
        update = IRAugmentedAssign(
            target=IRName(result_type=IRType("int"), name=counter_name),
            op="+",
            value=IRLiteral(result_type=IRType("int"), value=1),
        )

        # Prepend element declaration to body
        elem_var = IRVariable(name=item_name, type=elem_type)
        elem_init = IRSubscript(
            result_type=elem_type,
            base=IRName(result_type=array_type, name=array_name),
            index=IRName(result_type=IRType("int"), name=counter_name),
        )
        for_body: list[IRStmt] = [IRDeclare(var=elem_var, init=elem_init)]
        for s in body:
            for_body.extend(self._build_stmt(s))

        return [IRFor(init=init, condition=condition, update=update, body=for_body)]

    def _build_enumerate_loop(
        self,
        target: ast.Tuple,
        array_node: ast.expr,
        body: list[ast.stmt],
    ) -> list[IRStmt]:
        """Build IR for: for i, item in enumerate(array)."""
        idx_name = target.elts[0].id  # type: ignore[attr-defined]
        item_name = target.elts[1].id  # type: ignore[attr-defined]

        array_expr_name = array_node
        if not isinstance(array_expr_name, ast.Name):
            raise TranspilerError("enumerate() argument must be a variable name")

        array_type = self.symbols.get(array_expr_name.id)
        if not array_type or array_type.array_size is None:
            raise TranspilerError("enumerate() argument must be an array")

        size = array_type.array_size
        elem_type = IRType(array_type.base)

        loop_var = IRVariable(name=idx_name, type=IRType("int"))
        self.symbols[idx_name] = IRType("int")
        self.symbols[item_name] = elem_type

        zero = IRLiteral(result_type=IRType("int"), value=0)
        init = IRDeclare(var=loop_var, init=zero)
        condition = IRBinOp(
            result_type=IRType("bool"),
            op="<",
            left=IRName(result_type=IRType("int"), name=idx_name),
            right=IRLiteral(result_type=IRType("int"), value=size),
        )
        update = IRAugmentedAssign(
            target=IRName(result_type=IRType("int"), name=idx_name),
            op="+",
            value=IRLiteral(result_type=IRType("int"), value=1),
        )

        # Prepend element declaration to body
        elem_var = IRVariable(name=item_name, type=elem_type)
        elem_init = IRSubscript(
            result_type=elem_type,
            base=IRName(result_type=array_type, name=array_expr_name.id),
            index=IRName(result_type=IRType("int"), name=idx_name),
        )
        for_body: list[IRStmt] = [IRDeclare(var=elem_var, init=elem_init)]
        for s in body:
            for_body.extend(self._build_stmt(s))

        return [IRFor(init=init, condition=condition, update=update, body=for_body)]

    def _build_expr_with_type_hint(
        self, node: ast.expr | None, type_hint: IRType
    ) -> IRExpr | None:
        """Build IR expression, using type hint for list/tuple literals."""
        if node is None:
            return None

        # If it's a list/tuple and we have an array type hint, build as array
        if isinstance(node, ast.List | ast.Tuple) and type_hint.array_size is not None:
            ir_args = [self._build_expr(e) for e in node.elts]
            # array_size == -1 means infer from literal
            if type_hint.array_size == -1:
                actual_type = IRType(base=type_hint.base, array_size=len(ir_args))
                return IRConstruct(result_type=actual_type, args=ir_args)
            if len(ir_args) != type_hint.array_size:
                raise TranspilerError(
                    f"Array size mismatch: expected {type_hint.array_size} elements, "
                    f"got {len(ir_args)}"
                )
            return IRConstruct(result_type=type_hint, args=ir_args)

        # If it's a list comprehension with array type hint, use that type
        if isinstance(node, ast.ListComp) and type_hint.array_size is not None:
            result = self._build_list_comprehension(node.elt, node.generators)
            # Override the inferred type with the type hint (unless -1 = infer)
            if isinstance(result, IRConstruct) and type_hint.array_size != -1:
                result.result_type = type_hint
            return result

        return self._build_expr(node)

    def _build_expr(self, node: ast.expr) -> IRExpr:
        """Build IR expression from AST expression."""
        match node:
            case ast.Constant(value=value):
                ir_type = infer_literal_type(value)
                return IRLiteral(result_type=ir_type, value=value)

            case ast.Name(id=name):
                ir_type = self.symbols.get(name, IRType("float"))
                return IRName(result_type=ir_type, name=name)

            case ast.BinOp(left=left, op=op, right=right):
                left_expr = self._build_expr(left)
                right_expr = self._build_expr(right)
                op_str = binop_to_str(type(op).__name__)

                # Convert % on floats/vectors to mod() function call
                if op_str == "%" and left_expr.result_type.base not in ("int", "uint"):
                    return IRCall(
                        result_type=left_expr.result_type,
                        func="mod",
                        args=[left_expr, right_expr],
                    )

                # Convert // (floor division) to floor(x / y) for floats
                # or int(x / y) for integers
                if op_str == "//":
                    div_result_type = self._infer_binop_type(left_expr, right_expr, "/")
                    div_expr = IRBinOp(
                        result_type=div_result_type,
                        op="/",
                        left=left_expr,
                        right=right_expr,
                    )
                    if left_expr.result_type.base in ("int", "uint"):
                        # Integer division - just use / which truncates in GLSL
                        return div_expr
                    # Float division - wrap in floor()
                    return IRCall(
                        result_type=div_result_type,
                        func="floor",
                        args=[div_expr],
                    )

                # Optimize power operator for common cases
                if op_str == "**":
                    return self._build_power_expr(left_expr, right_expr, right)

                result_type = self._infer_binop_type(left_expr, right_expr, op_str)
                return IRBinOp(
                    result_type=result_type, op=op_str, left=left_expr, right=right_expr
                )

            case ast.UnaryOp(op=op, operand=operand):
                operand_expr = self._build_expr(operand)
                op_str = unaryop_to_str(type(op).__name__)
                return IRUnaryOp(
                    result_type=operand_expr.result_type,
                    op=op_str,
                    operand=operand_expr,
                )

            case ast.Compare(left=left, ops=ops, comparators=comparators):
                return self._build_compare(left, ops, comparators)

            case ast.BoolOp(op=op, values=values):
                return self._build_boolop(op, values)

            case ast.Call(func=func, args=args, keywords=keywords):
                return self._build_call(func, args, keywords)

            case ast.Attribute(value=value, attr=attr):
                # Handle ctx.vs_uv -> vs_uv (context builtin access)
                if self._is_context_access(value, attr):
                    glsl_type, _ = CONTEXT_BUILTINS[attr]
                    return IRName(result_type=IRType(glsl_type), name=attr)

                base_expr = self._build_expr(value)
                if is_swizzle(attr):
                    result_type = swizzle_result_type(attr)
                    return IRSwizzle(
                        result_type=result_type, base=base_expr, components=attr
                    )
                result_type = self._field_access_type(base_expr.result_type, attr)
                return IRFieldAccess(
                    result_type=result_type, base=base_expr, field=attr
                )

            case ast.Subscript(value=value, slice=slice_):
                base_expr = self._build_expr(value)
                index_expr = self._build_negative_index(slice_, base_expr.result_type)
                result_type = subscript_result_type(base_expr.result_type)
                return IRSubscript(
                    result_type=result_type, base=base_expr, index=index_expr
                )

            case ast.IfExp(test=test, body=body, orelse=orelse):
                cond = self._build_expr(test)
                true_expr = self._build_expr(body)
                false_expr = self._build_expr(orelse)
                return IRTernary(
                    result_type=true_expr.result_type,
                    condition=cond,
                    true_expr=true_expr,
                    false_expr=false_expr,
                )

            case ast.Tuple(elts=elts) | ast.List(elts=elts):
                if len(elts) < 2 or len(elts) > 4:
                    raise TranspilerError(
                        f"List/tuple must have 2-4 elements for vec conversion, "
                        f"got {len(elts)}. Use list[T] for arrays."
                    )
                ir_args: list[IRExpr] = [self._build_expr(e) for e in elts]
                result_type = IRType(f"vec{len(ir_args)}")
                return IRConstruct(result_type=result_type, args=ir_args)

            case ast.ListComp(elt=elt, generators=generators):
                return self._build_list_comprehension(elt, generators)

            case ast.NamedExpr(target=target, value=value):
                # Walrus operator (:=) — hoist declaration as a pending statement
                value_expr = self._build_expr(value)
                if not isinstance(target, ast.Name):
                    raise TranspilerError(
                        "Walrus operator target must be a simple name"
                    )
                var_name = target.id
                ir_type = value_expr.result_type
                var = IRVariable(name=var_name, type=ir_type)
                self.symbols[var_name] = ir_type
                self._pending_stmts.append(IRDeclare(var=var, init=value_expr))
                return IRName(result_type=ir_type, name=var_name)

        raise TranspilerError(f"Unsupported expression: {type(node).__name__}")

    def _build_call(
        self, func: ast.expr, args: list[ast.expr], keywords: list[ast.keyword]
    ) -> IRExpr:
        """Build IR for a function call."""
        if isinstance(func, ast.Name):
            func_name = func.id

            # Check if it's a struct constructor
            if func_name in self.collected.structs:
                arg_exprs = [self._build_expr(a) for a in args]
                for kw in keywords:
                    arg_exprs.append(self._build_expr(kw.value))
                return IRConstruct(result_type=IRType(func_name), args=arg_exprs)

            # Type constructors
            if func_name in TYPE_CONSTRUCTORS:
                arg_exprs = [self._build_expr(a) for a in args]
                for kw in keywords:
                    arg_exprs.append(self._build_expr(kw.value))
                return IRConstruct(result_type=IRType(func_name), args=arg_exprs)

            # Handle sum() builtin
            if func_name == "sum":
                return self._build_sum_call(args)

            # Handle len() builtin
            if func_name == "len":
                return self._build_len_call(args)

            # Regular function call - filter out ShaderContext arguments
            filtered_args = self._filter_context_args(func_name, args)
            arg_exprs = [self._build_expr(a) for a in filtered_args]
            for kw in keywords:
                arg_exprs.append(self._build_expr(kw.value))

            # Fill in default parameter values for user-defined functions
            arg_exprs = self._fill_default_args(func_name, arg_exprs)

            # Handle min/max with more than 2 arguments by chaining calls
            if func_name in ("min", "max") and len(arg_exprs) > 2:
                return self._build_chained_minmax(func_name, arg_exprs)

            result_type = self._infer_call_type(func_name, arg_exprs)
            return IRCall(result_type=result_type, func=func_name, args=arg_exprs)

        if isinstance(func, ast.Attribute):
            base_expr = self._build_expr(func.value)
            method_name = func.attr
            method_arg_exprs = [self._build_expr(a) for a in args]
            for kw in keywords:
                method_arg_exprs.append(self._build_expr(kw.value))

            # Check if this is a struct method call
            base_type = base_expr.result_type.base
            struct_def = self.collected.structs.get(base_type)
            if struct_def and method_name in struct_def.methods:
                # Transform obj.method(args) -> StructName_method(obj, args)
                method_info = struct_def.methods[method_name]
                full_func_name = f"{base_type}_{method_name}"
                all_args = [base_expr, *method_arg_exprs]
                # Fill in default args for method
                all_args = self._fill_method_default_args(method_info, all_args)
                result_type = (
                    IRType(method_info.return_type)
                    if method_info.return_type
                    else IRType("void")
                )
                return IRCall(
                    result_type=result_type, func=full_func_name, args=all_args
                )

            # Regular method call (e.g., vec3 methods, builtins)
            all_args = [base_expr, *method_arg_exprs]
            result_type = self._infer_call_type(method_name, all_args)
            return IRCall(result_type=result_type, func=method_name, args=all_args)

        raise TranspilerError(f"Unsupported call: {type(func).__name__}")

    def _build_compare(
        self, left: ast.expr, ops: list[ast.cmpop], comparators: list[ast.expr]
    ) -> IRExpr:
        """Build IR for comparison expression."""
        if len(ops) == 1:
            left_expr = self._build_expr(left)
            right_expr = self._build_expr(comparators[0])
            op_str = cmpop_to_str(type(ops[0]).__name__)
            return IRBinOp(
                result_type=IRType("bool"), op=op_str, left=left_expr, right=right_expr
            )

        # Chain comparisons: a < b < c -> (a < b) && (b < c)
        parts = []
        current = left
        for op, comp in zip(ops, comparators, strict=False):
            left_expr = self._build_expr(current)
            right_expr = self._build_expr(comp)
            op_str = cmpop_to_str(type(op).__name__)
            parts.append(
                IRBinOp(
                    result_type=IRType("bool"),
                    op=op_str,
                    left=left_expr,
                    right=right_expr,
                )
            )
            current = comp

        result = parts[0]
        for part in parts[1:]:
            result = IRBinOp(
                result_type=IRType("bool"), op="and", left=result, right=part
            )
        return result

    def _build_boolop(self, op: ast.boolop, values: list[ast.expr]) -> IRExpr:
        """Build IR for boolean operation."""
        op_str = "and" if isinstance(op, ast.And) else "or"
        exprs = [self._build_expr(v) for v in values]
        result = exprs[0]
        for expr in exprs[1:]:
            result = IRBinOp(
                result_type=IRType("bool"), op=op_str, left=result, right=expr
            )
        return result

    def _infer_binop_type(self, left: IRExpr, right: IRExpr, op: str) -> IRType:
        """Infer result type of binary operation."""
        if op in ("==", "!=", "<", "<=", ">", ">=", "and", "or"):
            return IRType("bool")

        result_type_str = infer_binop_result_type(
            left.result_type.base, right.result_type.base
        )
        return IRType(result_type_str)

    def _infer_call_type(self, func_name: str, args: list[IRExpr]) -> IRType:
        """Infer result type of function call."""
        if func_name in self.collected.functions:
            func_info = self.collected.functions[func_name]
            if func_info.return_type:
                return IRType(func_info.return_type)

        if func_name in SCALAR_RESULT_FUNCTIONS:
            return IRType("float")

        if func_name in BOOL_RESULT_FUNCTIONS:
            return IRType("bool")

        if func_name in PRESERVE_TYPE_FUNCTIONS and args:
            return args[0].result_type

        if func_name in ("texture", "texelFetch"):
            return IRType("vec4")

        if func_name == "cross":
            return IRType("vec3")

        return IRType("float")

    def _field_access_type(self, base_type: IRType, field: str) -> IRType:
        """Get result type of struct field access."""
        struct_def = self.collected.structs.get(base_type.base)
        if struct_def:
            for f in struct_def.fields:
                if f.name == field:
                    return IRType(f.type_name)
        return IRType("float")

    def _is_context_access(self, value: ast.expr, attr: str) -> bool:
        """Check if this is a context builtin access (e.g., ctx.vs_uv)."""
        if self._context_param_name is None:
            return False
        if not isinstance(value, ast.Name):
            return False
        if value.id != self._context_param_name:
            return False
        return attr in CONTEXT_BUILTINS

    def _build_negative_index(self, slice_: ast.expr, base_type: IRType) -> IRExpr:
        """Build index expression, converting negative indices to positive.

        For vec2/vec3/vec4 and arrays with known size, converts negative indices
        at compile time (e.g., v[-1] on vec4 becomes v[3]).
        """
        # Check if it's a negative constant via UnaryOp (e.g., -1)
        if (
            isinstance(slice_, ast.UnaryOp)
            and isinstance(slice_.op, ast.USub)
            and isinstance(slice_.operand, ast.Constant)
            and isinstance(slice_.operand.value, int)
        ):
            neg_index = -slice_.operand.value
            size = get_indexable_size(base_type)
            if size is not None and neg_index < 0:
                # Convert negative index to positive
                positive_index = size + neg_index
                return IRLiteral(result_type=IRType("int"), value=positive_index)
        # Also handle direct negative constant (e.g., ast.Constant with value=-1)
        if (
            isinstance(slice_, ast.Constant)
            and isinstance(slice_.value, int)
            and slice_.value < 0
        ):
            size = get_indexable_size(base_type)
            if size is not None:
                positive_index = size + slice_.value
                return IRLiteral(result_type=IRType("int"), value=positive_index)
        return self._build_expr(slice_)

    def _build_power_expr(
        self, base_expr: IRExpr, exp_expr: IRExpr, exp_node: ast.expr
    ) -> IRExpr:
        """Build optimized power expression.

        Optimizes only when exponent is a constant integer:
        - x**0.5 → sqrt(x)
        - x**2 → x * x
        - x**3 → x * x * x
        - x**4 → x * x * x * x
        - Others → pow(x, y)
        """
        result_type = base_expr.result_type

        # Check for constant exponent
        if isinstance(exp_node, ast.Constant):
            exp_val = exp_node.value
            # x**0.5 → sqrt(x)
            if exp_val == 0.5:
                return IRCall(result_type=result_type, func="sqrt", args=[base_expr])
            # Only optimize integer exponents 2, 3, 4
            # (float exponents like 3.0 should use pow())
            if isinstance(exp_val, int):
                # x**2 → x * x
                if exp_val == 2:
                    return IRBinOp(
                        result_type=result_type, op="*", left=base_expr, right=base_expr
                    )
                # x**3 → x * x * x
                if exp_val == 3:
                    x_squared = IRBinOp(
                        result_type=result_type, op="*", left=base_expr, right=base_expr
                    )
                    return IRBinOp(
                        result_type=result_type, op="*", left=x_squared, right=base_expr
                    )
                # x**4 → x * x * x * x (as (x*x) * (x*x) for efficiency)
                # But we emit it as x * x * x * x for consistency
                if exp_val == 4:
                    x2 = IRBinOp(
                        result_type=result_type, op="*", left=base_expr, right=base_expr
                    )
                    x3 = IRBinOp(
                        result_type=result_type, op="*", left=x2, right=base_expr
                    )
                    return IRBinOp(
                        result_type=result_type, op="*", left=x3, right=base_expr
                    )

        # Default: use pow()
        return IRCall(result_type=result_type, func="pow", args=[base_expr, exp_expr])

    def _build_list_comprehension(
        self, elt: ast.expr, generators: list[ast.comprehension]
    ) -> IRExpr:
        """Build IR for list comprehension by unrolling at compile time.

        Supports:
        - [expr for var in range(n)]
        - [expr for var in range(n) if condition]
        - [expr for var1 in range(n) for var2 in range(m)]
        Where ranges must be compile-time constants.
        """
        elements: list[IRExpr] = []
        self._unroll_generators(elt, generators, 0, {}, elements)

        if not elements:
            raise TranspilerError("List comprehension produced empty result")

        # Infer element type from first element
        elem_type = elements[0].result_type
        result_type = IRType(base=elem_type.base, array_size=len(elements))

        return IRConstruct(result_type=result_type, args=elements)

    def _unroll_generators(
        self,
        elt: ast.expr,
        generators: list[ast.comprehension],
        gen_idx: int,
        substitutions: dict[str, int],
        elements: list[IRExpr],
    ) -> None:
        """Recursively unroll comprehension generators."""
        if gen_idx >= len(generators):
            # All generators exhausted - evaluate the element expression
            substituted_elt = elt
            for var_name, var_val in substitutions.items():
                substituted_elt = substitute_name(substituted_elt, var_name, var_val)
            elem_expr = self._build_expr(substituted_elt)
            elements.append(elem_expr)
            return

        gen = generators[gen_idx]

        if not isinstance(gen.target, ast.Name):
            raise TranspilerError(
                "List comprehension target must be a simple variable name"
            )
        loop_var = gen.target.id

        # Parse the iterator - must be range()
        if not isinstance(gen.iter, ast.Call):
            raise TranspilerError("List comprehension iterator must be range()")
        if not isinstance(gen.iter.func, ast.Name) or gen.iter.func.id != "range":
            raise TranspilerError("List comprehension iterator must be range()")

        # Parse range arguments
        start, end, step = self._parse_range_args(gen.iter.args)

        old_symbol = self.symbols.get(loop_var)
        globals_dict = self.collected.globals

        for i in range(start, end, step):
            self.symbols[loop_var] = IRType("int")

            # Check if conditions (filters) - evaluate at compile time
            if gen.ifs:
                skip = False
                for if_node in gen.ifs:
                    substituted_cond = if_node
                    for var_name, var_val in substitutions.items():
                        substituted_cond = substitute_name(
                            substituted_cond, var_name, var_val
                        )
                    substituted_cond = substitute_name(substituted_cond, loop_var, i)
                    cond_val = eval_constant(substituted_cond, globals_dict)
                    if cond_val is None:
                        raise TranspilerError(
                            "List comprehension 'if' condition must be evaluable "
                            "at compile time"
                        )
                    if not cond_val:
                        skip = True
                        break
                if skip:
                    continue

            new_subs = {**substitutions, loop_var: i}
            self._unroll_generators(elt, generators, gen_idx + 1, new_subs, elements)

        # Restore old symbol
        if old_symbol is not None:
            self.symbols[loop_var] = old_symbol
        elif loop_var in self.symbols:
            del self.symbols[loop_var]

    def _parse_range_args(self, range_args: list[ast.expr]) -> tuple[int, int, int]:
        """Parse range() arguments, all must be compile-time constants."""
        globals_dict = self.collected.globals
        if len(range_args) == 1:
            end_val = eval_constant(range_args[0], globals_dict)
            if end_val is None:
                raise TranspilerError(
                    "List comprehension range() args must be compile-time constants"
                )
            return 0, end_val, 1
        elif len(range_args) == 2:
            start_val = eval_constant(range_args[0], globals_dict)
            end_val = eval_constant(range_args[1], globals_dict)
            if start_val is None or end_val is None:
                raise TranspilerError(
                    "List comprehension range() args must be compile-time constants"
                )
            return start_val, end_val, 1
        elif len(range_args) == 3:
            start_val = eval_constant(range_args[0], globals_dict)
            end_val = eval_constant(range_args[1], globals_dict)
            step_val = eval_constant(range_args[2], globals_dict)
            if start_val is None or end_val is None or step_val is None:
                raise TranspilerError(
                    "List comprehension range() args must be compile-time constants"
                )
            return start_val, end_val, step_val
        else:
            raise TranspilerError("range() requires 1-3 arguments")

    def _build_sum_call(self, args: list[ast.expr]) -> IRExpr:
        """Build IR for sum() builtin.

        Handles:
        - sum(array_name) -> unrolled addition of all elements
        - sum(generator_expr) -> unrolled addition from generator
        """
        if len(args) != 1:
            raise TranspilerError("sum() requires exactly 1 argument")

        arg = args[0]

        # sum(array_name) - array variable reference
        if isinstance(arg, ast.Name):
            array_type = self.symbols.get(arg.id)
            if array_type and array_type.array_size is not None:
                elem_type = IRType(array_type.base)
                array_ref = IRName(result_type=array_type, name=arg.id)
                result: IRExpr = IRSubscript(
                    result_type=elem_type,
                    base=array_ref,
                    index=IRLiteral(result_type=IRType("int"), value=0),
                )
                for i in range(1, array_type.array_size):
                    elem = IRSubscript(
                        result_type=elem_type,
                        base=array_ref,
                        index=IRLiteral(result_type=IRType("int"), value=i),
                    )
                    result = IRBinOp(
                        result_type=elem_type, op="+", left=result, right=elem
                    )
                return result

        # sum(generator_expr) or sum(list_comp) - unroll and chain additions
        if isinstance(arg, ast.GeneratorExp | ast.ListComp):
            elements: list[IRExpr] = []
            self._unroll_generators(arg.elt, arg.generators, 0, {}, elements)
            if not elements:
                raise TranspilerError("sum() over empty generator")
            gen_result: IRExpr = elements[0]
            for el in elements[1:]:
                gen_result = IRBinOp(
                    result_type=gen_result.result_type,
                    op="+",
                    left=gen_result,
                    right=el,
                )
            return gen_result

        # Fallback: try to build the expression normally
        arg_expr = self._build_expr(arg)
        return arg_expr

    def _build_len_call(self, args: list[ast.expr]) -> IRExpr:
        """Build IR for len() builtin.

        Returns compile-time constant for arrays and vectors.
        """
        if len(args) != 1:
            raise TranspilerError("len() requires exactly 1 argument")

        arg = args[0]
        if isinstance(arg, ast.Name):
            var_type = self.symbols.get(arg.id)
            if var_type:
                # Array with known size
                if var_type.array_size is not None:
                    return IRLiteral(
                        result_type=IRType("int"), value=var_type.array_size
                    )
                # Vector types
                size = get_indexable_size(var_type)
                if size is not None:
                    return IRLiteral(result_type=IRType("int"), value=size)

        raise TranspilerError("len() argument must be an array or vector variable")

    def _build_chained_minmax(self, func_name: str, arg_exprs: list[IRExpr]) -> IRExpr:
        """Build chained min/max calls for more than 2 arguments.

        min(a, b, c, d) -> min(min(min(a, b), c), d)
        """
        result_type = arg_exprs[0].result_type
        result = IRCall(
            result_type=result_type, func=func_name, args=[arg_exprs[0], arg_exprs[1]]
        )
        for arg in arg_exprs[2:]:
            result = IRCall(result_type=result_type, func=func_name, args=[result, arg])
        return result

    def _build_tuple_unpack(self, target: ast.Tuple, value: ast.expr) -> list[IRStmt]:
        """Build IR for tuple unpacking assignment.

        For simple values (variables, swizzles), generates direct access:
            x, y = vs_uv  ->  float x = vs_uv.x; float y = vs_uv.y;

        For complex values (function calls), generates temp variable:
            r, theta = get_polar(p)  ->  vec2 _tmp0 = get_polar(p);
                                         float r = _tmp0.x; float theta = _tmp0.y;
        """
        elts = target.elts
        num_vars = len(elts)
        if num_vars < 2 or num_vars > 4:
            raise TranspilerError(
                f"Tuple unpacking supports 2-4 elements, got {num_vars}"
            )

        # Swizzle components for extraction
        swizzle_map = ["x", "y", "z", "w"]
        result: list[IRStmt] = []

        # Check if value is "simple" (can access multiple times without side effects)
        is_simple = self._is_simple_expr(value)

        if is_simple:
            # Direct access without temp variable
            value_expr = self._build_expr(value)

            for i, elt in enumerate(elts):
                if not isinstance(elt, ast.Name):
                    raise TranspilerError("Tuple unpacking target must be simple names")

                component = swizzle_map[i]
                var_name = elt.id
                var_type = IRType("float")

                # Create the swizzle access on the original expression
                swizzle_expr = IRSwizzle(
                    result_type=var_type, base=value_expr, components=component
                )

                var = IRVariable(name=var_name, type=var_type)
                self.symbols[var_name] = var_type
                result.append(IRDeclare(var=var, init=swizzle_expr))
        else:
            # Complex expression - use temp variable
            value_expr = self._build_expr(value)
            vec_type = IRType(f"vec{num_vars}")

            tmp_name = f"_tmp{self._tmp_var_counter}"
            self._tmp_var_counter += 1
            tmp_var = IRVariable(name=tmp_name, type=vec_type)
            self.symbols[tmp_name] = vec_type

            result.append(IRDeclare(var=tmp_var, init=value_expr))

            for i, elt in enumerate(elts):
                if not isinstance(elt, ast.Name):
                    raise TranspilerError("Tuple unpacking target must be simple names")

                component = swizzle_map[i]
                var_name = elt.id
                var_type = IRType("float")

                tmp_ref = IRName(result_type=vec_type, name=tmp_name)
                swizzle_expr = IRSwizzle(
                    result_type=var_type, base=tmp_ref, components=component
                )

                var = IRVariable(name=var_name, type=var_type)
                self.symbols[var_name] = var_type
                result.append(IRDeclare(var=var, init=swizzle_expr))

        return result

    def _is_simple_expr(self, node: ast.expr) -> bool:
        """Check if an expression is simple (no side effects, can be duplicated)."""
        if isinstance(node, ast.Name):
            return True
        if isinstance(node, ast.Attribute):
            # Allow swizzle access on simple base
            return self._is_simple_expr(node.value)
        if isinstance(node, ast.Subscript):
            # Allow indexing on simple base with constant index
            return self._is_simple_expr(node.value) and isinstance(
                node.slice, ast.Constant
            )
        return False

    def _fill_default_args(
        self, func_name: str, arg_exprs: list[IRExpr]
    ) -> list[IRExpr]:
        """Fill in default argument values for missing parameters.

        For user-defined functions with default parameters, if fewer arguments
        are provided than the function expects, fills in the defaults.
        """
        func_info = self.collected.functions.get(func_name)
        if not func_info or not func_info.param_defaults:
            return arg_exprs

        # Count expected params (excluding ShaderContext)
        expected_params = 0
        for arg in func_info.node.args.args:
            if not self._is_context_param(arg):
                expected_params += 1

        # If we already have all args, nothing to fill
        if len(arg_exprs) >= expected_params:
            return arg_exprs

        # Fill in defaults from the end
        # Python defaults are aligned to the end of params
        num_defaults = len(func_info.param_defaults)
        num_missing = expected_params - len(arg_exprs)

        # Which default indices we need (from the end)
        # e.g., if we have 3 params (a, b, c) with 2 defaults (b_def, c_def)
        # and only 1 arg provided, we need both defaults
        # defaults are stored for the LAST num_defaults params
        start_default_idx = num_defaults - num_missing

        result = list(arg_exprs)
        for i in range(start_default_idx, num_defaults):
            default_str = func_info.param_defaults[i]
            if not default_str:
                raise TranspilerError(
                    f"Missing required argument for function {func_name}"
                )
            # Parse the default value string into an IR expression
            result.append(self._parse_default_value(default_str, func_info, i))

        return result

    def _parse_default_value(
        self, value_str: str, func_info: "FunctionInfo", default_idx: int
    ) -> IRExpr:
        """Parse a default value string into an IR expression."""
        # Get the type of the parameter this default is for
        # defaults align to the end of parameters
        num_params = len(func_info.param_types)
        num_defaults = len(func_info.param_defaults)
        param_idx = num_params - num_defaults + default_idx
        param_type = func_info.param_types[param_idx] or "float"

        # Try to parse as a simple literal or constructor
        try:
            node = ast.parse(value_str, mode="eval").body
            if isinstance(node, ast.Constant):
                ir_type = infer_literal_type(node.value)
                return IRLiteral(result_type=ir_type, value=node.value)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                # vec2/vec3/vec4 constructor
                func_name = node.func.id
                if func_name in TYPE_CONSTRUCTORS:
                    args = [self._build_expr(a) for a in node.args]
                    return IRConstruct(result_type=IRType(func_name), args=args)
        except Exception:
            pass

        # Fallback: create a literal with the string value
        return IRLiteral(result_type=IRType(param_type), value=float(value_str))


def build_ir(
    collected: CollectedInfo,
    entry_point: str,
    stage: ShaderStage = ShaderStage.FRAGMENT,
) -> ShaderIR:
    """Build ShaderIR from CollectedInfo."""
    builder = IRBuilder(collected, entry_point)
    return builder.build(stage)
