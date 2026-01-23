"""Build IR from CollectedInfo."""

import ast
import re

from py2glsl.context import CONTEXT_BUILTINS
from py2glsl.transpiler.constants import (
    AST_BINOP_MAP,
    AST_CMPOP_MAP,
    AST_UNARYOP_MAP,
    BOOL_RESULT_FUNCTIONS,
    MATRIX_TO_VECTOR,
    MAX_SWIZZLE_LENGTH,
    PRESERVE_TYPE_FUNCTIONS,
    SCALAR_RESULT_FUNCTIONS,
    SWIZZLE_CHARS,
    TYPE_CONSTRUCTORS,
    VECTOR_ELEMENT_TYPE,
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
from py2glsl.transpiler.models import CollectedInfo, FunctionInfo, TranspilerError
from py2glsl.transpiler.type_checker import infer_binop_result_type


class IRBuilder:
    """Builds ShaderIR from CollectedInfo."""

    def __init__(self, collected: CollectedInfo, entry_point: str):
        self.collected = collected
        self.entry_point_name = entry_point
        self.symbols: dict[str, IRType] = {}
        self._context_param_name: str | None = None
        self._tmp_var_counter: int = 0

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
        """Convert FunctionInfos to IRFunctions."""
        result = []
        for func_name, func_info in self.collected.functions.items():
            is_entry = func_name == self.entry_point_name
            ir_func = self._build_function(func_info, is_entry)
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
        """Infer the IR type of an expression without building it."""
        match node:
            case ast.Constant(value=value):
                return self._infer_literal_type(value)
            case ast.Name(id=name):
                return self.symbols.get(name, IRType("float"))
            case ast.Call(func=func):
                if isinstance(func, ast.Name):
                    func_name = func.id
                    # Check structs
                    if func_name in self.collected.structs:
                        return IRType(func_name)
                    # Check type constructors
                    if func_name in TYPE_CONSTRUCTORS:
                        return IRType(func_name)
                    # Check functions
                    if func_name in self.collected.functions:
                        ret = self.collected.functions[func_name].return_type
                        if ret:
                            return IRType(ret)
                return IRType("float")
            case ast.BinOp(left=left):
                return self._infer_expr_type(left)
            case ast.UnaryOp(operand=operand):
                return self._infer_expr_type(operand)
            case ast.Attribute(value=value, attr=attr):
                base_type = self._infer_expr_type(value)
                if self._is_swizzle(attr):
                    return self._swizzle_result_type(base_type, attr)
                return self._field_access_type(base_type, attr)
            case ast.Subscript(value=value):
                base_type = self._infer_expr_type(value)
                return self._subscript_result_type(base_type)
        return IRType("float")

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
                            vec_type = IRType(f"vec{len(elts)}")
                            vec_construct = IRConstruct(result_type=vec_type, args=args)
                            return [IRReturn(value=vec_construct)]
                    return [IRReturn(value=self._build_expr(value))]
                return [IRReturn()]

            case ast.Assign(targets=targets, value=value):
                result: list[IRStmt] = []

                # Handle tuple unpacking: a, b = func()
                if len(targets) == 1 and isinstance(targets[0], ast.Tuple):
                    return self._build_tuple_unpack(targets[0], value)

                value_expr = self._build_expr(value)
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
                type_name = self._get_type_from_annotation(ann)
                ir_type = self._parse_type_string(type_name)
                if isinstance(target, ast.Name):
                    init = None
                    if value:
                        init = self._build_expr_with_type_hint(value, ir_type)
                        # If size was inferred (-1), use the actual type from init
                        if ir_type.array_size == -1 and init is not None:
                            ir_type = init.result_type
                    var = IRVariable(name=target.id, type=ir_type)
                    self.symbols[target.id] = ir_type
                    return [IRDeclare(var=var, init=init)]
                return []

            case ast.AugAssign(target=target, op=op, value=value):
                op_str = self._binop_to_str(op)
                target_expr = self._build_expr(target)
                value_expr = self._build_expr(value)
                return [
                    IRAugmentedAssign(target=target_expr, op=op_str, value=value_expr)
                ]

            case ast.If(test=test, body=body, orelse=orelse):
                # Pre-scan to hoist variable declarations assigned in branches
                hoisted = self._hoist_if_variables(node)
                cond = self._build_expr(test)
                then_body = []
                for s in body:
                    then_body.extend(self._build_stmt(s))
                else_body = []
                for s in orelse:
                    else_body.extend(self._build_stmt(s))
                return [
                    *hoisted,
                    IRIf(condition=cond, then_body=then_body, else_body=else_body),
                ]

            case ast.For(target=target, iter=iter_expr, body=body):
                return self._build_for_loop(target, iter_expr, body)

            case ast.While(test=test, body=body):
                cond = self._build_expr(test)
                while_body = []
                for s in body:
                    while_body.extend(self._build_stmt(s))
                return [IRWhile(condition=cond, body=while_body)]

            case ast.Expr(value=value):
                return [IRExprStmt(expr=self._build_expr(value))]

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
        """Build IR for a for loop (only supports range())."""
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
                ir_type = self._infer_literal_type(value)
                return IRLiteral(result_type=ir_type, value=value)

            case ast.Name(id=name):
                ir_type = self.symbols.get(name, IRType("float"))
                return IRName(result_type=ir_type, name=name)

            case ast.BinOp(left=left, op=op, right=right):
                left_expr = self._build_expr(left)
                right_expr = self._build_expr(right)
                op_str = self._binop_to_str(op)

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
                op_str = self._unaryop_to_str(op)
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
                if self._is_swizzle(attr):
                    result_type = self._swizzle_result_type(base_expr.result_type, attr)
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
                result_type = self._subscript_result_type(base_expr.result_type)
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
            op_str = self._cmpop_to_str(ops[0])
            return IRBinOp(
                result_type=IRType("bool"), op=op_str, left=left_expr, right=right_expr
            )

        # Chain comparisons: a < b < c -> (a < b) && (b < c)
        parts = []
        current = left
        for op, comp in zip(ops, comparators, strict=False):
            left_expr = self._build_expr(current)
            right_expr = self._build_expr(comp)
            op_str = self._cmpop_to_str(op)
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

    def _binop_to_str(self, op: ast.operator) -> str:
        """Convert AST operator to string."""
        return AST_BINOP_MAP.get(type(op).__name__, "+")

    def _unaryop_to_str(self, op: ast.unaryop) -> str:
        """Convert AST unary operator to string."""
        return AST_UNARYOP_MAP.get(type(op).__name__, "-")

    def _cmpop_to_str(self, op: ast.cmpop) -> str:
        """Convert AST comparison operator to string."""
        return AST_CMPOP_MAP.get(type(op).__name__, "==")

    def _get_type_from_annotation(self, ann: ast.expr | None) -> str:
        """Extract type name from annotation."""
        if ann is None:
            return "float"
        if isinstance(ann, ast.Name):
            return ann.id
        if isinstance(ann, ast.Constant):
            return str(ann.value)
        # Handle list[T] syntax for arrays
        if isinstance(ann, ast.Subscript) and isinstance(ann.value, ast.Name):
            type_name = ann.value.id
            if type_name == "list":
                elem_type = self._get_type_from_annotation(ann.slice)
                return f"{elem_type}[]"
        return "float"

    def _parse_type_string(self, type_str: str) -> IRType:
        """Parse type string like 'float[3]' or 'float[]' into IRType."""
        # Match 'type[size]' with explicit size
        match = re.match(r"(\w+)\[(\d+)\]", type_str)
        if match:
            base = match.group(1)
            size = int(match.group(2))
            return IRType(base=base, array_size=size)
        # Match 'type[]' without size (size to be inferred from literal)
        match = re.match(r"(\w+)\[\]", type_str)
        if match:
            base = match.group(1)
            # array_size=-1 means "infer from literal"
            return IRType(base=base, array_size=-1)
        return IRType(base=type_str)

    def _infer_literal_type(self, value: object) -> IRType:
        """Infer IR type from literal value."""
        if isinstance(value, bool):
            return IRType("bool")
        if isinstance(value, int):
            return IRType("int")
        if isinstance(value, float):
            return IRType("float")
        return IRType("float")

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

    def _is_swizzle(self, attr: str) -> bool:
        """Check if attribute access is a vector swizzle."""
        if not attr or len(attr) > MAX_SWIZZLE_LENGTH:
            return False
        return all(c in SWIZZLE_CHARS for c in attr)

    def _swizzle_result_type(self, _base_type: IRType, components: str) -> IRType:
        """Get result type of swizzle operation."""
        n = len(components)
        if n == 1:
            return IRType("float")
        return IRType(f"vec{n}")

    def _field_access_type(self, base_type: IRType, field: str) -> IRType:
        """Get result type of struct field access."""
        struct_def = self.collected.structs.get(base_type.base)
        if struct_def:
            for f in struct_def.fields:
                if f.name == field:
                    return IRType(f.type_name)
        return IRType("float")

    def _subscript_result_type(self, base_type: IRType) -> IRType:
        """Get result type of subscript operation."""
        # Array subscript returns the element type
        if base_type.array_size is not None:
            return IRType(base_type.base)

        base = base_type.base
        # Check vector types (vec, ivec, uvec, bvec)
        for prefix, element_type in VECTOR_ELEMENT_TYPE.items():
            if base.startswith(prefix):
                return IRType(element_type)
        # Matrix subscript returns corresponding vector
        if base in MATRIX_TO_VECTOR:
            return IRType(MATRIX_TO_VECTOR[base])
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
            size = self._get_indexable_size(base_type)
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
            size = self._get_indexable_size(base_type)
            if size is not None:
                positive_index = size + slice_.value
                return IRLiteral(result_type=IRType("int"), value=positive_index)
        return self._build_expr(slice_)

    def _get_indexable_size(self, ir_type: IRType) -> int | None:
        """Get the size of an indexable type (vector or array)."""
        # Check for array types
        if ir_type.array_size is not None:
            return ir_type.array_size
        # Check for vector types (vec2, vec3, vec4, ivec2, etc.)
        base = ir_type.base
        for prefix in ("vec", "ivec", "uvec", "bvec"):
            if base.startswith(prefix) and len(base) == len(prefix) + 1:
                try:
                    return int(base[-1])
                except ValueError:
                    pass
        return None

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

        Supports: [expr for var in range(n)]
        Where n must be a constant.

        Example: [i * 2.0 for i in range(4)]
        Becomes: float[4](0.0, 2.0, 4.0, 6.0)
        """
        if len(generators) != 1:
            raise TranspilerError(
                "List comprehensions with multiple 'for' clauses are not supported"
            )

        gen = generators[0]

        # Check for conditions (if clauses)
        if gen.ifs:
            raise TranspilerError(
                "List comprehensions with 'if' conditions are not supported"
            )

        # Get loop variable name
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
        range_args = gen.iter.args
        start: int
        end: int
        step: int
        if len(range_args) == 1:
            end_val = self._eval_constant(range_args[0])
            if end_val is None:
                raise TranspilerError(
                    "List comprehension range() args must be compile-time constants"
                )
            start, end, step = 0, end_val, 1
        elif len(range_args) == 2:
            start_val = self._eval_constant(range_args[0])
            end_val = self._eval_constant(range_args[1])
            if start_val is None or end_val is None:
                raise TranspilerError(
                    "List comprehension range() args must be compile-time constants"
                )
            start, end, step = start_val, end_val, 1
        elif len(range_args) == 3:
            start_val = self._eval_constant(range_args[0])
            end_val = self._eval_constant(range_args[1])
            step_val = self._eval_constant(range_args[2])
            if start_val is None or end_val is None or step_val is None:
                raise TranspilerError(
                    "List comprehension range() args must be compile-time constants"
                )
            start, end, step = start_val, end_val, step_val
        else:
            raise TranspilerError("range() requires 1-3 arguments")

        # Generate the elements by evaluating the expression for each iteration
        elements: list[IRExpr] = []
        old_symbol = self.symbols.get(loop_var)

        for i in range(start, end, step):
            # Temporarily bind loop variable to current value
            self.symbols[loop_var] = IRType("int")

            # Create a modified AST with the loop variable replaced by constant
            substituted_elt = self._substitute_name(elt, loop_var, i)
            elem_expr = self._build_expr(substituted_elt)
            elements.append(elem_expr)

        # Restore old symbol
        if old_symbol is not None:
            self.symbols[loop_var] = old_symbol
        elif loop_var in self.symbols:
            del self.symbols[loop_var]

        if not elements:
            raise TranspilerError("List comprehension produced empty result")

        # Infer element type from first element
        elem_type = elements[0].result_type
        result_type = IRType(base=elem_type.base, array_size=len(elements))

        return IRConstruct(result_type=result_type, args=elements)

    def _eval_constant(self, node: ast.expr) -> int | None:
        """Evaluate an AST node as a compile-time constant integer."""
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name) and node.id in self.collected.globals:
            # Check if it's a known constant (globals store tuple of (type, value))
            type_str, value_str = self.collected.globals[node.id]
            if type_str == "int":
                return int(value_str)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = self._eval_constant(node.operand)
            if inner is not None:
                return -inner
        if isinstance(node, ast.BinOp):
            left = self._eval_constant(node.left)
            right = self._eval_constant(node.right)
            if left is not None and right is not None:
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.FloorDiv):
                    return left // right
        return None

    def _substitute_name(self, node: ast.expr, name: str, value: int) -> ast.expr:
        """Substitute a name with a constant value in an AST expression."""
        if isinstance(node, ast.Name) and node.id == name:
            return ast.Constant(value=value)
        if isinstance(node, ast.BinOp):
            return ast.BinOp(
                left=self._substitute_name(node.left, name, value),
                op=node.op,
                right=self._substitute_name(node.right, name, value),
            )
        if isinstance(node, ast.UnaryOp):
            return ast.UnaryOp(
                op=node.op,
                operand=self._substitute_name(node.operand, name, value),
            )
        if isinstance(node, ast.Call):
            return ast.Call(
                func=node.func,
                args=[self._substitute_name(a, name, value) for a in node.args],
                keywords=[
                    ast.keyword(
                        arg=kw.arg,
                        value=self._substitute_name(kw.value, name, value),
                    )
                    for kw in node.keywords
                ],
            )
        if isinstance(node, ast.IfExp):
            return ast.IfExp(
                test=self._substitute_name(node.test, name, value),
                body=self._substitute_name(node.body, name, value),
                orelse=self._substitute_name(node.orelse, name, value),
            )
        if isinstance(node, ast.Compare):
            return ast.Compare(
                left=self._substitute_name(node.left, name, value),
                ops=node.ops,
                comparators=[
                    self._substitute_name(c, name, value) for c in node.comparators
                ],
            )
        if isinstance(node, ast.Subscript):
            return ast.Subscript(
                value=self._substitute_name(node.value, name, value),
                slice=self._substitute_name(node.slice, name, value),
                ctx=node.ctx,
            )
        if isinstance(node, ast.Attribute):
            return ast.Attribute(
                value=self._substitute_name(node.value, name, value),
                attr=node.attr,
                ctx=node.ctx,
            )
        if isinstance(node, ast.Tuple):
            return ast.Tuple(
                elts=[self._substitute_name(e, name, value) for e in node.elts],
                ctx=node.ctx,
            )
        if isinstance(node, ast.List):
            return ast.List(
                elts=[self._substitute_name(e, name, value) for e in node.elts],
                ctx=node.ctx,
            )
        # For constants and other nodes, return as-is
        return node

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
                ir_type = self._infer_literal_type(node.value)
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
