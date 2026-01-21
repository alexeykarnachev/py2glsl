"""Build IR from CollectedInfo."""

import ast

from py2glsl.context import CONTEXT_BUILTINS
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

# GLSL type constructors that can be called as functions
TYPE_CONSTRUCTORS = frozenset(
    {
        "vec2",
        "vec3",
        "vec4",
        "ivec2",
        "ivec3",
        "ivec4",
        "uvec2",
        "uvec3",
        "uvec4",
        "mat2",
        "mat3",
        "mat4",
        "float",
        "int",
        "bool",
    }
)


class IRBuilder:
    """Builds ShaderIR from CollectedInfo."""

    def __init__(self, collected: CollectedInfo, entry_point: str):
        self.collected = collected
        self.entry_point_name = entry_point
        self.symbols: dict[str, IRType] = {}
        self._context_param_name: str | None = None

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

        # Add globals as constants
        for name, (type_name, value) in self.collected.globals.items():
            variables.append(
                IRVariable(
                    name=name,
                    type=IRType(type_name),
                    storage=StorageClass.CONST,
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

        # Add globals to symbols
        for name, (type_name, _) in self.collected.globals.items():
            self.symbols[name] = IRType(type_name)

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
        return IRType("float")

    def _build_stmt(self, node: ast.stmt) -> list[IRStmt]:
        """Build IR statement(s) from AST statement."""
        match node:
            case ast.Return(value=value):
                if value:
                    return [IRReturn(value=self._build_expr(value))]
                return [IRReturn()]

            case ast.Assign(targets=targets, value=value):
                result: list[IRStmt] = []
                value_expr = self._build_expr(value)
                for target in targets:
                    if isinstance(target, ast.Name) and target.id not in self.symbols:
                        # First assignment to a new variable - emit declaration
                        ir_type = value_expr.result_type
                        var = IRVariable(name=target.id, type=ir_type)
                        self.symbols[target.id] = ir_type
                        result.append(IRDeclare(var=var, init=value_expr))
                    else:
                        target_expr = self._build_expr(target)
                        result.append(IRAssign(target=target_expr, value=value_expr))
                return result

            case ast.AnnAssign(target=target, annotation=ann, value=value):
                type_name = self._get_type_from_annotation(ann)
                ir_type = IRType(type_name)
                if isinstance(target, ast.Name):
                    var = IRVariable(name=target.id, type=ir_type)
                    self.symbols[target.id] = ir_type
                    init = self._build_expr(value) if value else None
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
                index_expr = self._build_expr(slice_)
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
                ir_args: list[IRExpr] = [self._build_expr(e) for e in elts]
                result_type = IRType(f"vec{len(ir_args)}")
                return IRConstruct(result_type=result_type, args=ir_args)

        raise TranspilerError(f"Unsupported expression: {type(node).__name__}")

    def _build_call(
        self, func: ast.expr, args: list[ast.expr], keywords: list[ast.keyword]
    ) -> IRExpr:
        """Build IR for a function call."""
        arg_exprs = [self._build_expr(a) for a in args]
        # Add keyword arguments as positional (GLSL structs use positional)
        for kw in keywords:
            arg_exprs.append(self._build_expr(kw.value))

        if isinstance(func, ast.Name):
            func_name = func.id

            # Check if it's a struct constructor
            if func_name in self.collected.structs:
                return IRConstruct(result_type=IRType(func_name), args=arg_exprs)

            # Type constructors
            if func_name in TYPE_CONSTRUCTORS:
                return IRConstruct(result_type=IRType(func_name), args=arg_exprs)

            # Regular function call
            result_type = self._infer_call_type(func_name, arg_exprs)
            return IRCall(result_type=result_type, func=func_name, args=arg_exprs)

        if isinstance(func, ast.Attribute):
            base_expr = self._build_expr(func.value)
            method_name = func.attr
            all_args = [base_expr, *arg_exprs]
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
        match op:
            case ast.Add():
                return "+"
            case ast.Sub():
                return "-"
            case ast.Mult():
                return "*"
            case ast.Div():
                return "/"
            case ast.Mod():
                return "%"
            case ast.Pow():
                return "**"
            case ast.BitAnd():
                return "&"
            case ast.BitOr():
                return "|"
            case ast.BitXor():
                return "^"
            case ast.LShift():
                return "<<"
            case ast.RShift():
                return ">>"
        return "+"

    def _unaryop_to_str(self, op: ast.unaryop) -> str:
        """Convert AST unary operator to string."""
        match op:
            case ast.USub():
                return "-"
            case ast.UAdd():
                return "+"
            case ast.Not():
                return "not"
            case ast.Invert():
                return "~"
        return "-"

    def _cmpop_to_str(self, op: ast.cmpop) -> str:
        """Convert AST comparison operator to string."""
        match op:
            case ast.Eq():
                return "=="
            case ast.NotEq():
                return "!="
            case ast.Lt():
                return "<"
            case ast.LtE():
                return "<="
            case ast.Gt():
                return ">"
            case ast.GtE():
                return ">="
        return "=="

    def _get_type_from_annotation(self, ann: ast.expr | None) -> str:
        """Extract type name from annotation."""
        if ann is None:
            return "float"
        if isinstance(ann, ast.Name):
            return ann.id
        if isinstance(ann, ast.Constant):
            return str(ann.value)
        return "float"

    def _infer_literal_type(self, value: object) -> IRType:
        """Infer IR type from literal value."""
        if isinstance(value, bool):
            return IRType("bool")
        if isinstance(value, int):
            return IRType("int")
        if isinstance(value, float):
            return IRType("float")
        return IRType("float")

    def _infer_binop_type(self, left: IRExpr, _right: IRExpr, op: str) -> IRType:
        """Infer result type of binary operation."""
        if op in ("==", "!=", "<", "<=", ">", ">=", "and", "or"):
            return IRType("bool")
        return left.result_type

    def _infer_call_type(self, func_name: str, args: list[IRExpr]) -> IRType:
        """Infer result type of function call."""
        if func_name in self.collected.functions:
            func_info = self.collected.functions[func_name]
            if func_info.return_type:
                return IRType(func_info.return_type)

        scalar_result = {"length", "distance", "dot", "determinant"}
        if func_name in scalar_result:
            return IRType("float")

        bool_result = {"any", "all", "not"}
        if func_name in bool_result:
            return IRType("bool")

        preserve_type = {
            "abs",
            "sign",
            "floor",
            "ceil",
            "fract",
            "mod",
            "min",
            "max",
            "clamp",
            "mix",
            "step",
            "smoothstep",
            "sin",
            "cos",
            "tan",
            "asin",
            "acos",
            "atan",
            "sinh",
            "cosh",
            "tanh",
            "exp",
            "log",
            "exp2",
            "log2",
            "sqrt",
            "inversesqrt",
            "pow",
            "normalize",
            "reflect",
            "refract",
        }
        if func_name in preserve_type and args:
            return args[0].result_type

        if func_name in ("texture", "texelFetch"):
            return IRType("vec4")

        if func_name == "cross":
            return IRType("vec3")

        return IRType("float")

    def _is_swizzle(self, attr: str) -> bool:
        """Check if attribute access is a vector swizzle."""
        if not attr or len(attr) > 4:
            return False
        return all(c in "xyzwrgba" for c in attr)

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
        base = base_type.base
        if base.startswith("vec") or base.startswith("ivec") or base.startswith("uvec"):
            if base.startswith("ivec"):
                return IRType("int")
            if base.startswith("uvec"):
                return IRType("uint")
            return IRType("float")
        if base.startswith("mat"):
            size = base[3] if len(base) > 3 else "4"
            return IRType(f"vec{size}")
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


def build_ir(
    collected: CollectedInfo,
    entry_point: str,
    stage: ShaderStage = ShaderStage.FRAGMENT,
) -> ShaderIR:
    """Build ShaderIR from CollectedInfo."""
    builder = IRBuilder(collected, entry_point)
    return builder.build(stage)
