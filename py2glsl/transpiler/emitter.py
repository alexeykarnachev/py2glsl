"""Code emitter that generates target code from IR."""

from collections import defaultdict

from py2glsl.transpiler.constants import OPERATOR_PRECEDENCE
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
    IRUnaryOp,
    IRVariable,
    IRWhile,
    ShaderIR,
    StorageClass,
)
from py2glsl.transpiler.target import Target


def _topological_sort(functions: list[IRFunction]) -> list[IRFunction]:
    """Sort functions so that callees come before callers."""
    func_map = {f.name: f for f in functions}
    func_names = set(func_map.keys())

    # Build dependency graph: func -> set of functions it calls
    dependencies: dict[str, set[str]] = defaultdict(set)
    for func in functions:
        dependencies[func.name] = _find_called_functions(func, func_names)

    # Count dependencies for each function
    in_degree: dict[str, int] = {name: len(deps) for name, deps in dependencies.items()}

    # Start with functions that don't call any other user functions
    queue = [name for name, degree in in_degree.items() if degree == 0]
    result: list[IRFunction] = []
    visited: set[str] = set()

    while queue:
        name = queue.pop(0)
        if name in visited:
            continue
        visited.add(name)

        # Check if all dependencies are processed
        if all(dep in visited for dep in dependencies[name]):
            result.append(func_map[name])
            # Add functions that depend on this one
            for other_name, deps in dependencies.items():
                if (
                    name in deps
                    and other_name not in visited
                    and all(d in visited for d in dependencies[other_name])
                ):
                    queue.append(other_name)
        else:
            # Dependencies not ready, process them first
            for dep in dependencies[name]:
                if dep not in visited:
                    queue.insert(0, dep)
            queue.append(name)

    # Add any remaining functions (in case of cycles or disconnected)
    for func in functions:
        if func not in result:
            result.append(func)

    return result


def _find_called_functions(func: IRFunction, all_func_names: set[str]) -> set[str]:
    """Find all user-defined functions called by this function."""
    called: set[str] = set()
    _visit_stmts(func.body, called, all_func_names)
    return called


def _visit_stmts(stmts: list[IRStmt], called: set[str], func_names: set[str]) -> None:
    """Visit statements to find function calls."""
    for stmt in stmts:
        _visit_stmt(stmt, called, func_names)


def _visit_stmt(stmt: IRStmt, called: set[str], func_names: set[str]) -> None:
    """Visit a statement to find function calls."""
    match stmt:
        case IRDeclare(init=init):
            if init:
                _visit_expr(init, called, func_names)
        case IRAssign(value=value):
            _visit_expr(value, called, func_names)
        case IRAugmentedAssign(value=value):
            _visit_expr(value, called, func_names)
        case IRReturn(value=value):
            if value:
                _visit_expr(value, called, func_names)
        case IRIf(condition=cond, then_body=then_b, else_body=else_b):
            _visit_expr(cond, called, func_names)
            _visit_stmts(then_b, called, func_names)
            _visit_stmts(else_b, called, func_names)
        case IRFor(init=init, condition=cond, update=update, body=body):
            if init:
                _visit_stmt(init, called, func_names)
            if cond:
                _visit_expr(cond, called, func_names)
            if update:
                _visit_stmt(update, called, func_names)
            _visit_stmts(body, called, func_names)
        case IRWhile(condition=cond, body=body):
            _visit_expr(cond, called, func_names)
            _visit_stmts(body, called, func_names)
        case IRExprStmt(expr=expr):
            _visit_expr(expr, called, func_names)


def _visit_expr(expr: IRExpr, called: set[str], func_names: set[str]) -> None:
    """Visit an expression to find function calls."""
    match expr:
        case IRCall(func=func_name, args=args):
            if func_name in func_names:
                called.add(func_name)
            for arg in args:
                _visit_expr(arg, called, func_names)
        case IRBinOp(left=left, right=right):
            _visit_expr(left, called, func_names)
            _visit_expr(right, called, func_names)
        case IRUnaryOp(operand=operand):
            _visit_expr(operand, called, func_names)
        case IRConstruct(args=args):
            for arg in args:
                _visit_expr(arg, called, func_names)
        case IRSwizzle(base=base):
            _visit_expr(base, called, func_names)
        case IRFieldAccess(base=base):
            _visit_expr(base, called, func_names)
        case IRSubscript(base=base, index=index):
            _visit_expr(base, called, func_names)
            _visit_expr(index, called, func_names)
        case IRTernary(condition=cond, true_expr=t, false_expr=f):
            _visit_expr(cond, called, func_names)
            _visit_expr(t, called, func_names)
            _visit_expr(f, called, func_names)


def _get_precedence(expr: IRExpr) -> int:
    """Get the precedence of an expression for parenthesization."""
    match expr:
        case IRBinOp(_, op, _, _):
            return OPERATOR_PRECEDENCE.get(op, 0)
        case IRUnaryOp():
            return OPERATOR_PRECEDENCE.get("unary", 8)
        case IRTernary():
            return OPERATOR_PRECEDENCE.get("?", 14)
        case IRCall() | IRConstruct():
            return OPERATOR_PRECEDENCE.get("call", 9)
        case IRFieldAccess() | IRSwizzle() | IRSubscript():
            return OPERATOR_PRECEDENCE.get("member", 10)
        case _:
            # Literals, names - highest precedence (no parens needed)
            return 100


class Emitter:
    """Generates target code from IR using a Target."""

    def __init__(self, target: Target):
        self.target = target

    def emit(self, shader: ShaderIR) -> str:
        """Generate complete shader code."""
        lines: list[str] = []

        # Version/header
        version = self.target.version_directive()
        if version:
            lines.append(version)
            # Precision qualifiers (for ES targets)
            precision = self.target.precision_qualifiers()
            if precision:
                lines.extend(precision)
            lines.append("")

        # Predefined uniforms (for targets like Shadertoy)
        predefined = self.target.get_predefined_uniforms()
        for name, type_name in predefined.items():
            lines.append(f"uniform {type_name} {name};")
        if predefined:
            lines.append("")

        # Struct definitions
        for struct in shader.structs:
            lines.extend(self._emit_struct(struct))
            lines.append("")

        # Variable declarations (uniforms, inputs, outputs)
        # Skip for export-only targets like Shadertoy
        if not self.target.is_export_only():
            for var in shader.variables:
                lines.append(self._emit_variable_decl(var))
            if shader.variables:
                lines.append("")

        # Helper functions (non-entry-point), sorted by dependencies
        helper_funcs = [f for f in shader.functions if not f.is_entry_point]
        sorted_helpers = _topological_sort(helper_funcs)
        for func in sorted_helpers:
            lines.extend(self._emit_function(func))
            lines.append("")

        # Entry point function
        entry_func = next(
            (f for f in shader.functions if f.name == shader.entry_point), None
        )
        if entry_func:
            lines.extend(self._emit_function(entry_func))
            lines.append("")

            # Entry point wrapper (main())
            inputs = [v for v in shader.variables if v.storage == StorageClass.INPUT]
            outputs = [v for v in shader.variables if v.storage == StorageClass.OUTPUT]
            lines.extend(
                self.target.entry_point_wrapper(
                    shader.stage, entry_func, inputs, outputs
                )
            )

        return "\n".join(lines)

    def _emit_struct(self, struct: IRStruct) -> list[str]:
        lines = [f"struct {struct.name} {{"]
        for name, type_ in struct.fields:
            type_str = self.target.type_name(type_)
            lines.append(f"    {type_str} {name};")
        lines.append("};")
        return lines

    def _emit_variable_decl(self, var: IRVariable) -> str:
        qualifier = self.target.storage_qualifier(var.storage)
        type_str = self.target.type_name(var.type)
        if var.init_value is not None:
            if qualifier:
                return f"{qualifier} {type_str} {var.name} = {var.init_value};"
            return f"{type_str} {var.name} = {var.init_value};"
        if qualifier:
            return f"{qualifier} {type_str} {var.name};"
        return f"{type_str} {var.name};"

    def _emit_function(self, func: IRFunction) -> list[str]:
        return_type = (
            self.target.type_name(func.return_type) if func.return_type else "void"
        )
        params = self._emit_params(func.params)
        lines = [f"{return_type} {func.name}({params}) {{"]
        for stmt in func.body:
            lines.extend(self._emit_stmt(stmt, indent=1))
        lines.append("}")
        return lines

    def _emit_params(self, params: list[IRParameter]) -> str:
        parts = []
        for p in params:
            type_str = self.target.type_name(p.type)
            if p.qualifier:
                parts.append(f"{p.qualifier} {type_str} {p.name}")
            else:
                parts.append(f"{type_str} {p.name}")
        return ", ".join(parts)

    def _emit_stmt(self, stmt: IRStmt, indent: int = 0) -> list[str]:
        prefix = "    " * indent

        match stmt:
            case IRDeclare(var, init):
                base_type = self.target.type_name(var.type)
                # Handle array types: float arr[3] instead of float[3] arr
                if var.type.array_size is not None:
                    var_decl = f"{base_type} {var.name}[{var.type.array_size}]"
                else:
                    var_decl = f"{base_type} {var.name}"
                if init:
                    expr_str = self._emit_expr(init)
                    return [f"{prefix}{var_decl} = {expr_str};"]
                return [f"{prefix}{var_decl};"]

            case IRAssign(target, value):
                target_str = self._emit_expr(target)
                value_str = self._emit_expr(value)
                return [f"{prefix}{target_str} = {value_str};"]

            case IRAugmentedAssign(target, op, value):
                target_str = self._emit_expr(target)
                value_str = self._emit_expr(value)
                return [f"{prefix}{target_str} {op}= {value_str};"]

            case IRReturn(value):
                if value:
                    return [f"{prefix}return {self._emit_expr(value)};"]
                return [f"{prefix}return;"]

            case IRIf(condition, then_body, else_body):
                cond_str = self._emit_expr(condition)
                lines = [f"{prefix}if ({cond_str}) {{"]
                for s in then_body:
                    lines.extend(self._emit_stmt(s, indent + 1))
                if else_body:
                    lines.append(f"{prefix}}} else {{")
                    for s in else_body:
                        lines.extend(self._emit_stmt(s, indent + 1))
                lines.append(f"{prefix}}}")
                return lines

            case IRFor(init, condition, update, body):
                init_str = self._emit_for_init(init) if init else ""
                cond_str = self._emit_expr(condition) if condition else ""
                update_str = self._emit_for_update(update) if update else ""
                lines = [f"{prefix}for ({init_str}; {cond_str}; {update_str}) {{"]
                for s in body:
                    lines.extend(self._emit_stmt(s, indent + 1))
                lines.append(f"{prefix}}}")
                return lines

            case IRWhile(condition, body):
                cond_str = self._emit_expr(condition)
                lines = [f"{prefix}while ({cond_str}) {{"]
                for s in body:
                    lines.extend(self._emit_stmt(s, indent + 1))
                lines.append(f"{prefix}}}")
                return lines

            case IRExprStmt(expr):
                return [f"{prefix}{self._emit_expr(expr)};"]

            case IRBreak():
                return [f"{prefix}break;"]

            case IRContinue():
                return [f"{prefix}continue;"]

        return []

    def _emit_for_init(self, stmt: IRStmt) -> str:
        match stmt:
            case IRDeclare(var, init):
                type_str = self.target.type_name(var.type)
                if init:
                    return f"{type_str} {var.name} = {self._emit_expr(init)}"
                return f"{type_str} {var.name}"
            case IRAssign(target, value):
                return f"{self._emit_expr(target)} = {self._emit_expr(value)}"
        return ""

    def _emit_for_update(self, stmt: IRStmt) -> str:
        match stmt:
            case IRAssign(target, value):
                return f"{self._emit_expr(target)} = {self._emit_expr(value)}"
            case IRAugmentedAssign(target, op, value):
                return f"{self._emit_expr(target)} {op}= {self._emit_expr(value)}"
            case IRExprStmt(expr):
                return self._emit_expr(expr)
        return ""

    def _emit_expr(self, expr: IRExpr, parent_precedence: int = 0) -> str:
        """Emit an expression, adding parentheses only when necessary.

        Args:
            expr: The IR expression to emit
            parent_precedence: Precedence of parent operator (0 = top-level/statement)
        """
        result = self._emit_expr_inner(expr)
        # Add parens if this expression has lower precedence than parent
        expr_prec = _get_precedence(expr)
        if parent_precedence > 0 and expr_prec < parent_precedence:
            return f"({result})"
        return result

    def _emit_expr_inner(self, expr: IRExpr) -> str:
        """Emit expression without outer parentheses."""
        match expr:
            case IRLiteral(result_type, value):
                return self.target.literal(value, result_type)

            case IRName(_, name):
                # Apply builtin mapping (e.g., u_time -> iTime for Shadertoy)
                return self.target.map_builtin(name)

            case IRBinOp(_, op, left, right):
                op_str = self.target.operator(op)
                my_prec = OPERATOR_PRECEDENCE.get(op, 0)
                # Handle power operator specially
                if op == "**":
                    left_str = self._emit_expr(left, 0)
                    right_str = self._emit_expr(right, 0)
                    return f"pow({left_str}, {right_str})"
                # Emit children with current precedence to determine if they need parens
                left_str = self._emit_expr(left, my_prec)
                # Right side: use slightly higher precedence for left-associative ops
                # to force parens on right child with same precedence
                right_str = self._emit_expr(right, my_prec + 1)
                return f"{left_str} {op_str} {right_str}"

            case IRUnaryOp(_, op, operand):
                op_str = self.target.operator(op)
                unary_prec = OPERATOR_PRECEDENCE.get("unary", 8)
                operand_str = self._emit_expr(operand, unary_prec)
                return f"{op_str}{operand_str}"

            case IRCall(_, func, args):
                func_name = self.target.builtin_function(func) or func
                args_str = ", ".join(self._emit_expr(a, 0) for a in args)
                return f"{func_name}({args_str})"

            case IRConstruct(result_type, args):
                base_type = self.target.type_name(result_type)
                # Handle array constructors: float[3](...) instead of float(...)
                if result_type.array_size is not None:
                    type_str = f"{base_type}[{result_type.array_size}]"
                else:
                    type_str = base_type
                args_str = ", ".join(self._emit_expr(a, 0) for a in args)
                return f"{type_str}({args_str})"

            case IRSwizzle(_, base, components):
                base_str = self._emit_expr(base, OPERATOR_PRECEDENCE.get("member", 10))
                return f"{base_str}.{components}"

            case IRFieldAccess(_, base, field):
                base_str = self._emit_expr(base, OPERATOR_PRECEDENCE.get("member", 10))
                return f"{base_str}.{field}"

            case IRSubscript(_, base, index):
                base_str = self._emit_expr(base, OPERATOR_PRECEDENCE.get("member", 10))
                index_str = self._emit_expr(index, 0)
                return f"{base_str}[{index_str}]"

            case IRTernary(_, cond, true_e, false_e):
                cond_str = self._emit_expr(cond, 0)
                true_str = self._emit_expr(true_e, 0)
                false_str = self._emit_expr(false_e, 0)
                return f"{cond_str} ? {true_str} : {false_str}"

        return ""
