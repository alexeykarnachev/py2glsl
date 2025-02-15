"""GLSL code formatting and generation utilities."""

import ast
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterator, Optional, Sequence

from py2glsl.transpiler.operators import (
    AUGASSIGN_OPERATORS,
    BINARY_OPERATORS,
    COMPARISON_OPERATORS,
    UNARY_OPERATORS,
)
from py2glsl.types import GLSLType


class ExprContext(Enum):
    """Expression generation context."""

    DEFAULT = auto()
    DECLARATION = auto()
    STATEMENT = auto()
    CONDITION = auto()
    LOOP = auto()


@dataclass
class CodeBlock:
    """Code block with proper indentation."""

    indent: int = 0
    lines: list[str] = field(default_factory=list)

    def add_line(self, line: str = "") -> None:
        """Add line with proper indentation."""
        if not line:
            if not self.lines or self.lines[-1]:
                self.lines.append("")
            return

        if line.startswith(("#", "layout", "in ", "out ", "uniform ")):
            self.lines.append(line)
            return

        self.lines.append("    " * self.indent + line)

    @contextmanager
    def indented(self) -> Iterator[None]:
        """Context manager for indented block."""
        self.indent += 1
        try:
            yield
        finally:
            self.indent -= 1

    def begin_block(self) -> None:
        """Begin a new block."""
        self.add_line("{")
        self.indent += 1

    def end_block(self) -> None:
        """End current block."""
        self.indent -= 1
        self.add_line("}")

    def get_code(self) -> str:
        """Get generated code."""
        return "\n".join(self.lines)


@dataclass
class ExprFormatter:
    """Expression formatter with operator precedence."""

    context: ExprContext = ExprContext.DEFAULT
    needs_parens: bool = False
    var_types: dict[str, GLSLType] = field(default_factory=dict)

    @contextmanager
    def parens(self) -> Iterator[None]:
        """Context manager for parentheses."""
        prev = self.needs_parens
        self.needs_parens = True
        try:
            yield
        finally:
            self.needs_parens = prev

    @contextmanager
    def in_context(self, ctx: ExprContext) -> Iterator[None]:
        """Context manager for expression context."""
        prev = self.context
        self.context = ctx
        try:
            yield
        finally:
            self.context = prev

    def format_expr(self, node: ast.AST, type_: Optional[GLSLType] = None) -> str:
        """Format expression with proper context."""
        with self.in_context(ExprContext.STATEMENT):
            expr = self._format_expr(node)
            if type_ and self.context == ExprContext.DECLARATION:
                return f"{type_} {expr}"
            return expr

    def _format_expr(self, node: ast.AST) -> str:
        """Internal expression formatting."""
        match node:
            case ast.Name():
                return node.id

            case ast.Constant():
                return self._format_constant(node.value)

            case ast.BinOp():
                return self._format_binary_op(node)

            case ast.Call():
                return self._format_call(node)

            case ast.Attribute():
                return self._format_attribute(node)

            case ast.Compare():
                return self._format_comparison(node)

            case ast.BoolOp():
                return self._format_bool_op(node)

            case ast.UnaryOp():
                return self._format_unary_op(node)

            case _:
                raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    def _format_constant(self, value: object) -> str:
        """Format constant value."""
        if isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, int):
            return str(value)
        elif isinstance(value, float):
            s = f"{value:g}"
            return f"{s}.0" if "." not in s and "e" not in s else s
        raise ValueError(f"Unsupported constant type: {type(value)}")

    def _format_binary_op(self, node: ast.BinOp) -> str:
        """Format binary operation with proper precedence."""
        if type(node.op) not in BINARY_OPERATORS:
            raise ValueError(f"Unsupported binary operator: {type(node.op)}")

        op = BINARY_OPERATORS[type(node.op)]

        # Handle operator precedence
        with self.parens():
            left = self._format_expr(node.left)
            right = self._format_expr(node.right)

        if self.needs_parens:
            return f"({left} {op} {right})"
        return f"{left} {op} {right}"

    def _format_call(self, node: ast.Call) -> str:
        """Format function call."""
        # Handle math module functions
        if isinstance(node.func, ast.Attribute) and isinstance(
            node.func.value, ast.Name
        ):
            if node.func.value.id == "math":
                func_name = node.func.attr
            else:
                func_name = f"{self._format_expr(node.func.value)}.{node.func.attr}"
        else:
            func_name = self._format_expr(node.func)

        args = [self._format_expr(arg) for arg in node.args]
        return f"{func_name}({', '.join(args)})"

    def _format_attribute(self, node: ast.Attribute) -> str:
        """Format attribute access (swizzling)."""
        value = self._format_expr(node.value)
        return f"{value}.{node.attr}"

    def _format_comparison(self, node: ast.Compare) -> str:
        """Format comparison operation."""
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ValueError("Only simple comparisons are supported")

        if type(node.ops[0]) not in COMPARISON_OPERATORS:
            raise ValueError(f"Unsupported comparison operator: {type(node.ops[0])}")

        op = COMPARISON_OPERATORS[type(node.ops[0])]
        left = self._format_expr(node.left)
        right = self._format_expr(node.comparators[0])

        if self.needs_parens:
            return f"({left} {op} {right})"
        return f"{left} {op} {right}"

    def _format_bool_op(self, node: ast.BoolOp) -> str:
        """Format boolean operation."""
        if not isinstance(node.op, ast.And):
            raise ValueError("Only 'and' operations are supported")

        with self.parens():
            values = [self._format_expr(val) for val in node.values]

        if self.needs_parens:
            return f"({' && '.join(values)})"
        return f"{' && '.join(values)}"

    def _format_unary_op(self, node: ast.UnaryOp) -> str:
        """Format unary operation."""
        if type(node.op) not in UNARY_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {type(node.op)}")

        op = UNARY_OPERATORS[type(node.op)]
        operand = self._format_expr(node.operand)

        if isinstance(node.op, ast.Not):
            return f"!{operand}"

        if self.needs_parens:
            return f"{op}({operand})"
        return f"{op}{operand}"


@dataclass
class StatementFormatter:
    """Statement formatter with proper indentation."""

    code: CodeBlock = field(default_factory=CodeBlock)
    expr: ExprFormatter = field(default_factory=ExprFormatter)

    def format_statement(self, node: ast.AST) -> None:
        """Format statement with proper indentation."""
        match node:
            case ast.Assign():
                self._format_assignment(node)

            case ast.AnnAssign():
                self._format_annotated_assignment(node)

            case ast.AugAssign():
                self._format_aug_assignment(node)

            case ast.If():
                self._format_if_statement(node)

            case ast.For():
                self._format_for_loop(node)

            case ast.While():
                self._format_while_loop(node)

            case ast.Return():
                self._format_return(node)

            case ast.Break():
                self.code.add_line("break;")

            case ast.Continue():
                self.code.add_line("continue;")

            case ast.Expr():
                expr = self.expr.format_expr(node.value)
                self.code.add_line(f"{expr};")

            case ast.Pass():
                pass

            case _:
                raise ValueError(f"Unsupported statement: {type(node)}")

    def _format_assignment(self, node: ast.Assign) -> None:
        """Format assignment statement."""
        value = self.expr.format_expr(node.value)
        for target in node.targets:
            if not isinstance(target, ast.Name):
                raise ValueError("Only simple assignments are supported")
            self.code.add_line(f"{target.id} = {value};")

    def _format_annotated_assignment(self, node: ast.AnnAssign) -> None:
        """Format annotated assignment statement."""
        if not isinstance(node.target, ast.Name):
            raise ValueError("Only simple assignments are supported")

        with self.expr.in_context(ExprContext.DECLARATION):
            if node.value:
                value = self.expr.format_expr(node.value)
                self.code.add_line(f"{node.target.id} = {value};")
            else:
                self.code.add_line(f"{node.target.id};")

    def _format_aug_assignment(self, node: ast.AugAssign) -> None:
        """Format augmented assignment statement."""
        if not isinstance(node.target, (ast.Name, ast.Attribute)):
            raise ValueError("Only simple augmented assignments are supported")

        if type(node.op) not in AUGASSIGN_OPERATORS:
            raise ValueError(
                f"Unsupported augmented assignment operator: {type(node.op)}"
            )

        op = AUGASSIGN_OPERATORS[type(node.op)]
        target = self.expr.format_expr(node.target)
        value = self.expr.format_expr(node.value)

        self.code.add_line(f"{target} {op} {value};")

    def _format_if_statement(self, node: ast.If) -> None:
        """Format if statement."""
        with self.expr.in_context(ExprContext.CONDITION):
            condition = self.expr.format_expr(node.test)

        self.code.add_line(f"if ({condition})")
        self.code.begin_block()

        for stmt in node.body:
            self.format_statement(stmt)

        self.code.end_block()

        if node.orelse:
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                # elif case
                self.code.add_line("else")
                self._format_if_statement(node.orelse[0])
            else:
                # else case
                self.code.add_line("else")
                self.code.begin_block()

                for stmt in node.orelse:
                    self.format_statement(stmt)

                self.code.end_block()

    def _format_for_loop(self, node: ast.For) -> None:
        """Format for loop."""
        if not isinstance(node.target, ast.Name):
            raise ValueError("Only simple loop variables are supported")

        if not (
            isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name)
        ):
            raise ValueError("Only range-based for loops are supported")

        if node.iter.func.id != "range":
            raise ValueError("Only range-based for loops are supported")

        # Handle range arguments
        if len(node.iter.args) == 1:
            end = self.expr.format_expr(node.iter.args[0])
            init = "0"
            cond = f"< {end}"
        elif len(node.iter.args) == 2:
            start = self.expr.format_expr(node.iter.args[0])
            end = self.expr.format_expr(node.iter.args[1])
            init = start
            cond = f"< {end}"
        else:
            raise ValueError("Range with step not supported")

        self.code.add_line(
            f"for (int {node.target.id} = {init}; {node.target.id} {cond}; {node.target.id}++)"
        )
        self.code.begin_block()

        for stmt in node.body:
            self.format_statement(stmt)

        self.code.end_block()

    def _format_while_loop(self, node: ast.While) -> None:
        """Format while loop."""
        with self.expr.in_context(ExprContext.CONDITION):
            condition = self.expr.format_expr(node.test)

        self.code.add_line(f"while ({condition})")
        self.code.begin_block()

        for stmt in node.body:
            self.format_statement(stmt)

        self.code.end_block()

    def _format_return(self, node: ast.Return) -> None:
        """Format return statement."""
        if node.value is None:
            self.code.add_line("return;")
        else:
            value = self.expr.format_expr(node.value)
            self.code.add_line(f"return {value};")
