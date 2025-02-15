"""Code block and scope management."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Set


@dataclass
class ScopeManager:
    """Manages variable scopes."""

    current_scope: str = "global"
    scope_stack: list[str] = field(default_factory=list)
    declared_vars: dict[str, set[str]] = field(
        default_factory=lambda: {"global": set()}
    )

    @contextmanager
    def scope(self, name: str) -> Iterator[None]:
        """Context manager for scope handling."""
        self.enter(name)
        try:
            yield
        finally:
            self.exit()

    def enter(self, name: str) -> None:
        """Enter a new scope."""
        self.scope_stack.append(self.current_scope)
        self.current_scope = name
        if name not in self.declared_vars:
            self.declared_vars[name] = set()

    def exit(self) -> None:
        """Exit current scope."""
        if self.scope_stack:
            self.current_scope = self.scope_stack.pop()
        else:
            self.current_scope = "global"

    def declare(self, name: str) -> None:
        """Declare a variable in current scope."""
        self.declared_vars[self.current_scope].add(name)

    def is_declared(self, name: str) -> bool:
        """Check if variable is declared in current scope."""
        return name in self.declared_vars[self.current_scope]


@dataclass
class CodeBlock:
    """Manages code block generation."""

    indent_level: int = 0
    lines: list[str] = field(default_factory=list)

    @contextmanager
    def block(self) -> Iterator[None]:
        """Context manager for code blocks."""
        self.add_line("{")
        self.indent_level += 1
        try:
            yield
        finally:
            self.indent_level -= 1
            self.add_line("}")

    def add_line(self, line: str = "") -> None:
        """Add line with proper indentation."""
        if not line:
            if not self.lines or self.lines[-1]:
                self.lines.append("")
            return

        if line.startswith(("#", "layout", "in ", "out ", "uniform ")):
            self.lines.append(line)
            return

        self.lines.append(f"{'    ' * self.indent_level}{line}")

    def get_code(self) -> str:
        """Get generated code."""
        return "\n".join(self.lines)
