"""Intermediate Representation for shader transpilation."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class ShaderStage(Enum):
    """Shader pipeline stage."""

    VERTEX = auto()
    FRAGMENT = auto()
    COMPUTE = auto()


class StorageClass(Enum):
    """How a variable is stored/accessed."""

    LOCAL = auto()
    CONST = auto()
    UNIFORM = auto()
    INPUT = auto()
    OUTPUT = auto()
    BUFFER = auto()
    TEXTURE = auto()
    IMAGE = auto()
    SHARED = auto()


# Types


@dataclass
class IRType:
    """Type in the IR."""

    base: str
    array_size: int | None = None

    def __str__(self) -> str:
        if self.array_size:
            return f"{self.base}[{self.array_size}]"
        return self.base


# Variables and Parameters


@dataclass
class IRVariable:
    """A variable declaration."""

    name: str
    type: IRType
    storage: StorageClass = StorageClass.LOCAL
    binding: int | None = None
    location: int | None = None
    init_value: str | None = None


@dataclass
class IRParameter:
    """Function parameter."""

    name: str
    type: IRType
    qualifier: str = ""


# Expressions


@dataclass
class IRExpr:
    """Base for all expressions."""

    result_type: IRType


@dataclass
class IRLiteral(IRExpr):
    """Literal value."""

    value: Any


@dataclass
class IRName(IRExpr):
    """Variable reference."""

    name: str


@dataclass
class IRBinOp(IRExpr):
    """Binary operation."""

    op: str
    left: IRExpr
    right: IRExpr


@dataclass
class IRUnaryOp(IRExpr):
    """Unary operation."""

    op: str
    operand: IRExpr


@dataclass
class IRCall(IRExpr):
    """Function call."""

    func: str
    args: list[IRExpr]


@dataclass
class IRSwizzle(IRExpr):
    """Vector swizzle (e.g., .xyz)."""

    base: IRExpr
    components: str


@dataclass
class IRFieldAccess(IRExpr):
    """Struct field access."""

    base: IRExpr
    field: str


@dataclass
class IRSubscript(IRExpr):
    """Array subscript."""

    base: IRExpr
    index: IRExpr


@dataclass
class IRTernary(IRExpr):
    """Ternary conditional."""

    condition: IRExpr
    true_expr: IRExpr
    false_expr: IRExpr


@dataclass
class IRConstruct(IRExpr):
    """Type constructor (e.g., vec3(1.0, 2.0, 3.0))."""

    args: list[IRExpr]


# Statements


@dataclass
class IRStmt:
    """Base for all statements."""

    pass


@dataclass
class IRDeclare(IRStmt):
    """Variable declaration."""

    var: IRVariable
    init: IRExpr | None = None


@dataclass
class IRAssign(IRStmt):
    """Assignment."""

    target: IRExpr
    value: IRExpr


@dataclass
class IRAugmentedAssign(IRStmt):
    """Augmented assignment (+=, -=, etc.)."""

    target: IRExpr
    op: str
    value: IRExpr


@dataclass
class IRReturn(IRStmt):
    """Return statement."""

    value: IRExpr | None = None


@dataclass
class IRIf(IRStmt):
    """If statement."""

    condition: IRExpr
    then_body: list[IRStmt] = field(default_factory=list)
    else_body: list[IRStmt] = field(default_factory=list)


@dataclass
class IRFor(IRStmt):
    """For loop."""

    init: IRStmt | None
    condition: IRExpr | None
    update: IRStmt | None
    body: list[IRStmt] = field(default_factory=list)


@dataclass
class IRWhile(IRStmt):
    """While loop."""

    condition: IRExpr
    body: list[IRStmt] = field(default_factory=list)


@dataclass
class IRExprStmt(IRStmt):
    """Expression as statement."""

    expr: IRExpr


@dataclass
class IRBreak(IRStmt):
    """Break statement."""

    pass


@dataclass
class IRContinue(IRStmt):
    """Continue statement."""

    pass


# Functions and Structs


@dataclass
class IRFunction:
    """Function definition."""

    name: str
    params: list[IRParameter]
    return_type: IRType | None
    body: list[IRStmt] = field(default_factory=list)
    is_entry_point: bool = False


@dataclass
class IRStruct:
    """Struct definition."""

    name: str
    fields: list[tuple[str, IRType]]


# Complete Shader


@dataclass
class ShaderIR:
    """Complete shader intermediate representation."""

    stage: ShaderStage
    structs: list[IRStruct] = field(default_factory=list)
    variables: list[IRVariable] = field(default_factory=list)
    functions: list[IRFunction] = field(default_factory=list)
    entry_point: str = ""
    workgroup_size: tuple[int, int, int] | None = None
