"""GLSL AST Node Definitions"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from py2glsl.types.base import GLSLType


@dataclass
class Node:
    """Base AST node"""

    lineno: int
    col_offset: int


@dataclass
class Expr(Node):
    """Base expression node"""

    pass


@dataclass
class Stmt(Node):
    """Base statement node"""

    pass


@dataclass
class FieldAccess(Expr):
    value: Expr
    field: str


# ====================
# Type Annotations
# ====================


@dataclass
class TypeAnnotation(Expr):
    """Type annotation"""

    type_name: str


@dataclass
class VectorType(TypeAnnotation):
    """Vector type annotation"""

    size: int


@dataclass
class MatrixType(TypeAnnotation):
    """Matrix type annotation"""

    rows: int
    cols: int


# ====================
# Module and Blocks
# ====================


@dataclass
class Module(Stmt):
    """Module node"""

    body: List[Stmt]


@dataclass
class Block(Stmt):
    """Block of statements"""

    body: List[Stmt]


# ====================
# Expressions
# ====================


@dataclass
class Literal(Expr):
    """Literal value"""

    value: Union[int, float, bool, str]
    type: Optional["GLSLType"] = None


@dataclass
class Name(Expr):
    """Variable reference"""

    id: str
    type: Optional["GLSLType"] = None


@dataclass
class BinaryOp(Expr):
    """Binary operation"""

    left: Expr
    op: str
    right: Expr
    type: Optional["GLSLType"] = None


@dataclass
class UnaryOp(Expr):
    """Unary operation"""

    op: str
    operand: Expr
    type: Optional["GLSLType"] = None


@dataclass
class Call(Expr):
    """Function call"""

    func: Expr
    args: List[Expr]
    type: Optional["GLSLType"] = None


@dataclass
class Attribute(Expr):
    """Attribute access"""

    value: Expr
    attr: str
    type: Optional["GLSLType"] = None


@dataclass
class Subscript(Expr):
    """Array subscript"""

    value: Expr
    slice: Expr
    type: Optional["GLSLType"] = None


# ====================
# Statements
# ====================


@dataclass
class FunctionDef(Stmt):
    """Function definition"""

    name: str
    params: List[Tuple[str, TypeAnnotation]]
    return_type: TypeAnnotation
    body: List[Stmt]


@dataclass
class Return(Stmt):
    """Return statement"""

    value: Optional[Expr]


@dataclass
class Assign(Stmt):
    """Assignment statement"""

    targets: List[Expr]
    value: Expr


@dataclass
class AnnAssign(Stmt):
    """Annotated assignment"""

    target: Expr
    annotation: Expr
    value: Optional[Expr]


@dataclass
class If(Stmt):
    """If statement"""

    test: Expr
    body: List[Stmt]
    orelse: List[Stmt]


@dataclass
class For(Stmt):
    """For loop"""

    target: Expr
    iter: Expr
    body: List[Stmt]
    orelse: List[Stmt]


@dataclass
class While(Stmt):
    """While loop"""

    test: Expr
    body: List[Stmt]
    orelse: List[Stmt]


@dataclass
class Break(Stmt):
    """Break statement"""

    pass


@dataclass
class Continue(Stmt):
    """Continue statement"""

    pass


# ====================
# Structure Nodes
# ====================


@dataclass
class StructDef(Stmt):
    name: str
    fields: List[Tuple[str, TypeAnnotation]]


@dataclass
class StructType(TypeAnnotation):
    name: str
    fields: Dict[str, TypeAnnotation]


# ====================
# Helper Nodes
# ====================


@dataclass
class Arg(Node):
    """Function argument"""

    arg: str
    annotation: Optional[TypeAnnotation]
    default: Optional[Expr]


@dataclass
class UniformDecl(Stmt):
    """Uniform declaration"""

    name: str
    type: TypeAnnotation


@dataclass
class AttributeDecl(Stmt):
    """Attribute declaration"""

    name: str
    type: TypeAnnotation


__all__ = [
    "Node",
    "Expr",
    "Stmt",
    "Module",
    "Block",
    "Literal",
    "Name",
    "BinaryOp",
    "UnaryOp",
    "Call",
    "Attribute",
    "Subscript",
    "FunctionDef",
    "Return",
    "Assign",
    "AnnAssign",
    "If",
    "For",
    "While",
    "Break",
    "Continue",
    "TypeAnnotation",
    "VectorType",
    "MatrixType",
    "Arg",
    "UniformDecl",
    "AttributeDecl",
]
