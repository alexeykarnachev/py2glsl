"""GLSL operator mappings."""

import ast
from typing import Dict, Type

# Binary operators
BINARY_OPERATORS: Dict[Type[ast.operator], str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.Mod: "%",
}

# Comparison operators
COMPARISON_OPERATORS: Dict[Type[ast.cmpop], str] = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
}

# Unary operators
UNARY_OPERATORS: Dict[Type[ast.unaryop], str] = {
    ast.UAdd: "+",
    ast.USub: "-",
    ast.Not: "!",
}

# Augmented assignment operators
AUGASSIGN_OPERATORS: Dict[Type[ast.operator], str] = {
    ast.Add: "+=",
    ast.Sub: "-=",
    ast.Mult: "*=",
    ast.Div: "/=",
    ast.Mod: "%=",
}
