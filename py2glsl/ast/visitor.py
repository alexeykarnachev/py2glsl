"""GLSL AST Visitor Pattern"""

from typing import Any, Generic, List, TypeVar

from .nodes import *

T = TypeVar("T")


class Visitor(Generic[T]):
    """Base visitor class for AST traversal"""

    def visit(self, node: Node) -> T:
        """Dispatch to appropriate visit method based on node type"""
        method_name = "visit_" + node.__class__.__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: Node) -> T:
        """Default visitor implementation"""
        raise NotImplementedError(f"No visit_{node.__class__.__name__} method defined")

    # ====================
    # Module and Blocks
    # ====================

    def visit_Module(self, node: Module) -> T:
        return self.visit_block(node.body)

    def visit_Block(self, node: Block) -> T:
        return self.visit_block(node.body)

    def visit_block(self, body: List[Stmt]) -> T:
        results = []
        for stmt in body:
            results.append(self.visit(stmt))
        return results

    # ====================
    # Expressions
    # ====================

    def visit_Literal(self, node: Literal) -> T:
        return node.value

    def visit_Name(self, node: Name) -> T:
        return node.id

    def visit_BinaryOp(self, node: BinaryOp) -> T:
        left = self.visit(node.left)
        right = self.visit(node.right)
        return (left, node.op, right)

    def visit_UnaryOp(self, node: UnaryOp) -> T:
        operand = self.visit(node.operand)
        return (node.op, operand)

    def visit_Call(self, node: Call) -> T:
        func = self.visit(node.func)
        args = [self.visit(arg) for arg in node.args]
        return (func, args)

    def visit_Attribute(self, node: Attribute) -> T:
        value = self.visit(node.value)
        return (value, node.attr)

    def visit_Subscript(self, node: Subscript) -> T:
        value = self.visit(node.value)
        slice = self.visit(node.slice)
        return (value, slice)

    # ====================
    # Statements
    # ====================

    def visit_FunctionDef(self, node: FunctionDef) -> T:
        args = [self.visit(arg) for arg in node.args]
        body = self.visit_block(node.body)
        return (node.name, args, body)

    def visit_Return(self, node: Return) -> T:
        if node.value:
            return self.visit(node.value)
        return None

    def visit_Assign(self, node: Assign) -> T:
        targets = [self.visit(target) for target in node.targets]
        value = self.visit(node.value)
        return (targets, value)

    def visit_AnnAssign(self, node: AnnAssign) -> T:
        target = self.visit(node.target)
        annotation = self.visit(node.annotation)
        value = self.visit(node.value) if node.value else None
        return (target, annotation, value)

    def visit_If(self, node: If) -> T:
        test = self.visit(node.test)
        body = self.visit_block(node.body)
        orelse = self.visit_block(node.orelse)
        return (test, body, orelse)

    def visit_For(self, node: For) -> T:
        target = self.visit(node.target)
        iter = self.visit(node.iter)
        body = self.visit_block(node.body)
        orelse = self.visit_block(node.orelse)
        return (target, iter, body, orelse)

    def visit_While(self, node: While) -> T:
        test = self.visit(node.test)
        body = self.visit_block(node.body)
        orelse = self.visit_block(node.orelse)
        return (test, body, orelse)

    def visit_Break(self, node: Break) -> T:
        return None

    def visit_Continue(self, node: Continue) -> T:
        return None

    # ====================
    # Type Annotations
    # ====================

    def visit_TypeAnnotation(self, node: TypeAnnotation) -> T:
        return node.type_name

    def visit_VectorType(self, node: VectorType) -> T:
        return f"vec{node.size}"

    def visit_MatrixType(self, node: MatrixType) -> T:
        return f"mat{node.rows}x{node.cols}"

    # ====================
    # Helper Nodes
    # ====================

    def visit_Arg(self, node: Arg) -> T:
        annotation = self.visit(node.annotation) if node.annotation else None
        default = self.visit(node.default) if node.default else None
        return (node.arg, annotation, default)

    def visit_UniformDecl(self, node: UniformDecl) -> T:
        type_annotation = self.visit(node.type)
        return (node.name, type_annotation)

    def visit_AttributeDecl(self, node: AttributeDecl) -> T:
        type_annotation = self.visit(node.type)
        return (node.name, type_annotation)


class Transformer(Visitor[Node]):
    """Base transformer class for AST modification"""

    def generic_visit(self, node: Node) -> Node:
        """Default transformer implementation"""
        for field, old_value in node.__dict__.items():
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, Node):
                        value = self.visit(value)
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, Node):
                new_node = self.visit(old_value)
                setattr(node, field, new_node)
        return node


__all__ = ["Visitor", "Transformer"]
