import ast
import inspect
import re
from dataclasses import dataclass
from inspect import Parameter, signature
from textwrap import dedent
from typing import Any, Callable, Dict, List, Set, Tuple

from loguru import logger

from py2glsl.glsl.types import mat3, mat4, vec2, vec3, vec4
from py2glsl.transpiler.glsl_builder import GLSLBuilder, GLSLCodeError
from py2glsl.transpiler.type_system import TypeInferer, TypeInfo

from .glsl_builder import GLSLBuilder, GLSLCodeError, TranspilerResult
from .utils import extract_function_body


class GLSLTypeError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


def transpile(func: Callable) -> TranspilerResult:
    validate_function_signature(func)
    source = inspect.getsource(func)
    tree = ast.parse(dedent(source))
    func_node = tree.body[0]

    # Pass the entire function node instead of just its body
    called_functions = _find_called_functions(
        func_node,
        func.__globals__,
        func.__name__,
    )

    logger.debug(f"Final called functions: {list(called_functions.keys())}")

    type_inferer = TypeInferer(called_functions=called_functions)
    uniforms, attributes = detect_interface(func, type_inferer)
    glsl_body = extract_function_body(func_node, type_inferer)

    builder = GLSLBuilder()
    builder.configure_shader_transpiler(
        uniforms=uniforms,
        attributes=attributes,
        func_name=func.__name__,
        shader_body=glsl_body,
        called_functions=called_functions,
    )
    return builder.build()


def _find_called_functions(
    node: ast.AST,
    global_scope: Dict[str, Any],
    current_func: str,
    visited: Set[str] | None = None,
    depth: int = 0,
) -> Dict[str, ast.FunctionDef]:
    """Recursively find all called functions in the AST with call graph analysis."""
    visited = visited or set()
    called_funcs = {}
    indent = "  " * depth

    logger.debug(
        f"{indent}🚀 Analyzing calls in {current_func} (node type: {type(node).__name__})"
    )

    class CallVisitor(ast.NodeVisitor):
        def __init__(self):
            self.calls: List[str] = []
            self._current_node = None

        def visit_Call(self, node: ast.Call):
            self._current_node = node
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                self.calls.append(func_name)
                logger.debug(
                    f"{indent}🔍 Found function call: {func_name} (line {node.lineno})"
                )
            elif isinstance(node.func, ast.Attribute):
                logger.debug(
                    f"{indent}⚠️  Skipping method call: {ast.unparse(node.func)}"
                )
            self.generic_visit(node)

    def process_function(func_node: ast.FunctionDef) -> Dict[str, ast.FunctionDef]:
        nonlocal depth
        func_name = func_node.name
        if func_name in visited:
            logger.debug(f"{indent}⏩ Already visited {func_name}, skipping")
            return {}
        visited.add(func_name)

        logger.debug(f"{indent}📖 Processing function: {func_name} (depth {depth})")
        visitor = CallVisitor()
        visitor.visit(func_node)

        functions_found = {}
        for call in visitor.calls:
            logger.debug(f"{indent}🔗 Resolving call to {call} in {func_name}")

            if call not in global_scope:
                logger.warning(
                    f"{indent}⚠️  Undefined function {call} called in {func_name}"
                )
                continue

            target_func = global_scope[call]
            if not inspect.isfunction(target_func):
                logger.debug(
                    f"{indent}⏩ Skipping non-function {call} ({type(target_func)})"
                )
                continue

            try:
                logger.debug(f"{indent}📦 Fetching source for {call}")
                source = inspect.getsource(target_func)
                child_tree = ast.parse(dedent(source))

                if not child_tree.body or not isinstance(
                    child_tree.body[0], ast.FunctionDef
                ):
                    logger.error(f"{indent}❌ Invalid function source for {call}")
                    continue

                child_func = child_tree.body[0]
                logger.debug(
                    f"{indent}🌳 Parsed AST for {call}:\n{ast.dump(child_func, indent=2)}"
                )

                logger.debug(f"{indent}🔄 Recursing into {call}'s dependencies")
                nested_calls = _find_called_functions(
                    child_func, global_scope, call, visited, depth + 1
                )
                functions_found.update(nested_calls)
                functions_found[call] = child_func
                logger.debug(f"{indent}✅ Finished processing {call}")

            except Exception as e:
                logger.error(f"{indent}❌ Failed to process {call}: {str(e)}")
                raise GLSLCodeError(f"Error in {call} dependency: {str(e)}") from e

        return functions_found

    # Handle function definition nodes
    if isinstance(node, ast.FunctionDef):
        return process_function(node)

    # Handle other nodes and child nodes
    logger.debug(f"{indent}🔎 Scanning children of {type(node).__name__} node")
    for child in ast.iter_child_nodes(node):
        child_funcs = {}

        if isinstance(child, ast.FunctionDef):
            child_funcs.update(process_function(child))
        else:
            child_funcs.update(
                _find_called_functions(
                    child, global_scope, current_func, visited, depth
                )
            )

        called_funcs.update(child_funcs)

    logger.debug(f"{indent}🏁 Completed analysis of {current_func}")
    return called_funcs


def detect_interface(
    func: Callable, inferer: TypeInferer
) -> Tuple[Dict[str, str], Dict[str, str]]:
    sig = signature(func)
    uniforms = {}
    attributes = {}

    for param in sig.parameters.values():
        # Parameter classification logic
        if param.default != Parameter.empty:
            uniforms[param.name] = get_glsl_type(param.annotation)
        elif param.kind == Parameter.POSITIONAL_ONLY:
            attributes[param.name] = get_glsl_type(param.annotation)
        else:
            uniforms[param.name] = get_glsl_type(param.annotation)

        # Populate type inferer
        py_type = param.annotation
        inferer.symbols[param.name] = TypeInfo.from_pytype(py_type)

    return uniforms, attributes


class FunctionCallVisitor(ast.NodeVisitor):
    def __init__(self, globals_dict: dict):
        self.globals = globals_dict
        self.valid_functions: dict[str, Callable] = {}
        self.call_graph: dict[str, list[str]] = {}
        self.top_level_calls: set[str] = set()
        self.current_func: str | None = None
        self.called_functions = {}

    def visit_FunctionDef(self, node: ast.FunctionDef):
        prev_func = self.current_func
        self.current_func = node.name
        try:
            if node.name not in self.valid_functions:
                self._process_call(node.name)
            self.generic_visit(node)
        finally:
            self.current_func = prev_func

    def visit_Call(self, node: ast.Call):
        # Process all function calls
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            self._process_call(func_name)

            # Track dependencies between functions
            if self.current_func and func_name in self.valid_functions:
                self.call_graph.setdefault(self.current_func, []).append(func_name)

        self.generic_visit(node)

    def _process_call(self, func_name: str):
        if func_name in self.valid_functions:
            return

        func = self.globals.get(func_name)
        if not func or not inspect.isfunction(func):
            return

        try:
            logger.debug(f"Registering function: {func_name}")
            self.valid_functions[func_name] = func
            func_ast = ast.parse(inspect.getsource(func))
            self.visit(func_ast)  # Process nested calls
        except Exception as e:
            logger.error(f"Error processing {func_name}: {str(e)}")
            raise GLSLCodeError(f"Failed to process {func_name}: {str(e)}")

    def _track_dependency(self, callee: str):
        if self.current_func:
            self.call_graph.setdefault(self.current_func, []).append(callee)

    def _handle_user_function(self, node: ast.Call) -> TypeInfo:
        func_name = node.func.id
        func = self.called_functions[func_name]

        # Verify argument count matches
        if len(node.args) != len(inspect.signature(func).parameters):
            raise GLSLTypeError(node, f"Argument count mismatch for {func_name}")

        # Return the function's annotated return type
        return_type = inspect.signature(func).return_annotation
        return TypeInfo.from_pytype(return_type)


@dataclass
class FunctionRegistry:
    functions: dict[str, Callable]
    dependencies: dict[str, list[str]]

    @classmethod
    def from_ast(cls, body: list[ast.stmt], globals_dict: dict) -> "FunctionRegistry":
        visitor = FunctionCallVisitor(globals_dict)
        for node in body:
            visitor.visit(node)

        ordered = []
        seen = set()

        def add_func(name: str):
            if name in seen or name not in visitor.valid_functions:
                return
            seen.add(name)
            for dep in visitor.call_graph.get(name, []):
                add_func(dep)
            ordered.append(visitor.valid_functions[name])

        for name in visitor.top_level_calls:
            add_func(name)

        return cls(
            functions={f.__name__: f for f in ordered}, dependencies=visitor.call_graph
        )

    def get_return_type(self, func_name: str) -> TypeInfo:
        func = self.functions[func_name]
        return TypeInfo.from_pytype(func.return_annotation)


def validate_function_signature(func: Callable) -> None:
    """Validate the shader function signature"""
    sig = signature(func)
    if sig.return_annotation != vec4:
        raise GLSLTypeError("Shader must return vec4")

    pos_params = [
        p for p in sig.parameters.values() if p.kind == Parameter.POSITIONAL_ONLY
    ]
    if len(pos_params) != 1 or pos_params[0].annotation != vec2:
        raise GLSLTypeError("Must have one positional-only vec2 parameter")


def get_glsl_type(py_type: type) -> str:
    """Map Python type to GLSL type string"""
    type_map = {
        vec2: "vec2",
        vec3: "vec3",
        vec4: "vec4",
        mat3: "mat3",
        mat4: "mat4",
        float: "float",
        int: "int",
        bool: "bool",
    }
    if py_type not in type_map:
        raise GLSLTypeError(f"Unsupported type: {py_type.__name__}")
    return type_map[py_type]
