import ast
import inspect
from enum import Enum
from typing import Callable, Dict, List, Set, Tuple, Union

from loguru import logger

BUILTINS = {
    "abs": ("float", ["float"]),
    "cos": ("float", ["float"]),
    "fract": ("vec2", ["vec2"]),
    "length": ("float", ["vec2"]),
    "sin": ("float", ["float"]),
    "float": ("float", ["int"]),
    "sqrt": ("float", ["float"]),
    "tan": ("float", ["float"]),
    "radians": ("float", ["float"]),
    "mix": ("vec3", ["vec3", "vec3", "float"]),
    "normalize": ("vec3", ["vec3"]),
    "cross": ("vec3", ["vec3", "vec3"]),
    "max": ("float", ["float", "float"]),
    "min": ("float", ["float", "float"]),
    "vec2": ("vec2", ["float", "float"]),
    "vec3": ("vec3", ["float", "float", "float"]),
    "vec4": ("vec4", ["float", "float", "float", "float"]),
    "distance": ("float", ["vec3", "vec3"]),
    "smoothstep": ("float", ["float", "float", "float"]),
}

DEFAULT_UNIFORMS = {
    "u_time": "float",
    "u_aspect": "float",
    "u_resolution": "vec2",
    "u_mouse_pos": "vec2",
    "u_mouse_uv": "vec2",
}


class TranspilerError(Exception):
    pass


class StructDefinition:
    def __init__(self, name: str, fields: List[Tuple[str, str, str | None]]):
        self.name = name
        self.fields = fields


class Operator(Enum):
    ADD = ("+", 6)
    SUB = ("-", 6)
    MULT = ("*", 7)
    DIV = ("/", 7)
    MOD = ("%", 7)
    EQ = ("==", 4)
    NE = ("!=", 4)
    LT = ("<", 5)
    GT = (">", 5)
    LE = ("<=", 5)
    GE = (">=", 5)
    AND = ("&&", 3)
    OR = ("||", 2)


class FunctionCollector(ast.NodeVisitor):
    def __init__(self):
        self.functions: Dict[str, Tuple[str, List[str], ast.FunctionDef]] = {}
        self.structs: Dict[str, StructDefinition] = {}
        self.globals: Dict[str, Tuple[str, str]] = {}
        self.current_context: List[str] = []

    def visit_Module(self, node: ast.Module) -> None:
        self.current_context.append("module")
        self.generic_visit(node)
        self.current_context.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.current_context.append("class")
        is_dataclass = any(
            isinstance(d, ast.Name) and d.id == "dataclass" for d in node.decorator_list
        )
        if is_dataclass:
            fields = []
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(
                    stmt.target, ast.Name
                ):
                    field_type = self._get_annotation_type(stmt.annotation)
                    if field_type:
                        default_value = None
                        if stmt.value:
                            default_value = self._generate_simple_expr(stmt.value)
                        fields.append((stmt.target.id, field_type, default_value))
            self.structs[node.name] = StructDefinition(node.name, fields)
        self.generic_visit(node)
        self.current_context.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.current_context.append("function")
        params = [self._get_annotation_type(arg.annotation) for arg in node.args.args]
        return_type = self._get_annotation_type(node.returns)
        if return_type is None:
            raise TranspilerError(
                f"Function '{node.name}' lacks return type annotation"
            )
        self.functions[node.name] = (return_type, params, node)
        self.generic_visit(node)
        self.current_context.pop()

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if self.current_context[-1] == "module" and isinstance(node.target, ast.Name):
            expr_type = self._get_annotation_type(node.annotation)
            if node.value and expr_type:
                value = self._generate_simple_expr(node.value)
                self.globals[node.target.id] = (expr_type, value)
        self.generic_visit(node)

    def _get_annotation_type(self, annotation: ast.expr) -> str | None:
        if not annotation:
            return None
        if isinstance(annotation, ast.Constant):
            return annotation.value
        if isinstance(annotation, ast.Name):
            return annotation.id
        raise TranspilerError(
            f"Unsupported annotation type: {type(annotation).__name__}"
        )

    def _generate_simple_expr(self, node: ast.expr) -> str:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return "true" if node.value else "false"
            return str(node.value)
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.BinOp):
            left = self._generate_simple_expr(node.left)
            right = self._generate_simple_expr(node.right)
            op_map = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
            op = op_map.get(type(node.op))
            if op:
                return f"({left} {op} {right})"
            raise TranspilerError(
                f"Unsupported operator in global: {type(node.op).__name__}"
            )
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BUILTINS:
                func_name = node.func.id
                expected_args = len(BUILTINS[func_name][1])
                if len(node.args) != expected_args:
                    raise TranspilerError(
                        f"Function '{func_name}' expects {expected_args} arguments, got {len(node.args)}"
                    )
                args = [self._generate_simple_expr(arg) for arg in node.args]
                return f"{func_name}({', '.join(args)})"
            raise TranspilerError(
                f"Unsupported function call in default value: {node.func}"
            )
        raise TranspilerError(
            f"Unsupported expression in global: {type(node).__name__}"
        )


class GLSLGenerator:
    OPERATOR_PRECEDENCE = {
        "=": 1,
        "||": 2,
        "&&": 3,
        "==": 4,
        "!=": 4,
        "<": 5,
        ">": 5,
        "<=": 5,
        ">=": 5,
        "+": 6,
        "-": 6,
        "*": 7,
        "/": 7,
        "%": 7,
        "unary": 8,
        "call": 9,
        "member": 9,
    }

    def __init__(self, collector: FunctionCollector, indent_size: int = 4):
        self.collector = collector
        self.builtins = BUILTINS
        self.code = ""
        self.indent_level = 0
        self.indent_size = indent_size

    def indent(self) -> str:
        return " " * (self.indent_level * self.indent_size)

    def generate_function(self, func_name: str, node: ast.FunctionDef) -> None:
        params = [
            f"{self._get_annotation_type(arg.annotation)} {arg.arg}"
            for arg in node.args.args
        ]
        return_type = self._get_annotation_type(node.returns)
        signature = f"{return_type} {func_name}({', '.join(params)})"
        symbols = {
            arg.arg: self._get_annotation_type(arg.annotation) for arg in node.args.args
        }
        body = self._generate_body(node.body, symbols)
        self.code += f"{signature} {{\n{body}}}\n"

    def _generate_body(self, body: List[ast.stmt], symbols: Dict[str, str]) -> str:
        code = ""
        for stmt in body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                target = stmt.target.id
                expr_type = self._get_annotation_type(stmt.annotation)
                expr = (
                    self._generate_expr(stmt.value, symbols, 0) if stmt.value else None
                )
                symbols[target] = expr_type
                code += f"{self.indent()}{expr_type} {target}{f' = {expr}' if expr else ''};\n"
            elif isinstance(stmt, ast.Assign):
                if isinstance(stmt.targets[0], ast.Name):
                    target = stmt.targets[0].id
                    expr_type = self._get_expr_type(stmt.value, symbols)
                    if (
                        isinstance(stmt.value, ast.Call)
                        and isinstance(stmt.value.func, ast.Name)
                        and stmt.value.func.id in self.collector.structs
                    ):
                        struct_name = stmt.value.func.id
                        expr = self._generate_expr(stmt.value, symbols, 0)
                        code += f"{self.indent()}{struct_name} {target} = {expr};\n"
                    elif target not in symbols:
                        expr = self._generate_expr(stmt.value, symbols, 0)
                        symbols[target] = expr_type
                        code += f"{self.indent()}{expr_type} {target} = {expr};\n"
                    else:
                        expr = self._generate_expr(stmt.value, symbols, 0)
                        code += f"{self.indent()}{target} = {expr};\n"
                elif isinstance(stmt.targets[0], ast.Attribute):
                    target = self._generate_expr(stmt.targets[0], symbols, 0)
                    expr = self._generate_expr(stmt.value, symbols, 0)
                    code += f"{self.indent()}{target} = {expr};\n"
            elif isinstance(stmt, ast.AugAssign):
                target = self._generate_expr(stmt.target, symbols, 0)
                value = self._generate_expr(stmt.value, symbols, 0)
                op_map = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
                op = op_map.get(type(stmt.op))
                if op:
                    code += f"{self.indent()}{target} = {target} {op} {value};\n"
                else:
                    raise TranspilerError(
                        f"Unsupported augmented operator: {type(stmt.op).__name__}"
                    )
            elif (
                isinstance(stmt, ast.For)
                and isinstance(stmt.iter, ast.Call)
                and getattr(stmt.iter.func, "id", None) == "range"
            ):
                target = stmt.target.id
                args = stmt.iter.args
                start = (
                    self._generate_expr(args[0], symbols, 0) if len(args) > 1 else "0"
                )
                end = self._generate_expr(args[1 if len(args) > 1 else 0], symbols, 0)
                step = (
                    self._generate_expr(args[2], symbols, 0) if len(args) > 2 else "1"
                )
                symbols[target] = "int"
                body_code = self._generate_body(stmt.body, symbols.copy())
                code += f"{self.indent()}for (int {target} = {start}; {target} < {end}; {target} += {step}) {{\n{body_code}{self.indent()}}}\n"
            elif isinstance(stmt, ast.While):
                condition = self._generate_expr(stmt.test, symbols, 0)
                body_code = self._generate_body(stmt.body, symbols.copy())
                code += f"{self.indent()}while ({condition}) {{\n{body_code}{self.indent()}}}\n"
            elif isinstance(stmt, ast.If):
                condition = self._generate_expr(stmt.test, symbols, 0)
                if_body = self._generate_body(stmt.body, symbols.copy())
                code += (
                    f"{self.indent()}if ({condition}) {{\n{if_body}{self.indent()}}}"
                )
                if stmt.orelse:
                    else_body = self._generate_body(stmt.orelse, symbols.copy())
                    code += f" else {{\n{else_body}{self.indent()}}}"
                code += "\n"
            elif isinstance(stmt, ast.Return):
                expr = self._generate_expr(stmt.value, symbols, 0)
                code += f"{self.indent()}return {expr};\n"
            elif isinstance(stmt, ast.Break):
                code += f"{self.indent()}break;\n"
            elif isinstance(stmt, ast.Pass):
                raise TranspilerError("Pass statements are not supported in GLSL")
            elif isinstance(stmt, ast.Expr):
                continue
            else:
                raise TranspilerError(f"Unsupported statement: {type(stmt).__name__}")
        return code

    def _generate_expr(
        self, node: ast.expr, symbols: Dict[str, str], parent_precedence: int = 0
    ) -> str:
        if isinstance(node, ast.Constant):
            return self._generate_constant(node)
        elif isinstance(node, ast.Name):
            return self._generate_name(node)
        elif isinstance(node, ast.BinOp):
            return self._generate_binop(node, symbols, parent_precedence)
        elif isinstance(node, ast.Compare):
            return self._generate_compare(node, symbols, parent_precedence)
        elif isinstance(node, ast.BoolOp):
            return self._generate_boolop(node, symbols, parent_precedence)
        elif isinstance(node, ast.Call):
            return self._generate_call(node, symbols)
        elif isinstance(node, ast.Attribute):
            return self._generate_attribute(node, symbols)
        raise TranspilerError(f"Unsupported expression: {type(node).__name__}")

    def _generate_constant(self, node: ast.Constant) -> str:
        if isinstance(node.value, float):
            return (
                f"{node.value:.1f}"
                if node.value == int(node.value)
                else str(node.value)
            )
        elif isinstance(node.value, bool):
            return "true" if node.value else "false"
        return str(node.value)

    def _generate_name(self, node: ast.Name) -> str:
        if node.id == "True":
            return "true"
        elif node.id == "False":
            return "false"
        return node.id

    def _generate_binop(
        self, node: ast.BinOp, symbols: Dict[str, str], parent_precedence: int
    ) -> str:
        op_map = {
            ast.Add: Operator.ADD,
            ast.Sub: Operator.SUB,
            ast.Mult: Operator.MULT,
            ast.Div: Operator.DIV,
            ast.Mod: Operator.MOD,
        }
        op = op_map.get(type(node.op))
        if not op:
            raise TranspilerError(f"Unsupported binary op: {type(node.op).__name__}")
        precedence = op.value[1]
        left = self._generate_expr(node.left, symbols, precedence)
        right = self._generate_expr(node.right, symbols, precedence)
        expr = f"{left} {op.value[0]} {right}"
        return f"({expr})" if precedence < parent_precedence else expr

    def _generate_compare(
        self, node: ast.Compare, symbols: Dict[str, str], parent_precedence: int
    ) -> str:
        if len(node.ops) != 1:
            raise TranspilerError("Only single comparisons are supported")
        op_map = {
            ast.Eq: Operator.EQ,
            ast.NotEq: Operator.NE,
            ast.Lt: Operator.LT,
            ast.LtE: Operator.LE,
            ast.Gt: Operator.GT,
            ast.GtE: Operator.GE,
        }
        op = op_map.get(type(node.ops[0]))
        if not op:
            raise TranspilerError(
                f"Unsupported comparison: {type(node.ops[0]).__name__}"
            )
        precedence = op.value[1]
        left = self._generate_expr(node.left, symbols, precedence)
        right = self._generate_expr(node.comparators[0], symbols, precedence)
        expr = f"{left} {op.value[0]} {right}"
        return f"({expr})" if precedence < parent_precedence else expr

    def _generate_boolop(
        self, node: ast.BoolOp, symbols: Dict[str, str], parent_precedence: int
    ) -> str:
        op_map = {ast.And: Operator.AND, ast.Or: Operator.OR}
        op = op_map.get(type(node.op))
        if not op:
            raise TranspilerError(f"Unsupported boolean op: {type(node.op).__name__}")
        precedence = op.value[1]
        values = [self._generate_expr(val, symbols, precedence) for val in node.values]
        expr = f" {op.value[0]} ".join(values)
        return f"({expr})" if precedence < parent_precedence else expr

    def _generate_call(self, node: ast.Call, symbols: Dict[str, str]) -> str:
        func_name = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else self._generate_expr(node.func, symbols, 0)
        )
        if func_name in self.collector.structs:
            struct_def = self.collector.structs[func_name]
            field_map = {f[0]: i for i, f in enumerate(struct_def.fields)}
            values = ["0"] * len(
                struct_def.fields
            )  # Default to 0 for unspecified fields
            for kw in node.keywords:
                if kw.arg not in field_map:
                    raise TranspilerError(
                        f"Unknown field '{kw.arg}' in struct '{func_name}'"
                    )
                values[field_map[kw.arg]] = self._generate_expr(kw.value, symbols, 0)
            # Fill in defaults for unspecified fields
            for i, (field_name, _, default_value) in enumerate(struct_def.fields):
                if values[i] == "0" and default_value is not None:
                    values[i] = default_value
            return f"{func_name}({', '.join(values)})"
        args = [self._generate_expr(arg, symbols, 0) for arg in node.args]
        if func_name in self.builtins or func_name in self.collector.functions:
            return f"{func_name}({', '.join(args)})"
        raise TranspilerError(f"Unknown function call: {func_name}")

    def _generate_attribute(self, node: ast.Attribute, symbols: Dict[str, str]) -> str:
        value = self._generate_expr(
            node.value, symbols, self.OPERATOR_PRECEDENCE["member"]
        )
        return f"{value}.{node.attr}"

    def _get_expr_type(self, node: ast.expr, symbols: Dict[str, str]) -> str:
        if isinstance(node, ast.Constant):
            return self._get_constant_type(node)
        elif isinstance(node, ast.Name):
            return self._get_variable_type(node, symbols)
        elif isinstance(node, ast.Call):
            return self._get_call_type(node, symbols)
        elif isinstance(node, ast.BinOp):
            return self._get_binop_type(node, symbols)
        elif isinstance(node, ast.Attribute):
            return self._get_attribute_type(node, symbols)
        raise TranspilerError(f"Cannot determine type for: {type(node).__name__}")

    def _get_constant_type(self, node: ast.Constant) -> str:
        if isinstance(node.value, float):
            return "float"
        elif isinstance(node.value, int):
            return "int"
        elif isinstance(node.value, bool):
            return "bool"
        raise TranspilerError(f"Unsupported constant type: {type(node.value)}")

    def _get_variable_type(self, node: ast.Name, symbols: Dict[str, str]) -> str:
        if node.id in symbols:
            return symbols[node.id]
        elif node.id in self.collector.globals:
            return self.collector.globals[node.id][0]
        raise TranspilerError(f"Undefined variable: {node.id}")

    def _get_call_type(self, node: ast.Call, symbols: Dict[str, str]) -> str:
        func_name = node.func.id if isinstance(node.func, ast.Name) else None
        if func_name in self.builtins:
            return self.builtins[func_name][0]
        elif func_name in self.collector.functions:
            return self.collector.functions[func_name][0]
        elif func_name in self.collector.structs:
            return func_name
        raise TranspilerError(f"Unknown function: {func_name}")

    def _get_binop_type(self, node: ast.BinOp, symbols: Dict[str, str]) -> str:
        left_type = self._get_expr_type(node.left, symbols)
        right_type = self._get_expr_type(node.right, symbols)
        if isinstance(node.op, ast.Mod):
            if left_type != "int" or right_type != "int":
                raise TranspilerError("Modulo requires integer operands")
            return "int"
        if left_type.startswith("vec") and right_type in ["float", "int"]:
            return left_type
        if right_type.startswith("vec") and left_type in ["float", "int"]:
            return right_type
        if left_type == right_type and left_type.startswith("vec"):
            return left_type
        if "float" in (left_type, right_type):
            return "float"
        return "int"

    def _get_attribute_type(self, node: ast.Attribute, symbols: Dict[str, str]) -> str:
        value_type = self._get_expr_type(node.value, symbols)
        if value_type in self.collector.structs:
            for fname, ftype, _ in self.collector.structs[value_type].fields:
                if fname == node.attr:
                    return ftype
            raise TranspilerError(f"Invalid attribute access: {node.attr}")
        if value_type.startswith("vec"):
            vec_dim = int(value_type[-1])
            swizzle_len = len(node.attr)
            valid_lengths = {1: "float", 2: "vec2", 3: "vec3", 4: "vec4"}
            if swizzle_len in valid_lengths and swizzle_len <= vec_dim:
                valid_components_map = {2: "xyrg", 3: "xyzrgb", 4: "xyzwrgba"}
                valid_components = valid_components_map[vec_dim]
                if all(c in valid_components for c in node.attr):
                    return valid_lengths[swizzle_len]
                raise TranspilerError(f"Invalid attribute access: {node.attr}")
            raise TranspilerError(f"Invalid attribute access: {node.attr}")
        raise TranspilerError(f"Invalid attribute access: {node.attr}")

    def _get_annotation_type(self, annotation: ast.expr) -> str | None:
        if not annotation:
            return None
        if isinstance(annotation, ast.Constant):
            return annotation.value
        if isinstance(annotation, ast.Name):
            return annotation.id
        raise TranspilerError(
            f"Unsupported annotation type: {type(annotation).__name__}"
        )


class Transpiler:
    def __init__(
        self,
        shader_input: Union[str, Callable],
        main_func: str = "main_shader",
        version: str = "460 core",
        indent_size: int = 4,
    ):
        self.shader_input = shader_input
        self.main_func = main_func
        self.version = version
        self.indent_size = indent_size
        self.tree: ast.AST | None = None
        self.collector = FunctionCollector()
        self.generator: GLSLGenerator | None = None

    def parse(self) -> None:
        if callable(self.shader_input):
            module = inspect.getmodule(self.shader_input)
            shader_code = inspect.getsource(module or self.shader_input)
            if self.shader_input.__name__ != self.main_func:
                self.main_func = self.shader_input.__name__
        else:
            shader_code = self.shader_input
        if not shader_code.strip():
            raise TranspilerError("Empty shader code provided")
        self.tree = ast.parse(shader_code)

    def collect(self) -> None:
        if not self.tree:
            raise TranspilerError("AST not parsed. Call parse() first.")
        self.collector.visit(self.tree)
        if self.main_func not in self.collector.functions:
            raise TranspilerError(f"Main function '{self.main_func}' not found")
        if not self.collector.functions[self.main_func][2].body:
            raise TranspilerError(f"Main function '{self.main_func}' has no body")
        self.generator = GLSLGenerator(self.collector, indent_size=self.indent_size)

    def generate(self) -> Tuple[str, Set[str]]:
        if not self.generator:
            raise TranspilerError("Generator not initialized. Call collect() first.")

        struct_code = "".join(
            f"struct {name} {{\n"
            + "".join(f"    {ftype} {fname};\n" for fname, ftype, _ in defn.fields)
            + "};\n\n"
            for name, defn in self.collector.structs.items()
        )

        global_code = "".join(
            f"const {type_name} {name} = {value};\n"
            for name, (type_name, value) in self.collector.globals.items()
        )

        used_uniforms = set()
        main_node = self.collector.functions[self.main_func][2]
        main_params = []
        for arg in main_node.args.args:
            param_name = arg.arg
            if param_name == "vs_uv":
                main_params.append("vs_uv")
            elif param_name in DEFAULT_UNIFORMS:
                main_params.append(param_name)
                used_uniforms.add(param_name)
            else:
                param_type = self._get_annotation_type(arg.annotation)
                if param_type:
                    main_params.append(param_name)
                    used_uniforms.add(param_name)
                else:
                    raise TranspilerError(
                        f"Parameter '{param_name}' lacks type annotation"
                    )

        uniform_code = "".join(
            f"uniform {DEFAULT_UNIFORMS[name]} {name};\n"
            for name in used_uniforms
            if name in DEFAULT_UNIFORMS
        ) + "".join(
            f"uniform {self._get_annotation_type(arg.annotation)} {arg.arg};\n"
            for arg in main_node.args.args
            if arg.arg not in set(DEFAULT_UNIFORMS.keys()) | {"vs_uv"}
        )

        for func_name, (_, _, node) in self.collector.functions.items():
            self.generator.generate_function(func_name, node)

        main_code = (
            "in vec2 vs_uv;\nout vec4 fragColor;\n\n"
            f"void main() {{\n    fragColor = {self.main_func}({', '.join(main_params)});\n}}\n"
        )

        glsl_code = f"#version {self.version}\n\n{uniform_code}{global_code}{struct_code}{self.generator.code}{main_code}"
        logger.debug(f"Generated GLSL:\n{glsl_code}")
        return glsl_code, used_uniforms

    def transpile(self) -> Tuple[str, Set[str]]:
        self.parse()
        self.collect()
        return self.generate()

    def _get_annotation_type(self, annotation: ast.expr) -> str | None:
        if not annotation:
            return None
        if isinstance(annotation, ast.Constant):
            return annotation.value
        if isinstance(annotation, ast.Name):
            return annotation.id
        raise TranspilerError(
            f"Unsupported annotation type: {type(annotation).__name__}"
        )


def transpile(
    shader_input: Union[str, Callable],
    main_func: str = "main_shader",
    version: str = "460 core",
    indent_size: int = 4,
) -> Tuple[str, Set[str]]:
    transpiler = Transpiler(shader_input, main_func, version, indent_size)
    return transpiler.transpile()
