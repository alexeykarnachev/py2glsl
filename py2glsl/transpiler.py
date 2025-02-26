import ast
import inspect
import textwrap
from typing import Any, Callable, Dict, List, Set, Tuple, Union

from loguru import logger


class TranspilerError(Exception):
    pass


class StructDefinition:
    def __init__(self, name: str, fields: List[Tuple[str, str, str | None]]):
        self.name = name
        self.fields = fields


class FunctionCollector(ast.NodeVisitor):
    def __init__(self):
        self.functions: Dict[str, Tuple[str, List[str], ast.FunctionDef]] = {}
        self.structs: Dict[str, StructDefinition] = {}
        self.globals: Dict[str, Tuple[str, str]] = {}

    def _get_annotation_type(self, annotation: ast.AST) -> str:
        if annotation is None:
            return None
        if isinstance(annotation, ast.Name):
            return annotation.id
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            return annotation.value
        return None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        params = [self._get_annotation_type(arg.annotation) for arg in node.args.args]
        return_type = self._get_annotation_type(node.returns)
        self.functions[node.name] = (return_type, params, node)
        logger.debug(
            f"Collected function: {node.name}, return_type: {return_type}, params: {params}"
        )
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
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
                    default_value = None
                    if stmt.value:
                        default_value = self._generate_simple_expr(stmt.value)
                    fields.append((stmt.target.id, field_type, default_value))
            self.structs[node.name] = StructDefinition(node.name, fields)
            logger.debug(
                f"Collected struct: {node.name}, fields: {[(f[0], f[1]) for f in fields]}"
            )
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name) and node.value:
            expr_type = self._get_annotation_type(node.annotation)
            try:
                value = self._generate_simple_expr(node.value)
                self.globals[node.target.id] = (expr_type, value)
                logger.debug(
                    f"Collected global: {node.target.id}, type: {expr_type}, value: {value}"
                )
            except TranspilerError:
                pass
        self.generic_visit(node)

    def _generate_simple_expr(self, node: ast.AST) -> str:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return str(node.value)
            elif isinstance(node.value, bool):
                return "true" if node.value else "false"
            elif isinstance(node.value, str):
                return node.value
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {
                "vec2",
                "vec3",
                "vec4",
            }:
                args = [self._generate_simple_expr(arg) for arg in node.args]
                return f"{node.func.id}({', '.join(args)})"
        raise TranspilerError("Unsupported expression in global or default value")


class GLSLGenerator:
    def __init__(self, collector: FunctionCollector, indent_size: int = 4):
        self.collector = collector
        self.indent_size = indent_size
        self.builtins = {
            "sin": ("float", ["float"]),
            "cos": ("float", ["float"]),
            "tan": ("float", ["float"]),
            "abs": ("float", ["float"]),
            "length": ("float", ["vec2"]),
            "max": ("float", ["float", "float"]),
            "min": ("float", ["float", "float"]),
            "mix": ("vec3", ["vec3", "vec3", "float"]),
            "normalize": ("vec3", ["vec3"]),
            "cross": ("vec3", ["vec3", "vec3"]),
            "radians": ("float", ["float"]),
            "float": ("float", ["int"]),
            "vec2": ("vec2", ["float", "float"]),
            "vec3": ("vec3", ["float", "float", "float"]),
            "vec4": ("vec4", ["float", "float", "float", "float"]),
        }
        self.OPERATOR_PRECEDENCE = {
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
            "member": 10,
        }

    def generate(self, main_func: str) -> Tuple[str, Set[str]]:
        logger.debug(f"Starting GLSL generation for main function: {main_func}")
        lines = []
        used_uniforms = set()

        main_func_node = self.collector.functions[main_func][2]
        if not main_func_node.body or isinstance(main_func_node.body[0], ast.Pass):
            raise TranspilerError("Pass statements are not supported in GLSL")

        lines.append("#version 460 core\n")

        for arg in main_func_node.args.args:
            if arg.arg != "vs_uv":
                param_type = self.collector.functions[main_func][1][
                    main_func_node.args.args.index(arg)
                ]
                lines.append(f"uniform {param_type} {arg.arg};")
                used_uniforms.add(arg.arg)

        for name, (type_name, value) in self.collector.globals.items():
            lines.append(f"const {type_name} {name} = {value};")

        for struct_name, struct_def in self.collector.structs.items():
            lines.append(f"struct {struct_name} {{")
            for field_name, field_type, _ in struct_def.fields:
                lines.append(f"    {field_type} {field_name};")
            lines.append("};\n")

        for func_name, (return_type, params, node) in self.collector.functions.items():
            is_main = func_name == main_func
            effective_return_type = "vec4" if is_main else return_type
            if not is_main and return_type is None:
                raise TranspilerError(
                    f"Helper function '{func_name}' lacks return type annotation"
                )

            param_str = ", ".join(
                f"{p_type} {arg.arg}" for p_type, arg in zip(params, node.args.args)
            )
            body = self._generate_body(
                node.body,
                {arg.arg: p_type for arg, p_type in zip(node.args.args, params)},
            )
            lines.append(f"{effective_return_type} {func_name}({param_str}) {{")
            lines.extend(body.splitlines())
            lines.append("}\n")
            logger.debug(f"Generated GLSL for function: {func_name}")

        lines.append("in vec2 vs_uv;\nout vec4 fragColor;\n\nvoid main() {")
        main_call_args = [arg.arg for arg in main_func_node.args.args]
        main_call_str = ", ".join(main_call_args)
        lines.append(f"    fragColor = {main_func}({main_call_str});")
        lines.append("}")

        glsl_code = "\n".join(lines)
        logger.debug(f"GLSL generation complete. Used uniforms: {used_uniforms}")
        return glsl_code, used_uniforms

    def _generate_body(self, body: List[ast.AST], symbols: Dict[str, str]) -> str:
        logger.debug(f"Generating body with initial symbols: {symbols}")
        code = []
        indent = "    "
        for stmt in body:
            if isinstance(stmt, ast.Assign):
                target = stmt.targets[0]
                expr = self._generate_expr(stmt.value, symbols, 0)
                if isinstance(target, ast.Name):
                    target_name = target.id
                    expr_type = self._get_expr_type(stmt.value, symbols)
                    if target_name not in symbols:
                        symbols[target_name] = expr_type
                        code.append(f"{indent}{expr_type} {target_name} = {expr};")
                    else:
                        code.append(f"{indent}{target_name} = {expr};")
                elif isinstance(target, ast.Attribute):
                    target_expr = self._generate_expr(target, symbols, 0)
                    code.append(f"{indent}{target_expr} = {expr};")
            elif isinstance(stmt, ast.AnnAssign):
                if isinstance(stmt.target, ast.Name):
                    target = stmt.target.id
                    expr_type = self.collector._get_annotation_type(stmt.annotation)
                    expr = (
                        self._generate_expr(stmt.value, symbols, 0)
                        if stmt.value
                        else None
                    )
                    symbols[target] = expr_type
                    code.append(
                        f"{indent}{expr_type} {target}{f' = {expr}' if expr else ''};"
                    )
            elif isinstance(stmt, ast.AugAssign):
                target = self._generate_expr(stmt.target, symbols, 0)
                value = self._generate_expr(stmt.value, symbols, 0)
                op_map = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
                op = op_map.get(type(stmt.op))
                if op:
                    code.append(f"{indent}{target} = {target} {op} {value};")
                else:
                    raise TranspilerError(
                        f"Unsupported augmented operator: {type(stmt.op).__name__}"
                    )
            elif isinstance(stmt, ast.For):
                if (
                    isinstance(stmt.iter, ast.Call)
                    and isinstance(stmt.iter.func, ast.Name)
                    and stmt.iter.func.id == "range"
                ):
                    target = stmt.target.id
                    args = stmt.iter.args
                    start = (
                        "0"
                        if len(args) == 1
                        else self._generate_expr(args[0], symbols, 0)
                    )
                    end = self._generate_expr(
                        args[1 if len(args) > 1 else 0], symbols, 0
                    )
                    step = (
                        "1"
                        if len(args) <= 2
                        else self._generate_expr(args[2], symbols, 0)
                    )
                    symbols[target] = "int"
                    body_code = self._generate_body(stmt.body, symbols.copy())
                    code.append(
                        f"{indent}for (int {target} = {start}; {target} < {end}; {target} += {step}) {{"
                    )
                    code.extend(body_code.splitlines())
                    code.append(f"{indent}}}")
                else:
                    raise TranspilerError("Only range-based for loops are supported")
            elif isinstance(stmt, ast.While):
                condition = self._generate_expr(stmt.test, symbols, 0)
                body_code = self._generate_body(stmt.body, symbols.copy())
                code.append(f"{indent}while ({condition}) {{")
                code.extend(body_code.splitlines())
                code.append(f"{indent}}}")
            elif isinstance(stmt, ast.If):
                condition = self._generate_expr(stmt.test, symbols, 0)
                code.append(f"{indent}if ({condition}) {{")
                code.extend(self._generate_body(stmt.body, symbols.copy()).splitlines())
                if stmt.orelse:
                    code.append(f"{indent}}} else {{")
                    code.extend(
                        self._generate_body(stmt.orelse, symbols.copy()).splitlines()
                    )
                code.append(f"{indent}}}")
            elif isinstance(stmt, ast.Return):
                expr = self._generate_expr(stmt.value, symbols, 0)
                code.append(f"{indent}return {expr};")
            elif isinstance(stmt, ast.Break):
                code.append(f"{indent}break;")
            elif isinstance(stmt, ast.Pass):
                raise TranspilerError("Pass statements are not supported in GLSL")
        return "\n".join(code)

    def _generate_expr(
        self,
        node: ast.AST,
        symbols: Dict[str, str],
        parent_precedence: int = 0,
    ) -> str:
        if isinstance(node, ast.Name):
            return node.id if node.id in symbols else node.id
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return str(node.value)
            elif isinstance(node.value, bool):
                # Use lowercase for GLSL boolean literals
                return "true" if node.value else "false"
        elif isinstance(node, ast.BinOp):
            op_map = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
            op = op_map.get(type(node.op))
            if not op:
                raise TranspilerError(
                    f"Unsupported binary op: {type(node.op).__name__}"
                )
            precedence = self.OPERATOR_PRECEDENCE[op]
            left = self._generate_expr(node.left, symbols, precedence)
            right = self._generate_expr(node.right, symbols, precedence)
            expr = f"{left} {op} {right}"
            return f"({expr})" if precedence < parent_precedence else expr
        elif isinstance(node, ast.Compare):
            if len(node.ops) == 1 and len(node.comparators) == 1:
                op_map = {
                    ast.Lt: "<",
                    ast.Gt: ">",
                    ast.LtE: "<=",
                    ast.GtE: ">=",
                    ast.Eq: "==",
                    ast.NotEq: "!=",
                }
                op = op_map.get(type(node.ops[0]))
                if not op:
                    raise TranspilerError(
                        f"Unsupported comparison op: {type(node.ops[0]).__name__}"
                    )
                precedence = self.OPERATOR_PRECEDENCE[op]
                left = self._generate_expr(node.left, symbols, precedence)
                right = self._generate_expr(node.comparators[0], symbols, precedence)
                expr = f"{left} {op} {right}"
                return f"({expr})" if precedence < parent_precedence else expr
            raise TranspilerError("Multiple comparisons not supported")
        elif isinstance(node, ast.BoolOp):
            op_map = {ast.And: "&&", ast.Or: "||"}
            op = op_map.get(type(node.op))
            if not op:
                raise TranspilerError(
                    f"Unsupported boolean op: {type(node.op).__name__}"
                )
            precedence = self.OPERATOR_PRECEDENCE[op]
            values = [
                self._generate_expr(val, symbols, precedence) for val in node.values
            ]
            expr = f" {op} ".join(values)
            return f"({expr})" if precedence < parent_precedence else expr
        elif isinstance(node, ast.Call):
            func_name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else self._generate_expr(node.func, symbols, 0)
            )
            logger.debug(f"Processing function call: {func_name}")
            if func_name in self.collector.functions:
                logger.debug(f"Found {func_name} in collected functions")
                args = [self._generate_expr(arg, symbols, 0) for arg in node.args]
                return f"{func_name}({', '.join(args)})"
            elif func_name in self.builtins:
                logger.debug(f"Found {func_name} in builtins")
                args = [self._generate_expr(arg, symbols, 0) for arg in node.args]
                return f"{func_name}({', '.join(args)})"
            elif func_name in self.collector.structs:
                logger.debug(f"Found {func_name} as struct constructor")
                struct_def = self.collector.structs[func_name]
                field_map = {f[0]: i for i, f in enumerate(struct_def.fields)}
                required_fields = sum(
                    1 for _, _, default in struct_def.fields if default is None
                )
                if node.keywords:
                    values = [None] * len(struct_def.fields)
                    provided_fields = set()
                    for kw in node.keywords:
                        if kw.arg not in field_map:
                            raise TranspilerError(
                                f"Unknown field '{kw.arg}' in struct '{func_name}'"
                            )
                        values[field_map[kw.arg]] = self._generate_expr(
                            kw.value, symbols, 0
                        )
                        provided_fields.add(kw.arg)
                    missing_fields = [
                        field_name
                        for field_name, _, default in struct_def.fields
                        if default is None and field_name not in provided_fields
                    ]
                    if missing_fields:
                        raise TranspilerError(
                            f"Wrong number of arguments for struct {func_name}: expected {len(struct_def.fields)}, "
                            f"got {len(provided_fields)} (missing required fields: {', '.join(missing_fields)})"
                        )
                    for i, (field_name, _, default) in enumerate(struct_def.fields):
                        if values[i] is None:
                            values[i] = default if default is not None else "0.0"
                    return f"{func_name}({', '.join(values)})"
                elif node.args:
                    if len(node.args) != len(struct_def.fields):
                        raise TranspilerError(
                            f"Wrong number of arguments for struct {func_name}: expected {len(struct_def.fields)}, got {len(node.args)}"
                        )
                    args = [self._generate_expr(arg, symbols, 0) for arg in node.args]
                    return f"{func_name}({', '.join(args)})"
                raise TranspilerError(
                    f"Struct '{func_name}' initialization requires arguments"
                )
            else:
                logger.error(f"Unknown function call detected: {func_name}")
                raise TranspilerError(f"Unknown function call: {func_name}")
        elif isinstance(node, ast.Attribute):
            value = self._generate_expr(
                node.value, symbols, self.OPERATOR_PRECEDENCE["member"]
            )
            return f"{value}.{node.attr}"
        elif isinstance(node, ast.IfExp):
            condition = self._generate_expr(node.test, symbols, 3)
            true_expr = self._generate_expr(node.body, symbols, 3)
            false_expr = self._generate_expr(node.orelse, symbols, 3)
            expr = f"{condition} ? {true_expr} : {false_expr}"
            return f"({expr})" if parent_precedence <= 3 else expr
        raise TranspilerError(f"Unsupported expression: {type(node).__name__}")

    def _get_expr_type(self, node: ast.AST, symbols: Dict[str, str]) -> str:
        if isinstance(node, ast.Name):
            if node.id in symbols:
                return symbols[node.id]
            raise TranspilerError(f"Undefined variable: {node.id}")
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return "float"
            elif isinstance(node.value, bool):
                return "bool"
        elif isinstance(node, ast.BinOp):
            left_type = self._get_expr_type(node.left, symbols)
            right_type = self._get_expr_type(node.right, symbols)
            if left_type == right_type and left_type.startswith("vec"):
                return left_type
            if left_type.startswith("vec") and right_type in ["float", "int"]:
                return left_type
            if right_type.startswith("vec") and left_type in ["float", "int"]:
                return right_type
            if "float" in (left_type, right_type):
                return "float"
            return "int"
        elif isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            if func_name in self.builtins:
                return self.builtins[func_name][0]
            elif func_name in self.collector.functions:
                return self.collector.functions[func_name][0] or "vec4"
            elif func_name in self.collector.structs:
                return func_name
            raise TranspilerError(f"Unknown function: {func_name}")
        elif isinstance(node, ast.Attribute):
            value_type = self._get_expr_type(node.value, symbols)
            if value_type in self.collector.structs:
                struct_def = self.collector.structs[value_type]
                for field_name, field_type, _ in struct_def.fields:
                    if field_name == node.attr:
                        return field_type
                raise TranspilerError(
                    f"Unknown field '{node.attr}' in struct '{value_type}'"
                )
            if value_type.startswith("vec"):
                swizzle_len = len(node.attr)
                valid_lengths = {1: "float", 2: "vec2", 3: "vec3", 4: "vec4"}
                vec_dim = int(value_type[-1])
                valid_components = "xyzw"[:vec_dim] + "rgba"[:vec_dim]
                if (
                    swizzle_len not in valid_lengths
                    or swizzle_len > vec_dim
                    or not all(c in valid_components for c in node.attr)
                ):
                    raise TranspilerError(
                        f"Invalid swizzle '{node.attr}' for {value_type}"
                    )
                return valid_lengths[swizzle_len]
            raise TranspilerError(
                f"Cannot determine type for attribute on: {value_type}"
            )
        elif isinstance(node, ast.IfExp):
            true_type = self._get_expr_type(node.body, symbols)
            false_type = self._get_expr_type(node.orelse, symbols)
            if true_type != false_type:
                raise TranspilerError(
                    f"Ternary expression types mismatch: {true_type} vs {false_type}"
                )
            return true_type
        elif isinstance(node, ast.Compare) or isinstance(node, ast.BoolOp):
            return "bool"
        raise TranspilerError(f"Cannot determine type for: {type(node).__name__}")


class Transpiler:
    def __init__(
        self,
        shader_input: Union[str, Dict[str, Callable]],
        main_func: str | None = None,
        globals: Dict[str, Any] | None = None,
    ):
        self.shader_input = shader_input
        self.main_func = main_func
        self.tree = None
        self.collector = FunctionCollector()
        self.generator = None
        self.globals = globals or {}
        logger.debug(
            f"Initializing Transpiler with main_func: {main_func}, globals: {self.globals}"
        )
        self.parse()
        self.collect()

        # Add explicitly passed globals to the collector
        for name, value in self.globals.items():
            if isinstance(value, (int, float)):
                type_name = "float" if isinstance(value, float) else "int"
                self.collector.globals[name] = (type_name, str(value))
            elif isinstance(value, bool):
                # Ensure booleans are properly converted to GLSL's lowercase true/false
                self.collector.globals[name] = ("bool", "true" if value else "false")

    def parse(self) -> None:
        logger.debug("Parsing shader input")
        if isinstance(self.shader_input, dict):
            # Parse all items in context together to include globals
            source_lines = []
            for name, obj in self.shader_input.items():
                try:
                    source = textwrap.dedent(inspect.getsource(obj))
                    source_lines.append(source)
                except (OSError, TypeError) as e:
                    raise TranspilerError(f"Failed to get source for {name}: {e}")
            full_source = "\n".join(source_lines)
            self.tree = ast.parse(full_source)
            if not self.main_func:
                for node in self.tree.body:
                    if isinstance(node, ast.FunctionDef):
                        self.main_func = node.name
                        break
        elif isinstance(self.shader_input, str):
            shader_code = textwrap.dedent(self.shader_input)
            if not shader_code:
                raise TranspilerError("Empty shader code provided")
            self.tree = ast.parse(shader_code)
            if not self.main_func:
                self.main_func = "shader"
        else:
            raise TranspilerError("Shader input must be a string or context dictionary")
        logger.debug("Parsing complete")

    def collect(self) -> None:
        logger.debug("Collecting functions, structs, and globals")
        if not self.tree:
            raise TranspilerError("AST not parsed. Call parse() first.")
        self.collector.visit(self.tree)
        logger.debug(
            f"Collection complete. Functions: {list(self.collector.functions.keys())}"
        )
        if self.main_func not in self.collector.functions:
            raise TranspilerError(f"Main function '{self.main_func}' not found")
        self.generator = GLSLGenerator(self.collector)

    def generate(self) -> Tuple[str, Set[str]]:
        logger.debug(f"Generating GLSL for: {self.main_func}")
        if not self.generator:
            raise TranspilerError("Generator not initialized. Call collect() first.")
        return self.generator.generate(self.main_func)


def transpile(*args, main_func: str = None, **kwargs) -> Tuple[str, Set[str]]:
    logger.debug(
        f"Transpiling with args: {args}, main_func: {main_func}, kwargs: {kwargs}"
    )

    # Handle global constants passed as kwargs
    global_constants = {}
    for name, value in kwargs.items():
        if name != "main_func" and not callable(value) and not isinstance(value, type):
            if isinstance(value, (int, float, bool)):
                global_constants[name] = value

    if len(args) == 1:
        if isinstance(args[0], str):
            return Transpiler(
                args[0], main_func=main_func, globals=global_constants
            ).generate()
        elif inspect.ismodule(args[0]):
            module = args[0]
            if hasattr(module, "__all__"):
                context = {name: getattr(module, name) for name in module.__all__}
            else:
                context = {
                    name: obj
                    for name, obj in module.__dict__.items()
                    if inspect.isfunction(obj)
                    or (inspect.isclass(obj) and hasattr(obj, "__dataclass_fields__"))
                }
            return Transpiler(
                context, main_func=main_func or "shader", globals=global_constants
            ).generate()
        else:
            main_item = args[0]
            if main_item.__name__.startswith("test_"):
                raise TranspilerError(
                    f"Main function '{main_item.__name__}' excluded due to 'test_' prefix. "
                    "Please rename it to avoid conflicts with test function naming conventions."
                )
            context = {main_item.__name__: main_item}
            return Transpiler(
                context,
                main_func=main_func or main_item.__name__,
                globals=global_constants,
            ).generate()
    else:
        context = {}
        for item in args:
            if inspect.isfunction(item):
                context[item.__name__] = item
            elif inspect.isclass(item) and hasattr(item, "__dataclass_fields__"):
                context[item.__name__] = item
            else:
                raise TranspilerError(
                    f"Unsupported item type in transpile args: {type(item)}"
                )
        return Transpiler(
            context, main_func=main_func, globals=global_constants
        ).generate()
