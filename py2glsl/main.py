import argparse
import ast
import sys
from dataclasses import dataclass

from loguru import logger

from py2glsl import render

# Configure Loguru for color-coded logging
logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{message}</cyan>",
)

# Built-ins for GLSL transpilation
builtins = {
    "abs": ("float", ["float"]),
    "cos": ("float", ["float"]),
    "fract": ("vec2", ["vec2"]),
    "length": ("float", ["vec2"]),
    "sin": ("float", ["float"]),
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
}


# Default uniforms
@dataclass
class Interface:
    u_time: "float"
    u_aspect: "float"
    u_resolution: "vec2"
    u_mouse_pos: "vec2"
    u_mouse_uv: "vec2"


default_uniforms = {
    "u_time": "float",
    "u_aspect": "float",
    "u_resolution": "vec2",
    "u_mouse_pos": "vec2",
    "u_mouse_uv": "vec2",
}


def pytype_to_glsl(pytype):
    """Convert Python type annotations to GLSL types."""
    return pytype


class StructDefinition:
    """Helper class to represent GLSL structs in Python."""

    def __init__(self, name, fields):
        self.name = name
        self.fields = fields  # List of (name, type) tuples


class FunctionCollector(ast.NodeVisitor):
    """Collect function, struct definitions, and module-level globals from the AST."""

    def __init__(self):
        self.functions = {}
        self.structs = {}
        self.globals = {}
        self.current_context = []  # Track context: 'module', 'class', or 'function'

    def visit_Module(self, node):
        """Set module context and visit children."""
        self.current_context.append("module")
        self.generic_visit(node)
        self.current_context.pop()

    def visit_ClassDef(self, node):
        """Handle struct definitions via Python classes."""
        self.current_context.append("class")
        if node.name.endswith("Struct"):
            struct_name = node.name.replace("Struct", "")
            fields = []
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(
                    stmt.target, ast.Name
                ):
                    fields.append((stmt.target.id, stmt.annotation.value))
            self.structs[struct_name] = StructDefinition(struct_name, fields)
        self.generic_visit(node)
        self.current_context.pop()

    def visit_FunctionDef(self, node):
        """Collect function definitions."""
        self.current_context.append("function")
        func_name = node.name
        params = [
            arg.annotation.value for arg in (node.args.posonlyargs + node.args.args)
        ]
        return_type = node.returns.value if node.returns else None
        if return_type is None:
            logger.error(f"Function '{func_name}' missing return type")
            raise ValueError(
                f"Function '{func_name}' must have a return type annotation"
            )
        self.functions[func_name] = (return_type, params, node)
        self.generic_visit(node)
        self.current_context.pop()

    def visit_AnnAssign(self, node):
        """Handle global variable declarations only at module level."""
        if (
            self.current_context
            and self.current_context[-1] == "module"
            and isinstance(node.target, ast.Name)
        ):
            target = node.target.id
            expr_type = node.annotation.value
            value = self._generate_expr(node.value) if node.value else None
            self.globals[target] = (expr_type, value)
        self.generic_visit(node)

    def _generate_expr(self, node):
        """Generate expression strings for globals."""
        if isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.BinOp):
            left = self._generate_expr(node.left)
            right = self._generate_expr(node.right)
            op_map = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
            op = op_map.get(type(node.op))
            if op is None:
                raise ValueError(
                    f"Unsupported binary operation in global: {type(node.op).__name__}"
                )
            return f"({left} {op} {right})"
        raise ValueError(
            f"Unsupported expression type in global: {type(node).__name__}"
        )


class GLSLGenerator:
    """Generate GLSL code from Python AST."""

    def __init__(self, functions, structs, globals, builtins, default_uniforms):
        self.functions = functions
        self.structs = structs
        self.globals = globals
        self.builtins = builtins
        self.default_uniforms = default_uniforms
        self.code = ""

    def generate(self, func_name, node):
        logger.info(f"Generating GLSL for function: {func_name}")
        params = [
            f"{pytype_to_glsl(arg.annotation.value)} {arg.arg}"
            for arg in (node.args.posonlyargs + node.args.args)
        ]
        return_type = pytype_to_glsl(node.returns.value)
        signature = f"{return_type} {func_name}({', '.join(params)})"
        body_code = self.generate_body(
            node.body,
            {
                arg.arg: pytype_to_glsl(arg.annotation.value)
                for arg in (node.args.posonlyargs + node.args.args)
            },
        )
        self.code += f"{signature} {{\n{body_code}}}\n"

    def generate_body(self, body, symbols):
        code = ""
        for stmt in body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                target = stmt.target.id
                expr_type = stmt.annotation.value
                if stmt.value:
                    expr_code = self.generate_expr(stmt.value, symbols)
                    if expr_type in self.structs and isinstance(stmt.value, ast.Call):
                        struct_name = expr_type
                        expr_code = expr_code.replace(
                            f"{struct_name}Struct(", f"{struct_name}("
                        )
                    code += f"    {expr_type} {target} = {expr_code};\n"
                else:
                    code += f"    {expr_type} {target};\n"
                symbols[target] = expr_type
            elif isinstance(stmt, ast.Assign):
                if isinstance(stmt.targets[0], ast.Name):
                    target = stmt.targets[0].id
                    expr_code = self.generate_expr(stmt.value, symbols)
                    expr_type = self.get_expr_type(stmt.value, symbols)
                    if target not in symbols:
                        symbols[target] = expr_type
                        code += f"    {expr_type} {target} = {expr_code};\n"
                    else:
                        code += f"    {target} = {expr_code};\n"
                elif isinstance(stmt.targets[0], ast.Attribute):
                    target = self.generate_expr(stmt.targets[0], symbols)
                    expr_code = self.generate_expr(stmt.value, symbols)
                    code += f"    {target} = {expr_code};\n"
                else:
                    logger.error(
                        f"Unsupported assignment target type: {type(stmt.targets[0])}"
                    )
                    raise ValueError(
                        f"Unsupported assignment target type: {type(stmt.targets[0]).__name__}"
                    )
            elif isinstance(stmt, ast.AugAssign):
                target = self.generate_expr(stmt.target, symbols)
                value_code = self.generate_expr(stmt.value, symbols)
                op_map = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
                op = op_map.get(type(stmt.op))
                if op is None:
                    logger.error(
                        f"Unsupported augmented assignment operation: {type(stmt.op).__name__}"
                    )
                    raise ValueError(
                        f"Unsupported augmented assignment operation: {type(stmt.op).__name__}"
                    )
                code += f"    {target} = ({target} {op} {value_code});\n"
            elif isinstance(stmt, ast.While):
                condition = self.generate_expr(stmt.test, symbols)
                body_code = self.generate_body(stmt.body, symbols.copy())
                code += f"    while ({condition}) {{\n{body_code}    }}\n"
            elif isinstance(stmt, ast.For):
                if (
                    isinstance(stmt.target, ast.Name)
                    and isinstance(stmt.iter, ast.Call)
                    and stmt.iter.func.id == "range"
                ):
                    target = stmt.target.id
                    args = stmt.iter.args
                    start = (
                        self.generate_expr(args[0], symbols) if len(args) > 1 else "0"
                    )
                    end = self.generate_expr(
                        args[1] if len(args) > 1 else args[0], symbols
                    )
                    step = (
                        self.generate_expr(args[2], symbols) if len(args) > 2 else "1"
                    )
                    symbols[target] = "int"
                    body_code = self.generate_body(stmt.body, symbols.copy())
                    code += f"    for (int {target} = {start}; {target} < {end}; {target} += {step}) {{\n{body_code}    }}\n"
                else:
                    logger.error("Only 'for ... in range()' loops are supported")
                    raise ValueError("Unsupported for loop type")
            elif isinstance(stmt, ast.If):
                condition = self.generate_expr(stmt.test, symbols)
                if_body = self.generate_body(stmt.body, symbols.copy())
                code += f"    if ({condition}) {{\n{if_body}    }}\n"
                if stmt.orelse:
                    else_body = self.generate_body(stmt.orelse, symbols.copy())
                    code += f"    else {{\n{else_body}    }}\n"
            elif isinstance(stmt, ast.Break):
                code += "    break;\n"
            elif isinstance(stmt, ast.Return):
                expr_code = self.generate_expr(stmt.value, symbols)
                code += f"    return {expr_code};\n"
            elif isinstance(stmt, ast.Pass):
                logger.error("Shader body cannot be empty (contains 'pass')")
                raise ValueError("Unsupported statement type: pass")
            else:
                logger.error(f"Unsupported statement type in body: {type(stmt)}")
                raise ValueError(f"Unsupported statement type: {type(stmt).__name__}")
        return code

    def generate_expr(self, node, symbols):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Call):
            func_name = node.func.id
            args = [self.generate_expr(arg, symbols) for arg in node.args]
            return f"{func_name}({', '.join(args)})"
        elif isinstance(node, ast.BinOp):
            left = self.generate_expr(node.left, symbols)
            right = self.generate_expr(node.right, symbols)
            op_map = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
            op = op_map.get(type(node.op))
            if op is None:
                logger.error(f"Unsupported binary operation: {type(node.op).__name__}")
                raise ValueError(
                    f"Unsupported expression type: {type(node.op).__name__}"
                )
            return f"({left} {op} {right})"
        elif isinstance(node, ast.Attribute):
            value = self.generate_expr(node.value, symbols)
            return f"{value}.{node.attr}"
        elif isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Compare):
            left = self.generate_expr(node.left, symbols)
            op = node.ops[0]
            right = self.generate_expr(node.comparators[0], symbols)
            op_map = {
                ast.Lt: "<",
                ast.LtE: "<=",
                ast.Gt: ">",
                ast.GtE: ">=",
                ast.Eq: "==",
                ast.NotEq: "!=",
            }
            op_str = op_map.get(type(op))
            if op_str is None:
                logger.error(f"Unsupported comparison operator: {type(op).__name__}")
                raise ValueError(
                    f"Unsupported comparison operator: {type(op).__name__}"
                )
            return f"{left} {op_str} {right}"
        elif isinstance(node, ast.BoolOp):
            values = [self.generate_expr(value, symbols) for value in node.values]
            op_map = {ast.Or: "||", ast.And: "&&"}
            op_str = op_map.get(type(node.op))
            if op_str is None:
                logger.error(f"Unsupported boolean operation: {type(node.op).__name__}")
                raise ValueError(
                    f"Unsupported boolean operation: {type(node.op).__name__}"
                )
            return " ".join(
                [f"({val})" for val in values[:1]]
                + [f"{op_str} ({val})" for val in values[1:]]
            )
        logger.error(f"Unsupported expression type: {type(node)}")
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")

    def get_expr_type(self, node, symbols):
        if isinstance(node, ast.Name):
            return symbols.get(node.id, "float")
        elif isinstance(node, ast.Call):
            func_name = node.func.id
            if func_name in self.builtins:
                return self.builtins[func_name][0]
            elif func_name in self.functions:
                return self.functions[func_name][0]
            raise ValueError(f"Unknown function: {func_name}")
        elif isinstance(node, ast.BinOp):
            left_type = self.get_expr_type(node.left, symbols)
            right_type = self.get_expr_type(node.right, symbols)
            if left_type == right_type:
                return left_type
            elif left_type.startswith("vec") and right_type == "float":
                return left_type
            elif right_type.startswith("vec") and left_type == "float":
                return right_type
            raise ValueError(f"Type mismatch: {left_type} and {right_type}")
        elif isinstance(node, ast.Attribute):
            base_type = self.get_expr_type(node.value, symbols)
            attr = node.attr
            if base_type == "vec2":
                if attr in ["x", "y"]:
                    return "float"
                elif attr == "xy":
                    return "vec2"
            elif base_type == "vec4":
                if attr in ["x", "y", "z", "w"]:
                    return "float"
                elif attr in ["xy", "xz", "xw", "yz", "yw", "zw"]:
                    return "vec2"
                elif attr in ["xyz", "xyw", "xzw", "yzw", "rgb"]:
                    return "vec3"
                elif attr in ["xyzw", "rgba"]:
                    return "vec4"
            elif base_type in self.structs:
                for field_name, field_type in self.structs[base_type].fields:
                    if field_name == attr:
                        return field_type
            logger.error(f"Invalid attribute '{attr}' for type '{base_type}'")
            raise ValueError(
                f"Cannot infer type for attribute '{attr}' of '{base_type}'"
            )
        elif isinstance(node, ast.Constant):
            return "float"
        elif isinstance(node, ast.Compare):
            return "bool"
        elif isinstance(node, ast.BoolOp):
            return "bool"
        logger.error(f"Cannot infer type for node: {type(node)}")
        raise ValueError(f"Cannot infer type for node: {type(node).__name__}")


def transpile(shader_code, main_func_name="main_shader"):
    """Transpile Python shader code to GLSL."""
    module_ast = ast.parse(shader_code)
    collector = FunctionCollector()
    collector.visit(module_ast)
    generator = GLSLGenerator(
        collector.functions,
        collector.structs,
        collector.globals,
        builtins,
        default_uniforms,
    )

    # Generate global variable declarations
    for var_name, (var_type, var_value) in collector.globals.items():
        if var_value:
            generator.code += f"{var_type} {var_name} = {var_value};\n"
        else:
            generator.code += f"{var_type} {var_name};\n"

    # Generate struct definitions
    for struct_name, struct_def in collector.structs.items():
        fields = [
            f"    {field_type} {field_name};"
            for field_name, field_type in struct_def.fields
        ]
        generator.code += f"struct {struct_name} {{\n" + "\n".join(fields) + "\n};\n"

    # Generate all functions
    for func_name, (_, _, node) in collector.functions.items():
        generator.generate(func_name, node)

    # Validate main function
    if main_func_name not in collector.functions:
        raise ValueError(f"Main shader function '{main_func_name}' not found")
    main_params = collector.functions[main_func_name][1]
    main_param_names = [
        arg.arg
        for arg in (
            collector.functions[main_func_name][2].args.posonlyargs
            + collector.functions[main_func_name][2].args.args
        )
    ]

    # Check which parameters are uniforms (all u_* parameters)
    used_uniforms = {
        param_name for param_name in main_param_names if param_name.startswith("u_")
    }

    # Build fragment shader
    fragment_shader = "#version 460 core\n"
    fragment_shader += f"in {main_params[0]} {main_param_names[0]};\n"  # vs_uv as input
    for param_type, param_name in zip(main_params[1:], main_param_names[1:]):
        if param_name in used_uniforms:  # All u_* parameters are uniforms
            fragment_shader += f"uniform {param_type} {param_name};\n"
    fragment_shader += "out vec4 fragColor;\n"
    fragment_shader += generator.code
    fragment_shader += "void main() {\n"
    fragment_shader += (
        f"    fragColor = {main_func_name}({', '.join(main_param_names)});\n"
    )
    fragment_shader += "}\n"

    logger.opt(colors=True).info(
        f"<yellow>Fragment Shader GLSL:\n{fragment_shader}</yellow>"
    )
    return fragment_shader, used_uniforms


# Shader code with globals
shader_code = """
PI: 'float' = 3.141592
RM_MAX_DIST: 'float' = 10000.0
RM_MAX_N_STEPS: 'int' = 64
RM_EPS: 'float' = 0.0001
NORMAL_DERIVATIVE_STEP: 'float' = 0.015

class RayMarchResultStruct:
    i: 'int'
    p: 'vec3'
    n: 'vec3'
    ro: 'vec3'
    rd: 'vec3'
    dist: 'float'
    sd_last: 'float'
    sd_min: 'float'
    sd_min_shape: 'float'

def get_sd_shape(p: 'vec3') -> 'float':
    d = length(max(abs(p) - 1.0, 0.0)) - 0.2
    return d

def march(ro: 'vec3', rd: 'vec3') -> 'RayMarchResult':
    rm: 'RayMarchResult' = RayMarchResult(0, ro, vec3(0.0), ro, rd, 0.0, 0.0, RM_MAX_DIST, RM_MAX_DIST)
    for i in range(RM_MAX_N_STEPS):
        rm.p = rm.p + rm.rd * rm.sd_last
        sd_step_shape = get_sd_shape(rm.p)
        rm.sd_last = sd_step_shape
        rm.sd_min_shape = min(rm.sd_min_shape, sd_step_shape)
        rm.sd_min = min(rm.sd_min, sd_step_shape)
        rm.dist = rm.dist + length(rm.p - rm.ro)
        if rm.sd_last < RM_EPS or rm.dist > RM_MAX_DIST:
            if rm.sd_last < RM_EPS:
                rm.n = vec3(1.0)
            break

    if rm.sd_last < RM_EPS:
        h = RM_EPS
        eps = vec3(h, 0.0, 0.0)
        rm.n = vec3(0.0)
        if rm.sd_last == rm.sd_min_shape:
            e = vec2(NORMAL_DERIVATIVE_STEP, 0.0)
            rm.n = normalize(vec3(
                get_sd_shape(rm.p + e.xyy) - get_sd_shape(rm.p - e.xyy),
                get_sd_shape(rm.p + e.yxy) - get_sd_shape(rm.p - e.yxy),
                get_sd_shape(rm.p + e.yyx) - get_sd_shape(rm.p - e.yyx)
            ))
    return rm

def sin01(x: 'float', a: 'float', f: 'float', phase: 'float') -> 'float':
    return a * 0.5 * (sin(PI * f * (x + phase)) + 1.0)

def attenuate(d: 'float', coeffs: 'vec3') -> 'float':
    return 1.0 / (coeffs.x + coeffs.y * d + coeffs.z * d * d)

def main_shader(vs_uv: 'vec2', u_time: 'float', u_aspect: 'float') -> 'vec4':
    screen_pos = vs_uv * 2.0 - 1.0
    screen_pos.x *= u_aspect

    fov = radians(70.0)
    screen_dist = 1.0 / tan(0.5 * fov)
    cam_pos = vec3(5.0 * sin(u_time), 5.0, 5.0 * cos(u_time))  # Use u_time to animate camera
    look_at = vec3(0.0, 0.0, 0.0)

    forward = normalize(look_at - cam_pos)
    world_up = vec3(0.0, 1.0, 0.0)
    right = normalize(cross(forward, world_up))
    up = normalize(cross(right, forward))

    screen_center = cam_pos + forward * screen_dist
    sp = screen_center + right * screen_pos.x + up * screen_pos.y

    ro0 = cam_pos
    rd0 = normalize(sp - cam_pos)
    ro1 = sp * 4.0
    rd1 = normalize(look_at - cam_pos)

    ro = mix(ro0, ro1, 1.0)
    rd = mix(rd0, rd1, 1.0)
    rm = march(ro, rd)

    color = vec3(0.0)
    d = abs(max(0.0, rm.sd_min_shape))
    a = attenuate(d, vec3(0.01, 8.0, 8.0))
    color = 1.0 * abs(rm.n)

    return vec4(color, 1.0)
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render shader")
    parser.add_argument(
        "--mode",
        choices=["animate", "array", "gif", "video", "image"],
        default="animate",
    )
    args = parser.parse_args()

    glsl_code, used_uniforms = transpile(shader_code)

    if args.mode == "animate":
        render.animate(glsl_code, used_uniforms)
    elif args.mode == "array":
        render.render_array(glsl_code)
    elif args.mode == "gif":
        render.render_gif(glsl_code)
    elif args.mode == "video":
        render.render_video(glsl_code)
    elif args.mode == "image":
        render.render_image(glsl_code)
