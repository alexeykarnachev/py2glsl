# Proposed Transpiler Architecture

## Current Flow (Problems)

```
Python AST → CollectedInfo → GLSL strings (directly)
                              ↑
                         Backend hardcodes:
                         - Entry point format
                         - Uniform detection (vs_uv magic)
                         - Fragment-shader-only
```

## Proposed Flow

```
Python Code
    ↓
[Parser] → Python AST
    ↓
[Collector] → CollectedInfo
    ↓
[IR Builder] → ShaderIR  ← NEW: Typed intermediate representation
    ↓
[Emitter] ← Dialect      ← NEW: Dialect defines syntax, Emitter generates
    ↓
Target Code (GLSL 460, GLSL 300es, WGSL, HLSL, etc.)
```

---

## 1. Intermediate Representation (IR)

The IR captures shader semantics independent of target language.

```python
# py2glsl/transpiler/ir.py

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


class ShaderStage(Enum):
    VERTEX = auto()
    FRAGMENT = auto()
    COMPUTE = auto()
    # Future: GEOMETRY, TESS_CONTROL, TESS_EVAL, RAYGEN, etc.


class StorageClass(Enum):
    """How a variable is stored/accessed."""
    LOCAL = auto()       # Function-local variable
    CONST = auto()       # Compile-time constant
    UNIFORM = auto()     # Uniform buffer / push constant
    INPUT = auto()       # Varying input (from previous stage)
    OUTPUT = auto()      # Varying output (to next stage)
    BUFFER = auto()      # Storage buffer (read/write)
    TEXTURE = auto()     # Sampled texture
    IMAGE = auto()       # Storage image (read/write)
    SHARED = auto()      # Workgroup shared memory (compute)


@dataclass
class IRType:
    """Type in the IR."""
    base: str                          # "float", "vec3", "mat4", "MyStruct"
    array_size: int | None = None      # None = scalar, N = array[N]

    def __str__(self) -> str:
        if self.array_size:
            return f"{self.base}[{self.array_size}]"
        return self.base


@dataclass
class IRVariable:
    """A variable declaration."""
    name: str
    type: IRType
    storage: StorageClass
    binding: int | None = None         # For uniforms/textures
    location: int | None = None        # For inputs/outputs


@dataclass
class IRParameter:
    """Function parameter."""
    name: str
    type: IRType
    qualifier: str = ""                # "in", "out", "inout"


@dataclass
class IRExpr:
    """Base for all expressions."""
    result_type: IRType


@dataclass
class IRLiteral(IRExpr):
    value: Any


@dataclass
class IRName(IRExpr):
    name: str


@dataclass
class IRBinOp(IRExpr):
    op: str                            # "+", "-", "*", "/", etc.
    left: IRExpr
    right: IRExpr


@dataclass
class IRCall(IRExpr):
    func: str
    args: list[IRExpr]


@dataclass
class IRSwizzle(IRExpr):
    base: IRExpr
    components: str                    # "xyz", "rg", etc.


@dataclass
class IRSubscript(IRExpr):
    base: IRExpr
    index: IRExpr


@dataclass
class IRTernary(IRExpr):
    condition: IRExpr
    true_expr: IRExpr
    false_expr: IRExpr


# Statements

@dataclass
class IRStmt:
    """Base for all statements."""
    pass


@dataclass
class IRDeclare(IRStmt):
    var: IRVariable
    init: IRExpr | None = None


@dataclass
class IRAssign(IRStmt):
    target: IRExpr
    value: IRExpr


@dataclass
class IRReturn(IRStmt):
    value: IRExpr | None


@dataclass
class IRIf(IRStmt):
    condition: IRExpr
    then_body: list[IRStmt]
    else_body: list[IRStmt]


@dataclass
class IRFor(IRStmt):
    init: IRStmt
    condition: IRExpr
    update: IRStmt
    body: list[IRStmt]


@dataclass
class IRWhile(IRStmt):
    condition: IRExpr
    body: list[IRStmt]


@dataclass
class IRExprStmt(IRStmt):
    """Expression as statement (e.g., function call)."""
    expr: IRExpr


# Functions and Shaders

@dataclass
class IRFunction:
    """A function definition."""
    name: str
    params: list[IRParameter]
    return_type: IRType | None
    body: list[IRStmt]
    is_entry_point: bool = False


@dataclass
class IRStruct:
    """A struct definition."""
    name: str
    fields: list[tuple[str, IRType]]   # (name, type) pairs


@dataclass
class ShaderIR:
    """Complete shader intermediate representation."""
    stage: ShaderStage
    structs: list[IRStruct]
    variables: list[IRVariable]        # Uniforms, inputs, outputs, etc.
    functions: list[IRFunction]
    entry_point: str                   # Name of entry point function

    # Compute-specific
    workgroup_size: tuple[int, int, int] | None = None
```

---

## 2. Parameter Qualifiers (Python Side)

Use type hints to specify storage class:

```python
# py2glsl/qualifiers.py

from typing import Generic, TypeVar

T = TypeVar("T")


class uniform(Generic[T]):
    """Marks a parameter as uniform input."""
    pass


class input_(Generic[T]):
    """Marks a parameter as varying input from previous stage."""
    pass


class output_(Generic[T]):
    """Marks a parameter as varying output to next stage."""
    pass


class texture(Generic[T]):
    """Marks a parameter as sampled texture."""
    pass


class buffer(Generic[T]):
    """Marks a parameter as storage buffer."""
    pass
```

**Usage in shader code:**

```python
from py2glsl.qualifiers import uniform, input_, output_, texture
from py2glsl.builtins import vec2, vec4, sampler2D

def fragment_shader(
    uv: input_[vec2],              # Varying from vertex shader
    u_time: uniform[float],        # Uniform
    u_texture: texture[sampler2D], # Sampled texture
) -> output_[vec4]:                # Fragment output
    color = texture_sample(u_texture, uv)
    return color * sin(u_time)
```

---

## 3. Shader Stage Decorators

```python
# py2glsl/decorators.py

from py2glsl.ir import ShaderStage


def fragment(func):
    """Mark function as fragment shader entry point."""
    func._shader_stage = ShaderStage.FRAGMENT
    return func


def vertex(func):
    """Mark function as vertex shader entry point."""
    func._shader_stage = ShaderStage.VERTEX
    return func


def compute(workgroup_size: tuple[int, int, int] = (1, 1, 1)):
    """Mark function as compute shader entry point."""
    def decorator(func):
        func._shader_stage = ShaderStage.COMPUTE
        func._workgroup_size = workgroup_size
        return func
    return decorator
```

**Usage:**

```python
from py2glsl.decorators import fragment, vertex, compute

@vertex
def vs_main(position: input_[vec3]) -> output_[vec4]:
    return vec4(position, 1.0)

@fragment
def fs_main(uv: input_[vec2]) -> output_[vec4]:
    return vec4(uv, 0.0, 1.0)

@compute(workgroup_size=(8, 8, 1))
def cs_main(global_id: input_[uvec3]) -> None:
    # Compute shader logic
    pass
```

---

## 4. Dialect System (Backend Abstraction)

Instead of backends generating code directly, separate into:
- **Dialect**: Defines syntax rules for a target language
- **Emitter**: Generates code using dialect rules

```python
# py2glsl/transpiler/dialect.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from py2glsl.transpiler.ir import *


@dataclass
class DialectConfig:
    """Configuration for a target dialect."""
    name: str
    version: str
    file_extension: str


class Dialect(ABC):
    """Defines syntax rules for a target language."""

    @abstractmethod
    def get_config(self) -> DialectConfig:
        """Get dialect configuration."""
        ...

    @abstractmethod
    def type_name(self, ir_type: IRType) -> str:
        """Convert IR type to target language type name."""
        ...

    @abstractmethod
    def builtin_function(self, name: str) -> str | None:
        """Map builtin function name, or None if not supported."""
        ...

    @abstractmethod
    def operator(self, op: str) -> str:
        """Map operator to target syntax."""
        ...

    @abstractmethod
    def storage_qualifier(self, storage: StorageClass) -> str:
        """Get storage qualifier syntax."""
        ...

    @abstractmethod
    def version_directive(self) -> str:
        """Get version/header directive."""
        ...

    @abstractmethod
    def entry_point_wrapper(
        self,
        stage: ShaderStage,
        entry_func: IRFunction,
        inputs: list[IRVariable],
        outputs: list[IRVariable],
    ) -> list[str]:
        """Generate entry point wrapper code."""
        ...


class GLSL460Dialect(Dialect):
    """GLSL 4.60 core dialect."""

    def get_config(self) -> DialectConfig:
        return DialectConfig("GLSL", "460 core", ".glsl")

    def type_name(self, ir_type: IRType) -> str:
        # GLSL types match IR types directly
        if ir_type.array_size:
            return f"{ir_type.base}[{ir_type.array_size}]"
        return ir_type.base

    def builtin_function(self, name: str) -> str | None:
        # Most builtins have same name in GLSL
        return name

    def operator(self, op: str) -> str:
        mapping = {"and": "&&", "or": "||", "not": "!"}
        return mapping.get(op, op)

    def storage_qualifier(self, storage: StorageClass) -> str:
        return {
            StorageClass.UNIFORM: "uniform",
            StorageClass.INPUT: "in",
            StorageClass.OUTPUT: "out",
            StorageClass.CONST: "const",
            StorageClass.BUFFER: "buffer",
            StorageClass.SHARED: "shared",
        }.get(storage, "")

    def version_directive(self) -> str:
        return "#version 460 core"

    def entry_point_wrapper(self, stage, entry_func, inputs, outputs):
        # Generate appropriate main() wrapper for stage
        ...


class GLSL300esDialect(Dialect):
    """GLSL ES 3.00 dialect (WebGL 2.0)."""

    def get_config(self) -> DialectConfig:
        return DialectConfig("GLSL ES", "300 es", ".glsl")

    def version_directive(self) -> str:
        return "#version 300 es\nprecision highp float;"

    # ... other methods with ES-specific behavior


class WGSLDialect(Dialect):
    """WebGPU Shading Language dialect."""

    def get_config(self) -> DialectConfig:
        return DialectConfig("WGSL", "1.0", ".wgsl")

    def type_name(self, ir_type: IRType) -> str:
        # WGSL uses different type names
        mapping = {
            "float": "f32",
            "int": "i32",
            "uint": "u32",
            "vec2": "vec2<f32>",
            "vec3": "vec3<f32>",
            "vec4": "vec4<f32>",
            "mat4": "mat4x4<f32>",
        }
        base = mapping.get(ir_type.base, ir_type.base)
        if ir_type.array_size:
            return f"array<{base}, {ir_type.array_size}>"
        return base

    def storage_qualifier(self, storage: StorageClass) -> str:
        # WGSL uses attributes, not qualifiers
        return ""  # Handled differently in entry_point_wrapper

    def version_directive(self) -> str:
        return ""  # WGSL has no version directive

    # ... WGSL-specific methods
```

---

## 5. Code Emitter

The emitter walks the IR and generates code using the dialect:

```python
# py2glsl/transpiler/emitter.py

from py2glsl.transpiler.ir import *
from py2glsl.transpiler.dialect import Dialect


class Emitter:
    """Generates target code from IR using a dialect."""

    def __init__(self, dialect: Dialect):
        self.dialect = dialect
        self.indent = 0

    def emit(self, shader: ShaderIR) -> str:
        """Generate complete shader code."""
        lines = []

        # Version/header
        version = self.dialect.version_directive()
        if version:
            lines.append(version)
            lines.append("")

        # Struct definitions
        for struct in shader.structs:
            lines.extend(self.emit_struct(struct))
            lines.append("")

        # Variable declarations (uniforms, inputs, outputs)
        for var in shader.variables:
            lines.append(self.emit_variable(var))
        if shader.variables:
            lines.append("")

        # Functions
        for func in shader.functions:
            if not func.is_entry_point:
                lines.extend(self.emit_function(func))
                lines.append("")

        # Entry point wrapper
        entry_func = next(f for f in shader.functions if f.name == shader.entry_point)
        inputs = [v for v in shader.variables if v.storage == StorageClass.INPUT]
        outputs = [v for v in shader.variables if v.storage == StorageClass.OUTPUT]
        lines.extend(self.dialect.entry_point_wrapper(
            shader.stage, entry_func, inputs, outputs
        ))

        return "\n".join(lines)

    def emit_struct(self, struct: IRStruct) -> list[str]:
        lines = [f"struct {struct.name} {{"]
        for name, type_ in struct.fields:
            type_str = self.dialect.type_name(type_)
            lines.append(f"    {type_str} {name};")
        lines.append("};")
        return lines

    def emit_variable(self, var: IRVariable) -> str:
        qualifier = self.dialect.storage_qualifier(var.storage)
        type_str = self.dialect.type_name(var.type)
        parts = [qualifier, type_str, var.name] if qualifier else [type_str, var.name]
        return " ".join(parts) + ";"

    def emit_function(self, func: IRFunction) -> list[str]:
        return_type = self.dialect.type_name(func.return_type) if func.return_type else "void"
        params = ", ".join(
            f"{self.dialect.type_name(p.type)} {p.name}"
            for p in func.params
        )

        lines = [f"{return_type} {func.name}({params}) {{"]
        for stmt in func.body:
            lines.extend(self.emit_stmt(stmt))
        lines.append("}")
        return lines

    def emit_stmt(self, stmt: IRStmt) -> list[str]:
        match stmt:
            case IRDeclare(var, init):
                type_str = self.dialect.type_name(var.type)
                if init:
                    return [f"    {type_str} {var.name} = {self.emit_expr(init)};"]
                return [f"    {type_str} {var.name};"]

            case IRAssign(target, value):
                return [f"    {self.emit_expr(target)} = {self.emit_expr(value)};"]

            case IRReturn(value):
                if value:
                    return [f"    return {self.emit_expr(value)};"]
                return ["    return;"]

            case IRIf(condition, then_body, else_body):
                lines = [f"    if ({self.emit_expr(condition)}) {{"]
                for s in then_body:
                    lines.extend("    " + line for line in self.emit_stmt(s))
                if else_body:
                    lines.append("    } else {")
                    for s in else_body:
                        lines.extend("    " + line for line in self.emit_stmt(s))
                lines.append("    }")
                return lines

            # ... other statement types

        return []

    def emit_expr(self, expr: IRExpr) -> str:
        match expr:
            case IRLiteral(_, value):
                return str(value)

            case IRName(_, name):
                return name

            case IRBinOp(_, op, left, right):
                op_str = self.dialect.operator(op)
                return f"({self.emit_expr(left)} {op_str} {self.emit_expr(right)})"

            case IRCall(_, func, args):
                func_name = self.dialect.builtin_function(func) or func
                args_str = ", ".join(self.emit_expr(a) for a in args)
                return f"{func_name}({args_str})"

            case IRSwizzle(_, base, components):
                return f"{self.emit_expr(base)}.{components}"

            case IRSubscript(_, base, index):
                return f"{self.emit_expr(base)}[{self.emit_expr(index)}]"

            case IRTernary(_, cond, true_e, false_e):
                return f"({self.emit_expr(cond)} ? {self.emit_expr(true_e)} : {self.emit_expr(false_e)})"

        return ""
```

---

## 6. New Transpile Flow

```python
# py2glsl/transpiler/__init__.py (new version)

def transpile(
    *args,
    dialect: Dialect | None = None,
    **kwargs,
) -> str:
    """Transpile Python shader to target language.

    Args:
        *args: Shader functions/classes
        dialect: Target dialect (default: GLSL460Dialect)
        **kwargs: Global constants

    Returns:
        Generated shader code
    """
    if dialect is None:
        dialect = GLSL460Dialect()

    # 1. Parse Python code to AST
    tree, main_func = parse_shader_code(args)

    # 2. Collect info (functions, structs, globals)
    collected = collect_info(tree)

    # 3. Build IR from collected info
    ir_builder = IRBuilder()
    shader_ir = ir_builder.build(collected, main_func, kwargs)

    # 4. Emit code using dialect
    emitter = Emitter(dialect)
    return emitter.emit(shader_ir)
```

---

## 7. Benefits of This Architecture

### Easy to Add New Backends

To add WebGPU (WGSL) support:
1. Create `WGSLDialect` class
2. Implement type mappings, operators, entry point format
3. Done - no changes to IR or emitter logic

### Easy to Add New Shader Stages

To add compute shaders:
1. Add `ShaderStage.COMPUTE` (already there)
2. Add `@compute` decorator
3. Update `entry_point_wrapper()` in dialects to handle compute
4. Done - IR already supports workgroup_size

### Easy to Add New Features

To add texture sampling:
1. Add `StorageClass.TEXTURE` (already there)
2. Add `texture[T]` qualifier
3. Add `texture_sample` to builtin functions
4. Dialects map to their texture syntax

### Cleaner Separation of Concerns

- **IR**: Captures shader semantics (what the shader does)
- **Dialect**: Defines syntax rules (how to write it)
- **Emitter**: Mechanical code generation (applying rules)
- **Collector**: Extracts info from Python AST
- **IRBuilder**: Transforms collected info to IR

---

## 8. Migration Path

Phase 1: Add IR alongside existing system
- Create ir.py with all IR types
- Create ir_builder.py to build IR from CollectedInfo
- Keep existing backends working

Phase 2: Add Dialect/Emitter
- Create dialect.py with Dialect ABC
- Create GLSL460Dialect matching current behavior
- Create emitter.py

Phase 3: Switch transpile() to use new system
- Update transpile() to use IRBuilder + Emitter
- Deprecate old backends.py

Phase 4: Add new dialects
- GLSL300esDialect (WebGL 2.0)
- WGSLDialect (WebGPU)
- HLSLDialect (DirectX)

Phase 5: Add new shader stages
- Vertex shaders
- Compute shaders
- Eventually: geometry, tessellation, raytracing
