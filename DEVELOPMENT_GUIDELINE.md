# Ultimate Implementation Guideline for py2glsl

## 1. Core Architecture

```python
# Core type system (types.py)
class glsl_type:
    """Base for all GLSL types with common operations"""
    def __add__(self, other): ...
    def __mul__(self, other): ...
    # etc.

class vec2(glsl_type):
    x: float
    y: float
    
    @property
    def xy(self) -> 'vec2': ...
    @property
    def yx(self) -> 'vec2': ...

# Similar for vec3, vec4
```

## 2. Implementation Phases

### Phase 1: Foundation
1. Basic type system
2. AST analysis
3. Simple GLSL generation
4. Core math functions

### Phase 2: Features
1. Control flow translation
2. Function lifting
3. Type inference
4. Error handling

### Phase 3: Advanced
1. Optimizations
2. Debug support
3. Tools integration
4. Documentation

## 3. Critical Components

```python
# AST transformation (transpiler.py)
class ShaderTranspiler:
    def analyze(self, func) -> ShaderAnalysis:
        """Extract shader structure and validate"""
    
    def transform(self, analysis: ShaderAnalysis) -> GLSLCode:
        """Convert to GLSL"""

# Type handling (types.py)
class TypeRegistry:
    """Manages GLSL type system"""
    def resolve_type(self, python_type) -> GLSLType: ...
    def validate_operation(self, op, *types) -> GLSLType: ...

# Code generation (codegen.py)
class GLSLGenerator:
    """Generates final GLSL code"""
    def generate_uniforms(self) -> str: ...
    def generate_functions(self) -> str: ...
    def generate_main(self) -> str: ...
```

## 4. Key Rules

### Type System
```python
# Must handle:
1. Type promotion (float -> vec2 -> vec3 -> vec4)
2. Swizzling (both read/write)
3. Operator overloading
4. Type validation
5. Precision rules
```

### AST Transformation
```python
# Must handle:
1. Function lifting
2. Control flow conversion
3. Variable scoping
4. Name mangling
5. Type inference
```

### Error Handling
```python
# Must provide:
1. Clear error messages
2. Source location
3. Type mismatch details
4. Performance warnings
5. Validation errors
```

## 5. Testing Strategy

```python
# test_types.py
def test_vector_operations():
    """Test all vector operations"""

# test_transpiler.py
def test_shader_conversion():
    """Test Python -> GLSL conversion"""

# test_integration.py
def test_real_shaders():
    """Test complete shader programs"""
```

## 6. Development Guidelines

### Code Style
```python
1. Use type hints everywhere
2. Document all public APIs
3. Keep functions focused
4. Use clear naming
5. Add detailed comments
```

### Performance
```python
1. Optimize AST traversal
2. Cache type information
3. Minimize string operations
4. Use efficient data structures
5. Profile critical paths
```

### Safety
```python
1. Validate all inputs
2. Check resource limits
3. Prevent infinite loops
4. Handle edge cases
5. Sanitize names
```

## 7. Project Structure

```
py2glsl/
├── __init__.py
├── types.py          # GLSL types
├── builtins.py       # Built-in functions
├── transpiler.py     # Main conversion logic
├── analyzer.py       # AST analysis
├── codegen.py        # GLSL generation
├── optimizer.py      # Optimization passes
├── errors.py         # Error types
└── utils.py          # Utilities

tests/
├── test_types.py
├── test_transpiler.py
├── test_shaders.py
└── test_integration.py
```

## 8. Implementation Order

1. Core Types
```python
# Implement in this order:
1. vec2 with basic ops
2. vec3 with swizzling
3. vec4 with type promotion
4. Basic math functions
```

2. AST Processing
```python
# Implement in this order:
1. Function analysis
2. Type inference
3. Validation
4. Transformation
```

3. Code Generation
```python
# Implement in this order:
1. Basic structure
2. Function generation
3. Control flow
4. Optimization
```

## 9. Additional Critical Points

### GLSL Version Handling
```python
# Modern GLSL focus
GLSL_VERSION = 460  # Using modern GLSL 4.60
OUTPUT_FORMAT = """
#version 460
{declarations}
{functions}
{shader_code}
"""
```

### Shader IO Convention
```python
# Fixed conventions
IN_UV_NAME = "vs_uv"      # Always use vs_uv for input
OUT_COLOR = "fs_color"    # Always use fs_color for output
```

### Function Requirements
```python
def validate_shader_function(func):
    """
    Rules:
    1. First arg must be vs_uv: vec2
    2. All other args must be uniforms (marked with *)
    3. Must return vec4
    4. No decorators needed - pure functions
    """
```

### Coordinate Spaces
```python
# Common space conversion functions
def uv_to_sp(uv: vec2) -> vec2:
    """UV [0,1] to screen [-1,1] space"""
    return uv * 2.0 - 1.0

# Let users handle their own coordinate transforms
# Don't auto-inject resolution adjustments
```

### Render Pipeline Integration
```python
def render_frame(
    shader_func,
    **uniforms
) -> None:
    """Single frame render"""

def animate(
    shader_func,
    fps: float = 60.0,
    **uniforms
) -> None:
    """Continuous animation"""
```

### Helper Functions
```python
# Common shader operations
def sdf_ops():
    """Signed Distance Field operations"""
    union = min
    intersection = max
    smooth_union = smooth_min

def noise():
    """Noise functions"""
    value_noise = ...
    perlin_noise = ...
    simplex_noise = ...
```

### Debug Support
```python
# Development tools
def print_glsl(shader_func) -> str:
    """Print generated GLSL with formatting"""

def validate_glsl(shader_func) -> list[str]:
    """Return validation errors"""

def preview_shader(shader_func, size=(512, 512)) -> None:
    """Open preview window"""
```

### Performance Guidelines
```python
PERFORMANCE_RULES = """
1. Avoid dynamic loops
2. Minimize branching
3. Watch precision requirements
4. Avoid dynamic indexing
5. Keep functions small
6. Reuse computed values
"""
```
