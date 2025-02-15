from py2glsl import animate, vec2, vec4, py2glsl
from py2glsl.builtins import sin, length

def plasma(vs_uv: vec2, *, u_time: float) -> vec4:
    # Center and scale UV coordinates
    uv = vs_uv * 2.0 - 1.0
    
    # Create animated plasma effect
    d = length(uv)
    color = sin(d * 10.0 - u_time * 2.0) * 0.5 + 0.5
    
    return vec4(color, color * 0.5, 1.0 - color, 1.0)

# Print generated GLSL code
result = py2glsl(plasma)
print("Fragment Shader:")
print(result.fragment_source)

# Display animation in real-time window
animate(plasma)

