.PHONY: all format typing test gold-generate examples

# Default: run everything
all: format typing test

# Format and lint
format:
	ruff check --fix py2glsl tests examples
	ruff format py2glsl tests examples

# Type check with mypy
typing:
	mypy py2glsl

# Run tests
test:
	pytest tests -v

# Generate/update gold file expected outputs and validate GLSL compiles
gold-generate:
	pytest tests/test_gold.py --generate

# Generate all outputs for each example shader
examples:
	@for f in examples/*.py; do \
		name=$$(basename "$$f" .py); \
		dir="examples/outputs/$$name"; \
		echo "========================================"; \
		echo "Processing: $$name"; \
		echo "========================================"; \
		mkdir -p "$$dir"; \
		\
		echo "  Rendering image..."; \
		py2glsl image "$$f" "$$dir/image.png" --width 800 --height 600 || echo "    Failed: image"; \
		\
		echo "  Rendering GIF (3s)..."; \
		py2glsl gif "$$f" "$$dir/animation.gif" --width 400 --height 300 --duration 3 --fps 30 || echo "    Failed: gif"; \
		\
		echo "  Rendering video (5s)..."; \
		py2glsl video "$$f" "$$dir/video.mp4" --width 800 --height 600 --duration 5 --fps 30 || echo "    Failed: video"; \
		\
		echo "  Exporting GLSL (OpenGL 4.6)..."; \
		py2glsl export "$$f" "$$dir/shader.opengl46.glsl" --target glsl || echo "    Failed: glsl"; \
		\
		echo "  Exporting GLSL (OpenGL 3.3)..."; \
		py2glsl export "$$f" "$$dir/shader.opengl33.glsl" --target opengl33 || echo "    Failed: opengl33"; \
		\
		echo "  Exporting GLSL (WebGL 2.0)..."; \
		py2glsl export "$$f" "$$dir/shader.webgl2.glsl" --target webgl || echo "    Failed: webgl"; \
		\
		echo "  Exporting Shadertoy..."; \
		py2glsl export "$$f" "$$dir/shader.shadertoy.glsl" --target shadertoy || echo "    Failed: shadertoy"; \
		\
		echo "  Done: $$dir/"; \
		echo ""; \
	done
	@echo "========================================"
	@echo "All examples processed!"
	@echo "Outputs saved to examples/outputs/<name>/"
	@echo "========================================"
