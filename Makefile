.PHONY: all format typing test examples

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

# Render all examples to images
examples:
	@mkdir -p examples/outputs
	@for f in examples/*.py; do \
		name=$$(basename "$$f" .py); \
		echo "Rendering $$name..."; \
		py2glsl image "$$f" "examples/outputs/$$name.png" --width 800 --height 600 || echo "  Failed: $$name"; \
	done
	@echo "Done. Outputs saved to examples/outputs/"
