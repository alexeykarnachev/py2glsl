.PHONY: all format typing test

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
