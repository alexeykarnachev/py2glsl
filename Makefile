.PHONY: format lint check test all

# Format code with ruff
format:
	ruff check --fix py2glsl tests examples
	ruff format py2glsl tests examples

# Lint without fixing
lint:
	ruff check py2glsl tests examples
	ruff format --check py2glsl tests examples

# Type check with mypy
typecheck:
	mypy py2glsl

# Run all checks (lint + typecheck)
check: lint typecheck

# Run tests
test:
	pytest tests -v

# Run everything (format, then check, then test)
all: format check test
