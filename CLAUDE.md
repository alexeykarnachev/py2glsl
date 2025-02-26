# py2glsl Development Guide

## Build Commands
- Test: `pytest`
- Single test: `pytest tests/path/to/test_file.py::TestClass::test_function -v`
- Type check: `mypy py2glsl`
- Lint: `ruff check py2glsl tests`
- Format: `ruff format py2glsl tests`
- Install dev dependencies: `uv sync`
- Install pre-commit hooks: `pre-commit install`

## Code Style
- Use double quotes for strings
- Type annotations required on all functions and variables
- Follow PEP 8 naming: snake_case for variables/functions, PascalCase for classes
- Maximum line length: 88 characters
- Use dataclasses for data structures
- Comprehensive error handling with custom exceptions
- Use type hints from `typing` module
- Docstrings for all modules, classes and functions

## Project Structure
- `/py2glsl`: Main package with transpiler modules
- `/tests`: Test modules mirroring structure of main package
- `/examples`: Example Python shader files