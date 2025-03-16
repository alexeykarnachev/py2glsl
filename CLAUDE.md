# py2glsl Development Guide

## Build Commands
- Test: `pytest`
- Single test: `pytest tests/path/to/test_file.py::TestClass::test_function -v`
- Type check: `mypy py2glsl`
- Lint: `ruff check py2glsl tests`
- Format: `ruff format py2glsl tests`
- Install dev dependencies: `uv sync`
- Install pre-commit hooks: `pre-commit install`

## Pre-commit Hooks
- Install: `pre-commit install`
- Run manually: `pre-commit run --all-files`
- Hooks include: ruff (linting), ruff-format (formatting), mypy (type checking)

## Code Style
- Use double quotes for strings
- Type annotations required on all functions and variables
- Follow PEP 8 naming: snake_case for variables/functions, PascalCase for classes
- IMPORTANT: GLSL-compatible types must use GLSL naming (vec2, vec3, vec4, mat2, etc.) NOT PascalCase
- Maximum line length: 88 characters
- Use dataclasses for data structures
- Comprehensive error handling with custom exceptions
- Use type hints from `typing` module
- Docstrings for all modules, classes and functions

## Project Structure
- `/py2glsl`: Main package with transpiler modules
- `/tests`: Test modules mirroring structure of main package
- `/examples`: Example Python shader files

## Code Generation Notes
- Pass statements are translated to GLSL comments (`// Pass statement (no-op)`)
- For loops over lists need size variable specified in globals (e.g., `list_name_size`)
- Augmented assignments (e.g., `a += b`) are converted to regular assignment form (`a = a + b`)
- GLSL doesn't support multiple assignment targets, each must be on a separate line
- Ensure test expectations match actual generated code structure and formatting

## Testing Best Practices
- After all unit tests pass, also run an example file to verify end-to-end functionality:
  ```bash
  python examples/001_ray_marching.py
  ```
- This ensures that changes don't break the actual shader generation and rendering
- Always address root causes methodically rather than implementing quick fixes, and verify all affected parts of the codebase

## GPU Testing
- GPU tests (`@pytest.mark.gpu`) run by default as part of the regular test suite
- To skip GPU tests (e.g., in environments without a GPU): `pytest -k "not gpu"` or set `NO_GPU=1 pytest`
- When adding GPU-intensive tests:
  1. Use the `@pytest.mark.gpu` decorator to mark tests that require a GPU
  2. Use isolated test directories to avoid file conflicts
  3. Implement retry logic for stability
  4. Clean up OpenGL resources properly to avoid context issues
  5. Always check for `HAS_GPU` condition before running GPU-intensive operations in tests