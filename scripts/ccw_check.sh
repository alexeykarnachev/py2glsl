#!/bin/bash
set -e

echo "🔍 Running complete verification..."

echo "├─ Running ruff..."
ruff check \
    py2glsl/ \
    tests/ \
    scripts/ \
    --select F,E,W,I,N,UP,B,A,C4,SIM,ERA,PL,RUF \
    --fix

echo "├─ Running mypy..."
mypy --strict --ignore-missing-imports py2glsl/ tests/ scripts/

echo "└─ Running tests..."
pytest -v tests/

echo "✨ All checks passed!"
