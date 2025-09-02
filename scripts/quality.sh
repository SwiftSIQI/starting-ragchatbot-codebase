#!/bin/bash
# Run all code quality checks and formatting

echo "=== Running code quality checks ==="

echo "1. Formatting code..."
./scripts/format.sh

echo "2. Running linters..."
./scripts/lint.sh

echo "3. Running tests..."
uv run pytest

echo "=== All quality checks complete ==="