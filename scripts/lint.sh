#!/bin/bash
# Run code quality checks

echo "Running flake8..."
uv run flake8 backend/ main.py --max-line-length=88 --extend-ignore=E203,W503

echo "Running mypy..."
uv run mypy backend/ main.py

echo "Code quality checks complete!"