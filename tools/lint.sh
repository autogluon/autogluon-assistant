#!/bin/bash
# Get the directory where the script is located and its parent directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
echo "Formatting Python files in $PARENT_DIR (excluding maab folder)..."

# Format with black
echo "Running black..."
black "$PARENT_DIR" --exclude "maab/|runs/"

# Run ruff with auto-fix
echo "Running ruff..."
ruff check --fix "$PARENT_DIR" --exclude "maab" --exclude "runs"

# Sort imports
echo "Running isort..."
isort "$PARENT_DIR" --skip-glob="*/maab/*" --skip-glob="*/runs/*"

echo "Done! All Python files in $PARENT_DIR have been formatted (excluding maab folder)."
