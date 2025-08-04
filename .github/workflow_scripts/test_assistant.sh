#!/bin/bash

MODULE=$1

set -ex

source $(dirname "$0")/env_setup.sh

install_all
setup_test_env

# Handle MCP module specially
if [ "$MODULE" = "mcp" ]; then
    echo "Setting up MCP test environment..."
    
    # Install MCP specific dependencies
    python -m pip install fastmcp aiohttp
    
    # Stop existing services to free up ports
    echo "Stopping existing services..."
    sudo lsof -ti:5000 | xargs -r sudo kill -9 || true
    pkill -f "mlzero-" || true
    pkill -f "autogluon.assistant.webui.backend" || true
    
    # Run MCP integration test
    python -m pytest -n 1 -vv -s --capture=tee-sys --log-cli-level=INFO tests/unittests/mcp/test_mcp_integration.py
else
    # Run standard unit tests for other modules
    python -m pytest -n 2 -vv -s --capture=tee-sys --log-cli-level=INFO --junitxml=results.xml tests/unittests/$MODULE/
fi
