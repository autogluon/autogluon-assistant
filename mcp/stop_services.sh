#!/bin/bash

# Stop services script for AutoGluon Assistant MCP Server

echo "=== Stopping AutoGluon Assistant Services ==="
echo

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

# Stop MCP Server
if [ -f .mcp.pid ]; then
    MCP_PID=$(cat .mcp.pid)
    echo -n "Stopping MCP Server (PID: $MCP_PID)... "
    if kill $MCP_PID 2>/dev/null; then
        echo -e "${GREEN}✓${NC}"
        rm .mcp.pid
    else
        echo -e "${RED}Process not found${NC}"
        rm .mcp.pid
    fi
else
    echo "MCP Server PID file not found"
fi

# Stop Flask Backend (if we started it)
if [ -f .flask.pid ]; then
    FLASK_PID=$(cat .flask.pid)
    echo -n "Stopping Flask Backend (PID: $FLASK_PID)... "
    if kill $FLASK_PID 2>/dev/null; then
        echo -e "${GREEN}✓${NC}"
        rm .flask.pid
    else
        echo -e "${RED}Process not found${NC}"
        rm .flask.pid
    fi
else
    echo "Flask Backend PID file not found (may have been started separately)"
fi

# Check for any remaining processes
echo
echo "Checking for remaining processes..."

# Check port 8000 (MCP)
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${RED}Warning: Port 8000 still in use${NC}"
    echo "You may need to manually kill the process:"
    echo "  lsof -ti:8000 | xargs kill -9"
fi

# Check port 5000 (Flask)
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${RED}Warning: Port 5000 still in use${NC}"
    echo "Flask backend may still be running (started separately)"
fi

echo
echo "Services stopped."
