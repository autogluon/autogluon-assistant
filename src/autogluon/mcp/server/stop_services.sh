#!/bin/bash

# Stop services script for AutoGluon Assistant MCP Server

echo "=== Stopping AutoGluon Assistant Services ==="
echo

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to kill process by port
kill_by_port() {
    local port=$1
    local service_name=$2
    
    echo -n "Checking $service_name on port $port... "
    
    # Get PID using port
    if command -v lsof &> /dev/null; then
        local pid=$(lsof -ti:$port)
    else
        local pid=$(netstat -tlnp 2>/dev/null | grep ":$port " | awk '{print $7}' | cut -d'/' -f1)
    fi
    
    if [ -n "$pid" ]; then
        echo -e "${YELLOW}Found process $pid${NC}"
        echo -n "Stopping $service_name (PID: $pid)... "
        if kill $pid 2>/dev/null; then
            echo -e "${GREEN}✓${NC}"
            # Wait a moment for process to terminate
            sleep 1
        else
            echo -e "${RED}Failed, trying force kill${NC}"
            kill -9 $pid 2>/dev/null
        fi
    else
        echo -e "${GREEN}Not running${NC}"
    fi
}

# Stop MCP Server by PID file or port
if [ -f .mcp.pid ]; then
    MCP_PID=$(cat .mcp.pid)
    echo -n "Stopping MCP Server (PID: $MCP_PID)... "
    if kill $MCP_PID 2>/dev/null; then
        echo -e "${GREEN}✓${NC}"
        rm .mcp.pid
    else
        echo -e "${RED}Process not found${NC}"
        rm .mcp.pid
        # Try to stop by port
        kill_by_port 8000 "MCP Server"
    fi
else
    # No PID file, try to stop by port
    kill_by_port 8000 "MCP Server"
fi

# Stop Flask Backend by PID file or port
if [ -f .flask.pid ]; then
    FLASK_PID=$(cat .flask.pid)
    echo -n "Stopping Flask Backend (PID: $FLASK_PID)... "
    if kill $FLASK_PID 2>/dev/null; then
        echo -e "${GREEN}✓${NC}"
        rm .flask.pid
    else
        echo -e "${RED}Process not found${NC}"
        rm .flask.pid
        # Try to stop by port
        kill_by_port 5000 "Flask Backend"
    fi
else
    # Check if Flask is running on port 5000
    echo -e "${YELLOW}Flask Backend may be running separately${NC}"
    kill_by_port 5000 "Flask Backend"
fi

# Also check for any remaining python processes running our scripts
echo
echo "Checking for remaining processes..."

# Kill any remaining server.py processes
pkill -f "server.py" 2>/dev/null && echo -e "${GREEN}Killed remaining server.py processes${NC}"

# Kill any remaining app.py processes (Flask)
pkill -f "webui/backend/app.py" 2>/dev/null && echo -e "${GREEN}Killed remaining Flask processes${NC}"

# Final check
echo
echo "Final port check..."
PORTS_IN_USE=false

# Check port 8000 (MCP)
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${RED}Warning: Port 8000 still in use${NC}"
    echo "You may need to manually kill the process:"
    echo "  lsof -ti:8000 | xargs kill -9"
    PORTS_IN_USE=true
else
    echo -e "${GREEN}✓ Port 8000 is free${NC}"
fi

# Check port 5000 (Flask)
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${RED}Warning: Port 5000 still in use${NC}"
    echo "This might be another Flask application"
    PORTS_IN_USE=true
else
    echo -e "${GREEN}✓ Port 5000 is free${NC}"
fi

echo
if [ "$PORTS_IN_USE" = true ]; then
    echo -e "${YELLOW}Some services may still be running. Use the commands above to force stop them.${NC}"
else
    echo -e "${GREEN}All services stopped successfully.${NC}"
fi
