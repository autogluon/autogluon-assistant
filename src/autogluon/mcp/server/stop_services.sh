#!/bin/bash

# Stop services script for AutoGluon Assistant MCP Server

echo "=== Stopping AutoGluon Assistant Services ==="
echo

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default ports
FLASK_PORT=5000
MCP_PORT=8000

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --flask-port|-f)
            FLASK_PORT="$2"
            shift 2
            ;;
        --server-port|-s)
            MCP_PORT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --flask-port, -f PORT    Flask backend port (default: 5000)"
            echo "  --server-port, -s PORT   MCP server port (default: 8000)"
            echo "  --help, -h               Show this help message"
            echo ""
            echo "Note: If services were started with custom ports, use the same ports here"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Try to read ports from saved files if they exist
if [ -f .flask.port ]; then
    SAVED_FLASK_PORT=$(cat .flask.port)
    if [ -z "$FLASK_PORT_SPECIFIED" ]; then
        FLASK_PORT=$SAVED_FLASK_PORT
        echo "Using saved Flask port: $FLASK_PORT"
    fi
fi

if [ -f .mcp.port ]; then
    SAVED_MCP_PORT=$(cat .mcp.port)
    if [ -z "$MCP_PORT_SPECIFIED" ]; then
        MCP_PORT=$SAVED_MCP_PORT
        echo "Using saved MCP port: $MCP_PORT"
    fi
fi

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
        kill_by_port $MCP_PORT "MCP Server"
    fi
else
    # No PID file, try to stop by port
    kill_by_port $MCP_PORT "MCP Server"
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
        kill_by_port $FLASK_PORT "Flask Backend"
    fi
else
    # Check if Flask is running on port
    echo -e "${YELLOW}Flask Backend may be running separately${NC}"
    kill_by_port $FLASK_PORT "Flask Backend"
fi

# Clean up port files
rm -f .flask.port .mcp.port

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

# Check MCP port
if lsof -Pi :$MCP_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${RED}Warning: Port $MCP_PORT still in use${NC}"
    echo "You may need to manually kill the process:"
    echo "  lsof -ti:$MCP_PORT | xargs kill -9"
    PORTS_IN_USE=true
else
    echo -e "${GREEN}✓ Port $MCP_PORT is free${NC}"
fi

# Check Flask port
if lsof -Pi :$FLASK_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${RED}Warning: Port $FLASK_PORT still in use${NC}"
    echo "This might be another Flask application"
    PORTS_IN_USE=true
else
    echo -e "${GREEN}✓ Port $FLASK_PORT is free${NC}"
fi

echo
if [ "$PORTS_IN_USE" = true ]; then
    echo -e "${YELLOW}Some services may still be running. Use the commands above to force stop them.${NC}"
else
    echo -e "${GREEN}All services stopped successfully.${NC}"
fi
