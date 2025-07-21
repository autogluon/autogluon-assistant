#!/bin/bash

# Start services script for AutoGluon Assistant MCP Server

set -e

echo "=== AutoGluon Assistant MCP Server Startup ==="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Using ports: Flask=$FLASK_PORT, MCP=$MCP_PORT"
echo

# Check if Flask backend is running
check_flask_backend() {
    echo -n "Checking Flask backend on port $FLASK_PORT... "
    if curl -s http://localhost:$FLASK_PORT/api/queue/info > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Running${NC}"
        return 0
    else
        echo -e "${RED}✗ Not running${NC}"
        return 1
    fi
}

# Start Flask backend
start_flask_backend() {
    echo "Starting Flask backend..."
    cd ../../../assistant/webui/backend
    
    # Start in background
    nohup python app.py > flask_backend.log 2>&1 &
    FLASK_PID=$!
    echo "Flask backend PID: $FLASK_PID"
    
    # Wait for it to start
    sleep 3
    
    if check_flask_backend; then
        echo -e "${GREEN}Flask backend started successfully${NC}"
        return 0
    else
        echo -e "${RED}Failed to start Flask backend${NC}"
        echo "Check flask_backend.log for details"
        return 1
    fi
}

# Start MCP Server
start_mcp_server() {
    echo "Starting MCP Server..."
    
    # Check if fastmcp is installed
    if ! python -c "import fastmcp" 2>/dev/null; then
        echo -e "${YELLOW}FastMCP not installed. Installing...${NC}"
        pip install fastmcp aiohttp
    fi
    
    python "$(dirname "$0")/server.py" --port $MCP_PORT &
    MCP_PID=$!
    echo "MCP Server PID: $MCP_PID"
    
    # Wait for it to start
    sleep 2
    
    # Check if running
    if curl -s http://localhost:$MCP_PORT > /dev/null 2>&1; then
        echo -e "${GREEN}MCP Server started successfully${NC}"
        
        # Save PIDs for shutdown
        echo $FLASK_PID > .flask.pid
        echo $MCP_PID > .mcp.pid
        
        # Also save ports for stop script
        echo $FLASK_PORT > .flask.port
        echo $MCP_PORT > .mcp.port
        
        return 0
    else
        echo -e "${RED}Failed to start MCP Server${NC}"
        kill $MCP_PID 2>/dev/null
        return 1
    fi
}

# Main startup sequence
main() {
    # Check/start Flask backend
    if ! check_flask_backend; then
        echo -e "${YELLOW}Flask backend not running. Attempting to start...${NC}"
        if ! start_flask_backend; then
            echo -e "${RED}Cannot proceed without Flask backend${NC}"
            exit 1
        fi
    fi
    
    # Start MCP Server
    if ! start_mcp_server; then
        echo -e "${RED}Failed to start MCP Server${NC}"
        exit 1
    fi
    
    echo
    echo -e "${GREEN}=== All services started successfully! ===${NC}"
    echo
    echo "Services running:"
    echo "  - Flask Backend: http://localhost:$FLASK_PORT"
    echo "  - MCP Server: http://localhost:$MCP_PORT/mcp"
    
    echo
    echo "To stop services, run: ./stop_services.sh"
    echo
    echo "Logs:"
    echo "  - Flask: ../src/autogluon/assistant/webui/backend/flask_backend.log"
    echo "  - MCP: Check terminal output or nohup.out"
}

# Run main
main
