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

# Parse arguments
REMOTE_MODE=false
for arg in "$@"; do
    case $arg in
        --remote)
            REMOTE_MODE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--remote]"
            echo "  --remote  Start server in remote mode (listen on all interfaces)"
            exit 0
            ;;
    esac
done

# Check if Flask backend is running
check_flask_backend() {
    echo -n "Checking Flask backend on port 5000... "
    if curl -s http://localhost:5000/api/queue/info > /dev/null 2>&1; then
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
    cd ../src/autogluon/assistant/webui/backend
    
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
    
    # Start MCP server
    if [ "$REMOTE_MODE" = true ]; then
        echo "Starting MCP server in REMOTE mode (listening on 0.0.0.0:8000)..."
        echo -e "${YELLOW}WARNING: Server will be accessible from any IP address${NC}"
        echo -e "${YELLOW}Make sure your EC2 security group is properly configured${NC}"
    else
        echo "Starting MCP server in LOCAL mode (listening on 127.0.0.1:8000)..."
    fi
    
    python ./mcp/server.py &
    MCP_PID=$!
    echo "MCP Server PID: $MCP_PID"
    
    # Wait for it to start
    sleep 2
    
    # Check if running
    if curl -s http://localhost:8000 > /dev/null 2>&1; then
        echo -e "${GREEN}MCP Server started successfully${NC}"
        
        # Save PIDs for shutdown
        echo $FLASK_PID > .flask.pid
        echo $MCP_PID > .mcp.pid
        
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
    echo "  - Flask Backend: http://localhost:5000"
    
    if [ "$REMOTE_MODE" = true ]; then
        # Get EC2 public IP
        PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "your-server-ip")
        echo "  - MCP Server: http://$PUBLIC_IP:8000 (remote access)"
        echo
        echo "Connect from your local machine:"
        echo "  python client_example.py /local/data /local/output --server http://$PUBLIC_IP:8000"
    else
        echo "  - MCP Server: http://localhost:8000 (local only)"
        echo
        echo "To test the setup, run:"
        echo "  cd examples"
        echo "  python client_example.py /path/to/data /path/to/output"
    fi
    
    echo
    echo "To stop services, run: ./stop_services.sh"
    echo
    echo "Logs:"
    echo "  - Flask: ../src/autogluon/assistant/webui/backend/flask_backend.log"
    echo "  - MCP: Check terminal output or nohup.out"
}

# Run main
main
