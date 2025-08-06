#!/bin/bash

# AWS Credentials Refresh Tool
# This script refreshes AWS temporary credentials using STS assume-role
# and saves them to a file that can be used by the SageMaker LLM provider.

# Default values
DEFAULT_REFRESH_INTERVAL=1800  # 30 minutes in seconds
DEFAULT_SESSION_NAME="mlzero-session"
DEFAULT_DURATION_SECONDS=3600  # 1 hour

# Usage information
print_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -r, --role-arn ARN       Required: AWS IAM Role ARN to assume"
    echo "  -o, --output-file FILE   Required: Path to save the credentials"
    echo "  -i, --interval SECONDS   Optional: Refresh interval in seconds (default: $DEFAULT_REFRESH_INTERVAL)"
    echo "  -s, --session-name NAME  Optional: Role session name (default: $DEFAULT_SESSION_NAME)"
    echo "  -d, --duration SECONDS   Optional: Session duration in seconds (default: $DEFAULT_DURATION_SECONDS)"
    echo "  -n, --no-loop            Optional: Run once, don't loop continuously"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --role-arn arn:aws:iam::123456789012:role/ExampleRole --output-file /path/to/credentials.txt"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -r|--role-arn)
            ROLE_ARN="$2"
            shift 2
            ;;
        -o|--output-file)
            CREDS_FILE="$2"
            shift 2
            ;;
        -i|--interval)
            REFRESH_INTERVAL="$2"
            shift 2
            ;;
        -s|--session-name)
            SESSION_NAME="$2"
            shift 2
            ;;
        -d|--duration)
            DURATION_SECONDS="$2"
            shift 2
            ;;
        -n|--no-loop)
            NO_LOOP=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Set defaults for optional parameters
REFRESH_INTERVAL=${REFRESH_INTERVAL:-$DEFAULT_REFRESH_INTERVAL}
SESSION_NAME=${SESSION_NAME:-$DEFAULT_SESSION_NAME}
DURATION_SECONDS=${DURATION_SECONDS:-$DEFAULT_DURATION_SECONDS}

# Check required parameters
if [ -z "$ROLE_ARN" ] || [ -z "$CREDS_FILE" ]; then
    echo "Error: Missing required parameters."
    print_usage
    exit 1
fi

# Check for required tools
for cmd in aws jq; do
    if ! command -v $cmd &> /dev/null; then
        echo "Error: $cmd command not found. Please install it."
        exit 1
    fi
done

# Function to refresh credentials
refresh_credentials() {
    echo "Refreshing AWS credentials at $(date)"
    
    # Clear any existing environment variables to ensure we're using the default credential provider chain
    unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
    
    # Assume the role
    CREDS=$(aws sts assume-role \
        --role-arn "$ROLE_ARN" \
        --role-session-name "$SESSION_NAME" \
        --duration-seconds "$DURATION_SECONDS")
    
    if [ $? -ne 0 ]; then
        echo "Error assuming role. Check permissions and network connectivity."
        return 1
    fi
    
    # Extract credentials
    ACCESS_KEY=$(echo "$CREDS" | jq -r '.Credentials.AccessKeyId')
    SECRET_KEY=$(echo "$CREDS" | jq -r '.Credentials.SecretAccessKey')
    SESSION_TOKEN=$(echo "$CREDS" | jq -r '.Credentials.SessionToken')
    EXPIRATION=$(echo "$CREDS" | jq -r '.Credentials.Expiration')
    
    # Create directory for credentials file if it doesn't exist
    mkdir -p "$(dirname "$CREDS_FILE")"
    
    # Save to credentials file
    echo "{
  \"Credentials\": {
    \"AccessKeyId\": \"$ACCESS_KEY\",
    \"SecretAccessKey\": \"$SECRET_KEY\",
    \"SessionToken\": \"$SESSION_TOKEN\",
    \"Expiration\": \"$EXPIRATION\"
  }
}" > "$CREDS_FILE"
    
    # Make sure only the current user can read the credentials file
    chmod 600 "$CREDS_FILE"
    
    echo "Credentials refreshed successfully. Expiration: $EXPIRATION"
    echo "Credentials saved to: $CREDS_FILE"
    
    # Also set them in the current environment
    export AWS_ACCESS_KEY_ID="$ACCESS_KEY"
    export AWS_SECRET_ACCESS_KEY="$SECRET_KEY" 
    export AWS_SESSION_TOKEN="$SESSION_TOKEN"
}

# Run once if no-loop option is set
if [ "$NO_LOOP" = true ]; then
    refresh_credentials
    exit $?
fi

# Main loop for continuous refresh
echo "Starting credential refresh process. Press Ctrl+C to stop."
echo "Refresh interval: $REFRESH_INTERVAL seconds"

while true; do
    refresh_credentials
    
    # Sleep for the refresh interval
    echo "Next refresh in $REFRESH_INTERVAL seconds (at $(date -d "+$REFRESH_INTERVAL seconds"))"
    sleep $REFRESH_INTERVAL
done
