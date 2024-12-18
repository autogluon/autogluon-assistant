#!/bin/bash

# Check if the script is being sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "This script must be sourced. Please run:"
    echo "source ${0}"
    exit 1
fi

# Colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Configuration file path
CONFIG_FILE="$HOME/.llm_config"

# Initialize temporary variables
tmp_AWS_DEFAULT_REGION=""
tmp_AWS_ACCESS_KEY_ID=""
tmp_AWS_SECRET_ACCESS_KEY=""
tmp_OPENAI_API_KEY=""
tmp_OPENAI_API_VERSION=""
tmp_AZURE_OPENAI_API_KEY=""
tmp_AZURE_OPENAI_ENDPOINT=""

# Function to print colored messages
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to print section header
print_header() {
    local message=$1
    echo
    print_color "$BLUE" "=== $message ==="
    echo
}

# Function to validate AWS region
validate_aws_region() {
    local region=$1
    # List of valid AWS regions
    # TODO: Raise warning if bedrock is not supported or is gated in region
    local valid_regions=("us-east-1" "us-east-2" "us-west-1" "us-west-2" "eu-west-1" "eu-central-1" "ap-southeast-1" "ap-southeast-2" "ap-northeast-1")
    
    for valid_region in "${valid_regions[@]}"; do
        if [ "$region" = "$valid_region" ]; then
            return 0
        fi
    done
    return 1
}

# Function to validate API keys
validate_openai_api_key() {
    local key=$1
    [[ $key =~ ^sk-[A-Za-z0-9_-]+$ ]] && return 0
    return 1
}

# Function to read existing configuration into temporary variables
read_existing_config() {
    if [ -f "$CONFIG_FILE" ]; then
        while IFS='=' read -r key value; do
            if [ -n "$key" ] && [ -n "$value" ]; then
                case "$key" in
                    "AWS_DEFAULT_REGION") tmp_AWS_DEFAULT_REGION="$value" ;;
                    "AWS_ACCESS_KEY_ID") tmp_AWS_ACCESS_KEY_ID="$value" ;;
                    "AWS_SECRET_ACCESS_KEY") tmp_AWS_SECRET_ACCESS_KEY="$value" ;;
                    "OPENAI_API_KEY") tmp_OPENAI_API_KEY="$value" ;;
                    "OPENAI_API_VERSION") tmp_OPENAI_API_VERSION="$value" ;;
                    "AZURE_OPENAI_API_KEY") tmp_AZURE_OPENAI_API_KEY="$value" ;;
                    "AZURE_OPENAI_ENDPOINT") tmp_AZURE_OPENAI_ENDPOINT="$value" ;;
                esac
            fi
        done < "$CONFIG_FILE"
    fi
}

# Function to save configuration and export variables
save_configuration() {
    local provider=$1
    
    # Create or truncate the config file
    echo "" > "$CONFIG_FILE" || { print_color "$RED" "Error: Cannot write to '$CONFIG_FILE'"; return 1; }
    
    if [ "$provider" = "bedrock" ]; then
        # Update AWS variables
        tmp_AWS_DEFAULT_REGION="$AWS_DEFAULT_REGION"
        tmp_AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID"
        tmp_AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY"
    elif [ "$provider" = "openai" ]; then
        # Update OpenAI variables
        tmp_OPENAI_API_KEY="$OPENAI_API_KEY"
        tmp_OPENAI_API_VERSION="$OPENAI_API_VERSION"
    elif [ "$provider" = "azure" ]; then
        # Update Azure variables
        tmp_AZURE_OPENAI_API_KEY="$AZURE_OPENAI_API_KEY"
        tmp_AZURE_OPENAI_ENDPOINT="$AZURE_OPENAI_ENDPOINT"
    fi
    
    # Save all configurations
    if [ -n "$tmp_AWS_ACCESS_KEY_ID" ]; then
        echo "AWS_DEFAULT_REGION=$tmp_AWS_DEFAULT_REGION" >> "$CONFIG_FILE"
        echo "AWS_ACCESS_KEY_ID=$tmp_AWS_ACCESS_KEY_ID" >> "$CONFIG_FILE"
        echo "AWS_SECRET_ACCESS_KEY=$tmp_AWS_SECRET_ACCESS_KEY" >> "$CONFIG_FILE"
    fi
    
    if [ -n "$tmp_OPENAI_API_KEY" ]; then
        echo "OPENAI_API_KEY=$tmp_OPENAI_API_KEY" >> "$CONFIG_FILE"
        echo "OPENAI_API_VERSION=$tmp_OPENAI_API_VERSION" >> "$CONFIG_FILE"
    fi

    if [ -n "$tmp_AZURE_OPENAI_API_KEY" ]; then
        echo "AZURE_OPENAI_API_KEY=$tmp_AZURE_OPENAI_API_KEY" >> "$CONFIG_FILE"
        echo "AZURE_OPENAI_ENDPOINT=$tmp_AZURE_OPENAI_ENDPOINT" >> "$CONFIG_FILE"
    fi
    
    # Export all variables
    if [ -n "$tmp_AWS_ACCESS_KEY_ID" ]; then
        export AWS_DEFAULT_REGION="$tmp_AWS_DEFAULT_REGION"
        export AWS_ACCESS_KEY_ID="$tmp_AWS_ACCESS_KEY_ID"
        export AWS_SECRET_ACCESS_KEY="$tmp_AWS_SECRET_ACCESS_KEY"
    fi
    
    if [ -n "$tmp_OPENAI_API_KEY" ]; then
        export OPENAI_API_KEY="$tmp_OPENAI_API_KEY"
        export OPENAI_API_VERSION="$tmp_OPENAI_API_VERSION"
    fi

    if [ -n "$tmp_AZURE_OPENAI_API_KEY" ]; then
        export AZURE_OPENAI_API_KEY="$tmp_AZURE_OPENAI_API_KEY"
        export AZURE_OPENAI_ENDPOINT="$tmp_AZURE_OPENAI_ENDPOINT"
    fi
    
    # Set proper permissions
    chmod 600 "$CONFIG_FILE"
    
    print_color "$GREEN" "Configuration saved to $CONFIG_FILE"
    print_color "$GREEN" "Variables have been exported in the current session"
}

# Function to check and backup existing configuration
check_existing_config() {
    if [ -f "$CONFIG_FILE" ]; then
        print_color "$BLUE" "Existing configuration found. Creating backup..."
        cp "$CONFIG_FILE" "${CONFIG_FILE}.backup"
        print_color "$GREEN" "Backup created at ${CONFIG_FILE}.backup"
    fi
}