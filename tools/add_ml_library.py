#!/usr/bin/env python3
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

from autogluon.assistant.tools_registry import ToolsRegistry


def get_user_input(prompt: str, required: bool = True, default: str = None) -> str:
    """Get user input with optional default value."""
    if default:
        prompt = f"{prompt} [{default}]: "
    else:
        prompt = f"{prompt}: "

    while True:
        value = input(prompt).strip()
        if value:
            return value
        if default is not None:
            return default
        if not required:
            return ""
        print("This field is required. Please provide a value.")


def get_list_input(prompt: str, required: bool = False) -> list:
    """Get a list of items from user input."""
    print(f"\n{prompt} (Enter empty line to finish)")
    items = []
    while True:
        item = input("- ").strip()
        if not item:
            if not items and required:
                print("At least one item is required.")
                continue
            break
        items.append(item)
    return items


def get_llm_config() -> dict:
    """Get LLM configuration from YAML config file."""
    config_path = get_user_input("Path to LLM config file (YAML)")

    with open(config_path, "r") as f:
        config = OmegaConf.load(f)
    return config.llm


def create_empty_catalog(registry: ToolsRegistry):
    """Create an empty catalog.json file with proper structure."""
    with open(registry.catalog_path, "w") as f:
        json.dump({"tools": {}}, f, indent=2)
    return registry.catalog_path


def register_tool_interactive():
    """Interactive function to register a new ML tool."""
    print("\n=== ML Tool Registration ===\n")

    # Initialize ToolsRegistry
    registry = ToolsRegistry()

    # Get existing tools and ensure catalog exists with proper structure
    try:
        existing_tools = registry.list_tools()
        if existing_tools:
            print("Existing tools:", ", ".join(existing_tools))
            print()
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        # Create or fix catalog.json if it doesn't exist or is invalid
        catalog_path = create_empty_catalog(registry)
        print(f"Created new tools catalog at: {catalog_path}")
        print()
        existing_tools = []

    # Get basic tool information
    name = get_user_input("Tool name")
    if name in existing_tools:
        print(f"Error: Tool '{name}' already exists.")
        return

    version = get_user_input("Version", default="0.1.0")
    description = get_user_input("Description")

    # Get optional information
    print("\nFeatures (e.g., 'classification', 'regression', etc.)")
    features = get_list_input("Enter tool features")

    print("\nRequirements (e.g., 'numpy>=1.20.0', 'torch>=1.9.0', etc.)")
    requirements = get_list_input("Enter tool requirements")

    print("\nPrompt templates (enter template strings for tool usage)")
    prompt_template = get_list_input("Enter prompt templates")

    # Get tutorials path and LLM options
    tutorials_path = get_user_input("Path to tutorials directory (optional)", required=False)
    tutorials_path = Path(tutorials_path) if tutorials_path else None

    # Initialize tutorial processing options
    condense_tutorials = False
    llm_config = None
    max_length = 9999

    if tutorials_path:
        # Get LLM config first as it's needed for both condensing and summarizing
        print("\nTutorial processing requires LLM configuration.")
        llm_config = get_llm_config()

        # Ask about condensing
        condense = input("\nDo you want to create condensed versions of tutorials? (y/N): ").lower()
        condense_tutorials = condense == "y"

        if condense_tutorials:
            max_length = int(get_user_input("Maximum length for condensed tutorials", default="9999"))

    # Confirm registration
    print("\nTool Registration Summary:")
    print(f"Name: {name}")
    print(f"Version: {version}")
    print(f"Description: {description}")
    print(f"Features: {', '.join(features) if features else 'None'}")
    print(f"Requirements: {', '.join(requirements) if requirements else 'None'}")
    print(f"Prompt Templates: {len(prompt_template)} templates")
    print(f"Tutorials Path: {tutorials_path or 'None'}")
    if tutorials_path:
        print(f"Create Condensed Versions: {'Yes' if condense_tutorials else 'No'}")
        if condense_tutorials:
            print(f"Max Length: {max_length}")
        print(f"LLM Config: {llm_config}")

    confirm = input("\nProceed with registration? (y/N): ").lower()
    if confirm != "y":
        print("Registration cancelled.")
        return

    try:
        # Register tool with all options
        registry.register_tool(
            name=name,
            version=version,
            description=description,
            features=features,
            requirements=requirements,
            prompt_template=prompt_template,
            tutorials_path=tutorials_path,
            condense=condense_tutorials,
            llm_config=llm_config,
            max_length=max_length,
        )
        print(f"\nSuccessfully registered tool: {name}")

    except Exception as e:
        print(f"\nError registering tool: {str(e)}")
        return


def main():
    try:
        register_tool_interactive()
    except KeyboardInterrupt:
        print("\nRegistration cancelled.")
        sys.exit(1)


if __name__ == "__main__":
    main()
