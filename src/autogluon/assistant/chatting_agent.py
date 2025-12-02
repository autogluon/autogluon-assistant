"""
Chatting agent entry point for AutoGluon Assistant.

This module provides a conversational interface for users to ask questions
about data analysis, machine learning, and related topics without executing code.
"""

import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

from .constants import DEFAULT_CONFIG_PATH, WEBUI_OUTPUT_DIR
from .rich_logging import configure_logging

logger = logging.getLogger(__name__)


def run_chat_agent(
    input_data_folder=None,
    output_folder=None,
    config_path=None,
    session_id=None,
    verbosity=1,
    interactive=True,
):
    """
    Run the AutoGluon Assistant in chat mode.

    Args:
        input_data_folder: Optional path to input data directory for context
        output_folder: Path to output directory for session logs
        config_path: Path to configuration file
        session_id: Optional session ID (generated if not provided)
        verbosity: Verbosity level
        interactive: Whether to run in interactive mode (default: True)

    Returns:
        ChattingManager instance
    """
    # Get the directory of the current file
    current_file_dir = Path(__file__).parent

    # Generate output folder if not provided
    if output_folder is None or not output_folder:
        working_dir = os.path.join(current_file_dir.parent.parent.parent, "chat_sessions")
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_uuid = uuid.uuid4()
        folder_name = f"chat-{current_datetime}-{random_uuid}"
        output_folder = os.path.join(working_dir, folder_name)

    # Create output directory
    output_dir = Path(output_folder).expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=False, exist_ok=True)

    # Configure logging
    configure_logging(verbosity=verbosity, output_dir=output_dir)

    # Import here to avoid circular imports
    from .managers.chatting_manager import ChattingManager

    # Log output directory for WebUI backend detection
    if os.environ.get("AUTOGLUON_WEBUI") == "true":
        logger.debug(f"{WEBUI_OUTPUT_DIR} {output_dir}")

    # Load configuration
    # First load default config
    default_config_path = Path(__file__).parent / "configs" / "chat_config.yaml"

    if not default_config_path.exists():
        # Fall back to regular default config if chat config doesn't exist
        default_config_path = DEFAULT_CONFIG_PATH

    if not default_config_path.exists():
        raise FileNotFoundError(f"Default config file not found: {default_config_path}")

    config = OmegaConf.load(default_config_path)

    # If config_path is provided, merge it with the default config
    if config_path is not None:
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        user_config = OmegaConf.load(config_path)
        config = OmegaConf.merge(config, user_config)

    # Create ChattingManager instance
    manager = ChattingManager(
        input_data_folder=input_data_folder,
        output_folder=output_folder,
        config=config,
        session_id=session_id,
    )

    # Initialize the manager (generate data context if input folder is provided)
    manager.initialize()

    logger.brief("=" * 80)
    logger.brief("AutoGluon Assistant - Chat Mode")
    logger.brief("=" * 80)

    if input_data_folder:
        logger.brief(f"Input data: {input_data_folder}")
        if manager.file_contents:
            logger.brief(f"Loaded {len(manager.file_contents)} file(s):")
            for file_path in list(manager.file_contents.keys())[:5]:  # Show first 5
                logger.brief(f"  - {file_path}")
            if len(manager.file_contents) > 5:
                logger.brief(f"  ... and {len(manager.file_contents) - 5} more")
    logger.brief(f"Session ID: {manager.session.session_id}")
    logger.brief(f"Output folder: {output_dir}")
    logger.brief("")
    logger.brief("Type your questions below. Type 'exit', 'quit', or 'bye' to end the session.")
    logger.brief("=" * 80)
    logger.brief("")

    # Run in interactive mode if requested
    if interactive:
        try:
            while True:
                # Get user input
                try:
                    user_input = input("\nYou: ").strip()
                except EOFError:
                    logger.brief("\nExiting chat session...")
                    break

                # Check for exit commands
                if user_input.lower() in ["exit", "quit", "bye", "q"]:
                    logger.brief("Goodbye!")
                    break

                # Skip empty input
                if not user_input:
                    continue

                # Process the message and get response
                try:
                    response = manager.chat(user_input)
                    logger.brief(f"\nAssistant: {response}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    logger.brief("I encountered an error. Please try again.")

        except KeyboardInterrupt:
            logger.brief("\n\nChat session interrupted by user.")
        finally:
            # Cleanup
            manager.cleanup()
            logger.brief(f"\nSession saved to: {output_dir}")
            logger.brief(f"Total messages: {len(manager.session.messages)}")

            # Show location of Q&A markdown files
            qa_folder = os.path.join(output_dir, "conversations")
            if os.path.exists(qa_folder):
                qa_files = [f for f in os.listdir(qa_folder) if f.endswith(".md")]
                if qa_files:
                    logger.brief(f"Q&A conversations saved to: {qa_folder}")
                    logger.brief(f"  {len(qa_files)} markdown file(s) created")

    return manager


def run_chat_server(
    input_data_folder=None,
    output_folder=None,
    config_path=None,
    session_id=None,
    verbosity=1,
    port=5001,
):
    """
    Run the chat agent as a server (for integration with web UIs).

    Args:
        input_data_folder: Optional path to input data directory for context
        output_folder: Path to output directory for session logs
        config_path: Path to configuration file
        session_id: Optional session ID
        verbosity: Verbosity level
        port: Port to run the server on

    Returns:
        None
    """
    # This is a placeholder for future server implementation
    logger.info("Chat server mode is not yet implemented.")
    logger.info("Please use interactive mode for now.")

    # For now, just run in non-interactive mode
    manager = run_chat_agent(
        input_data_folder=input_data_folder,
        output_folder=output_folder,
        config_path=config_path,
        session_id=session_id,
        verbosity=verbosity,
        interactive=False,
    )

    return manager
