"""
Session-based manager for handling conversational interactions.

This manager maintains conversation history, provides context to the chat agent,
and manages tool selection without code execution.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """
    A single message in the chat conversation.
    """

    role: str  # 'user' or 'assistant'
    content: str
    timestamp: float = field(default_factory=lambda: time.time())

    def to_dict(self):
        """Convert to dictionary for storage/display."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
        }


@dataclass
class ChatSession:
    """
    A chat session containing conversation history and metadata.
    """

    session_id: str
    messages: List[ChatMessage] = field(default_factory=list)
    created_at: float = field(default_factory=lambda: time.time())
    last_updated: float = field(default_factory=lambda: time.time())

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        message = ChatMessage(role=role, content=content)
        self.messages.append(message)
        self.last_updated = time.time()
        return message

    def get_history(self, max_messages: Optional[int] = None) -> List[ChatMessage]:
        """
        Get conversation history.

        Args:
            max_messages: Maximum number of recent messages to return

        Returns:
            List of chat messages
        """
        if max_messages is None:
            return self.messages
        return self.messages[-max_messages:]

    def to_dict(self):
        """Convert to dictionary for storage/display."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "message_count": len(self.messages),
            "messages": [msg.to_dict() for msg in self.messages],
        }


class ChattingManager:
    """
    Manages chat sessions and provides context for conversational interactions.
    Unlike the NodeManager, this doesn't execute code but maintains conversation state.
    """

    def __init__(
        self,
        input_data_folder: Optional[str],
        output_folder: str,
        config: Any,
        session_id: Optional[str] = None,
    ):
        """
        Initialize the ChattingManager.

        Args:
            input_data_folder: Optional path to input data directory for context
            output_folder: Path to output directory for saving session logs
            config: Configuration object
            session_id: Optional session ID (generated if not provided)
        """
        # Store paths
        self.input_data_folder = input_data_folder
        self.output_folder = output_folder

        # Validate input data folder if provided
        if input_data_folder is not None:
            if not Path(input_data_folder).exists():
                raise FileNotFoundError(f"input_data_folder not found: {input_data_folder}")

        # Create output folder if it doesn't exist
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        self.config = config

        # Initialize or load session
        if session_id is None:
            session_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.session = ChatSession(session_id=session_id)

        # Try to load existing session
        self._load_session()

        # Store file contents directly
        self.file_contents = {}

        # Track what has been presented to avoid duplicates
        self.data_context_presented = False
        self.presented_tutorials = set()  # Set of tutorial identifiers

        # Initialize agents
        self._init_agents()

    def _init_agents(self):
        """Initialize required agents for chat functionality."""
        from ..agents import ChatAgent, RerankerAgent, RetrieverAgent, ToolSelectorAgent

        # Chat agent for handling conversations
        self.chat_agent = ChatAgent(
            config=self.config,
            manager=self,
            llm_config=self.config.chat_agent,
            prompt_template=None,
        )

        # Tool selector for identifying relevant tools from user questions
        self.tool_selector = ToolSelectorAgent(
            config=self.config,
            manager=self,
            llm_config=self.config.tool_selector,
            prompt_template=None,
        )

        # Retriever for fetching relevant tutorials
        self.retriever = RetrieverAgent(
            config=self.config,
            manager=self,
            llm_config=self.config.retriever,
            prompt_template=None,
        )

        # Reranker for selecting best tutorials
        self.reranker = RerankerAgent(
            config=self.config,
            manager=self,
            llm_config=self.config.reranker,
            prompt_template=None,
        )

    def initialize(self):
        """Initialize the manager with data context if available."""
        if self.input_data_folder is not None:
            logger.info("Reading input files...")
            self._read_input_files()
            logger.info("Data context initialized.")
        else:
            logger.info("No input data folder provided, skipping data context initialization.")

    def _read_input_files(self):
        """Read all files from the input path directly."""
        import os

        input_path = Path(self.input_data_folder)

        # Check if it's a file or directory
        if not input_path.exists():
            logger.error(f"Input path does not exist: {input_path}")
            self.file_contents = {}
            return

        self.file_contents = {}

        if input_path.is_file():
            # Single file
            logger.info(f"Reading file: {input_path.name}")
            try:
                content = self._read_file_content(input_path)
                self.file_contents[input_path.name] = content
            except Exception as e:
                logger.warning(f"Failed to read {input_path.name}: {e}")
        else:
            # Directory - read all files
            for root, dirs, files in os.walk(input_path):
                for file in files:
                    file_path = Path(root) / file
                    relative_path = file_path.relative_to(input_path)

                    logger.debug(f"Reading file: {relative_path}")
                    try:
                        content = self._read_file_content(file_path)
                        self.file_contents[str(relative_path)] = content
                    except Exception as e:
                        logger.warning(f"Failed to read {relative_path}: {e}")

        logger.info(f"Read {len(self.file_contents)} file(s)")

    def _read_file_content(self, file_path: Path, max_size_mb: int = None) -> str:
        """
        Read file content with size limit.

        Args:
            file_path: Path to the file
            max_size_mb: Maximum file size in MB (uses config if not provided)

        Returns:
            File content as string
        """
        if max_size_mb is None:
            max_size_mb = getattr(self.config, "max_file_size_mb", 10)

        # Check file size
        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            return f"[File too large: {size_mb:.2f} MB, skipped]"

        # Try to read as text
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return content
        except UnicodeDecodeError:
            # Try other encodings
            for encoding in ["latin-1", "cp1252", "iso-8859-1"]:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        content = f.read()
                    return content
                except:
                    continue
            # If all fail, it's likely binary
            return f"[Binary file: {file_path.suffix}, skipped]"
        except Exception as e:
            return f"[Error reading file: {e}]"

    def chat(self, user_message: str) -> str:
        """
        Process a user message and generate a response.

        Args:
            user_message: The user's question or message

        Returns:
            The assistant's response
        """

        iter_num = self.time_step + 1
        logger.debug(f"\n{'='*80}")
        logger.debug(f"CHAT ITERATION {iter_num}")
        logger.debug(f"{'='*80}")

        # Store current user message for the prompt
        self.current_user_message = user_message
        logger.debug(f"User message: {user_message}")

        # Add user message to history
        self.session.add_message(role="user", content=user_message)
        logger.debug(f"Session message count: {len(self.session.messages)}")

        # Identify relevant tool from user message
        logger.debug("Identifying relevant tools...")
        self.selected_tool = self._identify_tool_from_message(user_message)
        logger.debug(f"Selected tool: {self.selected_tool}")

        # Retrieve relevant tutorials for the selected tool
        if self.selected_tool:
            logger.debug(f"Retrieving tutorials for {self.selected_tool}...")
            self._retrieve_tutorials()
            logger.debug(f"Tutorial prompt length: {len(self.tutorial_prompt) if self.tutorial_prompt else 0} chars")
        else:
            self.tutorial_prompt = ""
            logger.debug("No tool selected, no tutorials retrieved")

        # Log the prompt that will be sent
        logger.debug(f"\n{'='*80}")
        logger.debug("PROMPT CONSTRUCTION")
        logger.debug(f"{'='*80}")
        logger.debug(f"Data context presented: {self.data_context_presented}")
        logger.debug(f"Presented tutorials count: {len(self.presented_tutorials)}")

        # Save user message to file
        self.save_and_log_states(user_message, "user_message.txt", per_iteration=True)

        # Generate response using chat agent
        logger.debug("\nGenerating response...")
        response = self.chat_agent()

        logger.debug(f"\n{'='*80}")
        logger.debug("RESPONSE RECEIVED")
        logger.debug(f"{'='*80}")
        logger.debug(f"Response length: {len(response)} chars")
        logger.debug(f"Response preview: {response[:200]}...")

        # Add assistant response to history
        self.session.add_message(role="assistant", content=response)

        # Mark data context as presented after first message
        if not self.data_context_presented and self.file_contents:
            self.data_context_presented = True

        # Save this Q&A exchange to markdown file
        self._save_qa_to_markdown(user_message, response)

        # Save session
        self._save_session()

        return response

    def _save_qa_to_markdown(self, question: str, answer: str):
        """
        Save the Q&A exchange to a markdown file.

        Args:
            question: User's question
            answer: Assistant's answer
        """
        # Create QA folder if it doesn't exist
        qa_folder = os.path.join(self.output_folder, "conversations")
        os.makedirs(qa_folder, exist_ok=True)

        # Use time_step as iteration number (number of Q&A pairs)
        iter_num = self.time_step

        # Create filename with iteration number
        filename = f"qa_iter{iter_num:03d}.md"
        filepath = os.path.join(qa_folder, filename)

        # Format as markdown
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        markdown_content = f"""# Q&A Exchange - Iteration {iter_num}

**Timestamp:** {timestamp}
**Session ID:** {self.session.session_id}

---

## Question

{question}

---

## Answer

{answer}

---

*Generated by AutoGluon Assistant Chat Mode*
"""

        # Write to file
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            logger.debug(f"Saved Q&A to {filepath}")
        except Exception as e:
            logger.warning(f"Failed to save Q&A to markdown: {e}")

    def _identify_tool_from_message(self, user_message: str) -> str:
        """
        Identify the most relevant tool/library from the user's message.

        Args:
            user_message: The user's question

        Returns:
            Tool name or empty string
        """
        # Store user input for tool selector
        self.user_input = user_message
        self.task_description = user_message

        try:
            # Use tool selector to identify relevant tool
            tools = self.tool_selector()
            if tools and len(tools) > 0:
                return tools[0]  # Return the top tool
        except Exception as e:
            logger.warning(f"Failed to identify tool: {e}")

        return ""

    def _retrieve_tutorials(self):
        """
        Retrieve and rerank tutorials for the selected tool.
        Only include tutorials that haven't been presented before.
        """
        try:
            # Retrieve tutorials
            self.tutorial_retrieval = self.retriever()

            # Rerank tutorials
            all_tutorials = self.reranker()

            # Filter out previously presented tutorials
            new_tutorials = self._filter_new_tutorials(all_tutorials)

            if new_tutorials:
                self.tutorial_prompt = new_tutorials
                # Track these tutorials as presented
                self._mark_tutorials_presented(new_tutorials)
            else:
                self.tutorial_prompt = ""

        except Exception as e:
            logger.warning(f"Failed to retrieve tutorials: {e}")
            self.tutorial_prompt = ""

    def _filter_new_tutorials(self, tutorials: str) -> str:
        """
        Filter out tutorials that have been presented before.

        Args:
            tutorials: The tutorial text

        Returns:
            Filtered tutorial text with only new tutorials
        """
        if not tutorials:
            return ""

        # Simple approach: split by tutorial sections and filter
        # This assumes tutorials have some identifiable structure
        lines = tutorials.split("\n")
        filtered_lines = []
        current_tutorial_hash = None

        for line in lines:
            # Simple hashing of tutorial content to detect duplicates
            if line.strip().startswith("#") or line.strip().startswith("##"):
                # This is a tutorial header
                current_tutorial_hash = hash(line.strip())

            if current_tutorial_hash not in self.presented_tutorials:
                filtered_lines.append(line)

        return "\n".join(filtered_lines) if filtered_lines else ""

    def _mark_tutorials_presented(self, tutorials: str):
        """
        Mark tutorials as presented to avoid showing them again.

        Args:
            tutorials: The tutorial text
        """
        if not tutorials:
            return

        lines = tutorials.split("\n")
        for line in lines:
            if line.strip().startswith("#") or line.strip().startswith("##"):
                tutorial_hash = hash(line.strip())
                self.presented_tutorials.add(tutorial_hash)

    def _load_session(self):
        """Load session from file if it exists."""
        import json

        session_file = os.path.join(self.output_folder, f"{self.session.session_id}.json")

        if os.path.exists(session_file):
            try:
                with open(session_file, "r") as f:
                    data = json.load(f)

                # Restore messages
                self.session.messages = [
                    ChatMessage(role=msg["role"], content=msg["content"], timestamp=msg["timestamp"])
                    for msg in data.get("messages", [])
                ]
                self.session.created_at = data.get("created_at", self.session.created_at)
                self.session.last_updated = data.get("last_updated", self.session.last_updated)

                logger.info(f"Loaded session {self.session.session_id} with {len(self.session.messages)} messages")
            except Exception as e:
                logger.warning(f"Failed to load session: {e}")

    def _save_session(self):
        """Save session to file."""
        import json

        session_file = os.path.join(self.output_folder, f"{self.session.session_id}.json")

        try:
            with open(session_file, "w") as f:
                json.dump(self.session.to_dict(), f, indent=2)
            logger.debug(f"Saved session to {session_file}")
        except Exception as e:
            logger.error(f"Failed to save session: {e}")

    def save_and_log_states(self, content, save_name, per_iteration=False, add_uuid=False, node=None):
        """
        Save states to a file and log them.

        Args:
            content: Content to save
            save_name: Name for the saved file
            per_iteration: Whether this is for a specific iteration
            add_uuid: Whether to add a UUID to the filename
            node: Unused (for compatibility)
        """
        import uuid

        if add_uuid:
            name, ext = os.path.splitext(save_name)
            uuid_suffix = str(uuid.uuid4()).replace("-", "")[:4]
            save_name = f"{name}_{uuid_suffix}{ext}"

        # Determine the save directory
        if per_iteration:
            # Save to iteration-specific folder
            iter_num = self.time_step
            iter_folder = os.path.join(self.output_folder, f"iter_{iter_num:03d}")
            states_dir = os.path.join(iter_folder, "states")
        else:
            states_dir = os.path.join(self.output_folder, "states")

        os.makedirs(states_dir, exist_ok=True)
        output_file = os.path.join(states_dir, save_name)

        logger.debug(f"Saving {output_file}...")

        with open(output_file, "w") as file:
            if content is not None:
                if isinstance(content, list):
                    file.write("\n".join(str(item) for item in content))
                else:
                    file.write(content)
            else:
                file.write("<None>")

    def log_agent_start(self, message: str):
        """Log agent start message."""
        logger.info(message)

    def log_agent_end(self, message: str):
        """Log agent end message."""
        logger.info(message)

    def cleanup(self):
        """Clean up resources."""
        # Save final session state
        self._save_session()

        # Cleanup retriever resources
        if hasattr(self, "retriever"):
            self.retriever.cleanup()

        logger.info("ChattingManager cleanup completed.")

    @property
    def time_step(self) -> int:
        """Get the current conversation step (number of exchanges)."""
        return len(self.session.messages) // 2

    @property
    def data_prompt(self) -> str:
        """Get data prompt for agents."""
        if not self.file_contents:
            return ""
        # Return simple file listing for tool selector
        return f"Files available: {', '.join(self.file_contents.keys())}"

    @property
    def description_files(self) -> list:
        """Get description files (empty for chat mode)."""
        return []

    @property
    def tool_prompt(self) -> str:
        """Get tool-specific prompt (empty string for chat mode)."""
        return ""

    @property
    def tutorial_retrieval(self) -> str:
        """Get tutorial retrieval (managed internally)."""
        return getattr(self, "_tutorial_retrieval", "")

    @tutorial_retrieval.setter
    def tutorial_retrieval(self, value: str):
        """Set tutorial retrieval."""
        self._tutorial_retrieval = value

    @property
    def previous_tutorial_prompt(self) -> str:
        """Get previous tutorial prompt (not used in chat mode)."""
        return ""

    @property
    def per_iteration_output_folder(self) -> str:
        """Get output folder (use session output folder for chat mode)."""
        return self.output_folder

    @property
    def python_code(self) -> str:
        """Get Python code (not applicable for chat mode)."""
        return ""

    @property
    def bash_script(self) -> str:
        """Get bash script (not applicable for chat mode)."""
        return ""

    @property
    def error_message(self) -> str:
        """Get error message (not applicable for chat mode)."""
        return ""

    @property
    def previous_error_message(self) -> str:
        """Get previous error message (not applicable for chat mode)."""
        return ""

    @property
    def all_previous_error_analyses(self) -> str:
        """Get all error analyses (not applicable for chat mode)."""
        return ""

    @property
    def validation_score(self) -> float:
        """Get validation score (not applicable for chat mode)."""
        return 0.0

    def get_file_contents_text(self, max_length_per_file: int = None) -> str:
        """
        Get formatted file contents for the prompt.

        Args:
            max_length_per_file: Maximum characters per file (uses config if not provided)

        Returns:
            Formatted file contents as string
        """
        if not self.file_contents:
            return ""

        if max_length_per_file is None:
            max_length_per_file = getattr(self.config, "max_length_per_file", 5000)

        sections = []
        for file_path, content in self.file_contents.items():
            sections.append(f"=== File: {file_path} ===")

            # Truncate if too long
            if len(content) > max_length_per_file:
                truncated = content[:max_length_per_file]
                sections.append(truncated)
                sections.append(
                    f"\n[... truncated, showing first {max_length_per_file} characters of {len(content)} total ...]"
                )
            else:
                sections.append(content)

            sections.append("")  # Empty line between files

        return "\n".join(sections)
