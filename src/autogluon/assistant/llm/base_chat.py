import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from pydantic import BaseModel, ConfigDict, Field
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


def log_retry_attempt(retry_state):
    """Custom callback to log each retry attempt"""
    if retry_state.outcome.failed:
        exception = retry_state.outcome.exception()
        attempt_number = retry_state.attempt_number
        logger.error(f"Attempt {attempt_number} failed: {type(exception).__name__}: {exception}")


class GlobalTokenTracker:
    """Singleton class to track token usage across all conversations."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalTokenTracker, cls).__new__(cls)
            cls._instance.total_input_tokens = 0
            cls._instance.total_output_tokens = 0
            cls._instance.conversations = {}  # Track per-conversation usage
            cls._instance.sessions = {}  # Track per-session usage
        return cls._instance

    def add_tokens(
        self,
        conversation_id: str,
        session_name: str,
        input_tokens: int,
        output_tokens: int,
    ):
        """Add token counts for a specific conversation and session."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        # Track conversation-level usage
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = {
                "input_tokens": 0,
                "output_tokens": 0,
            }

        self.conversations[conversation_id]["input_tokens"] += input_tokens
        self.conversations[conversation_id]["output_tokens"] += output_tokens

        # Track session-level usage
        if session_name not in self.sessions:
            self.sessions[session_name] = {"input_tokens": 0, "output_tokens": 0}

        self.sessions[session_name]["input_tokens"] += input_tokens
        self.sessions[session_name]["output_tokens"] += output_tokens

    def get_conversation_usage(self, conversation_id: str) -> Dict[str, Any]:
        """Get token usage for a specific conversation."""
        if conversation_id not in self.conversations:
            return {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }

        conv_usage = self.conversations[conversation_id]
        return {
            "input_tokens": conv_usage["input_tokens"],
            "output_tokens": conv_usage["output_tokens"],
            "total_conversation_tokens": conv_usage["input_tokens"] + conv_usage["output_tokens"],
        }

    def get_total_usage(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Get total token usage across all conversations and sessions."""
        usage_data = {
            "total": {
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_tokens": self.total_input_tokens + self.total_output_tokens,
            },
            "conversations": {},
            "sessions": {},
        }

        # Add conversation-level usage
        for conv_id, conv_usage in self.conversations.items():
            usage_data["conversations"][conv_id] = {
                "input_tokens": conv_usage["input_tokens"],
                "output_tokens": conv_usage["output_tokens"],
                "total_tokens": conv_usage["input_tokens"] + conv_usage["output_tokens"],
            }

        # Add session-level usage
        for session_name, session_usage in self.sessions.items():
            usage_data["sessions"][session_name] = {
                "input_tokens": session_usage["input_tokens"],
                "output_tokens": session_usage["output_tokens"],
                "total_tokens": session_usage["input_tokens"] + session_usage["output_tokens"],
            }

        # Save to file if path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(usage_data, f, indent=2)

        return usage_data


class BaseAssistantChat(BaseModel):
    """Base class for assistant chat models with LangGraph support."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    history_: List[Dict[str, Any]] = Field(default_factory=list)
    input_tokens_: int = Field(default=0)
    output_tokens_: int = Field(default=0)
    graph: Optional[Any] = Field(default=None, exclude=True)
    app: Optional[Any] = Field(default=None, exclude=True)
    memory: Optional[Any] = Field(default=None, exclude=True)
    token_tracker: GlobalTokenTracker = Field(default_factory=GlobalTokenTracker)
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_name: str = Field(default="default_session")

    def initialize_conversation(
        self,
        llm: Any,
        system_prompt: str = "You are a technical assistant that excels at working on data science tasks.",
    ) -> None:
        """Initialize conversation using LangGraph."""
        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        graph = StateGraph(state_schema=MessagesState)

        def call_model(state: MessagesState):
            prompt_messages = prompt_template.invoke(state)
            response = llm.invoke(prompt_messages)
            return {"messages": [response]}

        graph.add_edge(START, "model")
        graph.add_node("model", call_model)

        memory = MemorySaver()
        app = graph.compile(checkpointer=memory)

        self.graph = graph
        self.app = app
        self.memory = memory

    def describe(self) -> Dict[str, Any]:
        """Get model description and conversation history."""
        conversation_usage = self.token_tracker.get_conversation_usage(self.conversation_id)
        total_usage = self.token_tracker.get_total_usage()

        return {
            "history": self.history_,
            "conversation_tokens": conversation_usage,
            "total_tokens_across_all_conversations": total_usage,
            "session_name": self.session_name,
        }

    @retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=32, min=32, max=128), after=log_retry_attempt)
    def assistant_chat(self, message: str) -> str:
        """Send a message and get response using LangGraph."""
        if not self.app:
            raise RuntimeError("Conversation not initialized. Call initialize_conversation first.")

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        input_messages = [HumanMessage(content=message)]
        response = self.app.invoke({"messages": input_messages}, config)

        ai_message = response["messages"][-1]
        input_tokens = output_tokens = 0

        if hasattr(ai_message, "usage_metadata"):
            usage = ai_message.usage_metadata
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)

            # Update both instance and global tracking
            self.input_tokens_ += input_tokens
            self.output_tokens_ += output_tokens
            self.token_tracker.add_tokens(self.conversation_id, self.session_name, input_tokens, output_tokens)

        self.history_.append(
            {
                "input": message,
                "output": ai_message.content,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )

        return ai_message.content

    async def astream(self, message: str):
        """Stream responses using LangGraph."""
        if not self.app:
            raise RuntimeError("Conversation not initialized. Call initialize_conversation first.")

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        input_messages = [HumanMessage(content=message)]

        async for chunk, metadata in self.app.stream({"messages": input_messages}, config, stream_mode="messages"):
            if isinstance(chunk, AIMessage):
                yield chunk.content
