"""
Chat agent for handling conversational interactions.

This agent processes user questions and generates helpful responses without
executing code.
"""

import logging

from ..prompts import ChatPrompt
from .base_agent import BaseAgent
from .utils import init_llm

logger = logging.getLogger(__name__)


class ChatAgent(BaseAgent):
    """
    Agent for handling conversational interactions and answering user questions.

    Agent Input:
    - User question/message from the conversation history

    Agent Output:
    - Generated response to the user's question
    """

    def __init__(self, config, manager, llm_config, prompt_template):
        super().__init__(config=config, manager=manager)

        self.chat_llm_config = llm_config
        self.chat_prompt_template = prompt_template

        self.chat_prompt = ChatPrompt(
            llm_config=self.chat_llm_config,
            manager=self.manager,
            template=self.chat_prompt_template,
        )

        # Initialize LLM if using multi-turn
        if self.chat_llm_config.multi_turn:
            self.chat_llm = init_llm(
                llm_config=self.chat_llm_config,
                agent_name="chat_agent",
                multi_turn=self.chat_llm_config.multi_turn,
            )

    def __call__(self):
        """
        Generate a response to the user's question.

        Returns:
            str: Generated response
        """
        self.manager.log_agent_start("ChatAgent: generating response to user question.")

        # Build prompt for chat response
        prompt = self.chat_prompt.build()

        # Log the full prompt
        logger.debug(f"\n{'='*80}")
        logger.debug("FULL PROMPT SENT TO LLM")
        logger.debug(f"{'='*80}")
        logger.debug(prompt)
        logger.debug(f"{'='*80}\n")

        # Save prompt to file
        self.manager.save_and_log_states(prompt, "chat_prompt.txt", per_iteration=True)

        # Initialize LLM if not using multi-turn
        if not self.chat_llm_config.multi_turn:
            self.chat_llm = init_llm(
                llm_config=self.chat_llm_config,
                agent_name="chat_agent",
                multi_turn=self.chat_llm_config.multi_turn,
            )

        # Log LLM state
        logger.debug(f"Multi-turn enabled: {self.chat_llm_config.multi_turn}")
        logger.debug(f"LLM session name: {self.chat_llm.session_name}")
        logger.debug(f"LLM conversation ID: {self.chat_llm.conversation_id}")
        if hasattr(self.chat_llm, 'history_'):
            logger.debug(f"LLM history length: {len(self.chat_llm.history_)}")

        # Get response from LLM
        response = self.chat_llm.assistant_chat(prompt)

        # Log response
        logger.debug(f"\n{'='*80}")
        logger.debug("RAW LLM RESPONSE")
        logger.debug(f"{'='*80}")
        logger.debug(response)
        logger.debug(f"{'='*80}\n")

        # Save response to file
        self.manager.save_and_log_states(response, "chat_response.txt", per_iteration=True)

        # Parse the response
        parsed_response = self.chat_prompt.parse(response)

        self.manager.log_agent_end("ChatAgent: response generated successfully.")

        return parsed_response
