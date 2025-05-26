import logging
from typing import List

from ..prompts import RetrieverPrompt
from ..tools_registry import TutorialInfo
from .base_agent import BaseAgent
from .utils import init_llm

logger = logging.getLogger(__name__)


class RetrieverAgent(BaseAgent):
    """
    Agent for retrieving and selecting relevant tutorials based on task context.

    Agent Input: Task context, data info, user prompt, error info
    Agent Output: Formatted tutorial prompt with selected relevant tutorials
    """

    def __init__(self, config, manager, llm_config, prompt_template):
        super().__init__(config=config, manager=manager)
        self.retrieval_llm_config = llm_config
        self.retrieval_prompt_template = prompt_template
        self.retrieval_prompt = RetrieverPrompt(
            llm_config=self.retrieval_llm_config,
            manager=self.manager,
            template=self.retrieval_prompt_template,
        )

        if self.retrieval_llm_config.multi_turn:
            self.retrieval_llm = init_llm(
                llm_config=self.retrieval_llm_config,
                agent_name="retrieval",
                multi_turn=self.retrieval_llm_config.multi_turn,
            )

    def __call__(self):
        """Select relevant tutorials and format them into a prompt."""
        # Build prompt for tutorial selection
        prompt = self.retrieval_prompt.build()

        if not self.retrieval_llm_config.multi_turn:
            self.retrieval_llm = init_llm(
                llm_config=self.retrieval_llm_config,
                agent_name="retrieval",
                multi_turn=self.retrieval_llm_config.multi_turn,
            )

        response = self.retrieval_llm.assistant_chat(prompt)
        selected_tutorials = self.retrieval_prompt.parse(response)

        # Generate tutorial prompt using selected tutorials
        tutorial_prompt = self._generate_tutorial_prompt(selected_tutorials)

        return tutorial_prompt

    def _format_tutorial_content(
        self,
        tutorial: TutorialInfo,
        max_length: int,
    ) -> str:
        """Format a single tutorial's content with truncation if needed."""
        try:
            with open(tutorial.path, "r", encoding="utf-8") as f:
                content = f.read()

            # Truncate if needed
            if len(content) > max_length:
                content = content[:max_length] + "\n...(truncated)"

            formatted = f"""### {tutorial.title}
            
            {content}
            """
            return formatted

        except Exception as e:
            logger.warning(f"Error formatting tutorial {tutorial.path}: {e}")
            return ""

    def _generate_tutorial_prompt(self, selected_tutorials: List) -> str:
        """Generate formatted tutorial prompt from selected tutorials."""

        if not selected_tutorials:
            return ""

        # Get max tutorial length from config if available
        max_tutorial_length = self.config.max_tutorial_length

        # Format selected tutorials
        formatted_tutorials = []
        for tutorial in selected_tutorials:
            formatted = self._format_tutorial_content(tutorial, max_tutorial_length)
            if formatted:
                formatted_tutorials.append(formatted)

        if not formatted_tutorials:
            return ""

        return "\n\n".join(formatted_tutorials)
