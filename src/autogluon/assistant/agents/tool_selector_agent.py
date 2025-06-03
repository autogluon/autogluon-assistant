import logging
from typing import Tuple

from ..prompts import ToolSelectorPrompt
from .base_agent import BaseAgent
from .utils import init_llm

logger = logging.getLogger(__name__)


class ToolSelectorAgent(BaseAgent):
    """
    Select the most appropriate tool based on data description and task requirements.

    Agent Input:
    - data_prompt: Text string containing data prompt
    - description: Description of the task/data from previous analysis

    Agent Output:
    - str: Selected tool name
    """

    def __init__(self, config, manager, llm_config, prompt_template):
        super().__init__(config=config, manager=manager)

        self.tool_selector_llm_config = llm_config
        self.tool_selector_prompt_template = prompt_template

        self.tool_selector_prompt = ToolSelectorPrompt(
            llm_config=self.tool_selector_llm_config,
            manager=self.manager,
            template=self.tool_selector_prompt_template,
        )

        if self.tool_selector_llm_config.multi_turn:
            self.tool_selector_llm = init_llm(
                llm_config=self.tool_selector_llm_config,
                agent_name="tool_selector",
                multi_turn=self.tool_selector_llm_config.multi_turn,
            )

    def __call__(self) -> Tuple[str, str]:
        self.manager.log_agent_start("ToolSelectorAgent: choosing the most appropriate ML library for the task.")

        # Build prompt for tool selection
        prompt = self.tool_selector_prompt.build()

        if not self.tool_selector_llm_config.multi_turn:
            self.tool_selector_llm = init_llm(
                llm_config=self.tool_selector_llm_config,
                agent_name="tool_selector",
                multi_turn=self.tool_selector_llm_config.multi_turn,
            )

        response = self.tool_selector_llm.assistant_chat(prompt)

        selected_tool = self.tool_selector_prompt.parse(response)

        self.manager.log_agent_end(f"ToolSelectorAgent: selected {selected_tool}.")

        return selected_tool
