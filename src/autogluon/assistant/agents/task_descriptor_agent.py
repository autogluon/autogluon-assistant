import logging

from ..prompts import TaskDescriptorPrompt
from .base_agent import BaseAgent
from .utils import init_llm

logger = logging.getLogger(__name__)


class TaskDescriptorAgent(BaseAgent):
    """
    Generate task description based on data prompt, description files, and analysis.

    Agent Input:
    - data_prompt: Text string containing data prompt
    - description_files: List of description filenames
    - description_analysis: Analysis from previous step

    Agent Output:
    - Generated task description string
    """

    def __init__(self, config, llm_config, prompt_template):
        super().__init__(config=config)

        self.task_descriptor_llm_config = llm_config
        self.task_descriptor_prompt_template = prompt_template

        self.task_descriptor_prompt = TaskDescriptorPrompt(
            llm_config=self.task_descriptor_llm_config,
            template=self.task_descriptor_prompt_template,
        )

        if self.task_descriptor_llm_config.multi_turn:
            self.task_descriptor_llm = init_llm(
                llm_config=self.task_descriptor_llm_config,
                agent_name="task_descriptor",
                multi_turn=self.task_descriptor_llm_config.multi_turn,
            )

    def __call__(self, manager):
        """
        Generate task description based on provided data and analysis.

        Args:
            manager: Object containing data_prompt, description_files,
                            and description_analysis attributes

        Returns:
            str: Generated task description
        """
        # Build prompt for generating task description
        prompt = self.task_descriptor_prompt.build(manager)

        if not self.task_descriptor_llm_config.multi_turn:
            self.task_descriptor_llm = init_llm(
                llm_config=self.task_descriptor_llm_config,
                agent_name="task_descriptor",
                multi_turn=self.task_descriptor_llm_config.multi_turn,
            )

        response = self.task_descriptor_llm.assistant_chat(prompt)

        task_description = self.task_descriptor_prompt.parse(response)

        return task_description
