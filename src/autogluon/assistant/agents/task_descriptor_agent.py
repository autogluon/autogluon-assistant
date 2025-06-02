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

    def __init__(self, config, manager, llm_config, prompt_template):
        super().__init__(config=config, manager=manager)

        self.task_descriptor_llm_config = llm_config
        self.task_descriptor_prompt_template = prompt_template

        self.task_descriptor_prompt = TaskDescriptorPrompt(
            llm_config=self.task_descriptor_llm_config,
            manager=self.manager,
            template=self.task_descriptor_prompt_template,
        )

        if self.task_descriptor_llm_config.multi_turn:
            self.task_descriptor_llm = init_llm(
                llm_config=self.task_descriptor_llm_config,
                agent_name="task_descriptor",
                multi_turn=self.task_descriptor_llm_config.multi_turn,
            )

    def __call__(
        self,
    ):
        """
        Generate task description based on provided data and analysis.

        Returns:
            str: Generated task description
        """
        self.manager.log_agent_start(
            "TaskDescriptorAgent: generating a concise task description from source materials."
        )

        # Attach description file directly if within certain length
        description_files_contents = self.task_descriptor_prompt.get_description_files_contents(to_show=True)

        task_description = description_files_contents

        # Otherwise generate condensed task description
        prompt = self.task_descriptor_prompt.build()

        if not self.task_descriptor_llm_config.multi_turn:
            self.task_descriptor_llm = init_llm(
                llm_config=self.task_descriptor_llm_config,
                agent_name="task_descriptor",
                multi_turn=self.task_descriptor_llm_config.multi_turn,
            )

        response = self.task_descriptor_llm.assistant_chat(prompt)

        task_description = self.task_descriptor_prompt.parse(response)

        self.manager.log_agent_end("TaskDescriptorAgent: task description generated.")

        return task_description
