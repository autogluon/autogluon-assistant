import logging
from typing import List

from ..prompts import DescriptionFileRetrieverPrompt
from .base_agent import BaseAgent
from .utils import init_llm

logger = logging.getLogger(__name__)


class DescriptionFileRetrieverAgent(BaseAgent):
    """
    Identify potential description files from the data prompt.
    Only identifies files, does not read content.

    Agent Input:
    - data_prompt: Text string containing data prompt

    Agent Output:
    - List[str]: List of identified description filenames
    """

    def __init__(self, config, manager, llm_config, prompt_template):
        super().__init__(config=config, manager=manager)

        self.description_file_retriever_llm_config = llm_config
        self.description_file_retriever_prompt_template = prompt_template

        self.description_file_retriever_prompt = DescriptionFileRetrieverPrompt(
            llm_config=self.description_file_retriever_llm_config,
            manager=self.manager,
            template=self.description_file_retriever_prompt_template,
        )

        if self.description_file_retriever_llm_config.multi_turn:
            self.description_file_retriever_llm = init_llm(
                llm_config=self.description_file_retriever_llm_config,
                agent_name="description_file_retriever",
                multi_turn=self.description_file_retriever_llm_config.multi_turn,
            )

    def __call__(self) -> List[str]:
        self.manager.log_agent_start("DescriptionFileRetrieverAgent: identifying description files from data prompt.")

        # Build prompt for identifying description files
        prompt = self.description_file_retriever_prompt.build()

        if not self.description_file_retriever_llm_config.multi_turn:
            self.description_file_retriever_llm = init_llm(
                llm_config=self.description_file_retriever_llm_config,
                agent_name="description_file_retriever",
                multi_turn=self.description_file_retriever_llm_config.multi_turn,
            )

        response = self.description_file_retriever_llm.assistant_chat(prompt)

        description_files = self.description_file_retriever_prompt.parse(response)

        self.manager.log_agent_end("DescriptionFileRetrieverAgent: description file list extracted.")

        return description_files
