import logging

from ..prompts import ErrorAnalyzerPrompt
from .base_agent import BaseAgent
from .utils import init_llm

logger = logging.getLogger(__name__)


class ErrorAnalyzerAgent(BaseAgent):
    """
    Execute the code and give analysis.

    Agent Input:

    Agent Output:
    """

    def __init__(self, config, manager, llm_config, prompt_template):
        super().__init__(config=config, manager=manager)

        self.error_analyzer_llm_config = llm_config
        self.error_analyzer_prompt_template = prompt_template

        self.error_analyzer_prompt = ErrorAnalyzerPrompt(
            llm_config=self.error_analyzer_llm_config,
            manager=self.manager,
            template=self.error_analyzer_prompt_template,
        )

        if self.error_analyzer_llm_config.multi_turn:
            self.error_analyzer_llm = init_llm(
                llm_config=self.error_analyzer_llm_config,
                agent_name="error_analyzer",
                multi_turn=self.error_analyzer_llm_config.multi_turn,
            )

    def __call__(self):
        self.manager.log_agent_start(
            "ErrorAnalyzerAgent: analyzing previous error and preparing debugging suggestions."
        )

        # Build prompt for evaluating execution results
        prompt = self.error_analyzer_prompt.build()

        if not self.error_analyzer_llm_config.multi_turn:
            self.error_analyzer_llm = init_llm(
                llm_config=self.error_analyzer_llm_config,
                agent_name="error_analyzer",
                multi_turn=self.error_analyzer_llm_config.multi_turn,
            )

        response = self.error_analyzer_llm.assistant_chat(prompt)

        error_analysis = self.error_analyzer_prompt.parse(response)

        self.manager.log_agent_end("ErrorAnalyzerAgent: error analysis complete with summary and fix suggestions.")

        return error_analysis
