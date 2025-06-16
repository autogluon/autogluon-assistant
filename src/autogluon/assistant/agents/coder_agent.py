import logging

from ..prompts import BashCoderPrompt, PythonCoderPrompt
from .base_agent import BaseAgent
from .utils import init_llm

logger = logging.getLogger(__name__)


class CoderAgent(BaseAgent):
    """
    Execute the code and give analysis.

    Agent Input:

    Agent Output:
    """

    def __init__(self, config, manager, language, coding_mode, llm_config, prompt_template):
        super().__init__(config=config, manager=manager)
        assert language in ["bash", "python"]
        assert coding_mode in ["reader", "coder"]
        self.language = language
        self.coding_mode = coding_mode

        self.coder_llm_config = llm_config
        self.coder_prompt_template = prompt_template

        prompt_mapping = {
            "bash": {"reader": None, "coder": BashCoderPrompt},
            "python": {"reader": None, "coder": PythonCoderPrompt},
        }

        self.coder_prompt = prompt_mapping[language][coding_mode](
            llm_config=self.coder_llm_config,
            manager=self.manager,
            template=self.coder_prompt_template,
        )

        if self.coder_llm_config.multi_turn:
            self.coder_llm = init_llm(
                llm_config=self.coder_llm_config,
                agent_name=f"{self.language}_{self.coding_mode}",
                multi_turn=self.coder_llm_config.multi_turn,
            )

    def __call__(self):
        self.manager.log_agent_start("CoderAgent: starting to build and send code-generation prompt to the LLM.")

        # Build prompt for evaluating execution results
        prompt = self.coder_prompt.build()

        if not self.coder_llm_config.multi_turn:
            self.coder_llm = init_llm(
                llm_config=self.coder_llm_config,
                agent_name=f"{self.language}_{self.coding_mode}",
                multi_turn=self.coder_llm_config.multi_turn,
            )

        response = self.coder_llm.assistant_chat(prompt)

        generated_code = self.coder_prompt.parse(response)

        self.manager.log_agent_end("CoderAgent: code-generation prompt handled and code parsed from response.")

        return generated_code
