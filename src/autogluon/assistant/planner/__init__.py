from omegaconf import DictConfig

from .llm_planner import LLMPlanner


def get_planner(llm_config: DictConfig):
    return LLMPlanner(llm_config=llm_config)
