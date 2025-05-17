from .llm_coder import LLMCoder
from .utils import *


def generate_coder(llm_config, tutorial_link_for_rag=None):
    # TODO: implement coding_with_rag in rag_coder
    # Note: We have LLM-based tutorial selection in non-RAG mode
    if tutorial_link_for_rag:
        raise NotImplementedError
    else:
        return LLMCoder(llm_config)
