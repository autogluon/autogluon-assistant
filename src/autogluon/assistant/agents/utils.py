from datetime import datetime

from ..llm import ChatLLMFactory


def init_llm(llm_config, agent_name, multi_turn):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    if multi_turn:
        session_name = f"multi_turn_{agent_name}_{timestamp}"
    else:
        session_name = f"single_turn_{agent_name}_{timestamp}"
    llm = ChatLLMFactory.get_chat_model(llm_config, session_name=session_name)

    return llm
