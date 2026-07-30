import pytest

# These imports pull in optional heavy runtime dependencies; skip the whole
# module when they are unavailable so the suite still collects cleanly.
pytest.importorskip("langchain_openai")
pytest.importorskip("langchain_anthropic")
pytest.importorskip("langchain_aws")
pytest.importorskip("langgraph")


def test_minimax_is_registered_provider():
    from autogluon.assistant.llm.llm_factory import ChatLLMFactory

    assert "minimax" in ChatLLMFactory.get_valid_providers()


def test_minimax_regional_endpoints():
    from autogluon.assistant.llm.minimax_chat import MINIMAX_REGIONAL_ENDPOINTS

    assert set(MINIMAX_REGIONAL_ENDPOINTS) == {"global_en", "cn_zh"}
    assert MINIMAX_REGIONAL_ENDPOINTS["global_en"]["openai_base_url"] == "https://api.minimax.io/v1"
    assert MINIMAX_REGIONAL_ENDPOINTS["global_en"]["anthropic_base_url"] == "https://api.minimax.io/anthropic"
    assert MINIMAX_REGIONAL_ENDPOINTS["cn_zh"]["openai_base_url"] == "https://api.minimaxi.com/v1"
    assert MINIMAX_REGIONAL_ENDPOINTS["cn_zh"]["anthropic_base_url"] == "https://api.minimaxi.com/anthropic"


def test_minimax_models_include_current_releases():
    from autogluon.assistant.llm.minimax_chat import DEFAULT_MINIMAX_MODEL, MINIMAX_MODELS

    assert DEFAULT_MINIMAX_MODEL == "MiniMax-M3"
    assert "MiniMax-M3" in MINIMAX_MODELS
    assert "MiniMax-M2.7" in MINIMAX_MODELS
