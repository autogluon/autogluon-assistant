"""Real LangChain and SDK serialization against an in-memory HTTP transport."""

import copy
import json
from pathlib import Path

import httpx
import pytest
from omegaconf import OmegaConf

pytest.importorskip("langchain_openai")
pytest.importorskip("langchain_anthropic")
pytest.importorskip("langchain_aws")
pytest.importorskip("langgraph")

from anthropic import Anthropic, AsyncAnthropic  # noqa: E402
from openai import AsyncOpenAI, OpenAI  # noqa: E402

from autogluon.assistant.llm.llm_factory import ChatLLMFactory  # noqa: E402
from autogluon.assistant.llm.minimax_chat import MINIMAX_REGIONAL_ENDPOINTS, get_minimax_models  # noqa: E402

FLASH = "MiniMax-M3.1-Flash"
CASES = [
    "default",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
    "none",
    "runtime",
    "nested",
    "unrelated",
    "m3-default",
    "m3-enabled",
    "m3-disabled",
    "legacy",
    "custom",
]


def _response(request, body, api):
    if api == "openai":
        message = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        }
        if not body.get("stream"):
            return httpx.Response(200, request=request, json=message)
        chunk = dict(message, object="chat.completion.chunk")
        chunk.pop("usage")
        chunk["choices"] = [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]
        data = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"
    else:
        message = {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": body["model"],
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 1},
        }
        if not body.get("stream"):
            return httpx.Response(200, request=request, json=message)
        events = [
            ("message_start", {"message": dict(message, content=[], stop_reason=None)}),
            ("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "ok"}}),
            ("content_block_stop", {"index": 0}),
            (
                "message_delta",
                {"delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 1}},
            ),
            ("message_stop", {}),
        ]
        data = "".join(
            f"event: {event}\ndata: {json.dumps(dict(payload, type=event))}\n\n" for event, payload in events
        )
    return httpx.Response(200, request=request, headers={"content-type": "text/event-stream"}, content=data)


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["openai", "anthropic"])
@pytest.mark.parametrize("region", ["global_en", "cn_zh"])
@pytest.mark.parametrize("method", ["invoke", "stream", "ainvoke", "astream"])
@pytest.mark.parametrize("case", CASES)
async def test_factory_sdk_request_capture(monkeypatch, api, region, method, case):
    monkeypatch.setenv("MINIMAX_API_KEY", "test")
    config = {"provider": "minimax", "model": FLASH, "api": api, "region": region, "max_tokens": 4096}
    options = {}
    expected_effort = "max"
    if case in ("low", "medium", "high", "xhigh", "max", "none"):
        config.update(reasoning_effort=case, thinking={"type": "disabled"}, service_tier="priority")
        expected_effort = "max" if case == "none" else case
    elif case in ("runtime", "nested", "unrelated"):
        config["reasoning_effort"] = "max"
        options = {"reasoning_effort": "low"} if api == "openai" else {"output_config": {"effort": "low"}}
        expected_effort = "low"
        if case == "nested":
            options["extra_body"] = {
                "thinking": {"type": "disabled"},
                "service_tier": "priority",
                "reasoning_split": False,
            }
            if api == "openai":
                options["extra_body"]["reasoning_effort"] = "xhigh"
            else:
                options["extra_body"]["output_config"] = {"effort": "xhigh"}
            expected_effort = "xhigh"
        elif case == "unrelated":
            options["extra_body"] = {"metadata": {"user_id": "test"}}
    elif case.startswith("m3-"):
        config["model"] = "MiniMax-M3"
        if case != "m3-default":
            config.update(thinking={"enabled": case == "m3-enabled"}, reasoning_effort="low", service_tier="priority")
    elif case in ("legacy", "custom"):
        config["model"] = "MiniMax-M2.7" if case == "legacy" else "private-saved-model"
        config["thinking"] = {"type": "enabled", "budget_tokens": 1024}
    original = copy.deepcopy(options)
    captured = []
    base_url = MINIMAX_REGIONAL_ENDPOINTS[region][f"{api}_base_url"]

    def handler(request):
        body = json.loads(request.content)
        captured.append(body)
        suffix = "/chat/completions" if api == "openai" else "/v1/messages"
        assert str(request.url) == base_url + suffix
        return _response(request, body, api)

    chat = ChatLLMFactory.get_chat_model(OmegaConf.create(config), session_name="capture")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as async_client:
            if api == "openai":
                sync_sdk = OpenAI(api_key="test", base_url=base_url, http_client=client)
                async_sdk = AsyncOpenAI(api_key="test", base_url=base_url, http_client=async_client)
                chat.client = sync_sdk.chat.completions
                chat.async_client = async_sdk.chat.completions
            else:
                chat.__dict__["_client"] = Anthropic(api_key="test", base_url=base_url, http_client=client)
                chat.__dict__["_async_client"] = AsyncAnthropic(
                    api_key="test", base_url=base_url, http_client=async_client
                )
            if method == "invoke":
                assert chat.invoke("hello", **options).content == "ok"
            elif method == "ainvoke":
                assert (await chat.ainvoke("hello", **options)).content == "ok"
            elif method == "stream":
                assert "".join(chunk.text() for chunk in chat.stream("hello", **options)) == "ok"
            else:
                assert "".join([chunk.text() async for chunk in chat.astream("hello", **options)]) == "ok"
    assert options == original
    assert len(captured) == 1
    body = captured[0]
    assert body["model"] == config["model"]
    if config["model"] == FLASH:
        assert body["thinking"] == {"type": "adaptive"}
        actual_effort = body["reasoning_effort"] if api == "openai" else body["output_config"]["effort"]
        assert actual_effort == expected_effort
        assert "service_tier" not in body and "reasoning_split" not in body
    elif case.startswith("m3-"):
        if case == "m3-default":
            assert "thinking" not in body  # Keep the protocol's native default.
        else:
            assert body["thinking"] == {"type": "adaptive" if case == "m3-enabled" else "disabled"}
            assert body["service_tier"] == "priority"
    else:
        assert body["thinking"] == config["thinking"]


def test_current_catalog_and_yaml_defaults():
    for region in MINIMAX_REGIONAL_ENDPOINTS:
        assert get_minimax_models(region) == [FLASH, "MiniMax-M3"]
    choices = get_minimax_models()
    choices.append("mutated")
    assert get_minimax_models() == [FLASH, "MiniMax-M3"]
    with pytest.raises(ValueError, match="Invalid MiniMax region"):
        get_minimax_models("unknown")
    root = Path(__file__).resolve().parents[3]
    config = OmegaConf.load(root / "src/autogluon/assistant/configs/minimax.yaml")
    for name in ("llm", "python_coder", "bash_coder", "executer", "reader", "reranker"):
        assert config[name].model == FLASH
        assert config[name].reasoning_effort == "max"


def test_other_providers_still_validate_models(monkeypatch):
    monkeypatch.setattr(ChatLLMFactory, "get_valid_models", lambda provider: ["known-model"])
    config = OmegaConf.create({"provider": "openai", "model": "unknown-model"})
    with pytest.raises(ValueError, match="Invalid model"):
        ChatLLMFactory.get_chat_model(config, "validation")
