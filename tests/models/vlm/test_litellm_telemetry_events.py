# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

try:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
except ImportError:
    from opentelemetry.sdk.trace.export import InMemorySpanExporter

from openviking.models.vlm.backends import litellm_vlm
from openviking.models.vlm.backends.litellm_vlm import LiteLLMVLMProvider
from openviking.telemetry import tracer_module


def test_litellm_records_resolved_provider_model_and_typed_usage(monkeypatch):
    events = []
    monkeypatch.setattr(
        tracer_module.tracer,
        "add_event",
        lambda name, attributes=None: events.append((name, attributes)),
    )
    vlm = LiteLLMVLMProvider(
        {
            "provider": "litellm",
            "model": "claude-sonnet-test",
        }
    )
    monkeypatch.setattr(vlm, "update_token_usage", lambda **_kwargs: None)

    vlm._update_token_usage_from_response(
        SimpleNamespace(
            id="response-id",
            model="claude-response",
            choices=[SimpleNamespace(finish_reason="stop")],
            usage=SimpleNamespace(
                prompt_tokens=21,
                completion_tokens=8,
                total_tokens=29,
                prompt_tokens_details=None,
                completion_tokens_details=None,
            ),
        )
    )

    assert len(events) == 1
    name, attributes = events[0]
    assert name == "gen_ai.client.inference.operation.details"
    assert attributes["gen_ai.provider.name"] == "anthropic"
    assert attributes["gen_ai.request.model"] == "anthropic/claude-sonnet-test"
    assert attributes["gen_ai.usage.input_tokens"] == 21
    assert attributes["gen_ai.usage.output_tokens"] == 8
    assert type(attributes["gen_ai.usage.input_tokens"]) is int


def test_litellm_maps_moonshot_to_semantic_convention_provider(monkeypatch):
    events = []
    monkeypatch.setattr(
        tracer_module.tracer,
        "add_event",
        lambda name, attributes=None: events.append((name, attributes)),
    )
    vlm = LiteLLMVLMProvider({"provider": "litellm", "model": "kimi-k2"})

    vlm._record_inference_event(SimpleNamespace(), provider="moonshot")

    assert events[0][1]["gen_ai.provider.name"] == "moonshot_ai"


@pytest.mark.parametrize(
    ("model", "expected_provider"),
    [
        ("azure/deployment", "azure.ai.openai"),
        ("bedrock/anthropic.claude", "aws.bedrock"),
        ("vertex_ai/gemini-test", "gcp.vertex_ai"),
    ],
)
def test_litellm_explicit_routes_use_semantic_convention_provider(
    monkeypatch, model, expected_provider
):
    events = []
    monkeypatch.setattr(
        tracer_module.tracer,
        "add_event",
        lambda name, attributes=None: events.append((name, attributes)),
    )
    vlm = LiteLLMVLMProvider({"provider": "litellm", "model": model})
    monkeypatch.setattr(vlm, "update_token_usage", lambda **_kwargs: None)

    vlm._update_token_usage_from_response(SimpleNamespace(usage=None))

    assert events[0][1]["gen_ai.provider.name"] == expected_provider


@pytest.mark.asyncio
async def test_litellm_async_completion_exports_metadata_without_content(monkeypatch):
    sentinels = {
        "input": "LITELLM_INPUT_SECRET_17",
        "output": "LITELLM_OUTPUT_SECRET_23",
        "tool": "LITELLM_TOOL_SECRET_31",
    }
    response = SimpleNamespace(
        id="litellm-response-id",
        model="anthropic/claude-response",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=sentinels["output"], tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=4,
            completion_tokens=2,
            total_tokens=6,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        ),
    )

    async def acompletion(**_kwargs):
        return response

    monkeypatch.setattr(litellm_vlm, "acompletion", acompletion)
    vlm = LiteLLMVLMProvider({"provider": "litellm", "model": "claude-test", "max_retries": 0})
    monkeypatch.setattr(vlm, "update_token_usage", lambda **_kwargs: None)

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracer_module, "_otel_tracer", provider.get_tracer("test"))

    result = await vlm.get_completion_async(
        messages=[{"role": "user", "content": sentinels["input"]}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": sentinels["tool"],
                    "parameters": {"type": "object"},
                },
            }
        ],
    )

    assert result.content == sentinels["output"]
    span = exporter.get_finished_spans()[0]
    events = [
        event for event in span.events if event.name == "gen_ai.client.inference.operation.details"
    ]
    assert len(events) == 1
    assert events[0].attributes["gen_ai.provider.name"] == "anthropic"
    assert events[0].attributes["gen_ai.usage.input_tokens"] == 4
    exported = repr((span.attributes, tuple(span.events)))
    for sentinel in sentinels.values():
        assert sentinel not in exported


@pytest.mark.asyncio
async def test_litellm_async_failure_records_safe_explicit_route_metadata(monkeypatch):
    async def acompletion(**_kwargs):
        raise RuntimeError("LITELLM_PROVIDER_ERROR_SECRET_79")

    monkeypatch.setattr(litellm_vlm, "acompletion", acompletion)
    vlm = LiteLLMVLMProvider(
        {"provider": "litellm", "model": "bedrock/claude-test", "max_retries": 0}
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracer_module, "_otel_tracer", provider.get_tracer("test"))

    with pytest.raises(RuntimeError, match="LITELLM_PROVIDER_ERROR_SECRET_79"):
        await vlm.get_completion_async(prompt="LITELLM_REQUEST_SECRET_83")

    span = exporter.get_finished_spans()[0]
    events = [
        event for event in span.events if event.name == "gen_ai.client.inference.operation.details"
    ]
    assert len(events) == 1
    assert dict(events[0].attributes) == {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "aws.bedrock",
        "gen_ai.request.model": "bedrock/claude-test",
        "error.type": "builtins.RuntimeError",
    }
    exported = repr((span.attributes, tuple(span.events)))
    assert "LITELLM_PROVIDER_ERROR_SECRET_79" not in exported
    assert "LITELLM_REQUEST_SECRET_83" not in exported


@pytest.mark.parametrize(
    ("model", "expected_provider"),
    [
        ("azure_ai/claude-test", "azure.ai.inference"),
        ("azure_text/gpt-test", "azure.ai.openai"),
    ],
)
def test_litellm_azure_routes_are_not_conflated(monkeypatch, model, expected_provider):
    events = []
    monkeypatch.setattr(
        tracer_module.tracer,
        "add_event",
        lambda name, attributes=None: events.append((name, attributes)),
    )
    vlm = LiteLLMVLMProvider({"provider": "litellm", "model": model})

    vlm._record_inference_event(SimpleNamespace(), provider=model.partition("/")[0])

    assert events[0][1]["gen_ai.provider.name"] == expected_provider


def test_litellm_explicit_route_overrides_conflicting_provider_config(monkeypatch):
    events = []
    monkeypatch.setattr(
        tracer_module.tracer,
        "add_event",
        lambda name, attributes=None: events.append((name, attributes)),
    )
    vlm = LiteLLMVLMProvider(
        {
            "provider": "anthropic",
            "model": "bedrock/anthropic.claude",
            "api_key": "synthetic",
        }
    )

    provider, model = vlm._inference_event_target()
    vlm._record_inference_event(SimpleNamespace(), provider=provider, model=model)

    assert events[0][1]["gen_ai.provider.name"] == "aws.bedrock"
