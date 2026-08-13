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

from openviking.models.vlm.backends import volcengine_vlm
from openviking.models.vlm.backends.openai_vlm import OpenAIVLM
from openviking.models.vlm.backends.volcengine_vlm import VolcEngineVLM
from openviking.telemetry import tracer_module


@pytest.mark.asyncio
async def test_async_completion_exports_typed_metadata_without_content(monkeypatch):
    sentinels = {
        "prompt": "PROMPT_SECRET_7d0713",
        "message": "MESSAGE_SECRET_9af3ea",
        "tool": "TOOL_SECRET_a341c2",
        "body": "BODY_SECRET_06fc88",
        "header": "HEADER_SECRET_58e4d1",
        "output": "OUTPUT_SECRET_d497a5",
    }
    usage = SimpleNamespace(
        prompt_tokens=17,
        completion_tokens=5,
        total_tokens=22,
        prompt_tokens_details=SimpleNamespace(
            cached_tokens=3,
            cache_write_tokens=6,
            audio_tokens=2,
        ),
        completion_tokens_details=SimpleNamespace(
            reasoning_tokens=4,
            audio_tokens=1,
            accepted_prediction_tokens=2,
            rejected_prediction_tokens=1,
        ),
    )
    response = SimpleNamespace(
        id="chatcmpl-safe-id",
        model="gpt-safe-response-model",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=sentinels["output"], tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=usage,
    )
    received = {}

    async def create(**kwargs):
        received.update(kwargs)
        return response

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )
    vlm = OpenAIVLM(
        {
            "provider": "openai",
            "model": "gpt-safe-request-model",
            "extra_headers": {"x-sensitive": sentinels["header"]},
            "extra_request_body": {"metadata": sentinels["body"]},
        }
    )
    monkeypatch.setattr(vlm, "get_async_client", lambda: client)
    monkeypatch.setattr(vlm, "update_token_usage", lambda **_kwargs: None)

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracer_module, "_otel_tracer", provider.get_tracer("test"))

    result = await vlm.get_completion_async(
        prompt=sentinels["prompt"],
        messages=[{"role": "user", "content": sentinels["message"]}],
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
    assert received["messages"][0]["content"] == sentinels["message"]
    assert received["tools"][0]["function"]["description"] == sentinels["tool"]
    assert received["extra_body"]["metadata"] == sentinels["body"]

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "openai.vlm.call"
    gen_ai_events = [
        event for event in span.events if event.name == "gen_ai.client.inference.operation.details"
    ]
    assert len(gen_ai_events) == 1
    event = gen_ai_events[0]
    assert event.name == "gen_ai.client.inference.operation.details"
    assert dict(event.attributes) == {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "openai",
        "gen_ai.request.model": "gpt-safe-request-model",
        "gen_ai.response.id": "chatcmpl-safe-id",
        "gen_ai.response.model": "gpt-safe-response-model",
        "gen_ai.response.finish_reasons": ("stop",),
        "gen_ai.usage.input_tokens": 17,
        "gen_ai.usage.output_tokens": 5,
        "gen_ai.usage.cache_creation.input_tokens": 6,
        "gen_ai.usage.cache_read.input_tokens": 3,
        "gen_ai.usage.reasoning.output_tokens": 4,
        "openviking.gen_ai.usage.total_tokens": 22,
        "openviking.gen_ai.usage.input_audio_tokens": 2,
        "openviking.gen_ai.usage.output_audio_tokens": 1,
        "openviking.gen_ai.usage.accepted_prediction_tokens": 2,
        "openviking.gen_ai.usage.rejected_prediction_tokens": 1,
    }
    assert type(event.attributes["gen_ai.usage.input_tokens"]) is int
    assert type(event.attributes["gen_ai.usage.output_tokens"]) is int

    exported = repr((span.attributes, tuple(span.events)))
    for sentinel in sentinels.values():
        assert sentinel not in exported


@pytest.mark.parametrize(
    ("provider", "expected"),
    [("openai", "openai"), ("azure", "azure.ai.openai")],
)
def test_provider_name_uses_semantic_convention_value(monkeypatch, provider, expected):
    events = []
    monkeypatch.setattr(
        tracer_module.tracer,
        "add_event",
        lambda name, attributes=None: events.append((name, attributes)),
    )
    vlm = OpenAIVLM({"provider": provider, "model": "test-model"})

    vlm._record_inference_event(SimpleNamespace())

    assert events[0][1]["gen_ai.provider.name"] == expected


def test_stream_attribute_comes_from_actual_request(monkeypatch):
    events = []
    monkeypatch.setattr(
        tracer_module.tracer,
        "add_event",
        lambda name, attributes=None: events.append((name, attributes)),
    )
    vlm = OpenAIVLM({"provider": "openai", "model": "test-model"})

    vlm._record_inference_event(SimpleNamespace(), request={"stream": True})

    assert events[0][1]["gen_ai.request.stream"] is True


def test_configured_stream_is_not_reported_when_request_does_not_stream(monkeypatch):
    events = []
    monkeypatch.setattr(
        tracer_module.tracer,
        "add_event",
        lambda name, attributes=None: events.append((name, attributes)),
    )
    vlm = OpenAIVLM({"provider": "openai", "model": "test-model", "stream": True})

    vlm._record_inference_event(SimpleNamespace(), request={"model": "test-model"})

    assert "gen_ai.request.stream" not in events[0][1]


def test_completed_response_without_usage_still_records_request_metadata(monkeypatch):
    events = []
    monkeypatch.setattr(
        tracer_module.tracer,
        "add_event",
        lambda name, attributes=None: events.append((name, attributes)),
    )
    vlm = OpenAIVLM({"provider": "openai", "model": "test-model"})

    vlm._update_token_usage_from_response(
        SimpleNamespace(
            id="response-id",
            model="response-model",
            choices=[SimpleNamespace(finish_reason="length")],
            usage=None,
        )
    )

    assert events == [
        (
            "gen_ai.client.inference.operation.details",
            {
                "gen_ai.operation.name": "chat",
                "gen_ai.provider.name": "openai",
                "gen_ai.request.model": "test-model",
                "gen_ai.response.id": "response-id",
                "gen_ai.response.model": "response-model",
                "gen_ai.response.finish_reasons": ["length"],
            },
        )
    ]


def test_zero_token_counts_are_preserved(monkeypatch):
    events = []
    monkeypatch.setattr(
        tracer_module.tracer,
        "add_event",
        lambda name, attributes=None: events.append((name, attributes)),
    )
    vlm = OpenAIVLM({"provider": "openai", "model": "test-model"})
    monkeypatch.setattr(vlm, "update_token_usage", lambda **_kwargs: None)

    vlm._update_token_usage_from_response(
        SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                prompt_tokens_details=SimpleNamespace(cached_tokens=0),
                completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
            )
        )
    )

    attributes = events[0][1]
    assert attributes["gen_ai.usage.input_tokens"] == 0
    assert attributes["gen_ai.usage.output_tokens"] == 0
    assert attributes["gen_ai.usage.cache_read.input_tokens"] == 0
    assert attributes["gen_ai.usage.reasoning.output_tokens"] == 0
    assert attributes["openviking.gen_ai.usage.total_tokens"] == 0


def test_standalone_sync_completion_exports_default_request_model(monkeypatch):
    response = SimpleNamespace(
        id="sync-response-id",
        model="gpt-response-model",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="SYNC_OUTPUT_SECRET_45", tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=None,
    )
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: response),
        )
    )
    vlm = OpenAIVLM({"provider": "openai", "max_retries": 0})
    monkeypatch.setattr(vlm, "get_client", lambda: client)

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracer_module, "_otel_tracer", provider.get_tracer("test"))

    assert vlm.get_completion(prompt="SYNC_INPUT_SECRET_43") == "SYNC_OUTPUT_SECRET_45"

    span = exporter.get_finished_spans()[0]
    event = next(
        event for event in span.events if event.name == "gen_ai.client.inference.operation.details"
    )
    assert span.name == "openai.vlm.call"
    assert event.attributes["gen_ai.request.model"] == "gpt-4o-mini"
    exported = repr((span.attributes, tuple(span.events)))
    assert "SYNC_INPUT_SECRET_43" not in exported
    assert "SYNC_OUTPUT_SECRET_45" not in exported


@pytest.mark.asyncio
async def test_standalone_async_vision_completion_exports_event(monkeypatch):
    response = SimpleNamespace(
        id="vision-response-id",
        model="gpt-vision-response",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="VISION_OUTPUT_SECRET_51", tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=None,
    )

    async def create(**_kwargs):
        return response

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )
    vlm = OpenAIVLM({"provider": "openai", "model": "gpt-vision-request"})
    monkeypatch.setattr(vlm, "get_async_client", lambda: client)

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracer_module, "_otel_tracer", provider.get_tracer("test"))

    result = await vlm.get_vision_completion_async(
        messages=[{"role": "user", "content": "VISION_INPUT_SECRET_49"}]
    )

    assert result == "VISION_OUTPUT_SECRET_51"
    span = exporter.get_finished_spans()[0]
    events = [
        event for event in span.events if event.name == "gen_ai.client.inference.operation.details"
    ]
    assert span.name == "openai.vlm.call"
    assert len(events) == 1
    assert events[0].attributes["gen_ai.request.model"] == "gpt-vision-request"
    exported = repr((span.attributes, tuple(span.events)))
    assert "VISION_INPUT_SECRET_49" not in exported
    assert "VISION_OUTPUT_SECRET_51" not in exported


def test_volcengine_inherits_structured_inference_event(monkeypatch):
    events = []
    monkeypatch.setattr(
        tracer_module.tracer,
        "add_event",
        lambda name, attributes=None: events.append((name, attributes)),
    )
    vlm = VolcEngineVLM({"model": "doubao-test"})
    monkeypatch.setattr(vlm, "update_token_usage", lambda **_kwargs: None)

    vlm._update_token_usage_from_response(
        SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=9,
                completion_tokens=2,
                total_tokens=11,
                prompt_tokens_details=None,
                completion_tokens_details=None,
            )
        )
    )

    assert len(events) == 1
    assert events[0][0] == "gen_ai.client.inference.operation.details"
    assert events[0][1]["gen_ai.provider.name"] == "volcengine"
    assert events[0][1]["gen_ai.usage.input_tokens"] == 9


@pytest.mark.asyncio
async def test_async_failure_records_one_safe_terminal_event(monkeypatch):
    async def create(**_kwargs):
        raise RuntimeError("PROVIDER_ERROR_SECRET_47")

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )
    vlm = OpenAIVLM({"provider": "openai", "model": "gpt-failure-model", "max_retries": 0})
    monkeypatch.setattr(vlm, "get_async_client", lambda: client)

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracer_module, "_otel_tracer", provider.get_tracer("test"))

    with pytest.raises(RuntimeError, match="PROVIDER_ERROR_SECRET_47"):
        await vlm.get_completion_async(prompt="REQUEST_SECRET_53")

    span = exporter.get_finished_spans()[0]
    events = [
        event for event in span.events if event.name == "gen_ai.client.inference.operation.details"
    ]
    assert len(events) == 1
    assert dict(events[0].attributes) == {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "openai",
        "gen_ai.request.model": "gpt-failure-model",
        "error.type": "builtins.RuntimeError",
    }
    exported = repr((span.attributes, tuple(span.events)))
    assert "PROVIDER_ERROR_SECRET_47" not in exported
    assert "REQUEST_SECRET_53" not in exported
    exception_events = [event for event in span.events if event.name == "exception"]
    assert len(exception_events) == 1
    assert exception_events[0].attributes["exception.type"] == "builtins.RuntimeError"
    assert "exception.message" not in exception_events[0].attributes


@pytest.mark.asyncio
async def test_malformed_response_records_failure_without_success_event(monkeypatch):
    response = SimpleNamespace(
        id="malformed-response-id",
        model="gpt-response-model",
        choices=[],
        usage=SimpleNamespace(
            prompt_tokens=3,
            completion_tokens=1,
            total_tokens=4,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        ),
    )

    async def create(**_kwargs):
        return response

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )
    vlm = OpenAIVLM({"provider": "openai", "model": "gpt-malformed-model", "max_retries": 0})
    monkeypatch.setattr(vlm, "get_async_client", lambda: client)

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracer_module, "_otel_tracer", provider.get_tracer("test"))

    with pytest.raises(IndexError):
        await vlm.get_completion_async(prompt="safe synthetic input")

    span = exporter.get_finished_spans()[0]
    events = [
        event for event in span.events if event.name == "gen_ai.client.inference.operation.details"
    ]
    assert len(events) == 1
    assert events[0].attributes["error.type"] == "builtins.IndexError"
    assert events[0].attributes["gen_ai.response.id"] == "malformed-response-id"
    assert events[0].attributes["gen_ai.usage.input_tokens"] == 3
    assert events[0].attributes["gen_ai.usage.output_tokens"] == 1


@pytest.mark.asyncio
async def test_volcengine_async_completion_exports_metadata_without_content(monkeypatch):
    sentinels = {
        "input": "VOLC_INPUT_SECRET_59",
        "output": "VOLC_OUTPUT_SECRET_61",
        "tool": "VOLC_TOOL_SECRET_67",
    }
    response = SimpleNamespace(
        id="volc-response-id",
        model="doubao-response-model",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=sentinels["output"], tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=8,
            completion_tokens=3,
            total_tokens=11,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        ),
    )

    async def create(**_kwargs):
        return response

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )
    vlm = VolcEngineVLM({"model": "doubao-request-model", "max_retries": 0})
    monkeypatch.setattr(vlm, "get_async_client", lambda: client)
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
    assert events[0].attributes["gen_ai.provider.name"] == "volcengine"
    assert events[0].attributes["gen_ai.usage.output_tokens"] == 3
    exported = repr((span.attributes, tuple(span.events)))
    for sentinel in sentinels.values():
        assert sentinel not in exported


@pytest.mark.asyncio
async def test_volcengine_async_failure_records_once_after_retries(monkeypatch):
    attempts = 0

    async def create(**_kwargs):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("VOLC_PROVIDER_ERROR_SECRET_71")

    async def no_sleep(_delay):
        return None

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )
    vlm = VolcEngineVLM({"model": "doubao-failure-model", "max_retries": 1})
    monkeypatch.setattr(vlm, "get_async_client", lambda: client)
    monkeypatch.setattr(volcengine_vlm.asyncio, "sleep", no_sleep)

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracer_module, "_otel_tracer", provider.get_tracer("test"))

    with pytest.raises(RuntimeError, match="VOLC_PROVIDER_ERROR_SECRET_71"):
        await vlm.get_completion_async(prompt="VOLC_REQUEST_SECRET_73")

    assert attempts == 2
    span = exporter.get_finished_spans()[0]
    events = [
        event for event in span.events if event.name == "gen_ai.client.inference.operation.details"
    ]
    assert len(events) == 1
    assert events[0].attributes["error.type"] == "builtins.RuntimeError"
    assert "VOLC_PROVIDER_ERROR_SECRET_71" not in repr(events[0])
