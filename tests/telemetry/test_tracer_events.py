# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from openviking.session.memory.utils.json_parser import parse_json_with_stability
from openviking.telemetry import tracer_module


class _RecordingSpan:
    def __init__(self, *, end_time=None):
        self.end_time = end_time
        self.events = []
        self.statuses = []
        self.attributes = {}

    def add_event(self, name, attributes=None):
        self.events.append((name, attributes))

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def set_status(self, status):
        self.statuses.append(status)


def _enable_tracing(monkeypatch, span):
    monkeypatch.setattr(tracer_module, "_otel_tracer", object())
    monkeypatch.setattr(
        tracer_module,
        "otel_trace",
        SimpleNamespace(get_current_span=lambda: span),
    )


@pytest.fixture(autouse=True)
def _reset_content_capture(monkeypatch):
    monkeypatch.setattr(tracer_module, "_trace_capture_content", False)
    monkeypatch.setattr(tracer_module, "_trace_content_max_length", 4096)


def test_info_uses_stable_event_name_and_bounded_diagnostic_by_default(monkeypatch):
    span = _RecordingSpan()
    _enable_tracing(monkeypatch, span)

    tracer_module.tracer.info("MESSAGE_SECRET_iteration=7")

    assert len(span.events) == 1
    name, attributes = span.events[0]
    assert name == "openviking.log"
    assert attributes["openviking.log.message"] == "MESSAGE_SECRET_iteration=7"
    assert attributes["code.namespace"].endswith("test_tracer_events")
    assert attributes["code.function.name"] == (
        "test_tracer_events.test_info_uses_stable_event_name_and_bounded_diagnostic_by_default"
    )
    assert type(attributes["code.line.number"]) is int


def test_info_content_capture_is_opt_in_and_redacted(monkeypatch):
    span = _RecordingSpan()
    _enable_tracing(monkeypatch, span)
    tracer_module._configure_trace_content(True, 4096)

    tracer_module.tracer.info(
        'api_key="KEY_SECRET_17" Authorization: Bearer BEARER_SECRET_23 '
        "image=data:image/png;base64,IMAGE_SECRET_31"
    )

    _, attributes = span.events[0]
    message = attributes["openviking.log.message"]
    assert "KEY_SECRET_17" not in message
    assert "BEARER_SECRET_23" not in message
    assert "IMAGE_SECRET_31" not in message
    assert message.count("[redacted]") == 3


def test_sensitive_content_is_omitted_without_opt_in(monkeypatch):
    span = _RecordingSpan()
    _enable_tracing(monkeypatch, span)

    tracer_module.tracer.info("PROMPT_SECRET_37", contains_content=True)

    assert "openviking.log.message" not in span.events[0][1]


def test_sensitive_log_preserves_safe_typed_attributes(monkeypatch):
    span = _RecordingSpan()
    _enable_tracing(monkeypatch, span)

    tracer_module.tracer.info(
        "URI_SECRET_71",
        contains_content=True,
        attributes={"openviking.operation.count": 3, "openviking.success": True},
    )

    _, attributes = span.events[0]
    assert attributes["openviking.operation.count"] == 3
    assert type(attributes["openviking.operation.count"]) is int
    assert attributes["openviking.success"] is True
    assert "openviking.log.message" not in attributes
    assert "URI_SECRET_71" not in repr(attributes)


def test_decorator_arguments_and_result_are_content_opt_in(monkeypatch):
    span = _RecordingSpan()
    _enable_tracing(monkeypatch, span)

    @tracer_module.tracer("test.decorated", ignore_args=False, ignore_result=False)
    def decorated(value, *, metadata):
        return f"RESULT_SECRET_53:{value}:{metadata}"

    result = decorated("ARG_SECRET_59", metadata="KWARG_SECRET_61")

    assert result == "RESULT_SECRET_53:ARG_SECRET_59:KWARG_SECRET_61"
    assert "func_args" not in span.attributes
    assert "func_kwargs" not in span.attributes
    assert len(span.events) == 1
    assert span.events[0][0] == "openviking.log"
    assert "openviking.log.message" not in span.events[0][1]
    assert "SECRET" not in repr((span.attributes, span.events))


def test_decorator_content_capture_is_redacted_and_bounded(monkeypatch):
    span = _RecordingSpan()
    _enable_tracing(monkeypatch, span)
    tracer_module._configure_trace_content(True, 48)

    @tracer_module.tracer("test.decorated", ignore_args=False, ignore_result=False)
    def decorated(value):
        return value

    decorated("api_key=DECORATOR_SECRET_67 " + "x" * 80)

    assert "DECORATOR_SECRET_67" not in span.attributes["func_args"]
    assert len(span.attributes["func_args"]) == 48
    assert "DECORATOR_SECRET_67" not in span.events[0][1]["openviking.log.message"]
    assert len(span.events[0][1]["openviking.log.message"]) == 48


def test_sensitive_error_content_is_omitted_without_opt_in(monkeypatch):
    span = _RecordingSpan()
    _enable_tracing(monkeypatch, span)

    tracer_module.tracer.error(
        "MODEL_RESPONSE_SECRET_39",
        RuntimeError("PROVIDER_MESSAGE_SECRET_40"),
        console=False,
        contains_content=True,
    )

    name, attributes = span.events[0]
    assert name == "exception"
    assert attributes["exception.type"] == "builtins.RuntimeError"
    assert attributes["code.function.name"].endswith(
        "test_sensitive_error_content_is_omitted_without_opt_in"
    )
    assert "openviking.error.message" not in attributes
    assert "exception.message" not in attributes
    assert "exception.stacktrace" not in attributes


def test_content_capture_preserves_bounded_exception_stacktrace(monkeypatch):
    span = _RecordingSpan()
    _enable_tracing(monkeypatch, span)
    tracer_module._configure_trace_content(True, 4096)

    try:
        raise RuntimeError("STACKTRACE_MESSAGE_42")
    except RuntimeError as error:
        tracer_module.tracer.error("operation failed", error, console=False)

    _, attributes = span.events[0]
    assert "RuntimeError: STACKTRACE_MESSAGE_42" in attributes["exception.stacktrace"]


def test_default_diagnostic_redacts_project_credential_names(monkeypatch):
    span = _RecordingSpan()
    _enable_tracing(monkeypatch, span)

    tracer_module.tracer.info("OPENAI_API_KEY=OPENAI_SECRET_41 x-byteapm-appkey=BYTEAPM_SECRET_43")

    message = span.events[0][1]["openviking.log.message"]
    assert "OPENAI_SECRET_41" not in message
    assert "BYTEAPM_SECRET_43" not in message


def test_info_content_capture_is_bounded(monkeypatch):
    span = _RecordingSpan()
    _enable_tracing(monkeypatch, span)
    tracer_module._configure_trace_content(True, 24)

    tracer_module.tracer.info("x" * 100)

    message = span.events[0][1]["openviking.log.message"]
    assert len(message) == 24
    assert message.endswith("...[truncated]")


def test_add_event_preserves_attribute_types(monkeypatch):
    span = _RecordingSpan()
    _enable_tracing(monkeypatch, span)

    tracer_module.tracer.add_event(
        "gen_ai.client.inference.operation.details",
        {
            "gen_ai.usage.input_tokens": 17,
            "gen_ai.usage.output_tokens": 5,
            "openviking.cached": 3,
            "openviking.flag": True,
        },
    )

    _, attributes = span.events[0]
    assert attributes == {
        "gen_ai.usage.input_tokens": 17,
        "gen_ai.usage.output_tokens": 5,
        "openviking.cached": 3,
        "openviking.flag": True,
    }
    assert type(attributes["gen_ai.usage.input_tokens"]) is int
    assert type(attributes["gen_ai.usage.output_tokens"]) is int


def test_module_add_event_forwards_attributes(monkeypatch):
    span = _RecordingSpan()
    _enable_tracing(monkeypatch, span)

    tracer_module.add_event("test.event", {"count": 0})

    assert span.events == [("test.event", {"count": 0})]


def test_add_event_is_noop_when_tracing_is_disabled(monkeypatch):
    monkeypatch.setattr(tracer_module, "_otel_tracer", None)
    monkeypatch.setattr(
        tracer_module,
        "otel_trace",
        SimpleNamespace(
            get_current_span=lambda: (_ for _ in ()).throw(
                AssertionError("disabled tracing must not resolve the current span")
            )
        ),
    )

    tracer_module.tracer.add_event("test.event", {"count": 1})


def test_add_event_ignores_ended_span(monkeypatch):
    span = _RecordingSpan(end_time=1)
    _enable_tracing(monkeypatch, span)

    tracer_module.tracer.add_event("test.event", {"count": 1})

    assert span.events == []


def test_info_console_logging_is_unchanged_when_tracing_is_disabled(monkeypatch):
    messages = []
    fake_logger = SimpleNamespace(
        opt=lambda **_kwargs: SimpleNamespace(info=messages.append),
    )
    monkeypatch.setattr(tracer_module, "logger", fake_logger)
    monkeypatch.setattr(tracer_module, "_otel_tracer", None)

    tracer_module.tracer.info("still visible", console=True)

    assert messages == ["still visible"]


def test_init_from_server_config_forwards_content_settings(monkeypatch):
    captured = {}
    trace_config = SimpleNamespace(
        enabled=True,
        endpoint="",
        service_name="test",
        protocol="local",
        tls=SimpleNamespace(insecure=False),
        headers={},
        local_path="trace.jsonl",
        local_rotation_mb=40,
        local_backup_count=2,
        capture_content=True,
        content_max_length=1234,
    )
    server_config = SimpleNamespace(
        observability=SimpleNamespace(traces=trace_config),
    )
    monkeypatch.setattr(
        tracer_module,
        "init_tracer",
        lambda **kwargs: captured.update(kwargs) or "tracer",
    )

    result = tracer_module.init_tracer_from_server_config(server_config)

    assert result == "tracer"
    assert captured["capture_content"] is True
    assert captured["content_max_length"] == 1234


def test_disabled_trace_config_resets_content_capture(monkeypatch):
    monkeypatch.setattr(tracer_module, "_otel_tracer", object())
    monkeypatch.setattr(tracer_module, "_propagator", object())
    monkeypatch.setattr(tracer_module, "_trace_capture_content", True)
    server_config = SimpleNamespace(
        observability=SimpleNamespace(
            traces=SimpleNamespace(
                enabled=False,
                capture_content=False,
                content_max_length=4096,
            )
        )
    )

    assert tracer_module.init_tracer_from_server_config(server_config) is None
    assert tracer_module._otel_tracer is None
    assert tracer_module._propagator is None
    assert tracer_module._trace_capture_content is False


def test_failed_reinitialization_clears_old_tracer_and_does_not_enable_content(monkeypatch):
    monkeypatch.setattr(tracer_module, "_otel_tracer", object())
    monkeypatch.setattr(tracer_module, "_propagator", object())
    monkeypatch.setattr(tracer_module, "_trace_capture_content", False)
    monkeypatch.setattr(tracer_module, "OTLPGrpcSpanExporter", None)

    result = tracer_module.init_tracer(
        endpoint="localhost:4317",
        service_name="test",
        capture_content=True,
    )

    assert result is None
    assert tracer_module._otel_tracer is None
    assert tracer_module._propagator is None
    assert tracer_module._trace_capture_content is False


def test_invalid_content_limit_fails_before_exporter_creation(monkeypatch):
    exporter_calls = []
    monkeypatch.setattr(
        tracer_module,
        "OTLPGrpcSpanExporter",
        lambda **kwargs: exporter_calls.append(kwargs) or object(),
    )

    result = tracer_module.init_tracer(
        endpoint="localhost:4317",
        service_name="test",
        content_max_length=0,
    )

    assert result is None
    assert exporter_calls == []
    assert tracer_module._otel_tracer is None
    assert tracer_module._trace_capture_content is False


def test_validation_errors_do_not_export_model_content_by_default(monkeypatch):
    class _ExpectedInteger(BaseModel):
        count: int

    span = _RecordingSpan()
    _enable_tracing(monkeypatch, span)

    parse_json_with_stability(
        '{"count":"PYDANTIC_INPUT_SECRET_47"}',
        _ExpectedInteger,
    )

    assert "PYDANTIC_INPUT_SECRET_47" not in repr(span.events)


def test_real_sdk_reconfiguration_switches_cached_tracers_and_disables_export(tmp_path):
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    script = textwrap.dedent(
        f"""
        from opentelemetry import trace
        from openviking.telemetry import tracer_module

        first_path = {str(first_path)!r}
        second_path = {str(second_path)!r}

        assert tracer_module.init_tracer(
            endpoint="", service_name="test", protocol="local", local_path=first_path
        ) is not None
        cached_tracer = trace.get_tracer("cached")
        with cached_tracer.start_as_current_span("first-span"):
            pass
        assert tracer_module._owned_trace_provider.force_flush()

        assert tracer_module.init_tracer(
            endpoint="", service_name="test", protocol="local", local_path=second_path
        ) is not None
        with cached_tracer.start_as_current_span("second-span"):
            pass
        assert tracer_module._owned_trace_provider.force_flush()

        assert tracer_module.init_tracer(
            endpoint="", service_name="test", protocol="local", enabled=False
        ) is None
        with cached_tracer.start_as_current_span("disabled-span"):
            pass

        first = open(first_path, encoding="utf-8").read()
        second = open(second_path, encoding="utf-8").read()
        assert "first-span" in first
        assert "second-span" not in first
        assert "second-span" in second
        assert "disabled-span" not in second
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_real_sdk_preserves_preconfigured_host_provider(tmp_path):
    local_path = tmp_path / "openviking.jsonl"
    script = textwrap.dedent(
        f"""
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            SimpleSpanProcessor,
            SpanExporter,
            SpanExportResult,
        )

        class RecordingExporter(SpanExporter):
            def __init__(self):
                self.names = []

            def export(self, spans):
                self.names.extend(span.name for span in spans)
                return SpanExportResult.SUCCESS

        host_exporter = RecordingExporter()
        host_provider = TracerProvider()
        host_provider.add_span_processor(SimpleSpanProcessor(host_exporter))
        trace.set_tracer_provider(host_provider)
        host_tracer = trace.get_tracer("host")

        from openviking.telemetry import tracer_module

        with host_tracer.start_as_current_span("host-before"):
            pass
        assert tracer_module.init_tracer(
            endpoint="",
            service_name="openviking",
            protocol="local",
            local_path={str(local_path)!r},
        ) is not None
        assert trace.get_tracer_provider() is host_provider
        with tracer_module.get_tracer().start_as_current_span("openviking-span"):
            pass
        assert tracer_module._owned_trace_provider.force_flush()

        assert tracer_module.init_tracer(
            endpoint="", service_name="openviking", protocol="local", enabled=False
        ) is None
        assert trace.get_tracer_provider() is host_provider
        with host_tracer.start_as_current_span("host-after"):
            pass

        assert host_exporter.names == ["host-before", "host-after"]
        payload = open({str(local_path)!r}, encoding="utf-8").read()
        assert "openviking-span" in payload
        assert "host-before" not in payload
        assert "host-after" not in payload
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
