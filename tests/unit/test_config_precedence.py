"""Configuration precedence, highest first: explicit Settings(...) kwargs,
environment (ISC_ prefix, __ nesting), .env, config/local.yaml,
config/default.yaml. The bug this guards: the YAML was passed to Settings as
constructor kwargs, which pydantic-settings ranks ABOVE the environment, so no
ISC_* variable could override any key default.yaml sets -- including the
documented ISC_AGGREGATE__ENABLED=true flip for a live run."""

from __future__ import annotations

import os

import pytest

from isc.common.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _fresh_settings(monkeypatch):
    for name in list(os.environ):
        if name.startswith("ISC_"):
            monkeypatch.delenv(name)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_yaml_value_applies_without_env():
    assert get_settings().aggregate.enabled is False


def test_env_overrides_a_key_present_in_default_yaml(monkeypatch):
    monkeypatch.setenv("ISC_AGGREGATE__ENABLED", "true")
    assert get_settings().aggregate.enabled is True


def test_env_overrides_a_nested_llm_key(monkeypatch):
    monkeypatch.setenv("ISC_LLM__CHAT_MODEL", "some-other-model")
    assert get_settings().llm.chat_model == "some-other-model"


def test_explicit_kwargs_still_win(monkeypatch):
    monkeypatch.setenv("ISC_AGGREGATE__ENABLED", "true")
    assert Settings(aggregate={"enabled": False}).aggregate.enabled is False


def test_yaml_values_still_load():
    """llm.max_tokens is 8192 in default.yaml and 2048 in code, so this only
    passes if the YAML was actually read."""
    assert get_settings().llm.max_tokens == 8192
