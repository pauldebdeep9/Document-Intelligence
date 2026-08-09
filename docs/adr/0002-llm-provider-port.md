# ADR 0002: All model access goes through a provider port

**Status:** accepted · **Date:** 2026-08-09

## Context
Development runs against OpenAI because it is available now. The target
environment is Azure OpenAI. The two SDKs are close but not identical:
deployments instead of models, AAD instead of API keys, version-gated structured
output support.

The failure mode is not "the migration is hard" — it is that provider details
leak into extractors, retrievers and prompts, so the migration becomes a
refactor of the whole pipeline rather than a configuration change.

## Decision
- `isc/llm/ports.py` defines `ChatModel` and `EmbeddingModel` Protocols plus the
  value objects they exchange (`Message`, `LLMResult`, `Usage`).
- Only `openai_client.py` and `azure_openai_client.py` may import `openai`.
- `tests/unit/test_import_hygiene.py` enforces this by AST inspection.
- `registry.py` is the single place a provider name becomes a concrete class.

## Consequences
- Provider swap is one config value.
- Provider-specific features are unavailable unless promoted into the port,
  which is a deliberate friction.
- `LLMResult.model_confidence()` returns 0.75 when logprobs are absent, so a
  provider that hides them cannot make a field look auto-acceptable.

## Revisit if
A provider capability that materially improves extraction quality cannot be
expressed through the port.
