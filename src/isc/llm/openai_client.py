"""OpenAI implementation of the ChatModel / EmbeddingModel ports.

The only module (with azure_openai_client) permitted to import `openai`.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Sequence

from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel

from isc.common.config import Settings
from isc.common.errors import ProviderError
from isc.common.logging import get_logger
from isc.common.tracing import current_run, span
from isc.llm.cache import ResponseCache
from isc.llm.cost import estimate_usd
from isc.llm.ports import LLMResult, Message, Usage
from isc.llm.schema import to_strict_schema

log = get_logger("llm.openai")

_RETRYABLE = (RateLimitError, APITimeoutError)


class OpenAIChatModel:
    def __init__(self, settings: Settings, cache: ResponseCache) -> None:
        self._s = settings
        self._cache = cache
        self._client = OpenAI(timeout=settings.llm.timeout_s, max_retries=0)
        self.model = settings.llm.chat_model

    def complete(
        self,
        messages: Sequence[Message],
        *,
        schema: type[BaseModel] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [asdict(m) for m in messages],
            "temperature": self._s.llm.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self._s.llm.max_tokens,
            "logprobs": True,
        }
        if schema is not None:
            # Strict structured output. structured.py still validates: a provider
            # accepting the schema is not the same as the payload being correct.
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": to_strict_schema(schema),
                    "strict": True,
                },
            }

        key = self._cache.key_for("chat", payload)
        if (hit := self._cache.get(key)) is not None:
            return LLMResult.from_cache_payload(hit)

        with span("llm.complete", model=self.model, schema=schema.__name__ if schema else None):
            raw = self._call_with_retry(payload)

        choice = raw.choices[0]
        result = LLMResult(
            text=choice.message.content or "",
            model=raw.model,
            usage=_usage_from(raw),
            finish_reason=choice.finish_reason or "stop",
            mean_logprob=_mean_logprob(choice),
        )
        self._record(result)
        self._cache.put(key, result.to_cache_payload())
        return result

    def _call_with_retry(self, payload: dict[str, Any]) -> Any:
        delay = 1.0
        last: Exception | None = None
        for attempt in range(self._s.llm.max_retries):
            try:
                return self._client.chat.completions.create(**payload)
            except _RETRYABLE as exc:
                last = exc
                log.warning("retryable provider error (attempt %d): %s", attempt + 1, exc)
                time.sleep(delay)
                delay *= 2
            except APIError as exc:
                raise ProviderError(f"openai chat failed: {exc}") from exc
        raise ProviderError(f"openai chat failed after retries: {last}")

    def _record(self, result: LLMResult) -> None:
        if (run := current_run()) is not None:
            run.add_total("tokens.prompt", result.usage.prompt_tokens)
            run.add_total("tokens.completion", result.usage.completion_tokens)
            run.add_total("usd", estimate_usd(result.model, result.usage))


class OpenAIEmbeddingModel:
    _DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}

    def __init__(self, settings: Settings, cache: ResponseCache) -> None:
        self._s = settings
        self._cache = cache
        self._client = OpenAI(timeout=settings.llm.timeout_s, max_retries=0)
        self.model = settings.llm.embed_model

    @property
    def dimensions(self) -> int:
        return self._DIMS.get(self.model, 1536)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = [None] * len(texts)  # type: ignore[list-item]
        pending: list[tuple[int, str]] = []

        for i, t in enumerate(texts):
            key = self._cache.key_for("embed", {"model": self.model, "text": t})
            hit = self._cache.get(key)
            if hit is not None:
                out[i] = hit["vector"]
            else:
                pending.append((i, t))

        batch = self._s.llm.embed_batch_size
        for start in range(0, len(pending), batch):
            window = pending[start:start + batch]
            with span("llm.embed", model=self.model, n=len(window)):
                try:
                    resp = self._client.embeddings.create(
                        model=self.model, input=[t for _, t in window]
                    )
                except APIError as exc:
                    raise ProviderError(f"openai embed failed: {exc}") from exc
            for (idx, text), item in zip(window, resp.data, strict=True):
                out[idx] = item.embedding
                self._cache.put(
                    self._cache.key_for("embed", {"model": self.model, "text": text}),
                    {"vector": item.embedding},
                )
            if (run := current_run()) is not None:
                run.add_total("tokens.embed", resp.usage.total_tokens)
        return out


def _usage_from(raw: Any) -> Usage:
    u = getattr(raw, "usage", None)
    if u is None:
        return Usage()
    cached = 0
    details = getattr(u, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", 0) or 0
    return Usage(u.prompt_tokens, u.completion_tokens, cached)


def _mean_logprob(choice: Any) -> float | None:
    lp = getattr(choice, "logprobs", None)
    tokens = getattr(lp, "content", None) if lp else None
    if not tokens:
        return None
    values = [t.logprob for t in tokens if t.logprob is not None]
    return sum(values) / len(values) if values else None
