"""The provider boundary. Two Protocols and the value objects they exchange.

Hard rule, enforced by tests/unit/test_import_hygiene.py: no module outside
isc.llm may import `openai`. Every caller depends on these Protocols.

This is what makes the eventual OpenAI -> Azure OpenAI move a config change.
The Azure SDK is the same client with a deployment name instead of a model name
and AAD instead of an API key; if that difference is visible above this line, the
migration turns into a refactor of every extractor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str

    @staticmethod
    def system(content: str) -> Message:
        return Message("system", content)

    @staticmethod
    def user(content: str) -> Message:
        return Message("user", content)


@dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.cached_prompt_tokens + other.cached_prompt_tokens,
        )


@dataclass(frozen=True, slots=True)
class LLMResult:
    text: str
    model: str
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = "stop"
    # Mean per-token logprob when the provider exposes it. Feeds Signal.MODEL.
    mean_logprob: float | None = None
    # Whether this result was served from ResponseCache rather than the
    # provider. NOTE: because this dataclass is slots=True, this field's slot
    # descriptor would shadow any method of the same name -- that is why the
    # (de)serialisation helpers below are to_cache_payload/from_cache_payload,
    # not to_cache/from_cache. `LLMResult.from_cache` used to resolve to this
    # field's descriptor instead of the classmethod, so every cache hit raised
    # "'member_descriptor' object is not callable".
    from_cache: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    def to_cache_payload(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "cached_prompt_tokens": self.usage.cached_prompt_tokens,
            },
            "finish_reason": self.finish_reason,
            "mean_logprob": self.mean_logprob,
        }

    @classmethod
    def from_cache_payload(cls, data: dict[str, Any]) -> LLMResult:
        return cls(
            text=data["text"],
            model=data["model"],
            usage=Usage(**data["usage"]),
            finish_reason=data["finish_reason"],
            mean_logprob=data.get("mean_logprob"),
            from_cache=True,
        )

    def model_confidence(self) -> float:
        """Map logprob to [0,1]. Absent logprobs are treated as moderate, not certain.

        Deliberately conservative: a provider that hides logprobs must not be able
        to make a field look auto-acceptable.
        """
        import math

        if self.mean_logprob is None:
            return 0.75
        return max(0.0, min(1.0, math.exp(self.mean_logprob)))


@runtime_checkable
class ChatModel(Protocol):
    def complete(
        self,
        messages: Sequence[Message],
        *,
        schema: type[BaseModel] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult:
        """Return one completion. If `schema` is given, the provider must be asked
        for output conforming to it; validation still happens in structured.py."""
        ...


@runtime_checkable
class EmbeddingModel(Protocol):
    @property
    def dimensions(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Order-preserving. Batching is the implementation's problem, not the caller's."""
        ...
