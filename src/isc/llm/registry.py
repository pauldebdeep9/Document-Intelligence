"""Provider selection. The single place a provider name becomes a concrete class."""

from __future__ import annotations

from functools import lru_cache

from isc.common.config import Settings, get_settings
from isc.common.errors import ConfigError
from isc.llm.cache import ResponseCache
from isc.llm.ports import ChatModel, EmbeddingModel


def _cache(s: Settings) -> ResponseCache:
    return ResponseCache(s.cache.dir, s.cache.enabled)


@lru_cache(maxsize=4)
def get_chat_model(provider: str | None = None) -> ChatModel:
    s = get_settings()
    match provider or s.llm.provider:
        case "openai":
            from isc.llm.openai_client import OpenAIChatModel
            return OpenAIChatModel(s, _cache(s))
        case "azure":
            from isc.llm.azure_openai_client import AzureOpenAIChatModel
            return AzureOpenAIChatModel(s, _cache(s))
        case other:
            raise ConfigError(f"unknown llm provider: {other!r}")


@lru_cache(maxsize=4)
def get_embedding_model(provider: str | None = None) -> EmbeddingModel:
    s = get_settings()
    match provider or s.llm.provider:
        case "openai":
            from isc.llm.openai_client import OpenAIEmbeddingModel
            return OpenAIEmbeddingModel(s, _cache(s))
        case "azure":
            from isc.llm.azure_openai_client import AzureOpenAIEmbeddingModel
            return AzureOpenAIEmbeddingModel(s, _cache(s))
        case other:
            raise ConfigError(f"unknown llm provider: {other!r}")
