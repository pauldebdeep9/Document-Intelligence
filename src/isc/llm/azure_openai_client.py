"""Azure OpenAI implementation. Deliberately written now, while the shape is fresh.

Only three things differ from the OpenAI client, and all three are why the port
exists:

  1. Addressing      -> `deployment` name, not `model` name.
  2. Authentication  -> DefaultAzureCredential / managed identity, not an API key.
                        Key auth works but should not be what this repo models,
                        because Rockwell will not hand out static keys.
  3. API versioning  -> explicit api_version; structured-output support is
                        version-gated, so this is a real compatibility surface.

Left unwired until an Azure resource exists. The point is that the diff against
openai_client.py stays this small.
"""

from __future__ import annotations

from typing import Sequence

from pydantic import BaseModel

from isc.common.config import Settings
from isc.common.errors import ConfigError
from isc.llm.cache import ResponseCache
from isc.llm.ports import LLMResult, Message


class AzureOpenAIChatModel:
    def __init__(self, settings: Settings, cache: ResponseCache) -> None:
        if not settings.llm.chat_deployment:
            raise ConfigError("azure provider requires llm.chat_deployment")
        self._s = settings
        self._cache = cache
        # from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        # from openai import AzureOpenAI
        # token_provider = get_bearer_token_provider(
        #     DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
        # self._client = AzureOpenAI(
        #     azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        #     api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        #     azure_ad_token_provider=token_provider,
        # )
        raise NotImplementedError("wire up when the Azure resource exists")

    def complete(
        self,
        messages: Sequence[Message],
        *,
        schema: type[BaseModel] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult:
        # Identical body to OpenAIChatModel except: payload["model"] = deployment.
        raise NotImplementedError


class AzureOpenAIEmbeddingModel:
    def __init__(self, settings: Settings, cache: ResponseCache) -> None:
        raise NotImplementedError("wire up when the Azure resource exists")

    @property
    def dimensions(self) -> int:
        raise NotImplementedError

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError
