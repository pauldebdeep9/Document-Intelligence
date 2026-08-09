"""Layered configuration: default.yaml <- local.yaml <- environment.

Nested settings use a double underscore: ISC_LLM__CHAT_MODEL=gpt-4o.
Nothing reads os.environ directly outside this module.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from isc.common.errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "config"


class LLMSettings(BaseModel):
    provider: Literal["openai", "azure"] = "openai"
    chat_model: str = "gpt-4o-mini"
    embed_model: str = "text-embedding-3-small"
    # Azure addresses deployments, not models. Kept separate so the swap is config-only.
    chat_deployment: str | None = None
    embed_deployment: str | None = None
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_s: float = 60.0
    max_retries: int = 3
    embed_batch_size: int = 64


class CacheSettings(BaseModel):
    enabled: bool = True
    dir: Path = REPO_ROOT / ".cache"


class ChunkSettings(BaseModel):
    target_tokens: int = 512
    overlap_tokens: int = 64
    keep_tables_whole: bool = True
    max_table_tokens: int = 1500


class RetrievalSettings(BaseModel):
    top_k_dense: int = 20
    top_k_lexical: int = 20
    rrf_k: int = 60
    final_k: int = 8
    min_support_score: float = 0.35   # below this, answer/ abstains


class ThresholdSettings(BaseModel):
    auto_accept: float = 0.90
    review: float = 0.60
    reject: float = 0.30


class PathSettings(BaseModel):
    data: Path = REPO_ROOT / "data"
    runs: Path = REPO_ROOT / "runs"
    prompts: Path = CONFIG_DIR / "prompts"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ISC_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    env: str = "local"
    llm: LLMSettings = Field(default_factory=LLMSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    chunk: ChunkSettings = Field(default_factory=ChunkSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    thresholds: ThresholdSettings = Field(default_factory=ThresholdSettings)
    paths: PathSettings = Field(default_factory=PathSettings)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return data


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    merged = _deep_merge(
        _load_yaml(CONFIG_DIR / "default.yaml"),
        _load_yaml(CONFIG_DIR / "local.yaml"),
    )
    return Settings(**merged)


def load_prompt(relpath: str) -> str:
    """Prompts live on disk and are versioned in the filename, never inlined.

    relpath example: 'extract/purchase_order.v1.md'
    """
    path = get_settings().paths.prompts / relpath
    if not path.exists():
        raise ConfigError(f"prompt not found: {path}")
    return path.read_text()
