"""Layered configuration: default.yaml <- local.yaml <- environment.

Nested settings use a double underscore: ISC_LLM__CHAT_MODEL=gpt-4o.
Nothing reads os.environ directly outside this module.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from isc.common.errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "config"

# pydantic-settings reads .env to populate Settings below, but it does NOT
# export those values to os.environ -- and provider SDKs (OpenAI, etc.) read
# os.environ directly, not Settings. OPENAI_API_KEY also doesn't match
# env_prefix="ISC_", so Settings ignores it entirely regardless. Without this
# call the key sits in .env and get_chat_model() fails as "Missing
# credentials" even though the file looks correctly configured.
# override=False so a real environment variable (CI, Azure managed identity)
# always wins over the file.
load_dotenv(REPO_ROOT / ".env", override=False)


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
    # 800, not the original 1500: measured against the real 20-doc corpus
    # (see docs/adr/0007), per-row cost is ~43 tokens median, and 1500 lets
    # a 38-41 row table split into as few as 2 parts -- "rows 10-1180" is
    # barely more useful as a citation than "the table". 800 keeps small
    # tables (the common case, almost all under 250 tokens) whole while
    # forcing the four largest into 3-4 parts of ~15-18 rows each, roughly
    # the same retrieval-unit scale as a prose chunk at target_tokens=512.
    max_table_tokens: int = 800


class RetrievalSettings(BaseModel):
    top_k_dense: int = 20
    top_k_lexical: int = 20
    rrf_k: int = 60
    final_k: int = 8
    # 0.0 disables the LOW_SUPPORT gate -- measured, not guessed: it compares
    # against an RRF-fused score that is rank-bounded (~0.033 ceiling under
    # these settings, regardless of relevance) and against dense_score, whose
    # answerable/unanswerable distributions overlap too much on this corpus
    # to support any cutoff. See config/default.yaml's own comment and
    # docs/adr/0008. Kept as a real field, not removed: it becomes live again
    # with a reranker score built to express relevance.
    min_support_score: float = 0.0


class ThresholdSettings(BaseModel):
    auto_accept: float = 0.90
    review: float = 0.60
    reject: float = 0.30


_ROUTES = frozenset({"accept", "review", "low_confidence", "reject"})


class AggregateSettings(BaseModel):
    """aggregate/: answering total-spend and part-price questions from
    extracted records instead of chunks. See docs/adr/0011."""

    # On by default since AG-11, matching config/default.yaml: the live
    # retrieval eval (runs run_20260928T145343Z A flag off,
    # run_20260928T150038Z B1, run_20260928T150333Z B2) measured 9/9 on the
    # aggregate questions in both B runs, 0 misroutes, 0 leaks.
    # ISC_AGGREGATE__ENABLED=false turns it off.
    enabled: bool = True
    # Which Thresholds.route() bands may contribute to a computed figure.
    # Not a new threshold -- the bands are the existing extraction ones.
    # "review" is included by default because P1-03 measured 0 wrong of 93
    # review-band values; each one is still named in the answer as pending
    # review. A value outside these bands is excluded AND named, never
    # silently dropped.
    contributing_routes: tuple[str, ...] = ("accept", "review")

    @field_validator("contributing_routes")
    @classmethod
    def _known_routes(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if not v:
            raise ValueError("contributing_routes must name at least one route")
        unknown = set(v) - _ROUTES
        if unknown:
            raise ValueError(f"unknown routes {sorted(unknown)}; expected a subset of "
                             f"{sorted(_ROUTES)}")
        return v


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
    aggregate: AggregateSettings = Field(default_factory=AggregateSettings)
    paths: PathSettings = Field(default_factory=PathSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Highest first: explicit kwargs, environment, .env, then the YAML
        (local.yaml deep-merged over default.yaml; a missing file is
        skipped). The YAML must be a source ranked here, not constructor
        kwargs: kwargs outrank the environment, which made every key
        default.yaml sets impossible to override with ISC_*."""
        yaml_settings = YamlConfigSettingsSource(
            settings_cls,
            yaml_file=[CONFIG_DIR / "default.yaml", CONFIG_DIR / "local.yaml"],
            deep_merge=True,
        )
        return init_settings, env_settings, dotenv_settings, file_secret_settings, yaml_settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def load_prompt(relpath: str) -> str:
    """Prompts live on disk and are versioned in the filename, never inlined.

    relpath example: 'extract/purchase_order.v1.md'
    """
    path = get_settings().paths.prompts / relpath
    if not path.exists():
        raise ConfigError(f"prompt not found: {path}")
    return path.read_text()
