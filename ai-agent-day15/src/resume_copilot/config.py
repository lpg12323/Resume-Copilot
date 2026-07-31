"""Centralized application configuration for Resume Copilot.

This module uses pydantic-settings to load environment variables and
``.env`` files in a type-safe manner. All configurable values used
throughout the project should be accessed via ``settings``.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and ``.env`` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # LLM provider configuration
    # ------------------------------------------------------------------
    openai_api_key: str = Field(
        default="",
        description="API key for OpenAI-compatible LLM services.",
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for OpenAI-compatible API endpoints.",
    )
    model_name: str = Field(
        default="gpt-4o-mini",
        description="Model name used for analysis and rewriting.",
    )

    # ------------------------------------------------------------------
    # Application configuration
    # ------------------------------------------------------------------
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL.",
    )
    data_dir: Path = Field(
        default=Path("data"),
        description="Directory for input PDF resumes and JD examples.",
    )
    outputs_dir: Path = Field(
        default=Path("outputs"),
        description="Directory for generated Markdown resumes.",
    )

    # ------------------------------------------------------------------
    # LangGraph / LangChain configuration
    # ------------------------------------------------------------------
    llm_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for LLM generation.",
    )
    llm_max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum number of retries for LLM API calls.",
    )
    llm_timeout: int = Field(
        default=60,
        ge=1,
        description="Timeout in seconds for LLM API requests.",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        normalized = str(value).strip().upper()
        if normalized not in allowed:
            raise ValueError(
                f"Invalid LOG_LEVEL '{value}'. Allowed values: {', '.join(sorted(allowed))}"
            )
        return normalized

    @field_validator("data_dir", "outputs_dir", mode="before")
    @classmethod
    def _resolve_path(cls, value: str | Path) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[3] / path
        return path

    def ensure_directories(self) -> None:
        """Create configured data and output directories if they do not exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton instance of ``Settings``.

    The singleton pattern avoids re-reading the ``.env`` file on every
    import, while still allowing test code to instantiate ``Settings``
    directly with explicit values.
    """
    return Settings()


# Global settings shortcut. Prefer injecting ``settings`` in unit tests.
settings = get_settings()
