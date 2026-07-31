"""Tests for application configuration and logging setup."""

from io import StringIO
from pathlib import Path

import pytest

from resume_copilot.config import Settings, get_settings
from resume_copilot.logger import logger


def test_settings_default_values() -> None:
    """Default settings should be populated with sensible defaults."""
    settings = Settings()

    assert settings.model_name == "gpt-4o-mini"
    assert settings.openai_base_url == "https://api.openai.com/v1"
    assert settings.log_level == "INFO"
    assert settings.llm_temperature == pytest.approx(0.3)
    assert settings.llm_max_retries == 3
    assert settings.llm_timeout == 60


def test_settings_path_resolution() -> None:
    """Relative paths should be resolved against the project root."""
    settings = Settings()

    assert settings.data_dir.is_absolute()
    assert settings.outputs_dir.is_absolute()
    assert settings.data_dir.name == "data"
    assert settings.outputs_dir.name == "outputs"


def test_settings_log_level_validation() -> None:
    """Invalid log levels should raise a validation error."""
    with pytest.raises(ValueError):
        Settings(log_level="VERBOSE")


def test_settings_ensure_directories(tmp_path: Path) -> None:
    """ensure_directories should create data and outputs folders."""
    settings = Settings(data_dir=tmp_path / "data", outputs_dir=tmp_path / "outputs")

    settings.ensure_directories()

    assert settings.data_dir.exists()
    assert settings.outputs_dir.exists()


def test_settings_custom_values() -> None:
    """Settings should accept explicit overrides."""
    settings = Settings(
        openai_api_key="sk-test-key",
        model_name="gpt-4o",
        log_level="DEBUG",
        llm_temperature=0.7,
        llm_max_retries=5,
    )

    assert settings.openai_api_key == "sk-test-key"
    assert settings.model_name == "gpt-4o"
    assert settings.log_level == "DEBUG"
    assert settings.llm_temperature == pytest.approx(0.7)
    assert settings.llm_max_retries == 5


def test_get_settings_returns_singleton() -> None:
    """get_settings should return a cached singleton instance."""
    first = get_settings()
    second = get_settings()

    assert first is second


def test_logger_can_log_capture() -> None:
    """The configured loguru logger should be able to emit messages."""
    stream = StringIO()
    handler_id = logger.add(stream, level="INFO", format="{message}")

    try:
        logger.info("test_config_logger_message")
        captured = stream.getvalue()
        assert "test_config_logger_message" in captured
    finally:
        logger.remove(handler_id)


def test_logger_importable() -> None:
    """The global logger should be importable and usable."""
    assert logger is not None
    # loguru logger exposes bind / catch / etc.
    assert hasattr(logger, "info")
    assert hasattr(logger, "error")
