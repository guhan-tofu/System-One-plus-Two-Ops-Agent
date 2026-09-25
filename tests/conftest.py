from __future__ import annotations

from collections.abc import Iterator

import pytest

from sentinel.config import Settings, get_settings


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-live", action="store_true", default=False, help="run tests that hit real APIs"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-live"):
        return
    skip_live = pytest.mark.skip(reason="live test: pass --run-live to enable")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep tests independent of a developer's real `.env` and environment."""
    for var in (
        "AI_GATEWAY_API_KEY",
        "AI_GATEWAY_BASE_URL",
        "JEV_GATEWAY_MODEL",
        "JEV_API_KEY",
        "JEV_BASE_URL",
        "JEV_MODEL",
        "JEV_PROVIDER",
        "OPENAI_API_KEY",
        "OPENAI_MODEL_FAST",
        "OPENAI_MODEL_STRONG",
        "JEV_FALLBACK_TIER",
        "MODEL_TIERS_PATH",
        "JEV_ON_FAILURE",
        "THRESHOLDS_PATH",
        "TOOLS_POLICY_PATH",
        "DATABASE_URL",
        "SENTINEL_API_TOKEN",
        "LOG_LEVEL",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
