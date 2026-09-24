# Sentinel — conventions for Claude Code

- Read PLAN.md before starting. Work one phase at a time; stop and summarize at
  the end of each phase with how acceptance criteria were met.
- Python 3.12, uv. Commands: `uv run pytest`, `uv run ruff check .`,
  `uv run mypy src`, `uv run sentinel ...`.
- Secrets: only via config.Settings from .env. Never print, log, hardcode, or
  write keys into any file, test, or commit. If you see a key in a file, stop and
  tell me.
- Jev never executes anything; code executes. Deterministic policy beats model output.
- One judgment per Jev question. Keep state minimal and redacted.
- All external calls go through jev/provider.py or llm/openai_client.py.
- Every decision is written to the audit log with the returned model ID.
- New behavior needs tests; mock HTTP with respx. Live tests use @pytest.mark.live.
- Don't add dependencies without saying why.
