# Root conftest - controls default pytest discovery.
# Legacy distributed-tx tests under tests/legacy/: run explicitly, e.g.
#     pytest tests/legacy/ -v
# (Those modules reference APIs removed from agent.py.)
#
# Async: pytest-asyncio (pytest.ini / pyproject.toml asyncio_mode).

collect_ignore_glob = [
    ".venv/*",
    "_legacy_*/*",
    "tools/transcripts/*",
    "tests/legacy/*",
    # Original saga/idempotency files kept at repo root for git history only.
    "test_saga.py",
    "test_idempotency.py",
    "test_saga_compensation.py",
    "test_resilience.py",
]
