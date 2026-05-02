# Root conftest ¡X controls which paths pytest collects from by default.
# Legacy distributed-tx tests: run explicitly with  pytest tests/legacy/ -v
collect_ignore_glob = [
    ".venv/*",
    "_legacy_*/*",
    "tools/transcripts/*",
    "tests/legacy/*",
    # Originals kept in-place for git history; canonical copies in tests/legacy/
    "test_saga.py",
    "test_idempotency.py",
    "test_saga_compensation.py",
    "test_resilience.py",
]
