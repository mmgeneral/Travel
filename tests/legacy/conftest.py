# Legacy distributed-transaction tests — Saga, idempotency, resilience.
# These are kept as engineering reference for compensation patterns.
# They are intentionally excluded from the default pytest run (tests/ is not
# on the standard discovery path when running `pytest` from the project root).
#
# To run them explicitly:
#   pytest tests/legacy/ -v
import pytest

collect_ignore_glob = []  # noqa: F841 — nothing to ignore within this package
