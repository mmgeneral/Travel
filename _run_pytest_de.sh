#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
pytest test_decision_engine.py -v 2>&1 | tee pytest_decision_engine.log
echo "EXIT:${PIPESTATUS[0]}" >> pytest_decision_engine.log
