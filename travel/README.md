# Deprecated Mirror Directory

This `travel/` directory is a legacy mirror and is **not** the active code path.

## Active Source of Truth

Use the project root files as the primary implementation:

- `agent.py`
- `api.py`
- `saga.py`
- `shop_planning.py`
- `decision_engine.py`

## Runtime Entry Point

Use root entrypoint:

- `uvicorn api:app`

Do **not** use:

- `uvicorn travel.api:app`

## Why This Exists

This folder is kept temporarily for migration safety and historical reference.
It should not receive new feature changes.

