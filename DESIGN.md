# Travel Agent — Design Notes

See chat history for the full feature inventory. Key architectural decisions:

- **Saga-first**: booking side effects AND conversational state both flow through `saga.py`
- **Authority-domain search**: tabelog.com / official .jp sites prioritised over SEO blogs
- **Deterministic time arithmetic**: the LLM never computes times; the `TemporalEngine` does
- **Local-first**: Saga log + KB + preferences persisted to disk; cloud is optional

This file is a placeholder — fill in with project writeup as the MVP stabilises.

