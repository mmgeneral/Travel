"""Compatibility shim — use ``orchestrator.AgentOrchestrator`` in new code.

``api.py`` imports from ``orchestrator`` directly; this re-export avoids breaking
older imports/tests that still reference ``agent_orchestrator``.
"""

from orchestrator import AgentOrchestrator

__all__ = ["AgentOrchestrator"]
