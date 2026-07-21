"""In-memory transcript with optional JSONL persistence.

Design constraints
------------------
* Append-only: once a Turn is added it cannot be mutated.
* turns is exposed as a tuple to prevent external mutation.
* JSONL format: one JSON object per line, UTF-8, no BOM.
* Real-time incremental writes: each append() call flushes one line to disk.
* save_jsonl() / load_jsonl() allow full serialise / deserialise cycles.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

Speaker = Literal["professor", "student", "synthesizer", "user"]


@dataclass
class Turn:
    speaker: Speaker
    content: str
    timestamp: str   # ISO-8601 UTC
    turn_index: int  # 0-based, set by Transcript on append


class Transcript:
    """Ordered, append-only sequence of conversation turns."""

    def __init__(self, log_path: Path | None = None) -> None:
        self._turns: list[Turn] = []
        self.log_path = log_path

    @property
    def turns(self) -> tuple[Turn, ...]:
        """Immutable view -- prevents external code from mutating past turns."""
        return tuple(self._turns)

    def __len__(self) -> int:
        return len(self._turns)

    def append(self, speaker: Speaker, content: str) -> Turn:
        """Add a new turn and optionally persist it immediately."""
        turn = Turn(
            speaker=speaker,
            content=content,
            timestamp=datetime.now(timezone.utc).isoformat(),
            turn_index=len(self._turns),
        )
        self._turns.append(turn)
        if self.log_path is not None:
            self._append_jsonl(turn)
        return turn

    def render_for(self, audience: str) -> list[dict]:
        """Format transcript as a messages list for an LLM context window.

        Audience rules:
        - synthesizer sees every turn (including other synth turns)
        - professor / student see only public turns (synth meta-commentary hidden)
          own turns are role=assistant, others are role=user
        """
        messages: list[dict] = []
        for turn in self._turns:
            if audience == "synthesizer":
                role = "assistant" if turn.speaker in ("professor", "student", "synthesizer") else "user"
                messages.append({
                    "role": role,
                    "content": f"[{turn.speaker.upper()}]: {turn.content}",
                })
            else:
                if turn.speaker == "synthesizer":
                    continue
                role = "assistant" if turn.speaker == audience else "user"
                messages.append({
                    "role": role,
                    "content": f"[{turn.speaker.upper()}]: {turn.content}",
                })
        return messages

    def _append_jsonl(self, turn: Turn) -> None:
        """Append a single turn to the JSONL log (real-time, incremental)."""
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(turn), ensure_ascii=False) + "\n")

    def save_jsonl(self) -> None:
        """Rewrite the JSONL log with all current turns (full snapshot)."""
        if self.log_path is None:
            return
        self.log_path.write_text(
            "\n".join(json.dumps(asdict(t), ensure_ascii=False) for t in self._turns) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load_jsonl(cls, path: Path, *, log_path: Path | None = None) -> "Transcript":
        """Deserialise a JSONL file into a Transcript."""
        t = cls(log_path=log_path)
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                t._turns.append(Turn(**data))
        return t