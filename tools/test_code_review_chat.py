"""Tests for tools/code_review_chat.py.

Coverage
--------
* Transcript is append-only (Turn objects are immutable after creation).
* Transcript.turns returns a tuple, not a mutable list.
* Transcript renders different views for professor / student / synthesizer.
* JSONL save + load round-trips correctly.
* _build_student_messages passes STUDENT_SYSTEM prompt.
* _build_professor_messages passes PROFESSOR_SYSTEM prompt.
* _build_synthesizer_messages passes SYNTHESIZER_SYSTEM prompt.
* First student turn embeds codebase context; subsequent turns do not.
* Student uses TaskType.RETRIEVAL_REASONING.
* Professor uses TaskType.CRITIQUE.
* Synthesizer uses TaskType.SYNTHESIS.
* Synthesizer is NOT called automatically -- only on user trigger.
* Synthesizer IS called when _check_pause returns True.
* Synthesizer turn is written to transcript (not only printed).
* After pause -> synth -> resume, student/professor turns continue correctly.
* Synthesizer receives the full prior conversation (no synth turns in history).
* _read_codebase excludes tools/, __pycache__, .venv by default.
* _read_codebase respects user-supplied exclude patterns.
* run_chat returns the completed Transcript object.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from llm_router import LLMResponse, TaskType

from tools.personas import PROFESSOR_SYSTEM, STUDENT_SYSTEM, SYNTHESIZER_SYSTEM
from tools.transcript import Transcript, Turn
from tools.code_review_chat import (
    _build_professor_messages,
    _build_student_messages,
    _build_synthesizer_messages,
    _read_codebase,
    run_chat,
)


def _resp(content: str, task: TaskType | None = None) -> LLMResponse:
    return LLMResponse(
        content=content,
        model_used=f"mock-{task.value if task else 'x'}",
        tokens_in=10,
        tokens_out=20,
        latency_ms=50,
        cost_usd=0.0,
    )


def _make_router(
    student_replies: list[str] | None = None,
    professor_replies: list[str] | None = None,
    synth_replies: list[str] | None = None,
) -> MagicMock:
    counters: dict[TaskType, int] = {}

    def _complete(task: TaskType, messages, **kwargs) -> LLMResponse:
        idx = counters.get(task, 0)
        counters[task] = idx + 1
        table = {
            TaskType.RETRIEVAL_REASONING: student_replies or [f"student turn {idx}"],
            TaskType.CRITIQUE: professor_replies or [f"professor turn {idx}"],
            TaskType.SYNTHESIS: synth_replies or ["【目前共識】\n- ok\n【未解決爭論】\n- none\n【待開展】\n- tbd"],
        }
        replies = table.get(task, ["default"])
        return _resp(replies[idx % len(replies)], task)

    router = MagicMock()
    router.complete.side_effect = _complete
    return router


class TestTranscriptAppendOnly:
    def test_turns_returns_tuple(self):
        t = Transcript()
        t.append("student", "hello")
        assert isinstance(t.turns, tuple)

    def test_tuple_cannot_be_item_assigned(self):
        t = Transcript()
        t.append("student", "hello")
        with pytest.raises((TypeError, AttributeError)):
            t.turns[0] = Turn("professor", "hacked", "2026-01-01T00:00:00+00:00", 0)

    def test_turn_indices_are_sequential(self):
        t = Transcript()
        for i in range(5):
            turn = t.append("student" if i % 2 == 0 else "professor", f"msg {i}")
            assert turn.turn_index == i

    def test_len_reflects_append_count(self):
        t = Transcript()
        for _ in range(7):
            t.append("student", "x")
        assert len(t) == 7

    def test_existing_turns_unchanged_after_new_append(self):
        t = Transcript()
        t.append("student", "original content")
        t.append("professor", "something else")
        assert t.turns[0].content == "original content"
        assert t.turns[0].turn_index == 0


class TestTranscriptRenderFor:
    def _populated(self) -> Transcript:
        t = Transcript()
        t.append("student", "student observation")
        t.append("professor", "professor critique")
        t.append("synthesizer", "synth summary")
        t.append("student", "student reply")
        return t

    def test_student_view_hides_synthesizer(self):
        t = self._populated()
        msgs = t.render_for("student")
        combined = " ".join(m["content"] for m in msgs)
        assert "synth summary" not in combined

    def test_professor_view_hides_synthesizer(self):
        t = self._populated()
        msgs = t.render_for("professor")
        combined = " ".join(m["content"] for m in msgs)
        assert "synth summary" not in combined

    def test_synthesizer_view_sees_everything(self):
        t = self._populated()
        msgs = t.render_for("synthesizer")
        combined = " ".join(m["content"] for m in msgs)
        assert "student observation" in combined
        assert "professor critique" in combined
        assert "synth summary" in combined

    def test_own_turns_are_assistant_role(self):
        t = Transcript()
        t.append("professor", "my message")
        msgs = t.render_for("professor")
        assert msgs[0]["role"] == "assistant"

    def test_other_turns_are_user_role(self):
        t = Transcript()
        t.append("student", "student message")
        msgs = t.render_for("professor")
        assert msgs[0]["role"] == "user"


class TestTranscriptJSONL:
    def test_save_and_reload_preserves_all_turns(self, tmp_path):
        log = tmp_path / "t.jsonl"
        t = Transcript(log_path=log)
        t.append("student", "hello")
        t.append("professor", "world")
        t.save_jsonl()
        t2 = Transcript.load_jsonl(log)
        assert len(t2) == 2
        assert t2.turns[0].speaker == "student"
        assert t2.turns[0].content == "hello"

    def test_load_preserves_turn_indices(self, tmp_path):
        log = tmp_path / "t.jsonl"
        t = Transcript(log_path=log)
        for i in range(4):
            t.append("student", f"turn {i}")
        t.save_jsonl()
        t2 = Transcript.load_jsonl(log)
        assert [x.turn_index for x in t2.turns] == [0, 1, 2, 3]

    def test_incremental_append_writes_to_disk(self, tmp_path):
        log = tmp_path / "t.jsonl"
        t = Transcript(log_path=log)
        t.append("student", "first line")
        assert log.exists()
        lines = [line for line in log.read_text().splitlines() if line.strip()]
        assert len(lines) == 1

    def test_save_jsonl_produces_clean_file(self, tmp_path):
        log = tmp_path / "t.jsonl"
        t = Transcript(log_path=log)
        t.append("student", "a")
        t.append("professor", "b")
        t.save_jsonl()
        lines = [l for l in log.read_text().splitlines() if l.strip()]
        assert len(lines) == 2


class TestSystemPrompts:
    def test_student_messages_contain_student_system(self):
        t = Transcript()
        msgs = _build_student_messages(t, codebase_context="def foo(): pass")
        systems = [m["content"] for m in msgs if m["role"] == "system"]
        assert any(STUDENT_SYSTEM in s for s in systems)

    def test_professor_messages_contain_professor_system(self):
        t = Transcript()
        t.append("student", "I observed X")
        msgs = _build_professor_messages(t)
        systems = [m["content"] for m in msgs if m["role"] == "system"]
        assert any(PROFESSOR_SYSTEM in s for s in systems)

    def test_synthesizer_messages_contain_synthesizer_system(self):
        t = Transcript()
        t.append("student", "A")
        t.append("professor", "B")
        msgs = _build_synthesizer_messages(t)
        systems = [m["content"] for m in msgs if m["role"] == "system"]
        assert any(SYNTHESIZER_SYSTEM in s for s in systems)

    def test_first_student_turn_embeds_codebase(self):
        t = Transcript()
        codebase = "UNIQUE_CODEBASE_MARKER_XYZ"
        msgs = _build_student_messages(t, codebase_context=codebase)
        all_content = " ".join(m["content"] for m in msgs)
        assert codebase in all_content

    def test_subsequent_student_turn_uses_continuation_prompt(self):
        t = Transcript()
        t.append("student", "first obs")
        t.append("professor", "good point")
        msgs = _build_student_messages(t, codebase_context=None)
        all_content = " ".join(m["content"] for m in msgs)
        assert "繼續" in all_content

    def test_synthesizer_messages_exclude_synth_history(self):
        t = Transcript()
        t.append("student", "obs A")
        t.append("synthesizer", "PREVIOUS_SYNTH_SHOULD_NOT_APPEAR")
        t.append("professor", "critique B")
        msgs = _build_synthesizer_messages(t)
        all_content = " ".join(m["content"] for m in msgs)
        assert "PREVIOUS_SYNTH_SHOULD_NOT_APPEAR" not in all_content


class TestTaskTypeRouting:
    def test_student_uses_retrieval_reasoning(self, tmp_path):
        router = _make_router()
        with patch("tools.code_review_chat._check_pause", return_value=False):
            run_chat(target_dir=tmp_path, max_turns=1, router=router, _transcript=Transcript())
        student_calls = [c for c in router.complete.call_args_list if c.args[0] == TaskType.RETRIEVAL_REASONING]
        assert student_calls

    def test_professor_uses_critique(self, tmp_path):
        router = _make_router()
        with patch("tools.code_review_chat._check_pause", return_value=False):
            run_chat(target_dir=tmp_path, max_turns=2, router=router, _transcript=Transcript())
        prof_calls = [c for c in router.complete.call_args_list if c.args[0] == TaskType.CRITIQUE]
        assert prof_calls

    def test_synthesizer_not_called_without_user_trigger(self, tmp_path):
        router = _make_router()
        with patch("tools.code_review_chat._check_pause", return_value=False):
            run_chat(target_dir=tmp_path, max_turns=4, router=router, _transcript=Transcript())
        synth_calls = [c for c in router.complete.call_args_list if c.args[0] == TaskType.SYNTHESIS]
        assert not synth_calls, "Synthesizer must NOT auto-run"


class TestPauseFlow:
    def _make_pause_iter(self, seq: list[bool]):
        it = iter(seq)
        def _fn():
            return next(it, False)
        return _fn

    def test_synthesizer_called_when_pause_triggered(self, tmp_path):
        router = _make_router()
        t = Transcript()
        with patch("tools.code_review_chat._check_pause", side_effect=self._make_pause_iter([False, True, False, False])), \
             patch("sys.stdin") as mock_stdin:
            mock_stdin.readline.return_value = "\n"
            run_chat(target_dir=tmp_path, max_turns=2, router=router, _transcript=t)
        synth_calls = [c for c in router.complete.call_args_list if c.args[0] == TaskType.SYNTHESIS]
        assert synth_calls

    def test_synthesizer_appended_to_transcript(self, tmp_path):
        router = _make_router(synth_replies=["SYNTH_MARKER_12345"])
        t = Transcript()
        with patch("tools.code_review_chat._check_pause", side_effect=self._make_pause_iter([False, True, False, False])), \
             patch("sys.stdin") as mock_stdin:
            mock_stdin.readline.return_value = "\n"
            run_chat(target_dir=tmp_path, max_turns=2, router=router, _transcript=t)
        synth_turns = [turn for turn in t.turns if turn.speaker == "synthesizer"]
        assert synth_turns
        assert "SYNTH_MARKER_12345" in synth_turns[0].content

    def test_discussion_continues_after_resume(self, tmp_path):
        router = _make_router()
        t = Transcript()
        with patch("tools.code_review_chat._check_pause", side_effect=self._make_pause_iter([False, True, False, False, False])), \
             patch("sys.stdin") as mock_stdin:
            mock_stdin.readline.return_value = "\n"
            run_chat(target_dir=tmp_path, max_turns=4, router=router, _transcript=t)
        student_turns = [t_ for t_ in t.turns if t_.speaker == "student"]
        prof_turns = [t_ for t_ in t.turns if t_.speaker == "professor"]
        synth_turns = [t_ for t_ in t.turns if t_.speaker == "synthesizer"]
        assert len(student_turns) >= 2
        assert len(prof_turns) >= 1
        assert len(synth_turns) == 1

    def test_turn_indices_sequential_after_pause(self, tmp_path):
        router = _make_router()
        t = Transcript()
        with patch("tools.code_review_chat._check_pause", side_effect=self._make_pause_iter([False, True, False, False, False])), \
             patch("sys.stdin") as mock_stdin:
            mock_stdin.readline.return_value = "\n"
            run_chat(target_dir=tmp_path, max_turns=4, router=router, _transcript=t)
        indices = [turn.turn_index for turn in t.turns]
        assert indices == list(range(len(t)))

    def test_synthesizer_receives_full_prior_history(self, tmp_path):
        captured: list[list[dict]] = []

        def _complete(task, messages, **kwargs):
            if task == TaskType.SYNTHESIS:
                captured.append(messages)
            return _resp(f"response {task}", task)

        router = MagicMock()
        router.complete.side_effect = _complete

        t = Transcript()
        t.append("student", "STUDENT_SEED_CONTENT")
        t.append("professor", "PROFESSOR_SEED_CONTENT")

        with patch("tools.code_review_chat._check_pause", side_effect=self._make_pause_iter([True, False, False])), \
             patch("sys.stdin") as mock_stdin:
            mock_stdin.readline.return_value = "\n"
            run_chat(target_dir=tmp_path, max_turns=2, router=router, _transcript=t)

        assert captured
        all_content = " ".join(m["content"] for m in captured[0])
        assert "STUDENT_SEED_CONTENT" in all_content
        assert "PROFESSOR_SEED_CONTENT" in all_content

    def test_multiple_pauses_call_synthesizer_multiple_times(self, tmp_path):
        router = _make_router()
        t = Transcript()
        with patch("tools.code_review_chat._check_pause", side_effect=self._make_pause_iter([False, True, True, False])), \
             patch("sys.stdin") as mock_stdin:
            mock_stdin.readline.return_value = "\n"
            run_chat(target_dir=tmp_path, max_turns=4, router=router, _transcript=t)
        synth_turns = [turn for turn in t.turns if turn.speaker == "synthesizer"]
        assert len(synth_turns) == 2


class TestReadCodebase:
    def _write_py(self, directory: Path, name: str, content: str = "# stub") -> Path:
        p = directory / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return p

    def test_reads_py_files(self, tmp_path):
        self._write_py(tmp_path, "main.py", "def main(): pass")
        result = _read_codebase(tmp_path, [])
        assert "main.py" in result
        assert "def main(): pass" in result

    def test_excludes_pycache(self, tmp_path):
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        self._write_py(cache, "cached.py", "SHOULD_NOT_APPEAR")
        result = _read_codebase(tmp_path, [])
        assert "SHOULD_NOT_APPEAR" not in result

    def test_excludes_venv(self, tmp_path):
        venv = tmp_path / ".venv" / "lib"
        venv.mkdir(parents=True)
        self._write_py(venv, "site.py", "VENV_CONTENT")
        result = _read_codebase(tmp_path, [])
        assert "VENV_CONTENT" not in result

    def test_excludes_tools_dir(self, tmp_path):
        tools = tmp_path / "tools"
        tools.mkdir()
        self._write_py(tools, "code_review_chat.py", "THIS_TOOL_ITSELF")
        result = _read_codebase(tmp_path, [])
        assert "THIS_TOOL_ITSELF" not in result

    def test_respects_user_exclude_pattern(self, tmp_path):
        self._write_py(tmp_path, "test_something.py", "TEST_CONTENT")
        self._write_py(tmp_path, "main.py", "MAIN_CONTENT")
        result = _read_codebase(tmp_path, ["test_*.py"])
        assert "TEST_CONTENT" not in result
        assert "MAIN_CONTENT" in result

    def test_truncates_long_files(self, tmp_path):
        long_content = "\n".join(f"line {i}" for i in range(400))
        self._write_py(tmp_path, "big.py", long_content)
        result = _read_codebase(tmp_path, [])
        assert "truncated" in result.lower()

    def test_empty_dir_returns_empty_string(self, tmp_path):
        result = _read_codebase(tmp_path, [])
        assert result == ""

    def test_returns_path_header_per_file(self, tmp_path):
        self._write_py(tmp_path, "utils.py", "x = 1")
        result = _read_codebase(tmp_path, [])
        assert "=== utils.py ===" in result


class TestRunChatContract:
    def test_returns_transcript_object(self, tmp_path):
        router = _make_router()
        with patch("tools.code_review_chat._check_pause", return_value=False):
            result = run_chat(target_dir=tmp_path, max_turns=2, router=router, _transcript=Transcript())
        assert isinstance(result, Transcript)

    def test_max_turns_1_produces_student_only(self, tmp_path):
        router = _make_router()
        with patch("tools.code_review_chat._check_pause", return_value=False):
            t = run_chat(target_dir=tmp_path, max_turns=1, router=router, _transcript=Transcript())
        assert len(t) == 1
        assert t.turns[0].speaker == "student"

    def test_max_turns_4_produces_alternating_speakers(self, tmp_path):
        router = _make_router()
        with patch("tools.code_review_chat._check_pause", return_value=False):
            t = run_chat(target_dir=tmp_path, max_turns=4, router=router, _transcript=Transcript())
        speakers = [turn.speaker for turn in t.turns]
        assert speakers == ["student", "professor", "student", "professor"]

    def test_injected_transcript_returned_as_same_object(self, tmp_path):
        router = _make_router()
        existing = Transcript()
        existing.append("student", "pre-existing")
        with patch("tools.code_review_chat._check_pause", return_value=False):
            result = run_chat(target_dir=tmp_path, max_turns=2, router=router, _transcript=existing)
        assert result is existing
        assert result.turns[0].content == "pre-existing"
