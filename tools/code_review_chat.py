"""Code Review Chat — two-agent CLI discussion tool.

Two LLM agents (Professor + Student) read a codebase directory and discuss its
architecture in an infinite turn loop.  Press Enter at any time to pause and get
a structured three-section summary from a third Synthesizer agent.  Press Enter
again to resume.

Usage::

    python -m tools.code_review_chat /path/to/project
    python -m tools.code_review_chat /path/to/project --max-turns 4
    python -m tools.code_review_chat /path/to/project --resume tools/transcripts/2026_05_01.jsonl
    python -m tools.code_review_chat /path/to/project --exclude "test_*.py"
"""
from __future__ import annotations

import argparse
import fnmatch
import select
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from llm_router import LLMRouter, TaskType  # noqa: E402
from tools.personas import PROFESSOR_SYSTEM, STUDENT_SYSTEM, SYNTHESIZER_SYSTEM  # noqa: E402
from tools.transcript import Transcript  # noqa: E402

_TRANSCRIPTS_DIR = Path(__file__).parent / "transcripts"
_MAX_LINES_PER_FILE = 300
_ALWAYS_EXCLUDE_FRAGMENTS = {
    ".venv", "__pycache__", "node_modules", ".git", "tools/",
    ".mypy_cache", ".pytest_cache",
}


def _read_codebase(target_dir: Path, exclude_patterns: list[str]) -> str:
    parts: list[str] = []
    for py_file in sorted(target_dir.rglob("*.py")):
        rel = py_file.relative_to(target_dir)
        rel_str = str(rel).replace("\\", "/")
        if any(frag in rel_str for frag in _ALWAYS_EXCLUDE_FRAGMENTS):
            continue
        if any(fnmatch.fnmatch(rel_str, pat) for pat in exclude_patterns):
            continue
        try:
            raw = py_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = raw.splitlines()
        if len(lines) > _MAX_LINES_PER_FILE:
            shown = lines[:_MAX_LINES_PER_FILE]
            shown.append(f"# ... [{len(lines) - _MAX_LINES_PER_FILE} more lines truncated]")
            content = "\n".join(shown)
        else:
            content = "\n".join(lines)
        parts.append(f"=== {rel_str} ===\n{content}")
    return "\n\n".join(parts)


def _check_pause() -> bool:
    try:
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if ready:
            sys.stdin.readline()
            return True
    except (AttributeError, ValueError, OSError):
        pass
    return False


def _build_student_messages(transcript: Transcript, codebase_context: str | None) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": STUDENT_SYSTEM}]
    history = transcript.render_for("student")
    if not history and codebase_context:
        messages.append({
            "role": "user",
            "content": (
                "以下是你要分析的 codebase：\n\n"
                f"{codebase_context}\n\n"
                "請從你的視角報告你觀察到的架構事實。"
                "列點，引用具體的檔案名稱、行號、函式名稱。"
                "不要下「好/壞」的評價。"
            ),
        })
    else:
        messages.extend(history)
        messages.append({
            "role": "user",
            "content": "繼續你的觀察，或具體回應教授的最新意見。先回應教授的問題，再補充新的觀察。保持客觀，列點。",
        })
    return messages


def _build_professor_messages(transcript: Transcript) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": PROFESSOR_SYSTEM}]
    messages.extend(transcript.render_for("professor"))
    messages.append({
        "role": "user",
        "content": "根據學生的最新報告，給出你有立場的架構意見。明確同意或反對，引用工程概念，並附上具體改法方向。",
    })
    return messages


def _build_synthesizer_messages(transcript: Transcript) -> list[dict]:
    lines = []
    for t in transcript.turns:
        if t.speaker == "synthesizer":
            continue
        lines.append(f"[{t.speaker.upper()} | Turn {t.turn_index}]\n{t.content}")
    full_transcript = "\n\n".join(lines)
    return [
        {"role": "system", "content": SYNTHESIZER_SYSTEM},
        {"role": "user", "content": f"以下是完整的討論記錄：\n\n{full_transcript}\n\n請輸出三段摘要。"},
    ]


_SPEAKER_LABEL = {
    "student": "Student    \U0001f393",
    "professor": "Professor  \U0001f4d0",
    "synthesizer": "Synthesizer \U0001f4cb",
}


def _print_turn(turn_index: int, speaker: str, content: str) -> None:
    label = _SPEAKER_LABEL.get(speaker, speaker.capitalize())
    print(f"\n{'─' * 64}")
    print(f"[Turn {turn_index}]  {label}")
    print("─" * 64)
    print(content)


def _do_synthesis(transcript: Transcript, router: LLMRouter) -> None:
    print("\n\n[USER PRESSED ENTER → Calling Synthesizer…]")
    msgs = _build_synthesizer_messages(transcript)
    try:
        resp = router.complete(TaskType.SYNTHESIS, msgs)
        content = resp.content
    except Exception as exc:
        content = f"[Synthesizer failed: {exc}]"
    transcript.append("synthesizer", content)
    print("\n" + "═" * 64)
    print(content)
    print("═" * 64)
    print("\n\033[2m>>> [Press Enter to continue | Ctrl+C to quit]\033[0m")
    try:
        sys.stdin.readline()
    except (EOFError, KeyboardInterrupt):
        raise KeyboardInterrupt


def run_chat(
    target_dir: Path,
    max_turns: int | None = None,
    resume_path: Path | None = None,
    exclude_patterns: list[str] | None = None,
    *,
    router: LLMRouter | None = None,
    _transcript: Transcript | None = None,
) -> Transcript:
    """Run the professor/student discussion loop. Returns the final Transcript."""
    exclude_patterns = exclude_patterns or []
    _TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = _TRANSCRIPTS_DIR / f"transcript_{ts}.jsonl"

    if _transcript is not None:
        transcript = _transcript
    elif resume_path is not None:
        transcript = Transcript.load_jsonl(resume_path, log_path=log_path)
        print(f"[Resuming from {resume_path}  ({len(transcript)} turns loaded)]")
    else:
        transcript = Transcript(log_path=log_path)

    _router = router or LLMRouter()

    print(f"\n{'═' * 64}")
    print("  Code Review Chat")
    print(f"  Target : {target_dir}")
    print(f"  Log    : {log_path}")
    print(f"{'═' * 64}")
    print("[Press Enter at any time to pause. Ctrl+C to quit.]\n")

    codebase_context: str | None = None
    if _transcript is None and resume_path is None:
        print("[Reading codebase…] ", end="", flush=True)
        codebase_context = _read_codebase(target_dir, exclude_patterns)
        n_files = codebase_context.count("=== ")
        print(f"{n_files} .py files loaded\n")

    turn_count: int = len(transcript)
    is_first_student_turn: bool = (len(transcript) == 0)

    try:
        while True:
            if _check_pause():
                _do_synthesis(transcript, _router)

            turn_count += 1
            print(f"\n\033[2m>>> [Press Enter to pause | Ctrl+C to quit]  (Turn {turn_count} — Student…)\033[0m", end="\r", flush=True)

            student_msgs = _build_student_messages(
                transcript,
                codebase_context if is_first_student_turn else None,
            )
            is_first_student_turn = False

            try:
                s_resp = _router.complete(TaskType.RETRIEVAL_REASONING, student_msgs)
                s_content = s_resp.content
            except Exception as exc:
                s_content = f"[Student LLM error: {exc}]"

            transcript.append("student", s_content)
            _print_turn(turn_count, "student", s_content)

            if max_turns is not None and turn_count >= max_turns:
                break

            if _check_pause():
                _do_synthesis(transcript, _router)

            turn_count += 1
            print(f"\n\033[2m>>> [Press Enter to pause | Ctrl+C to quit]  (Turn {turn_count} — Professor…)\033[0m", end="\r", flush=True)

            prof_msgs = _build_professor_messages(transcript)
            try:
                p_resp = _router.complete(TaskType.CRITIQUE, prof_msgs)
                p_content = p_resp.content
            except Exception as exc:
                p_content = f"[Professor LLM error: {exc}]"

            transcript.append("professor", p_content)
            _print_turn(turn_count, "professor", p_content)

            if max_turns is not None and turn_count >= max_turns:
                break

    except KeyboardInterrupt:
        print("\n\n[Interrupted — saving transcript…]")

    transcript.save_jsonl()
    print(f"\n[Done. Transcript saved → {log_path}  ({len(transcript)} turns)]")
    return transcript


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m tools.code_review_chat",
        description="Two-agent (Professor / Student) code review CLI.",
    )
    parser.add_argument("target_dir", type=Path)
    parser.add_argument("--max-turns", type=int, default=None, metavar="N")
    parser.add_argument("--resume", type=Path, default=None, metavar="JSONL")
    parser.add_argument("--exclude", action="append", default=[], metavar="PATTERN")
    args = parser.parse_args(argv)
    if not args.target_dir.is_dir():
        parser.error(f"Not a directory: {args.target_dir}")
    run_chat(
        target_dir=args.target_dir.resolve(),
        max_turns=args.max_turns,
        resume_path=args.resume,
        exclude_patterns=args.exclude,
    )


if __name__ == "__main__":
    main()
