"""Strip legacy hardcoded *_build_shop_catalog* / *_build_shop_catalog_taipei* from agent.py.

Run once during migration cleanup; deletes the OLD block ending just before the
JSON-backed thin-wrapper definitions.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

START_MARKER = (
    '\ndef _build_shop_catalog() -> list[ShopProfile]:\n'
    '    return [\n'
    '        ShopProfile(\n'
    '            name="燃えよ麺助",'
)

END_MARKER = (
    '\ndef _build_shop_catalog() -> list[ShopProfile]:\n'
    '    """JSON-backed Kyoto seed catalog thin wrapper."""\n'
    '    return load_shop_catalog("kyoto")'
)


def main() -> None:
    path = ROOT / "agent.py"
    text = path.read_text(encoding="utf-8")
    start = text.find(START_MARKER)
    end = text.find(END_MARKER)
    if start == -1:
        raise SystemExit("START_MARKER not found — already stripped?")
    if end == -1:
        raise SystemExit("END_MARKER not found — JSON wrapper missing")
    if end <= start:
        raise SystemExit("malformed offsets")
    path.write_text(text[:start] + text[end:], encoding="utf-8")
    print(f"stripped {(end-start)//1024} KiB (~{end-start} bytes)")


if __name__ == "__main__":
    main()
