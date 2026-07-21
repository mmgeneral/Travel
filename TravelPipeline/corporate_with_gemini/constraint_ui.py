from __future__ import annotations

import json
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt


# 英文 → 中文標籤
_LABEL_MAP = {
    "min_rating": "最低評分",
    "max_price_level": "最高價格等級",
    "open_until": "營業至",
    "requires_air_conditioning": "需有空調",
    "requires_parking": "需有停車",
}


def run_constraint_confirmation(parsed: dict) -> dict:
    """
    Interactive terminal interface for confirming structured constraints of each
    pending_search.  Prompts the user to keep or discard each constraint field.
    """
    console = Console()
    pending_searches = parsed.get("pending_searches", [])
    if not isinstance(pending_searches, list):
        return parsed

    for ps in pending_searches:
        if not isinstance(ps, dict):
            continue
        sc = ps.get("structured_constraints", {})
        if not isinstance(sc, dict) or not sc:
            # 空的結構化約束：跳過確認，直接設定空的 confirmed_constraints
            ps["confirmed_constraints"] = {}
            continue

        # 將 sc 的 key 變成有序列表，以便顯示
        keys = list(sc.keys())
        # 狀態: True = 保留, False = 已移除
        status = {k: True for k in keys}

        while True:
            console.clear()
            console.print(
                f"[bold cyan]Slot ID:[/] {ps.get('slot_id', '(unknown)')}",
                justify="left",
            )
            must_have = ps.get("must_have", [])
            if must_have:
                console.print(
                    f"[yellow]must_have:[/] {' / '.join(must_have)}", justify="left"
                )

            # 建立表格
            table = Table(title="結構化約束確認")
            table.add_column("編號", style="bold cyan", no_wrap=True)
            table.add_column("約束名稱", style="white")
            table.add_column("系統判斷值", style="yellow")
            table.add_column("狀態", style="green")

            for idx, key in enumerate(keys):
                label = _LABEL_MAP.get(key, key)
                value = sc[key]
                if isinstance(value, bool):
                    display_val = "是" if value else "否"
                else:
                    display_val = str(value)
                status_str = "保留" if status[key] else "已移除"
                table.add_row(str(idx), label, display_val, status_str)

            console.print(table)

            user_input = Prompt.ask(
                "請輸入編號切換保留/移除（留空或輸入 done 完成）",
                default="done",
            )
            if user_input.strip() == "" or user_input.strip().lower() == "done":
                break

            try:
                idx = int(user_input.strip())
                if 0 <= idx < len(keys):
                    key = keys[idx]
                    status[key] = not status[key]
                else:
                    console.print(f"[red]編號 {idx} 超出範圍[/red]")
            except ValueError:
                console.print(f"[red]無效輸入：{user_input}[/red]")
                continue

        # 使用者完成後，將保留的約束存入 confirmed_constraints
        confirmed = {}
        for key in keys:
            if status.get(key, False):
                confirmed[key] = sc[key]
        ps["confirmed_constraints"] = confirmed

    return parsed
