"""
Unsat Plan Repair Demo - 失敗案例（套用還原作者邏輯的新解析器）
prompt 只給抽象約束名稱，不給具體數值，看 LLM 給出的建議能不能被
正確分類、正確抓出數字——預期：抓不到數字時應明確報失敗，不再
fallback 到任何猜測值。
"""
import os
import re
import requests
from z3 import Optimize, Int, sat, unsat

DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
DEEPSEEK_URL = 'https://api.deepseek.com/v1/chat/completions'

BUDGET = 4000
MINCOST_CITY1 = 2000
MINCOST_CITY2 = 3000

def call_llm(prompt: str) -> str:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError('DEEPSEEK_API_KEY 未設定')
    headers = {
        'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
        'Content-Type': 'application/json',
    }
    payload = {
        'model': 'deepseek-chat',
        'messages': [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt},
        ],
        'temperature': 0.0,
    }
    resp = requests.post(DEEPSEEK_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()['choices'][0]['message']['content'].strip()


def classify_suggestion(suggestion_text: str):
    text_lower = suggestion_text.lower()
    if 'budget' in text_lower:
        return 'budget'
    elif 'destination' in text_lower or 'city' in text_lower:
        return 'destination'
    elif 'transportation' in text_lower or 'flight' in text_lower:
        return 'transportation'
    elif 'house type' in text_lower or 'accommodation' in text_lower:
        return 'house_type'
    else:
        return None


def extract_budget_value(suggestion_text: str):
    match = re.search(r'\d+', suggestion_text)
    return int(match.group()) if match else None


def solve(budget_limit: int):
    s = Optimize()
    C1 = Int('C1')
    C2 = Int('C2')
    s.assert_and_track(C1 >= MINCOST_CITY1, 'mincost_city1')
    s.assert_and_track(C2 >= MINCOST_CITY2, 'mincost_city2')
    s.assert_and_track(C1 + C2 <= budget_limit, 'budget_upper')
    s.maximize(C1 + C2)
    result = s.check()
    core = [str(c) for c in s.unsat_core()] if result == unsat else []
    model_vals = {}
    if result == sat:
        m = s.model()
        model_vals = {'C1': m[C1].as_long(), 'C2': m[C2].as_long()}
    return result, core, model_vals


def run_demo():
    print("=" * 60)
    print("UNSAT PLAN REPAIR DEMO - 失敗案例（新解析器）")
    print("=" * 60)

    print(f"\n--- 約束設定 ---")
    print(f"  C1 >= {MINCOST_CITY1}   (城市1最低花費)")
    print(f"  C2 >= {MINCOST_CITY2}   (城市2最低花費)")
    print(f"  C1 + C2 <= {BUDGET}   (預算上限)")

    result, core, _ = solve(BUDGET)
    print(f"\n--- 第一次求解 ---")
    print(f"Z3 status: {result}")
    print(f"衝突約束: {core}")

    # 這次 prompt 只給抽象約束名稱，不給數值（對應失敗案例的原始設計）
    prompt = """The following constraints are in conflict:
  'mincost_city1'
  'mincost_city2'
  'budget_upper'
Please propose a repair using this format:
  Suggest[<your suggestion>]
Only output the action, nothing else."""

    print(f"\n--- LLM Prompt ---")
    print(prompt)

    print(f"\n--- LLM 回應 ---")
    reply = call_llm(prompt)
    print(f"[LLM raw reply] {reply}")

    suggestion_text = reply.split('[', 1)[1].rstrip(']') if '[' in reply else reply

    dimension = classify_suggestion(suggestion_text)
    print(f"    Classified dimension: {dimension or 'invalid (cannot classify)'}")

    if dimension is None:
        print("    [!] LLM suggestion could not be classified. Repair fails here.")
        print("\n" + "=" * 60)
        print("END OF DEMO")
        print("=" * 60)
        return

    if dimension != 'budget':
        print(f"    [!] Dimension '{dimension}' repair not implemented in this demo.")
        print("\n" + "=" * 60)
        print("END OF DEMO")
        print("=" * 60)
        return

    new_budget = extract_budget_value(suggestion_text)
    if new_budget is None:
        print("    [!] Suggestion was classified as 'budget' but no concrete "
              "number could be extracted from the text. This is the real "
              "failure point — explicit inability to proceed, not a fallback "
              "to a wrong number.")
        print("\n" + "=" * 60)
        print("END OF DEMO")
        print("=" * 60)
        return

    print(f"    Parsed budget from suggestion: {new_budget}")
    if new_budget > BUDGET:
        result2, _, vals = solve(new_budget)
        print(f"\n--- 修復後求解 (budget={new_budget}) ---")
        print(f"Z3 status: {result2}")
        if result2 == sat:
            print(f"C1 = {vals['C1']}, C2 = {vals['C2']}, 總花費 = {vals['C1']+vals['C2']}")
            print("✓ 修復成功")
    else:
        print("    [!] Extracted budget not greater than original; repair not applied.")

    print("\n" + "=" * 60)
    print("END OF DEMO")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
