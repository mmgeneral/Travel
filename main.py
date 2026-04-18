import os
import time
import json
import pynvml
import requests
from openai import OpenAI
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

# --- INFRASTRUCTURE: GPU WATCHDOG ---
class GPUWatchdog:
    def __init__(self, threshold=85):
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.threshold = threshold
        except Exception as e:
            print(f"Watchdog Init Failed: {e}")
            self.handle = None

    def wait_if_busy(self):
        if not self.handle: return
        while True:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                if util.gpu < self.threshold:
                    break
                print(f"⚠️ GPU busy ({util.gpu}%). Polling every 15s...")
            except: pass
            time.sleep(15)

# --- APPLICATION: MULTIMODAL AGENT ---
class AgentState(TypedDict):
    user_message: dict
    preferences: dict
    logs: List[str]

client = OpenAI(base_url=os.getenv("VLLM_URL"), api_key="not-needed")
watchdog = GPUWatchdog(threshold=int(os.getenv("GPU_COMPUTE_THRESHOLD", 85)))

def sync_preference_node(state: AgentState):
    watchdog.wait_if_busy()
    
    # Gemma 4 Thinking Mode for high-quality logic
    system_instruction = (
        "<|think|> You are a travel preference engine. Analyze text or images "
        "and update the preference JSON. Output ONLY valid JSON."
    )
    
    try:
        response = client.chat.completions.create(
            model="google/gemma-4-31B-it",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": state['user_message']}
            ],
            temperature=0.1
        )
        
        content = response.choices[0].message.content
        json_str = content[content.find("{"):content.rfind("}")+1]
        state['preferences'] = json.loads(json_str)
        state['logs'].append("Sync complete (Gemma-4 Vision)")
    except Exception as e:
        state['logs'].append(f"Sync error: {e}")
    return state

# --- ORCHESTRATION ---
workflow = StateGraph(AgentState)
workflow.add_node("sync", sync_preference_node)
workflow.set_entry_point("sync")
workflow.add_edge("sync", END)
app = workflow.compile()

def wait_for_vllm():
    url = os.getenv("VLLM_URL").replace("/v1", "/health")
    print(f"🔍 Pinging vLLM health check at {url}...")
    while True:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print("✅ vLLM Brain is healthy and model is loaded!")
                return True
        except:
            pass
        print("⏳ vLLM is still booting up (Downloading shards)... waiting 20s")
        time.sleep(20)

if __name__ == "__main__":
    print("🚀 Gemma-4 Multimodal Agent initialization...")
    
    # Block until the 4090 has finished loading the 31B model
    wait_for_vllm()
    
    test_input = [{"type": "text", "text": "I'm traveling with my toddler. Need parks nearby."}]
    initial_state = {"user_message": test_input, "preferences": {}, "logs": []}
    
    result = app.invoke(initial_state)
    print(f"🎉 Updated Preferences: {result['preferences']}")
