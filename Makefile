.PHONY: install test ollama-up ollama-pull vllm-up vllm-down code-review langfuse clean

install:
	pip install -r requirements.txt

test:
	.venv/bin/python3 -m pytest -v

# ¢w¢w Ollama (always-on, 1080 Ti) ¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w

ollama-up:
	@curl -sf http://localhost:11434/api/tags > /dev/null \
		&& echo "? Ollama responding at http://localhost:11434" \
		|| echo "? Ollama not responding ¡X start with: ollama serve"

ollama-pull:
	ollama pull qwen2.5:7b
	ollama pull qwen2.5:1.5b

# ¢w¢w vLLM (on-demand, lab 4090) ¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w

vllm-up:
	@echo "??  Using shared 4090. Check GPU memory first:"
	@nvidia-smi --query-gpu=memory.used,memory.free --format=csv -i 0 || true
	docker compose up -d vllm
	@echo "Waiting for vLLM to be ready¡K"
	@until curl -sf $${VLLM_URL:-http://localhost:8000}/v1/models > /dev/null; do sleep 2; done
	@echo "? vLLM ready"

vllm-down:
	docker compose down vllm

# ¢w¢w Code Review CLI ¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w

code-review:
	.venv/bin/python3 -m tools.code_review_chat .

code-review-timed:
	.venv/bin/python3 -m tools.code_review_chat . --max-turns 4

# ¢w¢w Langfuse (local tracing UI) ¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w

langfuse:
	docker run --rm -p 3000:3000 langfuse/langfuse:latest

# ¢w¢w Cleanup ¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w¢w

clean:
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -not -path "./.venv/*" -delete 2>/dev/null || true
