FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    curl \
    procps \
    && pip install --no-cache-dir nvitop \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    openai \
    pynvml \
    langgraph \
    python-dotenv

COPY . .

CMD ["python", "main.py"]
