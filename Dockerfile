FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml uv.lock /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY server /app/server
COPY models.py client.py openenv.yaml README.md /app/

RUN pip install --no-cache-dir -e .

ENV PYTHONPATH="/app"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
