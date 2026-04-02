FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata and requirements
COPY pyproject.toml uv.lock requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy models.py to root (required for imports)
COPY models.py /app/

# Copy server code
COPY *.py /app/server/
COPY scenarios/ /app/server/scenarios/

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# Set PYTHONPATH
ENV PYTHONPATH="/app"

# Expose port
EXPOSE 8000

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--ws-ping-interval", "300", "--ws-ping-timeout", "300"]
