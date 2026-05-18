# ── Dockerfile ────────────────────────────────────────────────────────────────
# Builds a production container for the Churn Prediction API.
#
# Build:    docker build -t churn-api .
# Run:      docker run -p 8000:8000 churn-api
# Test:     curl http://localhost:8000/health
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Don't write .pyc files; flush logs immediately
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/      ./src/
COPY models/   ./models/
COPY data/     ./data/

# Expose API port
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
