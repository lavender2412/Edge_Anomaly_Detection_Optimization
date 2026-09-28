# Lightweight inference image: LiteRT (TFLite runtime) + FastAPI, no TensorFlow.
# Build from the repository root so the model and data_utils.py are in context:
#   docker build -f api/Dockerfile -t edge-anomaly .
#   docker run --rm -p 8000:8000 --cpus=1 --memory=512m edge-anomaly
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NUM_THREADS=1

WORKDIR /app

COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

COPY data_utils.py .
COPY api/main.py api/main.py
COPY models/edge_int8.tflite models/thresholds.json models/

RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
