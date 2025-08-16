FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential swig ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app_rest.py ./app_rest.py
COPY templates ./templates

# carpetas de modelos y videos
RUN mkdir -p /app/models /app/videos

ENV HOST=0.0.0.0 \
    PORT=8000 \
    MODELS_DIR=/app/models \
    AUTO_PULL_S3=1

EXPOSE 8000
CMD ["python", "app_rest.py"]
