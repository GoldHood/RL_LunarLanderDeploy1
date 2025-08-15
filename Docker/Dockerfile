# Dockerfile (sin extensión)
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Dependencias de sistema (swig para box2d, build tools por si acaso)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential swig && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Dependencias (mejor cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Código
COPY app_rest.py ./app_rest.py
COPY templates ./templates

# 3) Directorio para modelos (si montas un volumen, se usará encima)
RUN mkdir -p /app/models

# Variables por defecto (puedes sobreescribir en compose)
ENV HOST=0.0.0.0 \
    PORT=8000 \
    MODELS_DIR=/app/models \
    AUTO_PULL_S3=1

EXPOSE 8000

# Arranque
CMD ["python", "app_rest.py"]
