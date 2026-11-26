# Hosted build: OpenRouter for chat, embeddings and try-on. No local models in the image.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    LLM_PROVIDER=openrouter \
    TRYON_PROVIDER=openrouter \
    TRUST_PROXY=1

WORKDIR /srv
COPY requirements-hosted.txt .
RUN pip install -r requirements-hosted.txt

COPY app/ app/
COPY web/ web/
# Catalog data, precomputed embeddings and the catalog photos (~9 MB).
COPY data/catalog/ data/catalog/

RUN useradd --create-home --uid 10001 app \
    && mkdir -p data/uploads data/tryon \
    && chown -R app:app data
USER app

# OPENROUTER_API_KEY must be passed at run time (never baked into the image).
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/api/status', timeout=4)"
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers"]
