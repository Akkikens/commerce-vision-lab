# Two targets:
#   docker build -t cvlab .                     -> slim image, stub backends (CI/demo)
#   docker build -t cvlab-ml --build-arg EXTRAS=ml,gen .  -> real models (large image)
FROM python:3.11-slim AS base
ARG EXTRAS=""
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN useradd --create-home --uid 10001 app
COPY pyproject.toml README.md ./
COPY src ./src
RUN if [ -n "$EXTRAS" ]; then pip install ".[${EXTRAS}]"; else pip install .; fi
USER app
ENV CVLAB_BACKEND=stub CVLAB_CACHE=/home/app/.cache/cvlab CVLAB_ARTIFACTS=/home/app/artifacts
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "cvlab.api:app", "--host", "0.0.0.0", "--port", "8000"]
