FROM public.ecr.aws/docker/library/python:3.13-slim

COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 /lambda-adapter /opt/extensions/lambda-adapter

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    AWS_LWA_READINESS_CHECK_PATH=/api/health \
    DATA_DIR=/var/task/data \
    REDIS_URL=disabled \
    DATABASE_URL=postgresql://none@127.0.0.1:1/none \
    CORS_ORIGINS=

WORKDIR /var/task

COPY requirements-serve.txt ./
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY synergy ./synergy
COPY deploy/artifacts/models ./data/models
COPY deploy/artifacts/processed ./data/processed
COPY deploy/artifacts/runs ./data/runs
COPY deploy/artifacts/serve ./data/serve

CMD ["python", "-m", "uvicorn", "synergy.api.server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
