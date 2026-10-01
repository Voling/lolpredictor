FROM public.ecr.aws/docker/library/python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/tmp/work \
    USE_TIMESCALE=0 \
    REDIS_URL=disabled \
    CORS_ORIGINS=

WORKDIR /var/task

COPY requirements-evaluate.txt ./
RUN pip install --no-cache-dir -r requirements-evaluate.txt && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY synergy ./synergy

CMD ["python", "-m", "synergy.worker"]
