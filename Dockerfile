FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 DATA_DIR=/app/data

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY synergy ./synergy
COPY lolpredictordjango ./lolpredictordjango

EXPOSE 8000
CMD ["gunicorn", "--chdir", "lolpredictordjango", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120", "lolpredictordjango.wsgi:application"]
