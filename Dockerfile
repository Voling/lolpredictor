FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 DATA_DIR=/app/data

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY synergy ./synergy

EXPOSE 8000
CMD ["uvicorn", "synergy.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
