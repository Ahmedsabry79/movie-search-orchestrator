FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && groupadd --gid 10001 movieapp \
    && useradd --uid 10001 --gid 10001 --create-home movieapp \
    && mkdir -p /app/data/raw /app/data/processed \
    && chown -R movieapp:movieapp /app/data
COPY --chown=movieapp:movieapp preprocessing.py .
COPY --chown=movieapp:movieapp app ./app
USER movieapp
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
