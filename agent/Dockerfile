FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

COPY pyproject.toml README.md ./
COPY . .

RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir --no-build-isolation -e . google-ai-generativelanguage && \
    chmod +x /app/entrypoint.sh

EXPOSE 8080

CMD ["/app/entrypoint.sh"]
