FROM python:3.12-slim

WORKDIR /app

# Copy uv binary directly from Astral's official image (faster & cleaner than pip install uv)
COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /uvx /bin/

COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

COPY app/ app/

# Create non-root user and give ownership of /app (including .venv)
RUN useradd --create-home appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Use Python stdlib urllib instead of curl (curl is not installed in python:3.12-slim)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

# Run uvicorn with module:app syntax (not app.main.py)
CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]