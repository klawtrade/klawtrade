FROM python:3.13-slim

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies (no dev deps in production)
RUN uv sync --frozen --no-dev

# Copy application code
COPY config/ config/
COPY src/ src/

# Create logs directory
RUN mkdir -p logs data

# Expose dashboard port
EXPOSE 8080

# Health check via dashboard API
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/state')" || exit 1

# Run the trading system
CMD ["uv", "run", "python", "-m", "src"]
