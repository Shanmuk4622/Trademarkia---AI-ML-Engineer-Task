# Use slim Python 3.10 image — minimises layer size while keeping a full CPython runtime.
# We avoid alpine because many scientific packages (scipy, numpy) need glibc and
# building manylinux wheels on musl libc causes subtle ABI issues.
FROM python:3.10-slim

# WORKDIR first so all subsequent paths are relative
WORKDIR /app

# ---- Install system deps ----
# libgomp1: required by scikit-learn/skfuzzy for OpenMP parallelism
# curl: health check in docker-compose
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
 && rm -rf /var/lib/apt/lists/*

# ---- Copy and install Python deps first (layer cache optimisation) ----
# Copying requirements before source so dep layer is only invalidated
# when requirements.txt changes, not on every code edit.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy project source ----
COPY src/ ./src/
COPY clusters/ ./clusters/
COPY embeddings/ ./embeddings/
COPY cache/ ./cache/

# Create cache dir in case it's empty on first run
RUN mkdir -p cache

# ---- Runtime configuration ----
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose the uvicorn port
EXPOSE 8000

# Single uvicorn command as required by the spec.
# --host 0.0.0.0 so the container port is reachable from outside.
# No --reload in production (would require watchfiles and adds overhead).
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
