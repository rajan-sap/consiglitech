# =============================================================================
# RAG Pipeline - Production Dockerfile
# =============================================================================
FROM python:3.10.16-slim AS base

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Dependencies Stage
# =============================================================================
FROM base AS dependencies

# Copy and install requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Production Stage
# =============================================================================
FROM dependencies AS production

# Copy application code
COPY ingestion/ ./ingestion/
COPY generation/ ./generation/
COPY retrieval/ ./retrieval/
COPY evaluation/ ./evaluation/
COPY main.py ./
COPY streamlit_app.py ./

# Create directories for data and vector store
RUN mkdir -p /app/data /app/chroma_db

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default: Run Streamlit app
# Override with: docker run <image> python main.py (for ingestion)
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
