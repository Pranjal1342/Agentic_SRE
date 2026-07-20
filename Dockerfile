FROM python:3.11-slim

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# System deps for psycopg/asyncpg and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embedding model at build time so runtime is air-gapped
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY --chown=user . .

# Build FAISS runbook index at image build time (baked in, read-only at runtime)
RUN python -c "from rag.runbook_rag import build_index; build_index()" || true

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

EXPOSE 7860

CMD ["python", "app.py"]
