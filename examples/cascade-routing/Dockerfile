# syntax=docker/dockerfile:1.6
# Multi-stage build: stage 1 compiles the LLM386 PyO3 wheel and the
# `llm386` CLI binary from source; stage 2 is a slim Python runtime that
# carries both into the final image.

# ---- builder ----
FROM rust:1.95-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config \
        libssl-dev \
        python3 \
        python3-dev \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir maturin

WORKDIR /llm386

# Source layout the wheel build needs: workspace root + crates + python.
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY python ./python

# Build the Python wheel (PyO3 native extension).
RUN cd python && maturin build --release --interpreter python3 --out /wheels

# Build the `llm386` CLI binary so the runtime image can inspect the store.
RUN cargo build --release -p llm386-cli && \
    cp target/release/llm386 /usr/local/bin/llm386


# ---- runtime ----
FROM python:3.11-slim

WORKDIR /app

# Install the LLM386 wheel.
COPY --from=builder /wheels/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# Carry the CLI binary too so users can inspect the store in the same image.
COPY --from=builder /usr/local/bin/llm386 /usr/local/bin/llm386

# Demo dependencies.
COPY examples/cascade-routing/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Demo source.
COPY examples/cascade-routing/agent.py .
COPY examples/cascade-routing/llm386.toml .

ENV PYTHONUNBUFFERED=1 \
    LLM386_STORE=/data/store \
    LLM386_PROFILES=/app/llm386.toml \
    LLM386_SESSION=1

CMD ["python", "-u", "agent.py"]
