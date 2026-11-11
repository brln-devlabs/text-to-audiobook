# Build stage
FROM rust:1-bookworm AS builder

WORKDIR /build

# cmake and clang for llama.cpp, LAME and SQLite
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    clang \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock ./
COPY src ./src

# Headless build: CLI with LLM and TTS, no GUI
RUN cargo build --release --locked --no-default-features --features llm,tts
# (the LLM is only loaded with USE_LLM = true)

# Final stage
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/text-to-audiobook /usr/local/bin/text-to-audiobook
COPY settings.json /app/settings.json

WORKDIR /app
RUN mkdir -p /app/input /app/output /app/cache /app/logs /app/models

ENV TTA_ROOT=/app
ENV HF_HOME=/app/models/huggingface

ENTRYPOINT ["text-to-audiobook"]
CMD ["run"]
