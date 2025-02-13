# Set default CUDA version that can be overridden during build
# Version should match available options at https://hub.docker.com/r/nvidia/cuda/tags
ARG CUDA_VERSION=12.4.1

# Stage 1: Rust builder
FROM rust:1.84-slim-bookworm AS rust-builder

RUN apt-get update && apt-get install -y \
  libssl3 \
  pkg-config \
  libssl-dev \
  g++ \
  build-essential \
  ca-certificates \
  libgflags-dev \
  libsnappy-dev \
  zlib1g-dev \
  libbz2-dev \
  libzstd-dev \
  libxml2-dev \
  protobuf-compiler \
  python3 \
  curl \
  git

ENV RUSTY_V8_MIRROR=https://github.com/denoland/rusty_v8/releases/download
ENV V8_FROM_SOURCE=0

WORKDIR /usr/src/app
COPY . .
RUN cargo build --release

# Stage 2: Flash-attention builder
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04 AS flash-builder

RUN apt-get update && apt-get install -y \
  python3 \
  python3-pip \
  python3-dev \
  build-essential \
  ninja-build

# Install build dependencies first
RUN pip install --upgrade pip \
  && pip install packaging \
  && pip install torch \
  && pip install --no-cache-dir --no-build-isolation flash-attn \
  && rm -rf /root/.cache/pip

# Stage 3: Final runtime
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu22.04

# Install Python and other dependencies
RUN apt-get update && apt-get install -y \
  python3 \
  python3-pip \
  python3-dev \
  build-essential \
  curl \
  && rm -f /usr/bin/python /usr/bin/pip \
  && ln -s /usr/bin/python3 /usr/bin/python \
  && ln -s /usr/bin/pip3 /usr/bin/pip \
  && rm -rf /var/lib/apt/lists/*

# Copy flash-attention from builder
COPY --from=flash-builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# Install grpcurl
RUN curl -sSL "https://github.com/fullstorydev/grpcurl/releases/download/v1.8.9/grpcurl_1.8.9_linux_x86_64.tar.gz" | tar -xz -C /usr/local/bin && \
  chmod +x /usr/local/bin/grpcurl

WORKDIR /app

# Copy requirements first to leverage caching
COPY src/ai_server/requirements.txt /app/requirements.txt

# Install Python dependencies (excluding flash-attn since we copied it)
RUN pip install --upgrade pip && \
  pip install --no-cache-dir -r requirements.txt && \
  rm -rf /root/.cache/pip

COPY src/ai_server /app/ai_server
COPY --from=rust-builder /usr/src/app/target/release/oramacore /app/oramacore

RUN mkdir -p /root/.cache/huggingface

RUN echo '#!/bin/bash\n\
  cd ai_server && python server.py &\n\
  \n\
  until grpcurl -plaintext localhost:50051 orama_ai_service.LLMService/CheckHealth 2>/dev/null | grep -q "\"status\": \"OK\""; do\n\
  echo "Waiting for Python gRPC server..."\n\
  sleep 5\n\
  done\n\
  \n\
  cd /app && ./oramacore\n\
  ' > /app/start.sh

RUN chmod +x /app/start.sh

EXPOSE 8080 50051

CMD ["/app/start.sh"]
