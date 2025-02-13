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

# Set environment variables for v8 build
ENV RUSTY_V8_MIRROR=https://github.com/denoland/rusty_v8/releases/download
ENV V8_FROM_SOURCE=0

WORKDIR /usr/src/app
COPY . .
RUN cargo build --release

FROM python:3.11-slim

RUN apt-get update && \
  apt-get install -y curl && \
  curl -sSL "https://github.com/fullstorydev/grpcurl/releases/download/v1.8.9/grpcurl_1.8.9_linux_x86_64.tar.gz" | tar -xz -C /usr/local/bin && \
  chmod +x /usr/local/bin/grpcurl && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src/ai_server /app/ai_server

RUN cd ai_server && pip install --no-cache-dir -r requirements.txt

COPY --from=rust-builder /usr/src/app/target/release/oramacore /app/oramacore

RUN mkdir -p /root/.cache/huggingface

RUN echo '#!/bin/bash\n\
  cd ai_server && python server.py &\n\
  \n\
  until grpcurl -plaintext localhost:50051 orama_ai_service.LLMService/CheckHealth 2>/dev/null | grep -q "\"status\": \"OK\""; do\n\
  echo "Waiting for Python gRPC server..."\n\
  sleep 1\n\
  done\n\
  \n\
  cd /app && ./oramacore\n\
  ' > /app/start.sh

RUN chmod +x /app/start.sh

EXPOSE 8080 50051

CMD ["/app/start.sh"]
