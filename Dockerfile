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

FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Install Python, pip and other dependencies
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

# Install grpcurl
RUN curl -sSL "https://github.com/fullstorydev/grpcurl/releases/download/v1.8.9/grpcurl_1.8.9_linux_x86_64.tar.gz" | tar -xz -C /usr/local/bin && \
  chmod +x /usr/local/bin/grpcurl

WORKDIR /app

# Copy requirements first to leverage caching
COPY src/ai_server/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && \
  pip install --no-cache-dir -r requirements.txt && \
  pip install --no-cache-dir --no-build-isolation flash-attn && \
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