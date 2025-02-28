# Set default CUDA version that can be overridden during build
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

# Stage 2: Final runtime
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu22.04

# Set non-interactive and timezone to avoid prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install Python 3.11 and other dependencies
RUN apt-get update && apt-get install -y \
  software-properties-common \
  curl \
  build-essential \
  tzdata \
  && ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime \
  && dpkg-reconfigure --frontend noninteractive tzdata \
  && add-apt-repository -y ppa:deadsnakes/ppa \
  && apt-get update \
  && apt-get install -y \
  python3.11 \
  python3.11-dev \
  python3.11-distutils \
  python3.11-venv \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
  && python3.11 get-pip.py \
  && rm get-pip.py

# Set Python 3.11 as the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
  && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create a virtual environment for better isolation
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy requirements first to leverage caching
COPY src/ai_server/requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && \
  pip install --no-cache-dir -r requirements.txt && \
  rm -rf /root/.cache/pip

COPY src/ai_server /app/ai_server
COPY --from=rust-builder /usr/src/app/target/release/oramacore /app/oramacore

RUN mkdir -p /root/.cache/huggingface

RUN echo '#!/bin/bash\n\
  cd ai_server && python server.py &\n\
  cd /app && ./oramacore\n\
  ' > /app/start.sh

RUN chmod +x /app/start.sh

EXPOSE 8080 50051

CMD ["/app/start.sh"]