FROM rust:1.85-slim-bookworm AS rust-builder

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

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
  libssl3 \
  ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=rust-builder /usr/src/app/target/release/oramacore /app/
COPY --from=rust-builder /usr/src/app/config.yaml /app/

ENV RUST_LOG=oramacore=trace

EXPOSE 8080

CMD ["./oramacore"]