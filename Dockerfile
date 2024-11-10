FROM rust:slim

RUN apt-get update && \
    apt-get install -y \
    pkg-config \
    libssl-dev \
    build-essential \
    g++ \
    clang \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN cargo build --release

WORKDIR /app/rustorama

CMD ["cargo", "run", "--release"]
