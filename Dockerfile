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

EXPOSE 8080

RUN cargo build --release

CMD ["./target/release/rustorama"]