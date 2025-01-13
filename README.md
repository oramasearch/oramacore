# Orama Core

🚧 Under active development 🚧

**Orama Core** is the actual database you need for your AI projects, answer engines, copilots, and search.

It includes a fully-fledged full-text search engine, vector database, LLM interface, and many more utilities.

## Requirements

To run **Orama Core** locally, you need to have the following programming languages installed:

- Python 3.11
- Rust 1.83.0

The Rust part of Orama Core communicates with Python via gRPC. So you'll also need to install a protobuf compiler:

```bash
apt-get install protobuf-compiler
```

After that, just install the dependencies:

```bash
cargo build
```

```bash
cd ./src/ai_server && pip install -r requirements.txt
```

An NVIDIA GPU is highly recommended for running the application.

## Getting Started

How to run:
```bash
RUST_LOG=trace PROTOC=/usr/bin/protoc cargo run --bin rustorama
```
or, for release mode:
```bash
RUST_LOG=trace PROTOC=/usr/bin/protoc cargo run --bin rustorama --release
```

The configuration file is located at `config.jsonc` and contains an example of the configuration.

## Tests

To run the tests, use:
```bash
cargo test
```

## E2E tests

Install `hurl`:
```
cargo install hurl
```

Run the tests:
```
hurl --very-verbose --test --variables-file api-test.hurl.property api-test.hurl
hurl --very-verbose --test --variables-file api-test.hurl.property embedding-api-test.hurl
```

NB: you need to have the server running before running the tests.

## License

AGPLv3