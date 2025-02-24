# OramaCore

🚧 Under active development. Do not use in production - APIs will change 🚧

**OramaCore** is the database you need for your AI projects, answer engines, copilots, and search.

It includes a fully-fledged full-text search engine, vector database, LLM interface, and many more utilities.

## Roadmap

- **v0.1.0**. ✅ ETA Jan 31st, 2025 (🚧 beta release)
    - ✅ Full-text search
    - ✅ Vector search
    - ✅ Search filters
    - ✅ Automatic embeddings generation
    - ✅ Built-in multiple LLM inference setup
    - ✅ Basic JavaScript integration
    - ✅ Disk persistence
    - ✅ Unified configuration
    - ✅ Dockerfile for load testing in production environment

- **v1.0.0**. ETA Feb 28th, 2025 (🎉 production ready!)
    - 🔜 Long-term user memory
    - 🚧 Multi-node setup
    - 🔜 Content expansion APIs
    - 🔜 JavaScript API integration
    - 🚧 Production-ready build
    - 🔜 Geosearch
    - 🚧 Zero-downtime upgrades
    - 🚧 Vector compression
    - 🚧 Benchmarks

## Requirements

To run **Orama Core** locally, you need to have the following programming languages installed:

- Python >= 3.11
- Rust >= 1.83.0

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
RUST_LOG=trace PROTOC=/usr/bin/protoc cargo run --bin oramacore
```
or, for release mode:
```bash
RUST_LOG=trace PROTOC=/usr/bin/protoc cargo run --bin oramacore --release
```

The configuration file is located at `config.jsonc` and contains an example of the configuration.

## Disk persistence

You can persist the database status on disk by runnng the following commands:

```bash
curl 'http://localhost:8080/v0/reader/dump_all' -X POST
```

```bash
curl 'http://localhost:8080/v0/writer/dump_all' -X POST
```

After killing and restarting the server, you'll find your data back in memory.

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

## Opentelemetry

OramaCore integrates Grafana and Prometheus for monitoring and collenting metrics.
Make sure your OramaCore configuration enables the prometheus exporter (`http.with_prometheus: true`).
Change the `otel/prometheus.yml` file to match your IP configuration and run the following command:
```
docker run --rm --name prometheus -d -p 9090:9090 -v $(pwd)/otel/prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus
docker run --rm --name grafana -d -p 3000:3000 grafana/grafana
```

We have a Grafana dashboard available at `otel/OramaCore Dashboard.json`. You can import it into your Grafana instance.

NB: the default username and password for Grafana are `admin` and `admin`.

## License

[AGPLv3](/LICENSE.md)

## Distributed
### RabbitMQ

```
docker run -it --rm --name rabbitmq -p 5552:5552 -p 15672:15672 -p 5672:5672  \
    -e RABBITMQ_SERVER_ADDITIONAL_ERL_ARGS='-rabbitmq_stream advertised_host localhost' \
    rabbitmq:4-management
docker exec rabbitmq rabbitmq-plugins enable rabbitmq_stream rabbitmq_stream_management
```