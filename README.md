# Rustorama

How to run:
```bash
RUST_LOG=trace cargo run --bin rustorama
```
or, for release mode:
```bash
RUST_LOG=trace cargo run --bin rustorama --release
```

The configuration file is located at `config.jsonc` and contains an example of the configuration.

NB: on MacOS, mistralrs uses `metal` as the default backend, for other OS, CPU is used as the default backend.

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
hurl --test --variables-file api-test.hurl.property api-test.hurl
```

NB: you need to have the server running before running the tests.


## Run embedding examples

### Product quantization

Download the dataset at [https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset](https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset) and place it under `/src/bin/datasets`.

Then:

```bash
cargo run --release --bin pq_bench
```
