# Rustorama

How to run:
```bash
cargo run -p rustorama
```
or, for release mode:
```bash
cargo run -p rustorama --release
```

The configuration file is located at `config.json` and contains an example of the configuration.

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
