# Tests

Reader-hardening coverage lives in `search_document_fetch.rs` (a blocked fetch
allows commit within one second), `health_ready.rs` (apply stall and commit
health), `receive_loop.rs` (silent source recovery, persistent stall, failed
recreation/probe/apply and panic exit), and `cjk_locales.rs` (regular and temporary
index creation, replacement and search for JP, KO and ZH, plus invalid input).
Run `cargo test --all` with the Python environment and test models described in
the root CLAUDE.md. Tests inject process exit and never terminate the test binary.

`replication_retention.rs` covers the replication refusal rules, readiness during
the initial catch-up and the shutdown commit; `receive_loop.rs` covers a refused
reconnect. `replication_retention_broker.rs` runs the refusal checks against a
disposable RabbitMQ broker. It is ignored by default; its header explains how to
run it with `--ignored`.
