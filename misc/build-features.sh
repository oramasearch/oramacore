cargo build --bin oramacore --no-default-features --features reader
cp target/debug/oramacore target/debug/oramacore-reader
cargo build --bin oramacore --no-default-features --features writer
cp target/debug/oramacore target/debug/oramacore-writer

ORAMACORE_HTTP_PORT=8080 target/debug/oramacore-writer &
ORAMACORE_HTTP_PORT=8081 target/debug/oramacore-reader &
