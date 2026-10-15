# Core implementation

The collection manager replicates writer operations into reader indexes. Its
read-side README describes storage and replication concurrency. `lock.rs` wraps
Tokio locks with metrics; `blocking_read` is reserved for blocking-pool closures
that own an `Arc` to the lock for the lifetime of synchronous I/O.
