# Operation transport

RabbitMQ consumers reconnect from the read side's next unapplied offset. Closing
the old consumer is best effort and bounded. A separate short-lived consumer at
`OffsetSpecification::Last` reads the last chunk to determine stream head, with
bounded creation, idle reads and close. Its offset is observation only; it never
advances the reader's apply position. In-memory receivers have no remote head.

Creating or recreating a RabbitMQ receiver first probes the stream's first
retained offset and head, and refuses a start offset the stream no longer holds
(`ReplicationRefusal`). Every delivery must carry the next expected offset. See
`docs/api.md` for the error codes; `src/tests/replication_retention_broker.rs`
runs these checks against a real broker.
