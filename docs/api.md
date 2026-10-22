# Health API

`GET /health` is liveness and returns HTTP 200 with `"up"`.
`GET /health/ready` returns a JSON object and HTTP 200 or 503 according to reader
progress. Its fields are `applied_offset`, `last_applied_offset` (the same value),
`last_received_offset`, `last_applied_at`, `last_received_at`, `last_commit_at`,
and `ready`. Timestamps are RFC 3339 UTC, or null before the event occurs in this
process. A writer-only node is ready and returns null progress fields.

A reader is unready when newer operations are known from a received message or
stream-head probe and no apply has completed for `apply_stall_threshold`, or
when its latest commit failed or its last successful commit is older than twice
`reader_side.config.commit_interval`. Startup uses process start time as the
baseline until the first apply or commit. Offset zero is a valid first message;
a head probe at zero with no completed apply counts as pending work. Readiness does not change `/health`.

Configure `reader_side.replication` with duration strings:

```yaml
replication:
  apply_stall_threshold: 5m
  consumer_stall_threshold: 2m
  consumer_check_interval: 30s
```

All three durations must be positive. Defaults apply when the section or a field
is absent. Every check interval, a bounded short-lived RabbitMQ consumer reads
from the last chunk to discover the head; the client does not expose a stream-head
metadata call. The probe has its own connection and consumer name and closes
when idle. The watchdog recreates a silent consumer from the next unapplied
offset when the head is ahead for the stall threshold. Applying an operation
does not count as consumer silence. A failed recreation or silence lasting twice
the threshold exits with status 1. Repeated head-probe failure with no received
operations for twice the threshold also exits, allowing a service manager to
restore a connection whose health cannot be established.

The receive task is supervised. A panic, receive/apply error, unrecoverable source
closure or exhausted reconnect logs the last successfully applied offset and
exits nonzero. Reconnecting a closed source is bounded to 120 seconds including
retries. Shutdown requests stop the loop without invoking the exit path. A queued
operation that cannot be decoded or applied can fail again after restart and
require operator intervention; the reader does not silently skip it. Reconnect
uses in-memory apply progress, while process restart replays from the persisted
commit offset.

A RabbitMQ reader starts from the first operation it has not applied: offset 0
when `read.info` is missing or stores 0, otherwise the stored offset plus one.
Before creating its consumer it probes the stream's first retained offset and
head, and refuses to start if the stream cannot replay from that offset, instead
of serving an index with a gap. The refusal is a startup error: the process exits
nonzero with the message on stderr. The message begins with a stable code:

- `REPLICATION_HISTORY_EXPIRED`: retention removed the required offset, or the
  broker delivered an offset other than the next expected one.
- `REPLICATION_STREAM_BEHIND`: the required offset is past the stream head, or the
  stream is empty and the reader has already applied operations.

An empty stream is accepted for a reader that has applied nothing. Recreating a
consumer runs the same checks, and a refused reconnect exits without retrying.
After starting, the reader stays unready until it has applied the stream head
observed at startup. Shutdown commits after the receive and commit loops stop,
so the stored offset never covers operations missing from the committed index.

Configure load-balancer health paths separately when deploying; this API change
does not change any target-group or probe configuration.
