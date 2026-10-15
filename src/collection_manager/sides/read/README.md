# Read side

Search holds index read guards through scoring, pin rules, facets and grouping,
then drops them before document retrieval and hooks. Missing documents are
omitted from hits and groups, including deletes between scoring and retrieval.
Shelf and batch-ID reads do not acquire index read guards.

Local and legacy global Zebo document reads acquire their guard inside
`spawn_blocking`, iterate and decode the complete result there, and return an
owned vector. Header and diagnostic reads also use the blocking pool. A cancelled
request cannot leave a borrowed guard behind in a running blocking task.

`replication.rs` owns receive/apply/commit progress, readiness, receive-task
supervision and silent-consumer recovery. Only successful operations advance
apply progress. See `docs/api.md` at the repository root for readiness settings
and exit behavior. Tests inject the exit callable and receiver source.
