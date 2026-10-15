# HTTP API

`/health` serves liveness. `/health/ready` exposes reader progress and returns
503 for stalled apply or failed/stale commits. Writer-only nodes return ready
with null progress fields. Response fields, defaults and supervision behavior
are documented in the repository's `docs/api.md`.
