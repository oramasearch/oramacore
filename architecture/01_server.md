# 01 - Orama Server Configuration

To bootstrap an Orama Server, you'll need a configuration file called `orama.yml` containing the following information:


```yml
# API key to be used to perform read and write operations on Orama
secret_key=mysupersecretkey

# Current node host
host=localhost

# Current node port
port=8080

# Current node RAFT server address
raft_address=http://127.0.0.1:8000

# RAFT peers
raft_peers=https://127.0.0.0:8000,https://127.0.0.0:8002,https://127.0.0.0:8003
```

## Health check route

Request:
```bash
curl http://localhost:8080/health
```

Response:
```text
ok
```

## RAFT status route

Request:
```bash
curl http://localhost:8080/cluster-info
```

Response:

```json
{
  "raft": {
    "is_leader": true,
    "leader_address": "http://localhost:8000"
  },
  "up_since": 1730452594663
}
```

## APIs

All APIs are versioned and prefixed with a `/v*` route group.