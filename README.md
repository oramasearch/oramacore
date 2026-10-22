![OramaCore](/misc/oramacore.png)

**OramaCore** is the AI runtime you need for your projects, answer engines,
copilots, and search.

It includes a fully-fledged full-text search engine, vector database, LLM
interface with action planning and reasoning, a JavaScript runtime to write and
run your own custom agents on your data, and many more utilities.

## Getting Started

The absolute easier way to get started is by following the
[docker-compose.yml](./docker-compose.yml) file that you can find in this
repository.

You can either clone the entire repo or setup `oramasearch/oramacore:latest` as
image in your `docker-compose.yml` file under the `oramacore` service.

Then compile your
[configuration file](https://docs.oramacore.com/docs/guide/configuration) and
run it:

```sh
docker compose up
```

This will create the following architecture, allowing you to perform
high-performance RAG with little to zero configuration.

![OramaCore Architecture](/docs/public/oramacore-arch.png)

An NVIDIA GPU is highly recommended for running the application. For production
usage, we recommend using minimum one NVIDIA A100. Optimal configuration would
include four NVIDIA H100.

## Document storage

Document storage normalizes each batch before reserving Zebo space: IDs are
sorted in release builds and duplicate IDs keep the last write in that batch.
This applies to global read commits, collection read commits, and collection
write inserts. Read commits normalize the whole uncommitted snapshot, including
documents accumulated across several updates. Before a commit, existing read
paths still return the first uncommitted copy of a repeated ID.

The Zebo dependency is pinned to an immutable git revision with release-mode
batch ordering checks. Replayed IDs across batches remain accepted with a
warning; a copy below a page's starting ID still routes to an earlier page.

## Available Dockerfiles

Depending on your machine, you may want to use different Docker images.

| Application   | CPU/GPU                              | Docker image                                                                                            |
| ------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| **OramaCore** | X86_64                               | [`oramasearch/oramacore`](https://hub.docker.com/r/oramasearch/oramacore)                               |
| **OramaCore** | ARM64 (Mac M series for example)     | [`oramasearch/oramacore-arm64`](https://hub.docker.com/r/oramasearch/oramacore-arm64)                   |
| **AI Server** | Any CPU architecture, no CUDA access | [`oramasearch/oramacore-ai-server`](https://hub.docker.com/r/oramasearch/oramacore-ai-server)           |
| **AI Server** | Any CPU architecture, CUDA available | [`oramasearch/oramacore-ai-server-cuda`](https://hub.docker.com/r/oramasearch/oramacore-ai-server-cuda) |

## Using the JavaScript SDK

You can install the official JavaScript SDK with npm:

```sh
npm i @orama/core
```

Then, you can start by creating a collection (a database index) with all of the
data you want to perform AI search & experiences on:

```js
import { OramaCoreManager } from "@orama/core";

const orama = new OramaCoreManager({
    url: "http://localhost:8080",
    masterAPIKey: "<master-api-key>", // The master API key set in your config file
});

const newCollection = await orama.createCollection({
    id: "products",
    writeAPIKey: "my-write-api-key", // A custom API key to perform write operations on your collection
    readAPIKey: "my-read-api-key", // A custom API key to perform read operations on your collection
});
```

Then, insert some data:

```js
import { CollectionManager } from "@orama/core";

const collection = new CollectionManager({
    url: "http://localhost:8080",
    collectionID: "<COLLECTION_ID>",
    writeAPIKey: "<write_api_key>",
});

// You can insert a single document
await collection.insert({
    title: "My first document",
    content: "This is the content of my first document.",
});

// Or you can insert multiple documents by passing an array of objects
await collection.insert([
    {
        title: "My first document",
        content: "This is the content of my first document.",
    },
    {
        title: "My second document",
        content: "This is the content of my second document.",
    },
]);
```

OramaCore will automatically generate highly optimized embeddings for you and
will store them inside its built-in vector database.

Now you can perform vector, hybrid, full-text search, or let OramaCore decide
which one is best for your specific query:

```js
import { CollectionManager } from "@orama/core";

const collection = new CollectionManager({
    url: "http://localhost:8080",
    collectionID: "<COLLECTION_ID>",
    readAPIKey: "<read_api_key>",
});

const results = await collection.search({
    term: "The quick brown fox",
    mode: "auto", // can be "fulltext", "vector", "hybrid", or "auto"
});
```

You can also perform **Answer Sessions** as you'd do on **Perplexity** or
**SearchGPT**, but on your own data!

```js
import { CollectionManager } from "@orama/core";

const collection = new CollectionManager({
    url: "http://localhost:8080",
    collectionID: "<COLLECTION_ID>",
    readAPIKey: "<read_api_key>",
});

const answerSession = collection.createAnswerSession({
    initialMessages: [
        {
            role: "user",
            content: "How do I install OramaCore?",
        },
        {
            role: "assistant",
            content:
                "You can install OramaCore by pulling the oramasearch/oramacore:latest Docker image",
        },
    ],
    events: {
        onStateChange(state) {
            console.log("State changed:", state);
        },
    },
});
```

Read more on the [official documentation](https://docs.oramacore.com/docs).

## Health checks

| Endpoint        | Purpose   | Response                                                                                       |
| --------------- | --------- | ---------------------------------------------------------------------------------------------- |
| `/health`       | Liveness  | Always `200` with `"up"` while the HTTP server runs.                                           |
| `/health/ready` | Readiness | `200` when the reader keeps up with the operation stream and commits on schedule, `503` if not. |

`/health/ready` returns `503` when the stream has an operation newer than the
last applied one and nothing has been applied for
`reader_side.replication.apply_stall_threshold` (default `5m`), or when the
last commit failed or is older than twice `reader_side.config.commit_interval`.
The body is the same for both status codes:

```json
{
  "applied_offset": 1042,
  "last_applied_offset": 1042,
  "last_received_offset": 1042,
  "last_applied_at": "2026-01-01T12:00:00.000Z",
  "last_received_at": "2026-01-01T12:00:00.000Z",
  "last_commit_at": "2026-01-01T11:59:00.000Z",
  "ready": true
}
```

Timestamps are `null` until the event first happens after startup. A node
running without the reader side always reports ready, with null progress fields.
`applied_offset` and `last_applied_offset` report the same successfully applied
position. See [the API notes](docs/api.md) for supervision and configuration.

Document retrieval runs on Tokio's blocking pool. Search releases its index read
locks after scoring, pin rules, facets and groups, before retrieving documents.
Documents deleted in that gap are omitted from hits and groups.

All collection languages are supported. Japanese, Korean and Chinese preserve
their locale identity and use the shared Unicode word tokenizer, splitting on
whitespace and punctuation without dictionary segmentation or stemming. The
writer resolves the same parser before publishing regular or temporary index
creation, and unknown language values are rejected when parsing API requests.

## License

[AGPLv3](/LICENSE.md)
