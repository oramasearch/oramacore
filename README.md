![OramaCore](/docs/public/oramacore.png)

**OramaCore** is the AI runtime you need for your projects, answer engines,
copilots, and search.

It includes a fully-fledged full-text search engine, vector database, LLM
interface with action planning and reasoning, a JavaScript runtime to write and
run your own custom agents on your data, and many more utilities.

## Getting Started

Pull the Docker image:

```sh
docker pull oramasearch/oramacore:latest
```

Then compile your
[configuration file](https://docs.oramacore.com/docs/guide/configuration) and
run it:

```sh
docker run \
  -p 8080:8080 \
  -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
  -v ./config.yaml:/app/config.yaml \
  --gpus all \
  oramacore
```

An NVIDIA GPU is highly recommended for running the application.

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
        { role: "user", content: "How do I install OramaCore?" },
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

## License

[AGPLv3](/LICENSE.md)
