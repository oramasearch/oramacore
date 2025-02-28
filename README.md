![OramaCore](/docs/public/oramacore.png)

**OramaCore** is the AI runtime you need for your projects, answer engines,
copilots, and search.

It includes a fully-fledged full-text search engine, vector database, LLM
interface with action planning and reasoning, and many more utilities.

## Requirements

Pull the Docker image:

```bash
docker pull oramasearch/oramacore:latest
```

Then compile your
[configuration file](https://docs.oramacore.com/docs/guide/configuration) and
run it:

```bash
docker run \
  -p 8080:8080 \
  -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
  -v ./config.yaml:/app/config.yaml \
  --gpus all \
  oramacore
```

An NVIDIA GPU is highly recommended for running the application.

## License

[AGPLv3](/LICENSE.md)
