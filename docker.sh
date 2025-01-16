docker build -t orama-core .

docker run -it \
  -p 8080:8080 \
  -p 50051:50051 \
  -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
  -v ./config.jsonc:/app/config.jsonc \
  --gpus all \
  orama-core