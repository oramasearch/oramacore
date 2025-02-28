docker build -t oramacore .

docker run -d \
  -p 8080:8080 \
  -p 50051:50051 \
  -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
  -v ./config.yaml:/app/config.yaml \
  --gpus all \
  oramasearch/oramacore:lastest


  docker run . \
  -p 8080:8080 \
  -p 50051:50051 \
  -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
  -v ./config.yaml:/app/config.yaml \
  --gpus all