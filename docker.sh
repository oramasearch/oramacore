docker build -t oramacore .

docker tag oramacore oramasearch/oramacore:1.0.0-rc1

docker run -d \
  -p 8080:8080 \
  -p 50051:50051 \
  -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
  -v ./config.yaml:/app/config.yaml \
  --gpus all \
  oramasearch/oramacore:lastest