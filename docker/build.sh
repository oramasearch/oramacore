#!/bin/bash
set -axe

VERSION=1.1.30

# Build OramaCore

## Build OramaCore for x86_64
docker buildx build --platform linux/amd64 -f Dockerfile-oramacore-x86 -t oramacore ..
docker tag oramacore oramasearch/oramacore:$VERSION
docker push oramasearch/oramacore:$VERSION
docker tag oramacore oramasearch/oramacore:latest
docker push oramasearch/oramacore:latest

## Build OramaCore for arm64
docker buildx build --platform linux/arm64 -f Dockerfile-oramacore-arm64 -t oramacore-arm64 ..
docker tag oramacore-arm64 oramasearch/oramacore-arm64:$VERSION
docker push oramasearch/oramacore-arm64:$VERSION
docker tag oramacore-arm64 oramasearch/oramacore-arm64:latest
docker push oramasearch/oramacore-arm64:latest

# Build OramaCore AI Server
cd ../src/ai_server

## Build AI Server without CUDA (CPU version)
# Create a non-CUDA version using a standard Python image
docker build -f ../../docker/Dockerfile-ai-server -t oramacore-ai-server .
docker tag oramacore-ai-server oramasearch/oramacore-ai-server:$VERSION
docker push oramasearch/oramacore-ai-server:$VERSION
docker tag oramacore-ai-server oramasearch/oramacore-ai-server:latest
docker push oramasearch/oramacore-ai-server:latest

## Build AI Server with CUDA
# The AI server needs to be built from the context of the ai_server directory
docker build -f ../../docker/Dockerfile-ai-server-cuda -t oramacore-ai-server-cuda .
docker tag oramacore-ai-server-cuda oramasearch/oramacore-ai-server-cuda:$VERSION
docker push oramasearch/oramacore-ai-server-cuda:$VERSION
docker tag oramacore-ai-server-cuda oramasearch/oramacore-ai-server-cuda:latest
docker push oramasearch/oramacore-ai-server-cuda:latest