
set -axe

VERSION=1.0.3

docker build -t oramacore .
docker tag oramacore oramasearch/oramacore:$VERSION
docker push oramasearch/oramacore:$VERSION
docker tag oramacore oramasearch/oramacore:latest
docker push oramasearch/oramacore:latest

cd src/ai_server && docker build -t oramacore-ai-server .
cd src/ai_server && docker tag oramacore-ai-server oramasearch/oramacore-ai-server:$VERSION
cd src/ai_server && docker push oramasearch/oramacore-ai-server:$VERSION
cd src/ai_server && docker tag oramacore-ai-server oramasearch/oramacore-ai-server:latest
cd src/ai_server && docker push oramasearch/oramacore-ai-server:latest
