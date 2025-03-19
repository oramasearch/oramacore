
set -axe

VERSION=1.0.2

docker build -t oramacore .
docker tag oramacore oramasearch/oramacore:$VERSION
docker push

cd src/ai_server && docker build -t oramacore-ai-server .
cd src/ai_server && docker tag oramacore-ai-server oramasearch/oramacore-ai-server:$VERSION
cd src/ai_server && docker push
