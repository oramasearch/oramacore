
set -axe

<<<<<<< HEAD
VERSION=1.1.1
=======
VERSION=1.1.2
>>>>>>> develop
docker build -t oramacore .
docker tag oramacore oramasearch/oramacore:$VERSION
docker push oramasearch/oramacore:$VERSION
docker tag oramacore oramasearch/oramacore:latest
docker push oramasearch/oramacore:latest

cd src/ai_server && docker build -t oramacore-ai-server . \
&& docker tag oramacore-ai-server oramasearch/oramacore-ai-server:$VERSION \
&& docker push oramasearch/oramacore-ai-server:$VERSION \
&& docker tag oramacore-ai-server oramasearch/oramacore-ai-server:latest \
&& docker push oramasearch/oramacore-ai-server:latest \
