docker build -t oramacore .
docker tag oramacore oramasearch/oramacore:1.0.2
docker push

cd src/ai_server && docker build -t oramacore-ai-server .
cd src/ai_server && docker tag oramacore-ai-server oramasearch/oramacore-ai-server:1.0.2
cd src/ai_server && docker push