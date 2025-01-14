# Health check
grpcurl -plaintext localhost:50051 orama_ai_service.HealthCheckService/CheckHealth

# Embeddings
grpcurl -d '{ "model": "BGESmall", "input": ["hello, world!", "hey there", "foo bar"], "intent": "passage" }' -plaintext localhost:50051 orama_ai_service.CalculateEmbeddingsService/GetEmbedding

# LLM - No streaming
grpcurl -d '{ "model": "google_query_translator", "prompt": "I am installing my Ryzen 9 9900X and I fear I bent some pins. What should I do? True story here." }' -plaintext localhost:50051 orama_ai_service.LLMService/CallLLM

# LLM - Streaming
grpcurl -d '{ "model": "google_query_translator", "prompt": "I am installing my Ryzen 9 9900X and I fear I bent some pins. What should I do? True story here." }' -plaintext localhost:50051 orama_ai_service.LLMService/CallLLMStream

# Vision
grpcurl -d '{ "image": "'$(base64 -w0 image.jpg)'", "text": "What is in this image?" }' -plaintext localhost:50051 orama_ai_service.VisionService/CallVision

# Get all available services
grpcurl -plaintext localhost:50051 list