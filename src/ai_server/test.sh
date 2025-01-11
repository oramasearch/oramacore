grpcurl -d '{ "model": "BGESmall", "input": ["hello, world!", "hey there", "foo bar"], "intent": "passage" }' -plaintext localhost:50051 orama_ai_service.CalculateEmbeddingsService/GetEmbedding

# LLM - No streaming
grpcurl -d '{ "model": "content_expansion", "prompt": "Tell me about Paris" }' -plaintext localhost:50051 orama_ai_service.LLMService/CallLLM

# LLM - Streaming
grpcurl -d '{ "model": "content_expansion", "prompt": "Tell me about Paris" }' -plaintext localhost:50051 orama_ai_service.LLMService/CallLLMStream

# Vision
grpcurl -d '{ "image": "'$(base64 -w0 image.jpg)'", "text": "What is in this image?" }' -plaintext localhost:50051 orama_ai_service.VisionService/CallVision

# Get all available services
grpcurl -plaintext localhost:50051 list