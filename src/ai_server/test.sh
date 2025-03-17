# Get all available services
grpcurl -plaintext localhost:50051 list

# Health check
grpcurl -plaintext localhost:50051 orama_ai_service.LLMService/CheckHealth

# Embeddings
grpcurl -plaintext -d '{ "model": "MultilingualE5Small", "input": ["The quick brown fox jumps over the lazy dog"], "intent": "passage" }' localhost:50051 orama_ai_service.LLMService/GetEmbedding