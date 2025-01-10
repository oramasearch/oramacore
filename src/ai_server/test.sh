grpcurl -d '{ "model": "BGESmall", "input": ["hello, world!", "hey there", "foo bar"], "intent": "passage" }' -plaintext localhost:50051 orama_ai_service.CalculateEmbeddingsService/GetEmbedding
