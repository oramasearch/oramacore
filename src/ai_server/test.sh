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

# LLM answer streaming
grpcurl -d '{ "context": "Beavers (genus Castor) are large, semiaquatic rodents of the Northern Hemisphere. There are two existing species: the North American beaver (Castor canadensis) and the Eurasian beaver (C. fiber). Beavers are the second-largest living rodents, after capybaras, weighing up to 50 kg (110 lb). They have stout bodies with large heads, long chisel-like incisors, brown or gray fur, hand-like front feet, webbed back feet, and tails that are flat and scaly. The two species differ in skull and tail shape and fur color. Beavers can be found in a number of freshwater habitats, such as rivers, streams, lakes and ponds. They are herbivorous, consuming tree bark, aquatic plants, grasses and sedges.", "prompt": "What do you know about beavers?", "conversation": [] }' -plaintext localhost:50051 orama_ai_service.LLMService/AnswerSessionStream

# Get all available services
grpcurl -plaintext localhost:50051 list