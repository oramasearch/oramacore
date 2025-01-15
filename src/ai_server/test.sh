# Get all available services
grpcurl -plaintext localhost:50051 list

# Health check
grpcurl -plaintext localhost:50051 orama_ai_service.LLMService/CheckHealth

# Embeddings
grpcurl -plaintext -d '{ "model": "BGESmall", "input": ["hello, world!", "hey there", "foo bar"], "intent": "passage" }' localhost:50051 orama_ai_service.LLMService/GetEmbedding

# Chat (non-streaming)
grpcurl -plaintext -d '{ "model": "google_query_translator", "prompt": "I am installing my Ryzen 9 9900X and I fear I bent some pins. What should I do?" }' localhost:50051 orama_ai_service.LLMService/Chat

# Chat (streaming)
grpcurl -plaintext localhost:50051 orama_ai_service.LLMService/ChatStream -d '{ "model": "google_query_translator", "prompt": "I am installing my Ryzen 9 9900X and I fear I bent some pins. What should I do?" }'

# Answer with context (streaming)
grpcurl -plaintext localhost:50051 orama_ai_service.LLMService/ChatStream -d '{ "model": "content_expansion", "prompt": "What do you know about beavers?", "conversation": { "messages": [ { "role": "USER", "content": "Beavers (genus Castor) are large, semiaquatic rodents of the Northern Hemisphere. There are two existing species: the North American beaver (Castor canadensis) and the Eurasian beaver (C. fiber). Beavers are the second-largest living rodents, after capybaras, weighing up to 50 kg (110 lb). They have stout bodies with large heads, long chisel-like incisors, brown or gray fur, hand-like front feet, webbed back feet, and tails that are flat and scaly. The two species differ in skull and tail shape and fur color. Beavers can be found in a number of freshwater habitats, such as rivers, streams, lakes and ponds. They are herbivorous, consuming tree bark, aquatic plants, grasses and sedges." } ] } }'