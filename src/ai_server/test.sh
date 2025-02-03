# Get all available services
grpcurl -plaintext localhost:50051 list

# Health check
grpcurl -plaintext localhost:50051 orama_ai_service.LLMService/CheckHealth

# Embeddings
grpcurl -plaintext -d '{ "model": "MultilingualE5Small", "input": ["The quick brown fox jumps over the lazy dog"], "intent": "passage" }' localhost:50051 orama_ai_service.LLMService/GetEmbedding

# Chat (non-streaming)
grpcurl -plaintext -H 'x-api-key: read_api_key' -d '{ "model": "google_query_translator", "prompt": "I am installing my Ryzen 9 9900X and I fear I bent some pins. What should I do?" }' localhost:50051 orama_ai_service.LLMService/Chat

# Chat (streaming)
grpcurl -plaintext -H 'x-api-key: read_api_key' -d '{ "model": "google_query_translator", "prompt": "I am installing my Ryzen 9 9900X and I fear I bent some pins. What should I do?" }' localhost:50051 orama_ai_service.LLMService/ChatStream

# Answer with context (streaming)
grpcurl -plaintext -H 'x-api-key: read_api_key' -d '{ "model": "answer", "prompt": "What do you know about beavers?", "conversation": { "messages": [ { "role": "USER", "content": "Beavers (genus Castor) are large, semiaquatic rodents of the Northern Hemisphere. There are two existing species: the North American beaver (Castor canadensis) and the Eurasian beaver (C. fiber). Beavers are the second-largest living rodents, after capybaras, weighing up to 50 kg (110 lb). They have stout bodies with large heads, long chisel-like incisors, brown or gray fur, hand-like front feet, webbed back feet, and tails that are flat and scaly. The two species differ in skull and tail shape and fur color. Beavers can be found in a number of freshwater habitats, such as rivers, streams, lakes and ponds. They are herbivorous, consuming tree bark, aquatic plants, grasses and sedges." } ] } }' localhost:50051 orama_ai_service.LLMService/ChatStream

# Answer with context (non-streaming)
grpcurl -plaintext -H 'x-api-key: read_api_key' -d '{ "model": "answer", "prompt": "What do you know about beavers?", "conversation": { "messages": [ { "role": "USER", "content": "Beavers (genus Castor) are large, semiaquatic rodents of the Northern Hemisphere. There are two existing species: the North American beaver (Castor canadensis) and the Eurasian beaver (C. fiber). Beavers are the second-largest living rodents, after capybaras, weighing up to 50 kg (110 lb). They have stout bodies with large heads, long chisel-like incisors, brown or gray fur, hand-like front feet, webbed back feet, and tails that are flat and scaly. The two species differ in skull and tail shape and fur color. Beavers can be found in a number of freshwater habitats, such as rivers, streams, lakes and ponds. They are herbivorous, consuming tree bark, aquatic plants, grasses and sedges." } ] } }' localhost:50051 orama_ai_service.LLMService/Chat

# Party Planner (streaming)
grpcurl -plaintext -H 'x-api-key: read_api_key' -d '{ "input": "I just started playing basketball and I want a good pair of shoes. Can you help me choose one?", "collection_id": "nike-data", "conversation": [] }' localhost:50051 orama_ai_service.LLMService/PlannedAnswer