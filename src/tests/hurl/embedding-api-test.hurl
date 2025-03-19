GET {{base_url}}/
HTTP 200

GET {{base_url}}/health
HTTP 200

GET http://www.randomnumberapi.com/api/v1.0/randomuuid
HTTP 200
[Captures]
collection_id: jsonpath "$[0]"

POST {{base_url}}/v1/collections/create
Authorization: Bearer my-master-api-key
{
    "id": "{{collection_id}}",
    "embeddings": {
        "model": "BGESmall",
        "document_fields": ["text"]
    },
    "read_api_key": "my-read-api-key",
    "write_api_key": "my-write-api-key"
}
HTTP 201

POST {{base_url}}/v1/collections/{{collection_id}}/insert
Authorization: Bearer my-write-api-key
[
    {
        "id": "1",
        "text": "The cat is sleeping on the table."
    },
    {
        "id": "2",
        "text": "A cat rests peacefully on the sofa."
    },
    {
        "id": "3",
        "text": "The dog is barking loudly in the yard."
    }
]
HTTP 200


POST {{base_url}}/v1/collections/{{collection_id}}/search?api-key=my-read-api-key
Authorization: Bearer my-read-api-key
[Options]
retry: 10
retry-interval: 500ms
{
    "mode": "vector",
    "term": "The feline is napping comfortably indoors.",
    "similarity": 0.6
}
HTTP 200
[Asserts]
jsonpath "$.hits[0].id" == "2"
