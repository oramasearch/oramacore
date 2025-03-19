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
    "read_api_key": "my-read-api-key",
    "write_api_key": "my-write-api-key"
}
HTTP 201

GET {{base_url}}/v1/collections/{{collection_id}}
Authorization: Bearer my-master-api-key
HTTP 200
[Asserts]
jsonpath "$.document_count" == 0


POST {{base_url}}/v1/collections/{{collection_id}}/insert
Authorization: Bearer my-write-api-key
[
    {
        "id": "1",
        "title": "The Beatles",
        "content": "The Beatles were an English rock band formed in Liverpool in 1960. With a line-up comprising John Lennon, Paul McCartney, George Harrison and Ringo Starr, they are regarded as the most influential band of all time.",
        "number": 1
    },
    {
        "id": "2",
        "title": "The Rolling Stones",
        "content": "The Rolling Stones are an English rock band formed in London in 1962. The first settled line-up consisted of Brian Jones, Ian Stewart, Mick Jagger, Keith Richards, Bill Wyman, and Charlie Watts.",
        "number": 2
    }
]
HTTP 200

GET {{base_url}}/v1/collections/{{collection_id}}
Authorization: Bearer my-master-api-key
HTTP 200
[Asserts]
jsonpath "$.document_count" == 2

POST {{base_url}}/v1/collections/{{collection_id}}/search?api-key=my-read-api-key
[Options]
retry: 10
retry-interval: 500ms
{
    "term": "beatles"
}
HTTP 200
[Asserts]
jsonpath "$.count" == 1
jsonpath "$.hits[0].id" == "1"