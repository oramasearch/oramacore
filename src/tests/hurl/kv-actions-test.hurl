
# First create a new collection for test purposes
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

# ------------------------------------------------------------------------------------
#
# Segments
#
# ------------------------------------------------------------------------------------

# Create a new segment
POST {{base_url}}/v1/collections/{{collection_id}}/segments/insert
Authorization: Bearer my-write-api-key
{
    "id": "1",
    "name": "My first segment",
    "description": "My first segment description",
    "goal": "My first segment goal"
}

HTTP 201

# Get the segment
GET {{base_url}}/v1/collections/{{collection_id}}/segments/get?api-key=my-read-api-key&segment_id=1
HTTP 200
[Asserts]
jsonpath "$.segment.id" == "1"
jsonpath "$.segment.name" == "My first segment"
jsonpath "$.segment.description" == "My first segment description"
jsonpath "$.segment.goal" == "My first segment goal"

# Get a non existing segment
GET {{base_url}}/v1/collections/{{collection_id}}/segments/get?api-key=my-read-api-key&segment_id=nonexisting
HTTP 200
[Asserts]
jsonpath "$.segment" == null

# Insert a second segment
POST {{base_url}}/v1/collections/{{collection_id}}/segments/insert
Authorization: Bearer my-write-api-key
{
    "id": "2",
    "name": "My second segment",
    "description": "My second segment description",
    "goal": "My second segment goal"
}

HTTP 201

# List all segments
GET {{base_url}}/v1/collections/{{collection_id}}/segments/all?api-key=my-read-api-key
HTTP 200
[Asserts]
jsonpath "$.segments" count == 2
jsonpath "$.segments[0].id" == "1"
jsonpath "$.segments[0].name" == "My first segment"
jsonpath "$.segments[0].description" == "My first segment description"
jsonpath "$.segments[0].goal" == "My first segment goal"
jsonpath "$.segments[1].id" == "2"
jsonpath "$.segments[1].name" == "My second segment"
jsonpath "$.segments[1].description" == "My second segment description"
jsonpath "$.segments[1].goal" == "My second segment goal"

# Update the first segment
POST {{base_url}}/v1/collections/{{collection_id}}/segments/update
Authorization: Bearer my-write-api-key
{
    "id": "1",
    "name": "My updated first segment",
    "description": "My updated first segment description",
    "goal": "My updated first segment goal"
}
HTTP 200
[Asserts]
jsonpath "$.success" == true

# Get updated segment
GET {{base_url}}/v1/collections/{{collection_id}}/segments/get?api-key=my-read-api-key&segment_id=1
HTTP 200
[Asserts]
jsonpath "$.segment.id" == "1"
jsonpath "$.segment.name" == "My updated first segment"
jsonpath "$.segment.description" == "My updated first segment description"
jsonpath "$.segment.goal" == "My updated first segment goal"

# Delete the first segment
POST {{base_url}}/v1/collections/{{collection_id}}/segments/delete
Authorization: Bearer my-write-api-key
{
    "id": "1"
}
HTTP 200
[Asserts]
jsonpath "$.success" == true

# List all segments - only one should be left
GET {{base_url}}/v1/collections/{{collection_id}}/segments/all?api-key=my-read-api-key
HTTP 200
[Asserts]
jsonpath "$.segments" count == 1
jsonpath "$.segments[0].id" == "2"
jsonpath "$.segments[0].name" == "My second segment"
jsonpath "$.segments[0].description" == "My second segment description"
jsonpath "$.segments[0].goal" == "My second segment goal"

# ------------------------------------------------------------------------------------
#
# Triggers
#
# ------------------------------------------------------------------------------------

# Create a new trigger linked to the second segment
POST {{base_url}}/v1/collections/{{collection_id}}/triggers/insert
Authorization: Bearer my-write-api-key
{
    "id": "1",
    "name": "My first trigger",
    "description": "My first trigger description",
    "response": "My first trigger response",
    "segment_id": "2"
}
HTTP 201
[Asserts]
jsonpath "$.success" == true
jsonpath "$.id" == "1"
jsonpath "$.trigger.id" == "1"
jsonpath "$.trigger.name" == "My first trigger"
jsonpath "$.trigger.description" == "My first trigger description"
jsonpath "$.trigger.response" == "My first trigger response"
jsonpath "$.trigger.segment_id" == "2"

# Get the trigger
GET {{base_url}}/v1/collections/{{collection_id}}/triggers/get?api-key=my-read-api-key&trigger_id=1
HTTP 200
[Asserts]
jsonpath "$.trigger.id" == "1"
jsonpath "$.trigger.name" == "My first trigger"
jsonpath "$.trigger.description" == "My first trigger description"
jsonpath "$.trigger.response" == "My first trigger response"
jsonpath "$.trigger.segment_id" == "2"

# Insert a second trigger
POST {{base_url}}/v1/collections/{{collection_id}}/triggers/insert
Authorization: Bearer my-write-api-key
{
    "id": "2",
    "name": "My second trigger",
    "description": "My second trigger description",
    "response": "My second trigger response",
    "segment_id": "2"
}
HTTP 201
[Asserts]
jsonpath "$.success" == true
jsonpath "$.id" == "2"
jsonpath "$.trigger.id" == "2"
jsonpath "$.trigger.name" == "My second trigger"
jsonpath "$.trigger.description" == "My second trigger description"
jsonpath "$.trigger.response" == "My second trigger response"
jsonpath "$.trigger.segment_id" == "2"

# List all triggers
GET {{base_url}}/v1/collections/{{collection_id}}/triggers/all?api-key=my-read-api-key
HTTP 200
[Asserts]
jsonpath "$.triggers" count == 2
jsonpath "$.triggers[0].id" == "1"
jsonpath "$.triggers[0].name" == "My first trigger"
jsonpath "$.triggers[0].description" == "My first trigger description"
jsonpath "$.triggers[0].goal" == "My first trigger goal"
jsonpath "$.triggers[1].id" == "2"
jsonpath "$.triggers[1].name" == "My second trigger"
jsonpath "$.triggers[1].description" == "My second trigger description"
jsonpath "$.triggers[1].goal" == "My second trigger goal"