# 02 - Indexes

Orama works by creating indexes and allowing you to select which index to query at runtime by specifying the index ID and - if needed - an API key.

### Create a new index

**Request**

- method: `POST`
- route: `/v1/index/create`
- auth: `bearer <orama_key>`
- body properties:
  - `id`: `string`
  - `name`: `string`, optional (default: random `string`)
  - `api_key`: `string`, optional (default: `null`)
  - `language`: `string`, optional (default: `en`)
  - `schema`: `OramaSchema`, optional (default: `null`)
  - `stop_words`: `string[]`, optional (default: `[]`)
  - `stemming`: `boolean`, optional (default: `true`)

**Response**

```json
{}
```

### Update an index

**Request**

- method: `PATCH`
- route: `/v1/index/update`
- auth: `bearer <index_key>`
- body properties:
  - `id`: `string`
  - `name`: `string`
  - `stop_words`: `string[]`, optional (default: `[]`)
  - `stemming`: `boolean`, optional (default: `true`)
  - `embeddings_model`: `string`, optional (default: `e5-small`)
  - `embeddings_properties`: `string[]`, optional (default: all string properties found)

**Response**

```json
{}
```

### Delete and index

**Request**

- method: `DELETE`
- route: `/v1/index/delete`
- auth: `bearer <orama_key>`
- body properties:
    - `id`: `string`

**Response**

```json
{}
```

### Get index info

**Request**

- method: `GET`
- route: `/v1/index/info`
- auth: `bearer <index_key>`
- body properties:
    - `id`: `string`

**Response**

```json5
{
  "created_at": "<timestamp>",
  "updated_at": "<timestamp>",
  "docs": "<number>",
  "size": "<number>", // size in MB
  "vectors_chunks": "<string>",
  "embeddings_model": "<string>",
}
```