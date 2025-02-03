use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::Result;
use http::uri::Scheme;
use redact::Secret;
use serde_json::json;
use tokio::time::sleep;

use crate::{
    ai::AIServiceConfig,
    build_orama,
    collection_manager::{
        dto::ApiKey,
        sides::{
            CollectionsWriterConfig, IndexesConfig, OramaModelSerializable, ReadSide, WriteSide,
        },
    },
    connect_write_and_read_side,
    test_utils::{create_grpc_server, generate_new_path},
    types::{CollectionId, DocumentList},
    web_server::HttpConfig,
    OramacoreConfig, ReadSideConfig, SideChannelType, WriteSideConfig,
};

fn create_oramacore_config() -> OramacoreConfig {
    OramacoreConfig {
        log: Default::default(),
        http: HttpConfig {
            host: "127.0.0.1".parse().unwrap(),
            port: 2222,
            allow_cors: false,
            with_prometheus: false,
        },
        ai_server: AIServiceConfig {
            host: "0.0.0.0".parse().unwrap(),
            port: 0,
            api_key: None,
            max_connections: 1,
            scheme: Scheme::HTTP,
        },
        writer_side: WriteSideConfig {
            master_api_key: ApiKey(Secret::new("my-master-api-key".to_string())),
            output: SideChannelType::InMemory,
            config: CollectionsWriterConfig {
                data_dir: generate_new_path(),
                embedding_queue_limit: 50,
                default_embedding_model: OramaModelSerializable(crate::ai::OramaModel::BgeSmall),
                // Lot of tests commit to test it.
                // So, we put an high value to avoid problems.
                insert_batch_commit_size: 10_000,
                javascript_queue_limit: 10_000,
                commit_interval: Duration::from_secs(3_000),
            },
        },
        reader_side: ReadSideConfig {
            input: SideChannelType::InMemory,
            config: IndexesConfig {
                data_dir: generate_new_path(),
                // Lot of tests commit to test it.
                // So, we put an high value to avoid problems.
                insert_batch_commit_size: 10_000,
                commit_interval: Duration::from_secs(3_000),
            },
        },
    }
}

async fn create(mut config: OramacoreConfig) -> Result<(Arc<WriteSide>, Arc<ReadSide>)> {
    if config.ai_server.port == 0 {
        let address = create_grpc_server().await?;
        config.ai_server.host = address.ip();
        config.ai_server.port = address.port();
        println!("AI server started on {}", address);
    }

    let (write_side, read_side, rec) = build_orama(config).await?;

    connect_write_and_read_side(rec, read_side.clone().unwrap());

    let write_side = write_side.unwrap();
    let read_side = read_side.unwrap();

    Ok((write_side, read_side))
}

#[tokio::test(flavor = "multi_thread")]
async fn test_simple_text_search() -> Result<()> {
    let (write_side, read_side) = create(create_oramacore_config()).await?;

    let collection_id = CollectionId("test-collection".to_string());
    create_collection(write_side.clone(), collection_id.clone()).await?;

    insert_docs(
        write_side.clone(),
        ApiKey(Secret::new("my-write-api-key".to_string())),
        collection_id.clone(),
        vec![
            json!({
                "id": "1",
                "name": "John Doe",
            }),
            json!({
                "id": "2",
                "name": "Jane Doe",
            }),
        ],
    )
    .await?;

    let result = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "Doe",
            })
            .try_into()?,
        )
        .await?;
    assert_eq!(result.count, 2);
    assert_eq!(result.hits.len(), 2);
    assert_eq!(result.hits[0].id, "1".to_string());
    assert_eq!(result.hits[1].id, "2".to_string());
    assert!(result.hits[0].score > 0.0);
    assert!(result.hits[1].score > 0.0);

    let result = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "John Doe",
            })
            .try_into()?,
        )
        .await?;
    assert_eq!(result.count, 2);
    assert_eq!(result.hits.len(), 2);
    assert_eq!(result.hits[0].id, "1".to_string());
    assert_eq!(result.hits[1].id, "2".to_string());
    // Multiple matches boost the score
    assert!(result.hits[0].score > result.hits[1].score);
    assert!(result.hits[0].score > 0.0);
    assert!(result.hits[1].score > 0.0);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_filter_on_unknown_field() -> Result<()> {
    let (write_side, read_side) = create(create_oramacore_config()).await?;

    let collection_id = CollectionId("test-collection".to_string());
    create_collection(write_side.clone(), collection_id.clone()).await?;

    let result = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "Doe",
                "where": {
                    "unknown_field": json!({
                        "eq": 1,
                    }),
                },
            })
            .try_into()?,
        )
        .await;
    assert!(result.is_err());
    assert_eq!(
        format!("{}", result.unwrap_err()),
        "Cannot filter by \"unknown_field\": unknown field".to_string(),
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_filter_field_with_from_filter_type() -> Result<()> {
    let (write_side, read_side) = create(create_oramacore_config()).await?;

    let collection_id = CollectionId("test-collection".to_string());
    create_collection(write_side.clone(), collection_id.clone()).await?;
    insert_docs(
        write_side.clone(),
        ApiKey(Secret::new("my-write-api-key".to_string())),
        collection_id.clone(),
        vec![
            json!({
                "id": "1",
                "name": "John Doe",
            }),
            json!({
                "id": "2",
                "name": "Jane Doe",
            }),
        ],
    )
    .await?;

    let result = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "Doe",
                "where": {
                    "name": json!({
                        "eq": 1,
                    }),
                },
            })
            .try_into()?,
        )
        .await;
    assert!(result.is_err());
    assert_eq!(
        format!("{}", result.unwrap_err()),
        "Filter on field \"name\"(Text(EN)) not supported".to_string(),
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_commit_and_load() -> Result<()> {
    let config = create_oramacore_config();
    let (write_side, read_side) = create(config.clone()).await?;

    let collection_id = CollectionId("test-collection".to_string());
    create_collection(write_side.clone(), collection_id.clone()).await?;
    insert_docs(
        write_side.clone(),
        ApiKey(Secret::new("my-write-api-key".to_string())),
        collection_id.clone(),
        vec![
            json!({
                "id": "1",
                "name": "John Doe",
            }),
            json!({
                "id": "2",
                "name": "Jane Doe",
            }),
        ],
    )
    .await?;

    let before_commit_result = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "Doe",
            })
            .try_into()?,
        )
        .await?;

    let before_commit_collection_lists = write_side
        .list_collections(ApiKey(Secret::new("my-master-api-key".to_string())))
        .await?;

    write_side.commit().await?;
    read_side.commit().await?;

    let after_commit_result = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "Doe",
            })
            .try_into()?,
        )
        .await?;

    assert_eq!(before_commit_result.count, after_commit_result.count);
    assert_eq!(
        before_commit_result.hits.len(),
        after_commit_result.hits.len()
    );
    assert_eq!(
        before_commit_result.hits[0].id,
        after_commit_result.hits[0].id
    );

    let after_commit_collection_lists = write_side
        .list_collections(ApiKey(Secret::new("my-master-api-key".to_string())))
        .await?;

    assert_eq!(
        before_commit_collection_lists,
        after_commit_collection_lists
    );

    let (write_side, read_side) = create(config.clone()).await?;

    let after_load_result = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "Doe",
            })
            .try_into()?,
        )
        .await?;

    assert_eq!(after_commit_result, after_load_result);

    let after_load_collection_lists = write_side
        .list_collections(ApiKey(Secret::new("my-master-api-key".to_string())))
        .await?;

    assert_eq!(after_commit_collection_lists, after_load_collection_lists);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_collection_id_already_exists() -> Result<()> {
    let (write_side, _) = create(create_oramacore_config()).await?;

    let collection_id = CollectionId("test-collection".to_string());
    create_collection(write_side.clone(), collection_id.clone()).await?;

    let output = write_side
        .create_collection(
            ApiKey(Secret::new("my-master-api-key".to_string())),
            json!({
                "id": collection_id.0.clone(),
                "read_api_key": "my-read-api-key",
                "write_api_key": "my-write-api-key",
            })
            .try_into()?,
        )
        .await;

    assert_eq!(
        format!("{}", output.err().unwrap()),
        "Collection \"test-collection\" already exists".to_string()
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_get_collections() -> Result<()> {
    let (write_side, _) = create(create_oramacore_config()).await?;

    let collection_ids: Vec<_> = (0..3)
        .map(|i| format!("my-test-collection-{}", i))
        .collect();
    for id in &collection_ids {
        create_collection(write_side.clone(), CollectionId(id.clone())).await?;
    }

    let collections = write_side
        .list_collections(ApiKey(Secret::new("my-master-api-key".to_string())))
        .await?;

    assert_eq!(collections.len(), 3);

    let ids: HashSet<_> = collections.into_iter().map(|c| c.id.0).collect();
    assert_eq!(ids, collection_ids.into_iter().collect());

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_search_documents_order() -> Result<()> {
    let (write_side, read_side) = create(create_oramacore_config()).await?;

    let collection_id = CollectionId("test-collection".to_string());
    create_collection(write_side.clone(), collection_id.clone()).await?;
    insert_docs(
        write_side.clone(),
        ApiKey(Secret::new("my-write-api-key".to_string())),
        collection_id.clone(),
        vec![
            json!({
                "id": "1",
                "text": "This is a long text with a lot of words",
            }),
            json!({
                "id": "2",
                "text": "This is a smaller text",
            }),
        ],
    )
    .await?;

    let output = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id,
            json!({
                "term": "text",
            })
            .try_into()
            .unwrap(),
        )
        .await?;

    assert_eq!(output.count, 2);
    assert_eq!(output.hits.len(), 2);
    assert_eq!(output.hits[0].id, "2");
    assert_eq!(output.hits[1].id, "1");
    assert!(output.hits[0].score > output.hits[1].score);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_search_documents_limit() -> Result<()> {
    let (write_side, read_side) = create(create_oramacore_config()).await?;

    let collection_id = CollectionId("test-collection".to_string());
    create_collection(write_side.clone(), collection_id.clone()).await?;
    insert_docs(
        write_side.clone(),
        ApiKey(Secret::new("my-write-api-key".to_string())),
        collection_id.clone(),
        (0..100).map(|i| {
            json!({
                "id": i.to_string(),
                "text": "text ".repeat(i + 1),
            })
        }),
    )
    .await?;

    let output = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id,
            json!({
                "term": "text",
                "limit": 10,
            })
            .try_into()
            .unwrap(),
        )
        .await?;

    assert_eq!(output.count, 100);
    assert_eq!(output.hits.len(), 10);
    assert_eq!(output.hits[0].id, "99");
    assert_eq!(output.hits[1].id, "98");
    assert_eq!(output.hits[2].id, "97");
    assert_eq!(output.hits[3].id, "96");
    assert_eq!(output.hits[4].id, "95");

    assert!(output.hits[0].score > output.hits[1].score);
    assert!(output.hits[1].score > output.hits[2].score);
    assert!(output.hits[2].score > output.hits[3].score);
    assert!(output.hits[3].score > output.hits[4].score);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_filter_number() -> Result<()> {
    let (write_side, read_side) = create(create_oramacore_config()).await?;

    let collection_id = CollectionId("test-collection".to_string());
    create_collection(write_side.clone(), collection_id.clone()).await?;
    insert_docs(
        write_side.clone(),
        ApiKey(Secret::new("my-write-api-key".to_string())),
        collection_id.clone(),
        (0..100).map(|i| {
            json!({
                "id": i.to_string(),
                "text": "text ".repeat(i + 1),
                "number": i,
            })
        }),
    )
    .await?;

    // EQ

    let output = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "text",
                "where": {
                    "number": {
                        "eq": 50,
                    },
                }
            })
            .try_into()?,
        )
        .await?;

    assert_eq!(output.count, 1);
    assert_eq!(output.hits.len(), 1);
    assert_eq!(output.hits[0].id, "50");

    // GT

    let output = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "text",
                "where": {
                    "number": {
                        "gt": 2
                    }
                }
            })
            .try_into()
            .unwrap(),
        )
        .await?;

    assert_eq!(output.count, 100 - 3);

    // GTE

    let output = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "text",
                "where": {
                    "number": {
                        "gte": 2
                    }
                }
            })
            .try_into()
            .unwrap(),
        )
        .await?;

    assert_eq!(output.count, 100 - 2);

    // LT

    let output = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "text",
                "where": {
                    "number": {
                        "lt": 2
                    }
                }
            })
            .try_into()
            .unwrap(),
        )
        .await?;

    assert_eq!(output.count, 2);

    // LTE

    let output = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "text",
                "where": {
                    "number": {
                        "lte": 2
                    }
                }
            })
            .try_into()
            .unwrap(),
        )
        .await?;

    assert_eq!(output.count, 3);

    // BETWEEN

    let output = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "text",
                "where": {
                    "number": {
                        "between": [2, 4],
                    }
                }
            })
            .try_into()
            .unwrap(),
        )
        .await?;

    assert_eq!(output.count, 3);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_facets_number() -> Result<()> {
    let (write_side, read_side) = create(create_oramacore_config()).await?;

    let collection_id = CollectionId("test-collection".to_string());
    create_collection(write_side.clone(), collection_id.clone()).await?;
    insert_docs(
        write_side.clone(),
        ApiKey(Secret::new("my-write-api-key".to_string())),
        collection_id.clone(),
        (0..100).map(|i| {
            json!({
                "id": i.to_string(),
                "text": "text ".repeat(i + 1),
                "number": i,
            })
        }),
    )
    .await?;

    let output = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id,
            json!({
                "term": "text",
                "facets": {
                    "number": {
                        "ranges": [
                            {
                                "from": 0,
                                "to": 10,
                            },
                            {
                                "from": 0.5,
                                "to": 10.5,
                            },
                            {
                                "from": -10,
                                "to": 10,
                            },
                            {
                                "from": -10,
                                "to": -1,
                            },
                            {
                                "from": 1,
                                "to": 100,
                            },
                            {
                                "from": 99,
                                "to": 105,
                            },
                            {
                                "from": 102,
                                "to": 105,
                            },
                        ],
                    }
                }
            })
            .try_into()?,
        )
        .await?;

    let facets = output.facets.expect("Facet should be there");
    let number_facet = facets
        .get("number")
        .expect("Facet on field 'number' should be there");

    assert_eq!(number_facet.count, 7);
    assert_eq!(number_facet.values.len(), 7);

    assert_eq!(
        number_facet.values,
        HashMap::from_iter(vec![
            ("-10--1".to_string(), 0),
            ("-10-10".to_string(), 11),
            ("0-10".to_string(), 11),
            ("0.5-10.5".to_string(), 10),
            ("1-100".to_string(), 99),
            ("102-105".to_string(), 0),
            ("99-105".to_string(), 1),
        ])
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_filter_bool() -> Result<()> {
    let (write_side, read_side) = create(create_oramacore_config()).await?;

    let collection_id = CollectionId("test-collection".to_string());
    create_collection(write_side.clone(), collection_id.clone()).await?;
    insert_docs(
        write_side.clone(),
        ApiKey(Secret::new("my-write-api-key".to_string())),
        collection_id.clone(),
        (0..100).map(|i| {
            json!({
                "id": i.to_string(),
                "text": "text ".repeat(i + 1),
                "bool": i % 2 == 0,
            })
        }),
    )
    .await?;

    let output = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "text",
                "where": {
                    "bool": true,
                }
            })
            .try_into()?,
        )
        .await?;

    assert_eq!(output.count, 50);
    assert_eq!(output.hits.len(), 10);
    for hit in output.hits.iter() {
        let id = hit.id.parse::<usize>().unwrap();
        assert_eq!(id % 2, 0);
    }

    let output = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id,
            json!({
                "term": "text",
                "where": {
                    "bool": false,
                }
            })
            .try_into()?,
        )
        .await?;

    assert_eq!(output.count, 50);
    assert_eq!(output.hits.len(), 10);
    for hit in output.hits.iter() {
        let id = hit.id.parse::<usize>().unwrap();
        assert_eq!(id % 2, 1);
    }

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_facets_bool() -> Result<()> {
    let (write_side, read_side) = create(create_oramacore_config()).await?;

    let collection_id = CollectionId("test-collection".to_string());
    create_collection(write_side.clone(), collection_id.clone()).await?;
    insert_docs(
        write_side.clone(),
        ApiKey(Secret::new("my-write-api-key".to_string())),
        collection_id.clone(),
        (0..100).map(|i| {
            json!({
                "id": i.to_string(),
                "text": "text ".repeat(i + 1),
                "bool": i % 2 == 0,
            })
        }),
    )
    .await?;

    let output = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id,
            json!({
                "term": "text",
                "facets": {
                    "bool": {
                        "true": true,
                        "false": true,
                    },
                }
            })
            .try_into()?,
        )
        .await?;

    let facets = output.facets.expect("Facet should be there");
    let bool_facet = facets
        .get("bool")
        .expect("Facet on field 'bool' should be there");

    assert_eq!(bool_facet.count, 2);
    assert_eq!(bool_facet.values.len(), 2);

    assert_eq!(
        bool_facet.values,
        HashMap::from_iter(vec![("true".to_string(), 50), ("false".to_string(), 50),])
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_facets_should_based_on_term() -> Result<()> {
    let (write_side, read_side) = create(create_oramacore_config()).await?;

    let collection_id = CollectionId("test-collection".to_string());
    create_collection(write_side.clone(), collection_id.clone()).await?;
    insert_docs(
        write_side.clone(),
        ApiKey(Secret::new("my-write-api-key".to_string())),
        collection_id.clone(),
        vec![
            json!({
                "id": "1",
                "text": "text",
                "bool": true,
                "number": 1,
            }),
            json!({
                "id": "2",
                "text": "text text",
                "bool": false,
                "number": 2,
            }),
            // This document doens't match the term
            // so it should not be counted in the facets
            json!({
                "id": "3",
                "text": "another",
                "bool": true,
                "number": 1,
            }),
        ],
    )
    .await?;

    let output = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id,
            json!({
                "term": "text",
                "facets": {
                    "bool": {
                        "true": true,
                        "false": true,
                    },
                    "number": {
                        "ranges": [
                            {
                                "from": 0,
                                "to": 10,
                            },
                        ],
                    },
                }
            })
            .try_into()?,
        )
        .await?;

    let facets = output.facets.expect("Facet should be there");
    let bool_facet = facets
        .get("bool")
        .expect("Facet on field 'bool' should be there");

    assert_eq!(bool_facet.count, 2);
    assert_eq!(bool_facet.values.len(), 2);

    assert_eq!(
        bool_facet.values,
        HashMap::from_iter(vec![("true".to_string(), 1), ("false".to_string(), 1),])
    );

    let number_facet = facets
        .get("number")
        .expect("Facet on field 'number' should be there");

    assert_eq!(number_facet.count, 1);
    assert_eq!(number_facet.values.len(), 1);

    assert_eq!(
        number_facet.values,
        HashMap::from_iter(vec![("0-10".to_string(), 2),])
    );

    Ok(())
}

#[ignore]
#[tokio::test(flavor = "multi_thread")]
async fn test_empty_term() -> Result<()> {
    let (write_side, read_side) = create(create_oramacore_config()).await?;

    let collection_id = CollectionId("test-collection".to_string());
    create_collection(write_side.clone(), collection_id.clone()).await?;
    insert_docs(
        write_side.clone(),
        ApiKey(Secret::new("my-write-api-key".to_string())),
        collection_id.clone(),
        vec![
            json!({
                "id": "doc1",
            }),
            json!({
                "id": "doc2",
            }),
            json!({
                "id": "doc3",
            }),
            json!({
                "id": "doc4",
            }),
            json!({
                "id": "doc5",
            }),
        ],
    )
    .await?;

    let output = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "doc",
            })
            .try_into()
            .unwrap(),
        )
        .await?;

    assert_eq!(output.hits.len(), 5);
    assert_eq!(output.count, 5);

    let output = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "",
            })
            .try_into()
            .unwrap(),
        )
        .await?;

    // This assertions fails. If the term is empty, the result is empty.
    // This is not the expected behavior: we should return all documents.
    // So, we need to fix this.
    // NB: the order of the documents is not important in this case,
    //     but we need to ensure that the order is consistent cross runs.
    // TODO: fix the search method to return all documents when the term is empty.
    assert_eq!(output.hits.len(), 5);
    assert_eq!(output.count, 5);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_vector_search_grpc() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();

    let (write_side, read_side) = create(create_oramacore_config()).await?;

    let collection_id = CollectionId("test-collection".to_string());
    write_side
        .create_collection(
            ApiKey(Secret::new("my-master-api-key".to_string())),
            json!({
                "id": collection_id.0.clone(),
                "embeddings": {
                    "model": "BGESmall",
                    "document_fields": ["text"],
                },
                "read_api_key": "my-read-api-key",
                "write_api_key": "my-write-api-key",
            })
            .try_into()?,
        )
        .await?;

    sleep(Duration::from_millis(100)).await;

    let mut docs = vec![
        json!({
            "id": "1",
            "text": "The cat is sleeping on the table.",
        }),
        json!({
            "id": "2",
            "text": "A cat rests peacefully on the sofa.",
        }),
        json!({
            "id": "3",
            "text": "The dog is barking loudly in the yard.",
        }),
    ];
    for i in 4..100 {
        docs.push(json!({
            "id": i.to_string(),
            "text": "foobar",
        }));
    }
    write_side
        .write(
            ApiKey(Secret::new("my-write-api-key".to_string())),
            collection_id.clone(),
            docs.try_into().unwrap(),
        )
        .await?;

    sleep(Duration::from_millis(500)).await;

    let output = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id,
            json!({
                "mode": "vector",
                "term": "The feline is napping comfortably indoors.",
            })
            .try_into()?,
        )
        .await?;

    // Due to the lack of a large enough dataset,
    // the search will not return 2 results as expected.
    // But it should return at least one result.
    // Anyway, the 3th document should not be returned.
    assert_ne!(output.count, 0);
    assert_ne!(output.hits.len(), 0);
    assert!(["1", "2"].contains(&output.hits[0].id.as_str()));

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_handle_bool() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();

    let config = create_oramacore_config();
    let (write_side, read_side) = create(config.clone()).await?;

    let collection_id = CollectionId("test-collection".to_string());
    create_collection(write_side.clone(), collection_id.clone()).await?;
    insert_docs(
        write_side.clone(),
        ApiKey(Secret::new("my-write-api-key".to_string())),
        collection_id.clone(),
        vec![
            json!({
                "id": "doc1",
                "bool": true,
            }),
            json!({
                "id": "doc2",
                "bool": false,
            }),
            json!({
                "id": "doc3",
                "bool": true,
            }),
            json!({
                "id": "doc4",
                "bool": false,
            }),
            json!({
                "id": "doc5",
                "bool": true,
            }),
        ],
    )
    .await?;

    let output = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "doc",
                "where": {
                    "bool": true,
                },
            })
            .try_into()
            .unwrap(),
        )
        .await?;
    assert_eq!(output.count, 3);

    read_side.commit().await?;

    let output = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "doc",
                "where": {
                    "bool": true,
                },
            })
            .try_into()
            .unwrap(),
        )
        .await?;
    assert_eq!(output.count, 3);

    let (_, read_side) = create(config).await?;

    let output = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "doc",
                "where": {
                    "bool": true,
                },
            })
            .try_into()
            .unwrap(),
        )
        .await?;
    assert_eq!(output.count, 3);

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_commit_and_load2() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let config = create_oramacore_config();

    let (write_side, read_side) = create(config.clone()).await?;

    let collection_id = CollectionId("test-collection".to_string());
    write_side
        .create_collection(
            ApiKey(Secret::new("my-master-api-key".to_string())),
            json!({
                "id": collection_id.0.clone(),
                "embeddings": {
                    "model_name": "gte-small",
                    "document_fields": ["name"],
                },
                "read_api_key": "my-read-api-key",
                "write_api_key": "my-write-api-key",
            })
            .try_into()?,
        )
        .await
        .unwrap();

    insert_docs(
        write_side.clone(),
        ApiKey(Secret::new("my-write-api-key".to_string())),
        collection_id.clone(),
        vec![
            json!({
                "id": "1",
                "name": "John Doe",
                "age": 20,
            }),
            json!({
                "id": "2",
                "name": "Jane Doe",
                "age": 21,
            }),
        ],
    )
    .await
    .unwrap();

    let before_commit_result = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "Doe",
                "where": {
                    "age": {
                        "eq": 20,
                    },
                }
            })
            .try_into()?,
        )
        .await
        .unwrap();
    assert_eq!(before_commit_result.count, 1);

    write_side.commit().await?;
    read_side.commit().await?;

    // After the commit, the result should be the same
    let after_commit_result = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "Doe",
                "where": {
                    "age": {
                        "eq": 20,
                    },
                }
            })
            .try_into()?,
        )
        .await
        .unwrap();

    assert_eq!(before_commit_result.count, after_commit_result.count);

    // After the commit, we can insert and search for new documents
    insert_docs(
        write_side.clone(),
        ApiKey(Secret::new("my-write-api-key".to_string())),
        collection_id.clone(),
        vec![json!({
            "id": "3",
            "name": "Foo Doe",
            "age": 20,
        })],
    )
    .await?;
    let result = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "Doe",
                "where": {
                    "age": {
                        "eq": 20,
                    },
                }
            })
            .try_into()?,
        )
        .await
        .unwrap();
    assert_eq!(result.count, 2);

    // We reload the read side
    let (write_side, read_side) = create(config.clone()).await?;
    let after_load_result = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "Doe",
                "where": {
                    "age": {
                        "eq": 20,
                    },
                }
            })
            .try_into()?,
        )
        .await
        .unwrap();

    assert_eq!(before_commit_result.count, after_load_result.count);

    // After the load we can insert and search for new documents
    insert_docs(
        write_side.clone(),
        ApiKey(Secret::new("my-write-api-key".to_string())),
        collection_id.clone(),
        vec![json!({
            "id": "3",
            "name": "Foo Doe",
            "age": 20,
        })],
    )
    .await
    .unwrap();

    let result = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "Doe",
                "where": {
                    "age": {
                        "eq": 20,
                    },
                }
            })
            .try_into()?,
        )
        .await
        .unwrap();
    assert_eq!(result.count, 2);

    write_side.commit().await.unwrap();
    read_side.commit().await.unwrap();

    let (_, read_side) = create(config.clone()).await?;

    let result = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "Doe",
                "where": {
                    "age": {
                        "eq": 20,
                    },
                }
            })
            .try_into()?,
        )
        .await
        .unwrap();
    assert_eq!(result.count, 2);

    let result = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "Doe",
                "mode": "vector",
                "where": {
                    "age": {
                        "eq": 20,
                    },
                }
            })
            .try_into()?,
        )
        .await
        .unwrap();
    // HSNW is a probabilistic algorithm, so the result may vary
    // but it should return at least one result.
    assert!(result.count > 0);

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 10)]
async fn test_read_commit_should_not_block_search() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let mut config = create_oramacore_config();
    config.reader_side.config.insert_batch_commit_size = 1_000_000;
    config.writer_side.config.insert_batch_commit_size = 1_000_000;

    let (write_side, read_side) = create(config.clone()).await?;

    let collection_id = CollectionId("test-collection".to_string());
    write_side
        .create_collection(
            ApiKey(Secret::new("my-master-api-key".to_string())),
            json!({
                "id": collection_id.0.clone(),
                "embeddings": {
                    "model_name": "gte-small",
                    "document_fields": ["name"],
                },
                "read_api_key": "my-read-api-key",
                "write_api_key": "my-write-api-key",
            })
            .try_into()?,
        )
        .await?;

    insert_docs(
        write_side.clone(),
        ApiKey(Secret::new("my-write-api-key".to_string())),
        collection_id.clone(),
        (0..100).map(|i| {
            json!({
                "id": i.to_string(),
                "text": "text ".repeat(i + 1),
            })
        }),
    )
    .await?;

    sleep(Duration::from_secs(1)).await;

    let commit_future = async {
        sleep(Duration::from_millis(5)).await;
        let commit_start = Instant::now();
        read_side.commit().await.unwrap();
        let commit_end = Instant::now();
        (commit_start, commit_end)
    };
    let search_future = async {
        sleep(Duration::from_millis(10)).await;
        let search_start = Instant::now();
        read_side
            .search(
                ApiKey(Secret::new("my-read-api-key".to_string())),
                collection_id.clone(),
                json!({
                    "term": "text",
                })
                .try_into()
                .unwrap(),
            )
            .await
            .unwrap();
        let search_end = Instant::now();
        (search_start, search_end)
    };

    let ((commit_start, commit_end), (search_start, search_end)) =
        tokio::join!(commit_future, search_future,);

    // The commit should start before the search start
    assert!(commit_start < search_start);
    // The commit should end after the search start
    assert!(commit_end > search_start);
    // The commit should end after the search end
    assert!(commit_end > search_end);

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 10)]
async fn test_delete_documents() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let mut config = create_oramacore_config();
    config.reader_side.config.insert_batch_commit_size = 1_000_000;
    config.writer_side.config.insert_batch_commit_size = 1_000_000;

    let (write_side, read_side) = create(config.clone()).await?;

    let collection_id = CollectionId("test-collection".to_string());
    write_side
        .create_collection(
            ApiKey(Secret::new("my-master-api-key".to_string())),
            json!({
                "id": collection_id.0.clone(),
                "embeddings": {
                    "model_name": "gte-small",
                    "document_fields": ["name"],
                },
                "read_api_key": "my-read-api-key",
                "write_api_key": "my-write-api-key",
            })
            .try_into()?,
        )
        .await?;

    let document_count = 10;
    insert_docs(
        write_side.clone(),
        ApiKey(Secret::new("my-write-api-key".to_string())),
        collection_id.clone(),
        (0..document_count).map(|i| {
            json!({
                "id": i.to_string(),
                "text": "text ".repeat(i + 1),
            })
        }),
    )
    .await?;

    let result = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "text",
            })
            .try_into()?,
        )
        .await?;
    assert_eq!(result.count, document_count);

    write_side
        .delete_documents(
            ApiKey(Secret::new("my-write-api-key".to_string())),
            collection_id.clone(),
            vec![(document_count - 1).to_string()],
        )
        .await?;
    sleep(Duration::from_millis(100)).await;
    let result = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "text",
            })
            .try_into()?,
        )
        .await?;
    assert_eq!(result.count, document_count - 1);

    write_side.commit().await.unwrap();
    read_side.commit().await.unwrap();
    let result = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "text",
            })
            .try_into()?,
        )
        .await?;
    assert_eq!(result.count, document_count - 1);

    write_side
        .delete_documents(
            ApiKey(Secret::new("my-write-api-key".to_string())),
            collection_id.clone(),
            vec![(document_count - 2).to_string()],
        )
        .await?;
    sleep(Duration::from_millis(100)).await;

    let result = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "text",
            })
            .try_into()?,
        )
        .await?;
    assert_eq!(result.count, document_count - 2);

    let (write_side, read_side) = create(config.clone()).await?;
    let result = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "text",
            })
            .try_into()?,
        )
        .await?;
    assert_eq!(result.count, document_count - 1);

    write_side
        .delete_documents(
            ApiKey(Secret::new("my-write-api-key".to_string())),
            collection_id.clone(),
            vec![(document_count - 2).to_string()],
        )
        .await?;
    sleep(Duration::from_millis(500)).await;

    let result = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "text",
            })
            .try_into()?,
        )
        .await?;
    assert_eq!(result.count, document_count - 2);

    write_side.commit().await.unwrap();
    read_side.commit().await.unwrap();

    let (_, read_side) = create(config.clone()).await?;
    let result = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "text",
            })
            .try_into()?,
        )
        .await?;
    assert_eq!(result.count, document_count - 2);

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 10)]
async fn test_array_types() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let config = create_oramacore_config();
    let (write_side, read_side) = create(config.clone()).await?;

    let collection_id = CollectionId("test-collection".to_string());
    write_side
        .create_collection(
            ApiKey(Secret::new("my-master-api-key".to_string())),
            json!({
                "id": collection_id.0.clone(),
                "read_api_key": "my-read-api-key",
                "write_api_key": "my-write-api-key",
            })
            .try_into()?,
        )
        .await?;

    let document_count = 10;
    insert_docs(
        write_side.clone(),
        ApiKey(Secret::new("my-write-api-key".to_string())),
        collection_id.clone(),
        (0..document_count).map(|i| {
            json!({
                "id": i.to_string(),
                "text": vec!["text ".repeat(i + 1)],
                "number": vec![i],
                "bool": vec![i % 2 == 0],
            })
        }),
    )
    .await?;
    sleep(Duration::from_millis(500)).await;

    let result = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "text",
            })
            .try_into()?,
        )
        .await?;
    assert_eq!(result.count, document_count);

    let result = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "text",
                "where": {
                    "number": {
                        "eq": 5,
                    },
                }
            })
            .try_into()?,
        )
        .await?;
    assert_eq!(result.count, 1);

    let result = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "text",
                "where": {
                    "bool": true,
                }
            })
            .try_into()?,
        )
        .await?;
    assert_eq!(result.count, 5);

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 10)]
async fn test_document_duplication() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let config = create_oramacore_config();
    let (write_side, read_side) = create(config.clone()).await?;

    let collection_id = CollectionId("test-collection".to_string());
    write_side
        .create_collection(
            ApiKey(Secret::new("my-master-api-key".to_string())),
            json!({
                "id": collection_id.0.clone(),
                "read_api_key": "my-read-api-key",
                "write_api_key": "my-write-api-key",
            })
            .try_into()?,
        )
        .await?;

    insert_docs(
        write_side.clone(),
        ApiKey(Secret::new("my-write-api-key".to_string())),
        collection_id.clone(),
        vec![json!({
            "id": "1",
            "text": "B",
        })],
    )
    .await?;
    insert_docs(
        write_side.clone(),
        ApiKey(Secret::new("my-write-api-key".to_string())),
        collection_id.clone(),
        vec![json!({
            "id": "1",
            "text": "C",
        })],
    )
    .await?;

    let result = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "B",
            })
            .try_into()?,
        )
        .await?;
    assert_eq!(result.count, 1);

    let result = read_side
        .search(
            ApiKey(Secret::new("my-read-api-key".to_string())),
            collection_id.clone(),
            json!({
                "term": "C",
            })
            .try_into()?,
        )
        .await?;
    assert_eq!(result.count, 0);

    Ok(())
}

async fn create_collection(write_side: Arc<WriteSide>, collection_id: CollectionId) -> Result<()> {
    write_side
        .create_collection(
            ApiKey(Secret::new("my-master-api-key".to_string())),
            json!({
                "id": collection_id.0.clone(),
                "read_api_key": "my-read-api-key",
                "write_api_key": "my-write-api-key",
            })
            .try_into()?,
        )
        .await?;
    sleep(Duration::from_millis(100)).await;

    Ok(())
}

async fn insert_docs<I>(
    write_side: Arc<WriteSide>,
    write_api_key: ApiKey,
    collection_id: CollectionId,
    docs: I,
) -> Result<()>
where
    I: IntoIterator<Item = serde_json::Value>,
{
    let document_list: Vec<serde_json::value::Value> = docs.into_iter().collect();
    let document_list: DocumentList = document_list.try_into()?;

    write_side
        .write(write_api_key, collection_id, document_list)
        .await?;

    sleep(Duration::from_millis(1_000)).await;

    Ok(())
}
