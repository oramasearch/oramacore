use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Duration,
};

use anyhow::Result;
use serde_json::json;
use tokio::time::sleep;

use crate::{
    build_orama,
    collection_manager::sides::{CollectionsWriterConfig, IndexesConfig, ReadSide, WriteSide},
    connect_write_and_read_side,
    embeddings::{
        fe::{FastEmbedModelRepoConfig, FastEmbedRepoConfig},
        EmbeddingConfig, ModelConfig,
    },
    test_utils::generate_new_path,
    types::{CollectionId, DocumentList},
    web_server::HttpConfig,
    OramacoreConfig, ReadSideConfig, SideChannelType, WriteSideConfig,
};

fn create_oramacore_config() -> OramacoreConfig {
    OramacoreConfig {
        http: HttpConfig {
            host: "127.0.0.1".parse().unwrap(),
            port: 2222,
            allow_cors: false,
            with_prometheus: false,
        },
        embeddings: EmbeddingConfig {
            preload: vec![],
            hugging_face: None,
            fastembed: None,
            models: HashMap::new(),
        },
        writer_side: WriteSideConfig {
            output: SideChannelType::InMemory,
            config: CollectionsWriterConfig {
                data_dir: generate_new_path(),
                embedding_queue_limit: 50,
            },
        },
        reader_side: ReadSideConfig {
            input: SideChannelType::InMemory,
            config: IndexesConfig {
                data_dir: generate_new_path(),
            },
        },
    }
}

async fn create(config: OramacoreConfig) -> Result<(Arc<WriteSide>, Arc<ReadSide>)> {
    let (write_side, read_side, rec) = build_orama(config).await?;

    connect_write_and_read_side(rec, read_side.clone().unwrap());

    let write_side = write_side.unwrap();
    let read_side = read_side.unwrap();

    Ok((write_side, read_side))
}

#[tokio::test]
async fn test_simple_text_search() -> Result<()> {
    let (write_side, read_side) = create(create_oramacore_config()).await?;

    let collection_id = CollectionId("test-collection".to_string());
    create_collection(write_side.clone(), collection_id.clone()).await?;

    insert_docs(
        write_side.clone(),
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

#[tokio::test]
async fn test_filter_on_unknown_field() -> Result<()> {
    let (write_side, read_side) = create(create_oramacore_config()).await?;

    let collection_id = CollectionId("test-collection".to_string());
    create_collection(write_side.clone(), collection_id.clone()).await?;

    let result = read_side
        .search(
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

#[tokio::test]
async fn test_filter_field_with_from_filter_type() -> Result<()> {
    let (write_side, read_side) = create(create_oramacore_config()).await?;

    let collection_id = CollectionId("test-collection".to_string());
    create_collection(write_side.clone(), collection_id.clone()).await?;
    insert_docs(
        write_side.clone(),
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
        "Filter on field \"name\"(Text(English)) not supported".to_string(),
    );

    Ok(())
}

#[tokio::test]
async fn test_commit_and_load() -> Result<()> {
    let config = create_oramacore_config();
    let (write_side, read_side) = create(config.clone()).await?;

    let collection_id = CollectionId("test-collection".to_string());
    create_collection(write_side.clone(), collection_id.clone()).await?;
    insert_docs(
        write_side.clone(),
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
            collection_id.clone(),
            json!({
                "term": "Doe",
            })
            .try_into()?,
        )
        .await?;

    let before_commit_collection_lists = write_side.list_collections().await;

    write_side.commit().await?;
    read_side.commit().await?;

    let after_commit_result = read_side
        .search(
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

    let after_commit_collection_lists = write_side.list_collections().await;

    assert_eq!(
        before_commit_collection_lists,
        after_commit_collection_lists
    );

    let (write_side, read_side) = create(config.clone()).await?;

    let after_load_result = read_side
        .search(
            collection_id.clone(),
            json!({
                "term": "Doe",
            })
            .try_into()?,
        )
        .await?;

    assert_eq!(after_commit_result, after_load_result);

    let after_load_collection_lists = write_side.list_collections().await;

    assert_eq!(after_commit_collection_lists, after_load_collection_lists);

    Ok(())
}

#[tokio::test]
async fn test_collection_id_already_exists() -> Result<()> {
    let (write_side, _) = create(create_oramacore_config()).await?;

    let collection_id = CollectionId("test-collection".to_string());
    create_collection(write_side.clone(), collection_id.clone()).await?;

    let output = write_side
        .create_collection(
            json!({
                "id": collection_id.0.clone(),
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

#[tokio::test]
async fn test_get_collections() -> Result<()> {
    let (write_side, _) = create(create_oramacore_config()).await?;

    let collection_ids: Vec<_> = (0..3)
        .map(|i| format!("my-test-collection-{}", i))
        .collect();
    for id in &collection_ids {
        create_collection(write_side.clone(), CollectionId(id.clone())).await?;
    }

    let collections = write_side.list_collections().await;

    assert_eq!(collections.len(), 3);

    let ids: HashSet<_> = collections.into_iter().map(|c| c.id.0).collect();
    assert_eq!(ids, collection_ids.into_iter().collect());

    Ok(())
}

#[tokio::test]
async fn test_search_documents_order() -> Result<()> {
    let (write_side, read_side) = create(create_oramacore_config()).await?;

    let collection_id = CollectionId("test-collection".to_string());
    create_collection(write_side.clone(), collection_id.clone()).await?;
    insert_docs(
        write_side.clone(),
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

#[tokio::test]
async fn test_search_documents_limit() -> Result<()> {
    let (write_side, read_side) = create(create_oramacore_config()).await?;

    let collection_id = CollectionId("test-collection".to_string());
    create_collection(write_side.clone(), collection_id.clone()).await?;
    insert_docs(
        write_side.clone(),
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

#[tokio::test]
async fn test_filter_number() -> Result<()> {
    let (write_side, read_side) = create(create_oramacore_config()).await?;

    let collection_id = CollectionId("test-collection".to_string());
    create_collection(write_side.clone(), collection_id.clone()).await?;
    insert_docs(
        write_side.clone(),
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

#[tokio::test]
async fn test_facets_number() -> Result<()> {
    let (write_side, read_side) = create(create_oramacore_config()).await?;

    let collection_id = CollectionId("test-collection".to_string());
    create_collection(write_side.clone(), collection_id.clone()).await?;
    insert_docs(
        write_side.clone(),
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

#[tokio::test]
async fn test_filter_bool() -> Result<()> {
    let (write_side, read_side) = create(create_oramacore_config()).await?;

    let collection_id = CollectionId("test-collection".to_string());
    create_collection(write_side.clone(), collection_id.clone()).await?;
    insert_docs(
        write_side.clone(),
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

#[tokio::test]
async fn test_facets_bool() -> Result<()> {
    let (write_side, read_side) = create(create_oramacore_config()).await?;

    let collection_id = CollectionId("test-collection".to_string());
    create_collection(write_side.clone(), collection_id.clone()).await?;
    insert_docs(
        write_side.clone(),
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

#[tokio::test]
async fn test_facets_should_based_on_term() -> Result<()> {
    let (write_side, read_side) = create(create_oramacore_config()).await?;

    let collection_id = CollectionId("test-collection".to_string());
    create_collection(write_side.clone(), collection_id.clone()).await?;
    insert_docs(
        write_side.clone(),
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

#[tokio::test]
async fn test_vector_search() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();

    let mut config = create_oramacore_config();
    config.embeddings.preload = vec!["gte-small".to_string()];
    config.embeddings.fastembed = Some(FastEmbedRepoConfig {
        cache_dir: std::env::temp_dir(),
    });
    config.embeddings.models.insert(
        "gte-small".to_string(),
        ModelConfig::Fastembed(FastEmbedModelRepoConfig {
            real_model_name: "Xenova/bge-small-en-v1.5".to_string(),
            dimensions: 384,
        }),
    );
    let (write_side, read_side) = create(config).await?;

    let collection_id = CollectionId("test-collection".to_string());
    write_side
        .create_collection(
            json!({
                "id": collection_id.0.clone(),
                "typed_fields": {
                    "vector": {
                        "mode": "embedding",
                        "model_name": "gte-small",
                        "document_fields": ["text"],
                    }
                }
            })
            .try_into()?,
        )
        .await?;

    sleep(Duration::from_millis(100)).await;

    insert_docs(
        write_side.clone(),
        collection_id.clone(),
        vec![
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
        ],
    )
    .await?;

    let output = read_side
        .search(
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

#[ignore]
#[tokio::test(flavor = "multi_thread")]
async fn test_empty_term() -> Result<()> {
    let (write_side, read_side) = create(create_oramacore_config()).await?;

    let collection_id = CollectionId("test-collection".to_string());
    create_collection(write_side.clone(), collection_id.clone()).await?;
    insert_docs(
        write_side.clone(),
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

#[cfg(feature = "test-python")]
#[tokio::test]
async fn test_vector_search_grpc() -> Result<()> {
    use crate::ai::grpc::GrpcModelConfig;

    let _ = tracing_subscriber::fmt::try_init();

    let mut config = create_oramacore_config();
    config.embeddings.models.insert(
        "BGESmall".to_string(),
        ModelConfig::Grpc(GrpcModelConfig {
            dimensions: 384,
            real_model_name: "BGESmall".to_string(),
        }),
    );

    let (write_side, read_side) = create(config).await?;

    let collection_id = CollectionId("test-collection".to_string());
    write_side
        .create_collection(
            json!({
                "id": collection_id.0.clone(),
                "typed_fields": {
                    "vector": {
                        "mode": "embedding",
                        "model_name": "BGESmall",
                        "document_fields": ["text"],
                    }
                }
            })
            .try_into()?,
        )
        .await?;

    sleep(Duration::from_millis(100)).await;

    write_side
        .write(
            collection_id.clone(),
            vec![
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
            ]
            .try_into()
            .unwrap(),
        )
        .await?;

    sleep(Duration::from_millis(500)).await;

    let output = read_side
        .search(
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

#[tokio::test]
async fn test_commit_and_load2() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();

    let _ = std::fs::remove_dir_all("/Users/allevo/repos/rustorama/.data");

    let mut config = create_oramacore_config();
    config.embeddings.preload = vec!["gte-small".to_string()];
    config.embeddings.fastembed = Some(FastEmbedRepoConfig {
        cache_dir: std::env::temp_dir(),
    });
    config.embeddings.models.insert(
        "gte-small".to_string(),
        ModelConfig::Fastembed(FastEmbedModelRepoConfig {
            real_model_name: "Xenova/bge-small-en-v1.5".to_string(),
            dimensions: 384,
        }),
    );
    config.reader_side.config.data_dir = "/Users/allevo/repos/rustorama/.data/read".into();
    config.writer_side.config.data_dir = "/Users/allevo/repos/rustorama/.data/write".into();
    let (write_side, read_side) = create(config.clone()).await?;

    let collection_id = CollectionId("test-collection".to_string());
    write_side
        .create_collection(
            json!({
                "id": collection_id.0.clone(),
                "typed_fields": {
                    "vector": {
                        "mode": "embedding",
                        "model_name": "gte-small",
                        "document_fields": ["name"],
                    }
                }
            })
            .try_into()?,
        )
        .await?;

    insert_docs(
        write_side.clone(),
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
    .await?;

    let before_commit_result = read_side
        .search(
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
        .await?;
    assert_eq!(before_commit_result.count, 1);

    write_side.commit().await?;
    read_side.commit().await?;

    // After the commit, the result should be the same
    let after_commit_result = read_side
        .search(
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
        .await?;

    assert_eq!(before_commit_result.count, after_commit_result.count);

    // After the commit, we can insert and search for new documents
    insert_docs(
        write_side.clone(),
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
        .await?;
    assert_eq!(result.count, 2);

    // We reload the read side
    let (write_side, read_side) = create(config.clone()).await?;
    let after_load_result = read_side
        .search(
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
        .await?;

    assert_eq!(before_commit_result.count, after_load_result.count);

    // After the load we can insert and search for new documents
    insert_docs(
        write_side.clone(),
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
        .await?;
    assert_eq!(result.count, 2);

    // write_side.commit().await?;
    read_side.commit().await?;

    let (_, read_side) = create(config.clone()).await?;
    let result = read_side
        .search(
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
        .await?;
    assert_eq!(result.count, 2);

    let result = read_side
        .search(
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
        .await?;
    assert_eq!(result.count, 1);

    Ok(())
}

async fn create_collection(write_side: Arc<WriteSide>, collection_id: CollectionId) -> Result<()> {
    write_side
        .create_collection(
            json!({
                "id": collection_id.0.clone(),
            })
            .try_into()?,
        )
        .await?;
    sleep(Duration::from_millis(100)).await;

    Ok(())
}

async fn insert_docs<I>(
    write_side: Arc<WriteSide>,
    collection_id: CollectionId,
    docs: I,
) -> Result<()>
where
    I: IntoIterator<Item = serde_json::Value>,
{
    let document_list: Vec<serde_json::value::Value> = docs.into_iter().collect();
    let document_list: DocumentList = document_list.try_into()?;

    write_side.write(collection_id, document_list).await?;

    sleep(Duration::from_millis(100)).await;

    Ok(())
}
