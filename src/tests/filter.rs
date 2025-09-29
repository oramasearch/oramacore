use serde_json::json;

use crate::{
    collection_manager::sides::{read::ReadError, write::index::EnumStrategy},
    tests::utils::{init_log, TestContext},
    types::SearchParams,
};

#[tokio::test(flavor = "multi_thread")]
async fn test_search_on_unknown_field() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();

    collection_client.create_index().await.unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "Doe",
                "where": {
                    "unknown_field": json!({
                        "eq": 1,
                    }),
                },
            })
            .try_into()
            .unwrap(),
        )
        .await;

    let error = result.unwrap_err();

    assert!(matches!(error, ReadError::FilterFieldNotFound(_)));

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_filter_number() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();
    let docs = (0..100)
        .map(|i| {
            json!({
                "id": i.to_string(),
                "text": "text ".repeat(i + 1),
                "number": i,
            })
        })
        .collect::<Vec<_>>();
    index_client
        .insert_documents(json!(docs).try_into().unwrap())
        .await
        .unwrap();

    // EQ

    let output = collection_client
        .search(
            json!({
                "term": "text",
                "where": {
                    "number": {
                        "eq": 50,
                    },
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.count, 1);
    assert_eq!(output.hits.len(), 1);
    assert_eq!(output.hits[0].id, format!("{}:50", index_client.index_id));

    // GT

    let output = collection_client
        .search(
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
        .await
        .unwrap();

    assert_eq!(output.count, 100 - 3);

    // GTE

    let output = collection_client
        .search(
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
        .await
        .unwrap();

    assert_eq!(output.count, 100 - 2);

    // LT

    let output = collection_client
        .search(
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
        .await
        .unwrap();

    assert_eq!(output.count, 2);

    // LTE

    let output = collection_client
        .search(
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
        .await
        .unwrap();

    assert_eq!(output.count, 3);

    // BETWEEN

    let output = collection_client
        .search(
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
        .await
        .unwrap();

    assert_eq!(output.count, 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_filter_bool() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();
    let docs = (0..100)
        .map(|i| {
            json!({
                "id": i.to_string(),
                "text": "text ".repeat(i + 1),
                "bool": i % 2 == 0,
            })
        })
        .collect::<Vec<_>>();
    index_client
        .insert_documents(json!(docs).try_into().unwrap())
        .await
        .unwrap();

    let output = collection_client
        .search(
            json!({
                "term": "text",
                "where": {
                    "bool": true,
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    println!("output: {output:#?}");

    assert_eq!(output.count, 50);
    assert_eq!(output.hits.len(), 10);
    for hit in output.hits.iter() {
        let id =
            serde_json::from_str::<serde_json::Value>(hit.document.as_ref().unwrap().inner.get())
                .unwrap()
                .get("id")
                .unwrap()
                .as_str()
                .unwrap()
                .parse::<usize>()
                .unwrap();
        assert_eq!(id % 2, 0);
    }

    let output = collection_client
        .search(
            json!({
                "term": "text",
                "where": {
                    "bool": false,
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.count, 50);
    assert_eq!(output.hits.len(), 10);
    for hit in output.hits.iter() {
        let id =
            serde_json::from_str::<serde_json::Value>(hit.document.as_ref().unwrap().inner.get())
                .unwrap()
                .get("id")
                .unwrap()
                .as_str()
                .unwrap()
                .parse::<usize>()
                .unwrap();
        assert_eq!(id % 2, 1);
    }

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_filter_string() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let docs = (0..100)
        .map(|i| {
            json!({
                "id": i.to_string(),
                "text": format!("text {}", i % 2 == 0),
            })
        })
        .collect::<Vec<_>>();
    index_client
        .insert_documents(json!(docs).try_into().unwrap())
        .await
        .unwrap();

    let output = collection_client
        .search(
            json!({
                "term": "text",
                "where": {
                    "text": "text true",
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.count, 50);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 10)]
async fn test_array_types() {
    init_log();

    let document_count = 10;

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();
    let docs = (0..document_count)
        .map(|i| {
            json!({
                "id": i.to_string(),
                "text": vec!["text ".repeat(i + 1)],
                "number": vec![i],
                "bool": vec![i % 2 == 0],
            })
        })
        .collect::<Vec<_>>();
    index_client
        .insert_documents(json!(docs).try_into().unwrap())
        .await
        .unwrap();

    let result = collection_client
        .search(
            json!({
                "term": "text",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, document_count);

    let result = collection_client
        .search(
            json!({
                "term": "text",
                "where": {
                    "number": {
                        "eq": 5,
                    },
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, 1);

    let result = collection_client
        .search(
            json!({
                "term": "text",
                "where": {
                    "bool": true,
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.count, 5);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_filter_and_or_not() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();
    let docs = (0..100)
        .map(|i| {
            json!({
                "id": i.to_string(),
                "text": format!("text {}", i % 3 == 0),
                "number": i,
                "bool": i % 2 == 0,
            })
        })
        .collect::<Vec<_>>();
    index_client
        .insert_documents(json!(docs).try_into().unwrap())
        .await
        .unwrap();

    let output = collection_client
        .search(
            json!({
                "term": "text",
                "where": {
                    "and": [
                        { "bool": true },
                        { "number": { "gte": 50 } },
                    ]
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.count, 25);

    let output = collection_client
        .search(
            json!({
                "term": "text",
                "where": {
                    "or": [
                        { "bool": true },
                        { "number": { "gte": 50 } },
                    ]
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.count, 75);

    let output = collection_client
        .search(
            json!({
                "term": "text",
                "where": {
                    "not": {
                        "bool": true,
                    }
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.count, 50);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_date() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    let result = index_client
        .insert_documents(
            json!([
                {
                    "id": "1",
                    "text": "test",
                    "date": "2023-01-01T00:00:00Z"
                },
                {
                    "id": "2",
                    "text": "test",
                    "date": "2023-01-02T00:00:00Z"
                },
                {
                    "id": "3",
                    "text": "test",
                    "date": "2023-01-03T00:00:00Z"
                }
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(result.inserted, 3);

    let search_params: SearchParams = json!({
        "term": "",
        "where": {
            "date": {
                "gt": "2023-01-01T00:00:00Z"
            }
        }
    })
    .try_into()
    .unwrap();
    let output = collection_client.search(search_params).await.unwrap();
    assert_eq!(output.count, 2);

    let search_params: SearchParams = json!({
        "term": "",
        "where": {
            "date": {
                "lt": "2023-01-03T00:00:00Z"
            }
        }
    })
    .try_into()
    .unwrap();
    let output = collection_client.search(search_params).await.unwrap();
    assert_eq!(output.count, 2);

    let search_params: SearchParams = json!({
        "term": "",
        "where": {
            "date": {
                "gte": "2023-01-01T00:00:00Z"
            }
        }
    })
    .try_into()
    .unwrap();
    let output = collection_client.search(search_params).await.unwrap();
    assert_eq!(output.count, 3);

    let search_params: SearchParams = json!({
        "term": "",
        "where": {
            "date": {
                "lte": "2023-01-03T00:00:00Z"
            }
        }
    })
    .try_into()
    .unwrap();
    let output = collection_client.search(search_params).await.unwrap();
    assert_eq!(output.count, 3);

    let search_params: SearchParams = json!({
        "term": "",
        "where": {
            "date": {
                "between": ["2023-01-01T00:00:01Z", "2023-01-02T23:59:59Z"]
            }
        }
    })
    .try_into()
    .unwrap();
    let output = collection_client.search(search_params).await.unwrap();
    assert_eq!(output.count, 1);

    test_context.commit_all().await.unwrap();

    let search_params: SearchParams = json!({
        "term": "",
        "where": {
            "date": {
                "gt": "2023-01-01T00:00:00Z"
            }
        }
    })
    .try_into()
    .unwrap();
    let output = collection_client.search(search_params).await.unwrap();
    assert_eq!(output.count, 2);

    let search_params: SearchParams = json!({
        "term": "",
        "where": {
            "date": {
                "lt": "2023-01-03T00:00:00Z"
            }
        }
    })
    .try_into()
    .unwrap();
    let output = collection_client.search(search_params).await.unwrap();
    assert_eq!(output.count, 2);

    let search_params: SearchParams = json!({
        "term": "",
        "where": {
            "date": {
                "gte": "2023-01-01T00:00:00Z"
            }
        }
    })
    .try_into()
    .unwrap();
    let output = collection_client.search(search_params).await.unwrap();
    assert_eq!(output.count, 3);

    let search_params: SearchParams = json!({
        "term": "",
        "where": {
            "date": {
                "lte": "2023-01-03T00:00:00Z"
            }
        }
    })
    .try_into()
    .unwrap();
    let output = collection_client.search(search_params).await.unwrap();
    assert_eq!(output.count, 3);

    let search_params: SearchParams = json!({
        "term": "",
        "where": {
            "date": {
                "between": ["2023-01-01T00:00:01Z", "2023-01-02T23:59:59Z"]
            }
        }
    })
    .try_into()
    .unwrap();
    let output = collection_client.search(search_params).await.unwrap();
    assert_eq!(output.count, 1);

    let test_context = test_context.reload().await;
    let collection_client = test_context
        .get_test_collection_client(
            collection_client.collection_id,
            collection_client.write_api_key,
            collection_client.read_api_key,
        )
        .unwrap();

    let search_params: SearchParams = json!({
        "term": "",
        "where": {
            "date": {
                "gt": "2023-01-01T00:00:00Z"
            }
        }
    })
    .try_into()
    .unwrap();
    let output = collection_client.search(search_params).await.unwrap();
    assert_eq!(output.count, 2);

    let search_params: SearchParams = json!({
        "term": "",
        "where": {
            "date": {
                "lt": "2023-01-03T00:00:00Z"
            }
        }
    })
    .try_into()
    .unwrap();
    let output = collection_client.search(search_params).await.unwrap();
    assert_eq!(output.count, 2);

    let search_params: SearchParams = json!({
        "term": "",
        "where": {
            "date": {
                "gte": "2023-01-01T00:00:00Z"
            }
        }
    })
    .try_into()
    .unwrap();
    let output = collection_client.search(search_params).await.unwrap();
    assert_eq!(output.count, 3);

    let search_params: SearchParams = json!({
        "term": "",
        "where": {
            "date": {
                "lte": "2023-01-03T00:00:00Z"
            }
        }
    })
    .try_into()
    .unwrap();
    let output = collection_client.search(search_params).await.unwrap();
    assert_eq!(output.count, 3);

    let search_params: SearchParams = json!({
        "term": "",
        "where": {
            "date": {
                "between": ["2023-01-01T00:00:01Z", "2023-01-02T23:59:59Z"]
            }
        }
    })
    .try_into()
    .unwrap();
    let output = collection_client.search(search_params).await.unwrap();
    assert_eq!(output.count, 1);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_enum_strategy() {
    init_log();

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client
        .create_index_with_explicit_type_strategy(crate::types::TypeParsingStrategies {
            enum_strategy: EnumStrategy::Explicit,
        })
        .await
        .unwrap();
    let docs = (0..100)
        .map(|i| {
            json!({
                "id": i.to_string(),
                "text": "text ".repeat(i + 1),
                "enum": if i % 2 == 0 { "enum('even')" } else { "enum(`odd`)" },
            })
        })
        .collect::<Vec<_>>();
    index_client
        .insert_documents(json!(docs).try_into().unwrap())
        .await
        .unwrap();

    let output = collection_client
        .search(
            json!({
                "term": "text",
                "where": {
                    "enum": "even"
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.count, 50);

    let stats = collection_client.reader_stats().await.unwrap();

    assert_eq!(
        stats.indexes_stats[0].type_parsing_strategies.enum_strategy,
        EnumStrategy::Explicit
    );

    test_context.commit_all().await.unwrap();

    let stats = collection_client.reader_stats().await.unwrap();

    assert_eq!(
        stats.indexes_stats[0].type_parsing_strategies.enum_strategy,
        EnumStrategy::Explicit
    );

    let new_test_context = test_context.reload().await;

    let new_collection = new_test_context
        .get_test_collection_client(
            collection_client.collection_id,
            collection_client.write_api_key,
            collection_client.read_api_key,
        )
        .unwrap();

    let output = new_collection
        .search(
            json!({
                "term": "text",
                "where": {
                    "enum": "even"
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.count, 50);

    let stats = new_collection.reader_stats().await.unwrap();

    assert_eq!(
        stats.indexes_stats[0].type_parsing_strategies.enum_strategy,
        EnumStrategy::Explicit
    );

    drop(new_test_context);
}
