#[path = "utils/start.rs"]
mod start;

use anyhow::Result;
use rustorama::{collection_manager::sides::read::CollectionsReader, types::CollectionId};
use serde_json::json;
use start::start_all;
use std::{sync::Arc, time::Duration};
use tokio::{sync::OnceCell, time::sleep};

static INSTANCE: OnceCell<(Arc<CollectionsReader>, CollectionId)> = OnceCell::const_new();

async fn init() -> (Arc<CollectionsReader>, CollectionId) {
    let (writer, reader, _) = start_all().await.unwrap();

    let collection_id = writer
        .create_collection(
            json!({
                "id": "test",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    writer
        .write(
            collection_id.clone(),
            json!([
                {
                    "id": "doc1",
                    "number": 1
                },
                {
                    "id": "doc2",
                    "number": 2
                },
                {
                    "id": "doc3",
                    "number": 3
                },
                {
                    "id": "doc4",
                    "number": 4
                },
                {
                    "id": "doc5",
                    "number": 5
                },
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    loop {
        sleep(Duration::from_millis(100)).await;
        let collection = match reader.get_collection(collection_id.clone()).await {
            Some(collection) => collection,
            None => continue,
        };
        let total = match collection.get_total_documents().await {
            Ok(total) => total,
            Err(_) => continue,
        };
        if total == 5 {
            break;
        }
    }

    (reader, collection_id)
}

#[tokio::test]
async fn test_number_filters_gt() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let (reader, collection_id) = INSTANCE.get_or_init(init).await;

    let collection = reader.get_collection(collection_id.clone()).await.unwrap();
    let output = collection
        .search(
            json!({
                "term": "doc",
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

    assert_eq!(output.hits.len(), 3);
    assert_eq!(output.count, 3);

    Ok(())
}

#[tokio::test]
async fn test_number_filters_gte() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let (reader, collection_id) = INSTANCE.get_or_init(init).await;

    let collection = reader.get_collection(collection_id.clone()).await.unwrap();
    let output = collection
        .search(
            json!({
                "term": "doc",
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

    assert_eq!(output.hits.len(), 4);
    assert_eq!(output.count, 4);

    Ok(())
}

#[tokio::test]
async fn test_number_filters_lt() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let (reader, collection_id) = INSTANCE.get_or_init(init).await;

    let collection = reader.get_collection(collection_id.clone()).await.unwrap();
    let output = collection
        .search(
            json!({
                "term": "doc",
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

    assert_eq!(output.hits.len(), 1);
    assert_eq!(output.count, 1);

    Ok(())
}

#[tokio::test]
async fn test_number_filters_lte() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let (reader, collection_id) = INSTANCE.get_or_init(init).await;

    let collection = reader.get_collection(collection_id.clone()).await.unwrap();
    let output = collection
        .search(
            json!({
                "term": "doc",
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

    assert_eq!(output.hits.len(), 2);
    assert_eq!(output.count, 2);

    Ok(())
}

#[tokio::test]
async fn test_number_filters_eq() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let (reader, collection_id) = INSTANCE.get_or_init(init).await;

    let collection = reader.get_collection(collection_id.clone()).await.unwrap();
    let output = collection
        .search(
            json!({
                "term": "doc",
                "where": {
                    "number": {
                        "eq": 2
                    }
                }
            })
            .try_into()
            .unwrap(),
        )
        .await?;

    assert_eq!(output.hits.len(), 1);
    assert_eq!(output.count, 1);

    Ok(())
}

#[tokio::test]
async fn test_number_filters_between() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let (reader, collection_id) = INSTANCE.get_or_init(init).await;

    let collection = reader.get_collection(collection_id.clone()).await.unwrap();
    let output = collection
        .search(
            json!({
                "term": "doc",
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

    assert_eq!(output.hits.len(), 3);
    assert_eq!(output.count, 3);

    Ok(())
}
