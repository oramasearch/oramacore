#[path = "utils/start.rs"]
mod start;

use anyhow::Result;
use rustorama::collection_manager::{sides::read::CollectionsReader, CollectionId};
use serde_json::json;
use start::start_all;
use std::{collections::HashSet, sync::Arc, time::Duration};
use tokio::{sync::OnceCell, time::sleep};

static INSTANCE: OnceCell<(Arc<CollectionsReader>, CollectionId)> = OnceCell::const_new();

async fn init() -> (Arc<CollectionsReader>, CollectionId) {
    let (writer, reader, _) = start_all().await.unwrap();

    let collection_id: CollectionId = writer
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
                    "bool": true
                },
                {
                    "id": "doc2",
                    "bool": true
                },
                {
                    "id": "doc3",
                    "bool": false
                },
                {
                    "id": "doc4",
                    "bool": true
                },
                {
                    "id": "doc5",
                    "bool": false
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
async fn test_bools() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();
    let (reader, collection_id) = INSTANCE.get_or_init(init).await;

    let collection = reader.get_collection(collection_id.clone()).await.unwrap();
    let output = collection
        .search(
            json!({
                "term": "doc",
                "where": {
                    "bool": true
                }
            })
            .try_into()
            .unwrap(),
        )
        .await?;

    assert_eq!(output.hits.len(), 3);
    assert_eq!(output.count, 3);

    let actual_doc_ids: HashSet<_> = output.hits.into_iter().map(|h| h.document.unwrap().get("id").unwrap().as_str().unwrap().to_string())
        .collect();

    assert_eq!(actual_doc_ids, HashSet::from_iter(["doc1".to_string(), "doc2".to_string(), "doc4".to_string()]));

    Ok(())
}
