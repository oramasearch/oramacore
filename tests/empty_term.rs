#[path = "utils/start.rs"]
mod start;

use anyhow::Result;
use serde_json::json;
use start::start_all;
use std::time::Duration;
use tokio::time::sleep;

#[ignore]
#[tokio::test(flavor = "multi_thread")]
async fn test_empty_term() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();

    let (writer, reader, _) = start_all().await?;

    let collection_id = writer
        .create_collection(
            json!({
                "id": "test",
            })
            .try_into()
            .unwrap(),
        )
        .await?;

    writer
        .write(
            collection_id.clone(),
            json!([
                {
                    "id": "doc1",
                },
                {
                    "id": "doc2",
                },
                {
                    "id": "doc3",
                },
                {
                    "id": "doc4",
                },
                {
                    "id": "doc5",
                },
            ])
            .try_into()
            .unwrap(),
        )
        .await?;

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

    let collection = reader.get_collection(collection_id.clone()).await.unwrap();
    let output = collection
        .search(
            json!({
                "term": "doc",
            })
            .try_into()
            .unwrap(),
        )
        .await?;

    assert_eq!(output.hits.len(), 5);
    assert_eq!(output.count, 5);

    let collection = reader.get_collection(collection_id.clone()).await.unwrap();
    let output = collection
        .search(
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
