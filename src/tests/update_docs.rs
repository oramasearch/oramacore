
use anyhow::bail;
use serde_json::json;

use crate::{
    tests::utils::{init_log, wait_for, TestContext},
    types::UpdateDocumentRequest,
};

#[tokio::test(flavor = "multi_thread")]
async fn test_update_docs_simple() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .insert_documents(
            json!([
                {
                    "id": "1",
                    "role": "juniordev",
                    "name": "Tommaso",
                },
                {
                    "id": "2",
                    "role": "juniordev",
                    "name": "Michele"
                },
                {
                    "id": "3",
                    "role": "juniordev",
                    "name": "MissingNo"
                }
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let output = collection_client
        .search(
            json!({
                "properties": [ "role" ],
                "term": "juniordev",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.count, 3);

    let req: UpdateDocumentRequest = serde_json::from_value(json!({
        "strategy": "merge",
        "documents": [
            {
                "id": "1",
                "role": "seniordev",
            },
            {
                "id": "2",
                "role": "seniordev",
            }
        ]
    }))
    .unwrap();
    let output = index_client.update_documents(req).await.unwrap();

    assert_eq!(output.inserted, 0);
    assert_eq!(output.updated, 2);
    assert_eq!(output.failed, 0);

    wait_for(&collection_client, |collection_client| {
        Box::pin(async move {
            let output = collection_client
                .search(
                    json!({
                        "properties": [ "role" ],
                        "term": "juniordev",
                    })
                    .try_into()
                    .unwrap(),
                )
                .await?;
            if output.count == 1 {
                Ok(())
            } else {
                bail!("Not updated yet")
            }
        })
    })
    .await
    .unwrap();

    let output = collection_client
        .search(
            json!({
                "properties": [ "role" ],
                "term": "seniordev",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.count, 2);

    drop(test_context);
}
