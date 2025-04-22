use anyhow::bail;
use futures::FutureExt;
use serde_json::json;

use crate::tests::utils::init_log;
use crate::tests::utils::wait_for;
use crate::tests::utils::TestContext;

#[tokio::test(flavor = "multi_thread")]
async fn test_replace_doc_on_insert() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();

    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .insert_documents(
            json!([
                {
                    "id": "1",
                    "text": "Tommaso",
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
                "term": "Tommaso",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.count, 1);
    assert_eq!(output.hits.len(), 1);

    index_client
        .unchecked_insert_documents(
            json!([
                {
                    "id": "1",
                    "text": "Michele",
                }
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    wait_for(&test_context, |test_context| {
        let collection_client = test_context
            .get_test_collection_client(
                collection_client.collection_id,
                collection_client.write_api_key,
                collection_client.read_api_key,
            )
            .unwrap();

        async move {
            let output = collection_client
                .search(
                    json!({
                        "term": "Tommaso",
                    })
                    .try_into()
                    .unwrap(),
                )
                .await
                .unwrap();
            if output.count == 0 {
                return Ok(());
            }

            bail!("still have Tommaso");
        }
        .boxed()
    })
    .await
    .unwrap();

    let output = collection_client
        .search(
            json!({
                "term": "Michele",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.count, 1);
    assert_eq!(output.hits.len(), 1);

    drop(test_context);
}
