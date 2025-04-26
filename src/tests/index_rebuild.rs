use std::time::Duration;

use anyhow::bail;
use futures::FutureExt;
use serde_json::json;
use tokio::time::sleep;

use crate::tests::utils::init_log;
use crate::tests::utils::wait_for;
use crate::tests::utils::TestContext;
use crate::types::LanguageDTO;

#[tokio::test(flavor = "multi_thread")]
async fn test_change_language() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();

    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .insert_documents(
            json!([ json!({ "id": "1", "text": "avvocata" }) ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let output = collection_client
        .search(
            json!({ "term": "avvocato" })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.count, 0);

    collection_client.rebuild_index(
        LanguageDTO::Italian,
    ).await.unwrap();

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
                    json!({ "term": "avvocato" })
                    .try_into()
                    .unwrap(),
                )
                .await
                .unwrap();
            if output.count == 1 {
                return Ok(());
            }
            bail!("Index not rebuilt yet")
        }.boxed()
    }).await.unwrap();


}
