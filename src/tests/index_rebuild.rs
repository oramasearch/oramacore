use serde_json::json;

use crate::tests::utils::init_log;
use crate::tests::utils::TestContext;

#[tokio::test(flavor = "multi_thread")]
async fn test_swap_index() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();

    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .insert_documents(
            json!([ json!({ "id": "1", "text": "Tommaso" }) ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    let output = collection_client
        .search(
            json!({ "term": "Tommaso" })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.count, 1);

    // let index_client = collection_client.create_index().await.unwrap();


}
