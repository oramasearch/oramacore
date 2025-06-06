use std::time::Duration;

use serde_json::json;
use tokio::time::sleep;

use crate::{
    tests::utils::{create_oramacore_config, init_log, TestContext}, types::{Document, DocumentList}
};

#[tokio::test(flavor = "multi_thread")]
async fn test_bug_1() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();
    let coll_id = collection_client.collection_id;
    let write_api_key = collection_client.write_api_key;
    let read_api_key = collection_client.read_api_key;
    let index_client = collection_client.create_index().await.unwrap();

    let docs = r#" [
        {"id":"4084278","ec_order_id":"3235693"}
    ]"#;
    let docs = serde_json::from_str::<Vec<Document>>(docs).unwrap();

    index_client
        .insert_documents(DocumentList(docs))
        .await
        .unwrap();

    test_context.commit_all().await.unwrap();
    let test_context = test_context.reload().await;

    let collection_client = test_context
        .get_test_collection_client(coll_id, write_api_key, read_api_key)
        .unwrap();

    let output = collection_client
        .search(
            json!({
                "term": "3235693",
                "properties": ["ec_order_id"],
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.count, 1);

    sleep(Duration::from_secs(1)).await;

    let output = collection_client
        .search(
            json!({
                "term": "",
                "where": {
                    "ec_order_id": "3235693",
                },
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.count, 1);

    drop(test_context);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_bug_2() {
    init_log();

    let mut config = create_oramacore_config();
    config.reader_side.config.insert_batch_commit_size = 1;
    let test_context = TestContext::new_with_config(config).await;

    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    index_client
        .insert_documents(
            json!([
                {"number": 55},
                {"number": 42},
            ])
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    sleep(Duration::from_secs(1)).await;

    drop(test_context);
}
