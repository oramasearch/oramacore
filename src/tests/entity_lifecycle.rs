use serde_json::json;

use crate::tests::utils::init_log;
use crate::tests::utils::TestContext;
use crate::types::DocumentList;

#[tokio::test(flavor = "multi_thread")]
async fn test_entity_lifecycle() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();

    let index_1_client = collection_client.create_index().await.unwrap();
    let stats = collection_client.reader_stats().await.unwrap();
    assert_eq!(stats.indexes_stats.len(), 1);

    let _ = collection_client.create_index().await.unwrap();
    let stats = collection_client.reader_stats().await.unwrap();
    assert_eq!(stats.indexes_stats.len(), 2);

    index_1_client.delete().await.unwrap();

    let err = index_1_client.insert_documents(DocumentList(vec![])).await;
    assert!(err.is_err());

    collection_client.delete().await.unwrap();

    let err = collection_client
        .search(
            json!({
                "term": "",
            })
            .try_into()
            .unwrap(),
        )
        .await;
    assert!(err.is_err());
}
