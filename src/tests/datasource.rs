use crate::collection_manager::sides::write::datasource::storage;
use crate::tests::utils::{setup_s3_container, wait_for, TestContext};
use crate::types::{CreateIndexRequest, IndexId};
use anyhow::bail;
use aws_sdk_s3::primitives::ByteStream;
use futures::FutureExt;
use serde_json::json;

#[tokio::test(flavor = "multi_thread")]
async fn test_sync_with_datasource() {
    let (s3_client, bucket_name, _container, endpoint_url) = setup_s3_container().await;

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();

    let index_id = IndexId::try_new("test-s3-index".to_string()).unwrap();
    let datasource_id = IndexId::try_new("s3-ds".to_string()).unwrap();

    let key1 = "test-file-1.json";
    let content1 = json!({ "text": "Hello, Oramacore! (file 1)"});
    let content_bytes1 = serde_json::to_vec(&content1).unwrap();

    s3_client
        .put_object()
        .bucket(&bucket_name)
        .key(key1)
        .body(ByteStream::from(content_bytes1.clone()))
        .send()
        .await
        .unwrap();

    test_context
        .writer
        .create_index(
            collection_client.write_api_key,
            collection_client.collection_id,
            CreateIndexRequest {
                index_id,
                embedding: None,
                type_strategy: Default::default(),
                datasource: Some(storage::DatasourceEntry {
                    id: datasource_id,
                    datasource: storage::DatasourceKind::S3(storage::S3 {
                        bucket: bucket_name,
                        region: "us-east-1".to_string(),
                        access_key_id: "test".to_string(),
                        secret_access_key: "test".to_string(),
                        endpoint_url: Some(endpoint_url),
                    }),
                }),
            },
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
                .search(json!({ "term": "hello", }).try_into().unwrap())
                .await
                .unwrap();
            if output.count == 1 {
                return Ok(());
            }

            bail!("Document not yet inserted")
        }
        .boxed()
    })
    .await
    .unwrap();
}
