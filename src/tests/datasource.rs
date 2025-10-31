use crate::collection_manager::sides::write::datasource;
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
                datasource: Some(datasource::Fetcher::S3(datasource::s3::S3Fetcher {
                    bucket: bucket_name,
                    region: "us-east-1".to_string(),
                    access_key_id: "test".to_string(),
                    secret_access_key: "test".to_string(),
                    endpoint_url: Some(endpoint_url),
                })),
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

    let base_dir = test_context
        .config
        .writer_side
        .config
        .data_dir
        .join("datasource");
    let db_name = datasource::s3::S3Fetcher::resy_db_filename(
        collection_client.collection_id.clone(),
        index_id.clone(),
    );
    let db_path = base_dir.join(db_name);

    // Wait for the resy file to be created by the sync operation
    wait_for(&test_context, |_| {
        let db_path_clone = db_path.clone();
        async move {
            if db_path_clone.exists() {
                Ok(())
            } else {
                bail!("Resy file not yet created by sync operation")
            }
        }
        .boxed()
    })
    .await
    .unwrap();

    assert!(
        db_path.exists(),
        "Resy file should exist after datasource creation"
    );

    test_context
        .writer
        .delete_index(
            collection_client.write_api_key.clone(),
            collection_client.collection_id.clone(),
            index_id.clone(),
        )
        .await
        .unwrap();

    assert!(
        !db_path.exists(),
        "Resy file should be deleted after datasource removal"
    );
}
