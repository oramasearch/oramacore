use crate::collection_manager::sides::write::datasource::storage;
use crate::tests::utils::{setup_s3_container, wait_for, TestContext};
use crate::types::{CollectionStatsRequest, CreateIndexRequest, IndexId};
use anyhow::bail;
use aws_sdk_s3::primitives::ByteStream;
use serde_json::json;

#[tokio::test]
async fn test_sync_with_datasource() {
    let (s3_client, bucket_name, _container, endpoint_url) = setup_s3_container().await;

    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();

    let index_id = IndexId::try_new("test-s3-index".to_string()).unwrap();
    let datasource_id = IndexId::try_new("s3-ds".to_string()).unwrap();

    let key1 = "test-file-1.json";
    let content1 = json!({ "message": "Hello, Oramacore! (file 1)", "bool": true });
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

    const EXPECTED_DOCUMENT_COUNT: usize = 1;
    wait_for(&test_context, |s| {
        async {
            let mut stats = collection_client.reader_stats().await.unwrap();
            assert_eq!(stats.indexes_stats.len(), 1)
            //
            // let reader = s.reader.clone();
            // let read_api_key = s.read_api_key;
            // let collection_id = s.collection_id;
            // async move {
            //     let stats = reader
            //         .collection_stats(
            //             read_api_key,
            //             collection_id,
            //             CollectionStatsRequest { with_keys: false },
            //         )
            //         .await?;
            //     let index_stats = stats
            //         .indexes_stats
            //         .iter()
            //         .find(|index| index.id == self.index_id)
            //         .ok_or_else(|| anyhow::anyhow!("Index not found"))?;
            //     if index_stats.document_count < EXPECTED_DOCUMENT_COUNT {
            //         bail!(
            //             "Document count mismatch: expected {}, got {}",
            //             EXPECTED_DOCUMENT_COUNT,
            //             index_stats.document_count
            //         );
            //     }
            //
            //     Ok(())
        }
        .boxed()
    })
    .await
    .unwrap();

    let res = collection_client
        .search(
            json!({
                "term": "hello",
                "where": {
                    "bool": true,
                }
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.count, 1);
}
