use anyhow::{Context, Result};
use aws_credential_types::provider::SharedCredentialsProvider;
use aws_credential_types::Credentials;
use aws_sdk_s3::{
    config::{BehaviorVersion, Region},
    Client,
};
use resy::remotes::aws::{Change, S3};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::path::PathBuf;
use tracing::error;

use crate::collection_manager::sides::write::datasource::{Operation, SyncUpdate};
use crate::types::{CollectionId, Document, DocumentList, IndexId};

use super::DatasourceError;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct S3Fetcher {
    pub bucket: String,
    pub region: String,
    pub access_key_id: String,
    pub secret_access_key: String,
    pub endpoint_url: Option<String>,
}

pub async fn validate_credentials(s3_fetcher: &S3Fetcher) -> Result<()> {
    let credentials = Credentials::new(
        s3_fetcher.access_key_id.clone(),
        s3_fetcher.secret_access_key.clone(),
        None,
        None,
        "resy",
    );

    let mut s3_config_builder = aws_sdk_s3::config::Builder::new()
        .credentials_provider(SharedCredentialsProvider::new(credentials))
        .region(Region::new(s3_fetcher.region.clone()))
        .behavior_version(BehaviorVersion::latest());

    if let Some(endpoint_url) = &s3_fetcher.endpoint_url {
        s3_config_builder = s3_config_builder.endpoint_url(endpoint_url);
        s3_config_builder = s3_config_builder.force_path_style(true);
    }

    let s3_config = s3_config_builder.build();
    let s3_client = Client::from_conf(s3_config);

    s3_client.list_buckets().send().await.context(
        "Failed to validate S3 credentials. Please check your credentials and permissions.",
    )?;

    Ok(())
}

const BATCH_SIZE: usize = 10;

pub async fn sync_s3_datasource(
    datasource_dir: PathBuf,
    collection_id: CollectionId,
    index_id: IndexId,
    s3_datasource: &S3Fetcher,
    event_sender: tokio::sync::mpsc::Sender<SyncUpdate>,
) -> Result<(), DatasourceError> {
    let credentials = Credentials::new(
        s3_datasource.access_key_id.clone(),
        s3_datasource.secret_access_key.clone(),
        None,
        None,
        "resy",
    );

    let mut s3_config_builder = aws_sdk_s3::config::Builder::new()
        .credentials_provider(SharedCredentialsProvider::new(credentials))
        .region(Region::new(s3_datasource.region.clone()))
        .behavior_version(BehaviorVersion::latest());

    if let Some(endpoint_url) = &s3_datasource.endpoint_url {
        s3_config_builder = s3_config_builder.endpoint_url(endpoint_url);
        s3_config_builder = s3_config_builder.force_path_style(true);
    }

    let s3_config = s3_config_builder.build();
    let s3_client = Client::from_conf(s3_config);
    let mut s3_diff_client = S3::from_client(s3_client.clone(), s3_datasource.bucket.clone());

    let mut docs_to_insert = Vec::with_capacity(BATCH_SIZE);
    let mut keys_to_remove = Vec::with_capacity(BATCH_SIZE);

    let db_name = format!("{}_{}.db", collection_id, index_id);
    let db_path = datasource_dir.join(db_name);
    if let Err(e) = s3_diff_client
        .stream_diff_and_update(db_path.as_path(), async |change| {
            match change {
                Change::Added(obj) => {
                    match fetch_document(&s3_client, &s3_datasource.bucket, &obj.key).await {
                        Ok(doc) => docs_to_insert.push(doc),
                        Err(e) => {
                            error!(error = ?e, key = %obj.key, "Failed to fetch new document");
                        }
                    }
                }
                Change::Modified { old: _, new } => {
                    match fetch_document(&s3_client, &s3_datasource.bucket, &new.key).await {
                        Ok(doc) => docs_to_insert.push(doc),
                        Err(e) => {
                            error!(error = ?e, key = %new.key, "Failed to fetch modified document");
                        }
                    }
                }
                Change::Deleted(obj) => {
                    keys_to_remove.push(obj.key);
                }
            }
            if docs_to_insert.len() >= BATCH_SIZE {
                let docs = std::mem::take(&mut docs_to_insert);
                event_sender
                    .send(SyncUpdate {
                        collection_id,
                        index_id,
                        operation: Operation::Insert(DocumentList(docs)),
                    })
                    .await
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;
            }
            if keys_to_remove.len() >= BATCH_SIZE {
                let keys = std::mem::take(&mut keys_to_remove);
                event_sender
                    .send(SyncUpdate {
                        collection_id,
                        index_id,
                        operation: Operation::Delete(keys),
                    })
                    .await
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;
            }
            Ok(())
        })
        .await
    {
        return Err(DatasourceError::SyncFailed(anyhow::anyhow!("{}", e)));
    }

    if !docs_to_insert.is_empty() {
        event_sender
            .send(SyncUpdate {
                collection_id,
                index_id,
                operation: Operation::Insert(DocumentList(docs_to_insert)),
            })
            .await?;
    }

    if !keys_to_remove.is_empty() {
        event_sender
            .send(SyncUpdate {
                collection_id,
                index_id,
                operation: Operation::Delete(keys_to_remove),
            })
            .await?;
    }

    Ok(())
}

async fn fetch_document(s3_client: &Client, bucket: &str, key: &str) -> Result<Document> {
    let obj = s3_client
        .get_object()
        .bucket(bucket)
        .key(key)
        .send()
        .await
        .with_context(|| format!("Failed to get object from S3 key: {}", key))?;

    let body = obj
        .body
        .collect()
        .await
        .context("Failed to read S3 object body")?;

    let mut inner = serde_json::from_slice::<Map<String, Value>>(&body.into_bytes())
        .context("Failed to parse document body as JSON")?;

    // adding an id key consistent with the bucket key, in order to recognize it to
    // update/delete it in future.
    inner.insert("id".to_string(), Value::String(key.to_string()));

    Ok(Document { inner })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::utils::setup_s3_container;
    use aws_sdk_s3::primitives::ByteStream;
    use fake::{Fake, Faker};
    use serde_json::json;
    use std::collections::HashSet;

    #[tokio::test]
    async fn test_sync_s3_datasource() {
        let (s3_client, bucket_name, _container, endpoint_url) = setup_s3_container().await;

        let s3_datasource = S3Fetcher {
            bucket: bucket_name.clone(),
            region: "us-east-1".to_string(),
            access_key_id: "test".to_string(),
            secret_access_key: "test".to_string(),
            endpoint_url: Some(endpoint_url),
        };

        let tmp_dir = tempfile::tempdir().unwrap();
        let datasource_dir = tmp_dir.path().to_path_buf();
        let collection_id = CollectionId::try_new(Faker.fake::<String>()).unwrap();
        let index_id = IndexId::try_new(Faker.fake::<String>()).unwrap();

        let (sync_sender, mut sync_receiver) = tokio::sync::mpsc::channel(100);

        // 1. Initial sync, no changes
        sync_s3_datasource(
            datasource_dir.clone(),
            collection_id,
            index_id,
            &s3_datasource,
            sync_sender.clone(),
        )
        .await
        .unwrap();

        assert!(sync_receiver.try_recv().is_err());

        // 2. Add BATCH_SIZE + 1 files to test batching
        let num_files = BATCH_SIZE + 1;
        let mut expected_keys = HashSet::new();
        for i in 0..num_files {
            let key = format!("test-file-{}.json", i);
            let content = json!({ "message": format!("Hello, Oramacore! (file {})", i) });
            let content_bytes = serde_json::to_vec(&content).unwrap();

            s3_client
                .put_object()
                .bucket(&bucket_name)
                .key(&key)
                .body(ByteStream::from(content_bytes.clone()))
                .send()
                .await
                .unwrap();
            expected_keys.insert(key);
        }

        sync_s3_datasource(
            datasource_dir.clone(),
            collection_id,
            index_id,
            &s3_datasource,
            sync_sender.clone(),
        )
        .await
        .unwrap();

        let mut received_keys = HashSet::new();

        // Receive the first batch
        let sync_update = sync_receiver.recv().await.unwrap();
        assert_eq!(sync_update.collection_id, collection_id);
        assert_eq!(sync_update.index_id, index_id);
        if let Operation::Insert(docs) = sync_update.operation {
            assert_eq!(docs.0.len(), BATCH_SIZE);
            for doc in docs.0 {
                received_keys.insert(doc.inner["id"].as_str().unwrap().to_string());
            }
        } else {
            panic!("Expected Insert operation");
        }

        // Receive the second batch
        let sync_update = sync_receiver.recv().await.unwrap();
        assert_eq!(sync_update.collection_id, collection_id);
        assert_eq!(sync_update.index_id, index_id);
        if let Operation::Insert(docs) = sync_update.operation {
            assert_eq!(docs.0.len(), 1);
            for doc in docs.0 {
                received_keys.insert(doc.inner["id"].as_str().unwrap().to_string());
            }
        } else {
            panic!("Expected Insert operation");
        }
        assert_eq!(received_keys, expected_keys);
        assert!(sync_receiver.try_recv().is_err());

        // 3. Modify one, delete one
        let key_to_modify = "test-file-0.json";
        let key_to_delete = "test-file-1.json";

        let updated_content = json!({ "message": "Hello, Oramacore! Updated." });
        let updated_content_bytes = serde_json::to_vec(&updated_content).unwrap();
        s3_client
            .put_object()
            .bucket(&bucket_name)
            .key(key_to_modify)
            .body(ByteStream::from(updated_content_bytes.clone()))
            .send()
            .await
            .unwrap();

        s3_client
            .delete_object()
            .bucket(&bucket_name)
            .key(key_to_delete)
            .send()
            .await
            .unwrap();

        sync_s3_datasource(
            datasource_dir.clone(),
            collection_id,
            index_id,
            &s3_datasource,
            sync_sender.clone(),
        )
        .await
        .unwrap();

        let mut insert_received = false;
        let mut delete_received = false;

        for _ in 0..2 {
            let sync_update = sync_receiver.recv().await.unwrap();
            match sync_update.operation {
                Operation::Insert(docs) => {
                    assert_eq!(docs.0.len(), 1);
                    let doc = &docs.0[0];
                    assert_eq!(doc.inner.get("id").unwrap().as_str().unwrap(), key_to_modify);
                    assert_eq!(
                        doc.inner.get("message").unwrap().as_str().unwrap(),
                        "Hello, Oramacore! Updated."
                    );
                    insert_received = true;
                }
                Operation::Delete(keys) => {
                    assert_eq!(keys.len(), 1);
                    assert_eq!(keys[0], key_to_delete);
                    delete_received = true;
                }
            }
        }

        assert!(insert_received);
        assert!(delete_received);
        assert!(sync_receiver.try_recv().is_err());
    }
}
