use anyhow::{Context, Result};
use aws_credential_types::provider::SharedCredentialsProvider;
use aws_credential_types::Credentials;
use aws_sdk_s3::{
    config::{BehaviorVersion, Region},
    Client,
};
use resy::remotes::aws::{Change, S3};
use serde_json::{Map, Value};
use std::path::PathBuf;
use tracing::error;

use crate::collection_manager::sides::write::datasource::storage;
use crate::collection_manager::sides::write::datasource::{Operation, SyncUpdate};
use crate::types::{CollectionId, Document, DocumentList, IndexId};

pub async fn sync_s3_datasource(
    datasource_dir: &PathBuf,
    collection_id: CollectionId,
    index_id: IndexId,
    datasource_id: IndexId,
    s3_datasource: &storage::S3,
    event_sender: tokio::sync::mpsc::Sender<SyncUpdate>,
) -> anyhow::Result<()> {
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
    }

    let s3_client = Client::from_conf(s3_config_builder.build());
    let mut s3_diff_client = S3::from_client(s3_client.clone(), s3_datasource.bucket.clone());

    const BATCH_SIZE: usize = 10;
    let mut docs_to_insert = Vec::with_capacity(BATCH_SIZE);
    let mut keys_to_remove = Vec::with_capacity(BATCH_SIZE);

    let db_path = datasource_dir.join(datasource_id.as_str());
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
                    .await?;
            }

            if keys_to_remove.len() >= BATCH_SIZE {
                let keys = std::mem::take(&mut keys_to_remove);
                event_sender
                    .send(SyncUpdate {
                        collection_id,
                        index_id,
                        operation: Operation::Delete(keys),
                    })
                    .await?;
            }

            Ok(())
        })
        .await
    {
        error!(error = ?e, "S3 diff stream failed");
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
