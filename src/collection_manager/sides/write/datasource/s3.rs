use std::path::PathBuf;
use std::sync::Arc;

use aws_credential_types::provider::SharedCredentialsProvider;
use aws_credential_types::Credentials;
use aws_sdk_s3::{
    config::{BehaviorVersion, Region},
    Client,
};
use resy::remotes::aws::{Change, S3};
use serde_json::{Map, Value};
use tracing::{error, info};

use crate::collection_manager::sides::write::datasource::storage;
use crate::collection_manager::sides::write::WriteSide;
use crate::types::{CollectionId, Document, DocumentList, IndexId, WriteApiKey};

pub async fn sync_s3_datasource(
    write_side: Arc<WriteSide>,
    collection_id: CollectionId,
    index_id: IndexId,
    datasource_id: IndexId,
    s3_datasource: &storage::S3,
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
    let mut s3 = S3::from_client(s3_client.clone(), s3_datasource.bucket.clone());

    // FIXME: this vector could become quite large on the first call.
    // The chunking should be done at this level.
    let mut keys_to_add: Vec<String> = Vec::new();
    let mut keys_to_remove: Vec<String> = Vec::new();
    let db_path = PathBuf::from("./resy").join(datasource_id.as_str());
    if let Err(e) = s3
        .stream_diff_and_update(db_path.as_path(), |change| {
            match change {
                Change::Added(obj) => {
                    keys_to_add.push(obj.key.to_string());
                }
                Change::Modified { old: _, new } => {
                    keys_to_add.push(new.key.to_string());
                }
                Change::Deleted(obj) => {
                    keys_to_remove.push(obj.key.to_string());
                }
            }
            Ok(())
        })
        .await
    {
        error!(error = ?e, "Failed to stream from s3");
        return Ok(());
    }

    // Handle additions/modifications
    if !keys_to_add.is_empty() {
        let mut docs_to_insert = Vec::new();
        for key_chunk in keys_to_add.chunks(100) {
            for key in key_chunk {
                match s3_client
                    .get_object()
                    .bucket(s3_datasource.bucket.as_str())
                    .key(key.as_str())
                    .send()
                    .await
                {
                    Ok(obj) => match obj.body.collect().await {
                        Ok(body) => {
                            match serde_json::from_slice::<Map<String, Value>>(&body.into_bytes()) {
                                Ok(inner) => docs_to_insert.push(Document { inner }),
                                Err(e) => {
                                    error!(error = ?e, "Failed to parse document from S3 key {}", key)
                                }
                            }
                        }
                        Err(e) => error!(error = ?e, "Failed to read body from S3 key {}", key),
                    },
                    Err(e) => error!(error = ?e, "Failed to get object from S3 key {}", key),
                }
            }

            if !docs_to_insert.is_empty() {
                info!(
                    "Inserting {} documents from datasource",
                    docs_to_insert.len()
                );
                let document_list = DocumentList(std::mem::take(&mut docs_to_insert));
                if let Err(e) = write_side
                    .insert_documents(
                        WriteApiKey::ApiKey(write_side.master_api_key),
                        collection_id,
                        index_id,
                        document_list,
                    )
                    .await
                {
                    error!(error = ?e, "Failed to insert documents from datasource");
                }
            }
        }
    }

    // Handle deletions
    if !keys_to_remove.is_empty() {
        for key_chunk in keys_to_remove.chunks(100) {
            info!("Deleting {} documents from datasource", key_chunk.len());
            let docs_to_delete = key_chunk.to_vec();
            if let Err(e) = write_side
                .delete_documents(
                    WriteApiKey::ApiKey(write_side.master_api_key),
                    collection_id,
                    index_id,
                    docs_to_delete,
                )
                .await
            {
                error!(error = ?e, "Failed to delete documents from datasource");
            }
        }
    }

    Ok(())
}
