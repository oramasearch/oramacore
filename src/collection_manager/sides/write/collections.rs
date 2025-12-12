use std::collections::HashMap;
use std::ops::Deref;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use chrono::Utc;
use futures::future::join_all;
use serde::{Deserialize, Serialize};
use tracing::{error, info};

use crate::collection_manager::sides::write::collection::CreateEmptyCollection;
use crate::collection_manager::sides::write::context::WriteSideContext;
use crate::collection_manager::sides::write::WriteError;
use crate::collection_manager::sides::{CollectionWriteOperation, OperationSender, WriteOperation};
use crate::lock::{OramaAsyncLock, OramaAsyncLockReadGuard};
use crate::metrics::commit::COMMIT_CALCULATION_TIME;
use crate::metrics::CollectionCommitLabels;
use crate::python::embeddings::Model;
use crate::types::{CollectionId, DocumentId};
use crate::types::{CreateCollection, DescribeCollectionResponse, LanguageDTO};
use oramacore_lib::fs::{create_if_not_exists, BufferedFile};
use oramacore_lib::nlp::locales::Locale;

use super::collection::CollectionWriter;
use super::CollectionsWriterConfig;

pub struct CollectionsWriter {
    collections: OramaAsyncLock<HashMap<CollectionId, CollectionWriter>>,
    config: CollectionsWriterConfig,
    context: WriteSideContext,
    default_model: Model,
    data_dir: PathBuf,
}

impl CollectionsWriter {
    pub async fn try_load(
        config: CollectionsWriterConfig,
        context: WriteSideContext,
        global_document_id: u64,
    ) -> Result<Self> {
        let mut collections: HashMap<CollectionId, CollectionWriter> = Default::default();
        let default_model = config.default_embedding_model;

        let data_dir = config.data_dir.join("collections");
        create_if_not_exists(&data_dir).context("Cannot create data directory")?;

        let info_path = data_dir.join("info.json");
        info!("Loading collections from disk from {:?}", info_path);
        let collection_info = match BufferedFile::open(info_path)
            .and_then(|file| file.read_json_data::<CollectionsInfo>())
        {
            std::result::Result::Ok(CollectionsInfo::V1(info)) => info,
            Err(_) => {
                info!(
                    "No collections found in data directory {:?}. Create new instance",
                    data_dir
                );
                return Ok(CollectionsWriter {
                    // collection_options: Default::default(),
                    collections: OramaAsyncLock::new("collections", Default::default()),
                    config,
                    context,
                    default_model,
                    data_dir,
                });
            }
        };

        for collection_id in collection_info.collection_ids {
            let collection_dir = data_dir.join(collection_id.as_str());

            // If the collection is not loaded correctly, we bail out the error
            // and we abort the start up process
            // Should we instead ignore it?
            // TODO: think about it
            let collection =
                CollectionWriter::try_load(collection_dir, context.clone(), global_document_id)
                    .await?;
            collections.insert(collection_id, collection);
        }

        let writer = CollectionsWriter {
            collections: OramaAsyncLock::new("collections", collections),
            config,
            context,
            default_model,
            data_dir,
        };

        Ok(writer)
    }

    pub async fn get_collection<'s, 'coll>(
        &'s self,
        id: CollectionId,
    ) -> Option<CollectionReadLock<'coll>>
    where
        's: 'coll,
    {
        let r = self.collections.read("get_collection").await;
        CollectionReadLock::try_new(r, id)
    }

    pub async fn create_collection(
        &self,
        collection_option: CreateCollection,
        sender: OperationSender,
        global_document_id: u64,
    ) -> Result<(), WriteError> {
        let CreateCollection {
            id,
            description,
            mcp_description,
            language,
            embeddings_model,
            write_api_key,
            read_api_key,
        } = collection_option;

        info!("Creating collection {:?}", id);

        let language = language.unwrap_or(LanguageDTO::English);
        let default_locale: Locale = language.into();

        let req = CreateEmptyCollection {
            id,
            description: description.clone(),
            default_locale,
            embeddings_model: embeddings_model.unwrap_or(self.default_model),
            write_api_key,
            read_api_key,
        };
        let collection = CollectionWriter::empty(
            self.data_dir.join(id.as_str()),
            req,
            self.context.clone(),
            global_document_id,
        )?;

        let mut collections = self.collections.write("create_collection").await;

        if collections.contains_key(&id) {
            return Err(WriteError::CollectionAlreadyExists(id));
        }

        collections.insert(id, collection);
        drop(collections);

        sender
            .send(WriteOperation::CreateCollection2 {
                id,
                read_api_key,
                write_api_key,
                description,
                mcp_description,
                default_locale,
            })
            .await
            .context("Cannot send create collection")?;

        Ok(())
    }

    pub async fn update_collection_mcp_description(
        &self,
        collection_id: CollectionId,
        mcp_description: Option<String>,
        sender: OperationSender,
    ) -> Result<(), WriteError> {
        info!("Updating collection {:?}", collection_id);

        let collections = self
            .collections
            .read("update_collection_mcp_description")
            .await;
        if !collections.contains_key(&collection_id) {
            return Err(WriteError::CollectionNotFound(collection_id));
        }
        drop(collections);

        sender
            .send(WriteOperation::Collection(
                collection_id,
                CollectionWriteOperation::UpdateMcpDescription { mcp_description },
            ))
            .await
            .context("Cannot send update collection")?;

        Ok(())
    }

    pub async fn list(&self) -> Vec<DescribeCollectionResponse> {
        let collections = self.collections.read("list").await;

        let mut r = vec![];
        for collection in collections.values() {
            r.push(collection.as_dto().await);
        }
        r
    }

    pub async fn commit(&self) -> Result<()> {
        info!("Committing collections");

        // During the commit, we don't accept any new write operation
        // This `write` lock is held until the commit is done
        let mut collections = self.collections.write("commit").await;
        info!("locked");

        let data_dir = &self.config.data_dir.join("collections");
        create_if_not_exists(data_dir).context("Cannot create data directory")?;

        let futures: Vec<_> = collections
            .iter_mut()
            .map(|(collection_id, collection)| {
                let collection_dir = data_dir.join(collection_id.as_str());
                async move {
                    let m = COMMIT_CALCULATION_TIME.create(CollectionCommitLabels {
                        collection: collection_id.as_str().to_string(),
                        side: "write",
                    });
                    let r = collection.commit(collection_dir).await;

                    drop(m);

                    r
                }
            })
            .collect();
        join_all(futures)
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;

        info!("unlocked");

        let info_path = data_dir.join("info.json");
        info!("Committing info at {:?}", info_path);
        BufferedFile::create_or_overwrite(info_path)
            .context("Cannot create info.json")?
            .write_json_data(&CollectionsInfo::V1(CollectionsInfoV1 {
                collection_ids: collections.keys().cloned().collect::<Vec<_>>(),
            }))
            .context("Cannot write info.json")?;

        // Now it is safe to drop the lock
        // because we save everything to disk
        drop(collections);
        info!("Collections committed");

        Ok(())
    }

    pub async fn delete_collection(&self, collection_id: CollectionId) -> Option<Vec<DocumentId>> {
        let mut collections = self.collections.write("delete_collection").await;
        let collection = collections.remove(&collection_id);

        let collection = match collection {
            None => return None,
            Some(coll) => coll,
        };

        let data_dir = &self.config.data_dir.join("collections");
        let collection_dir = data_dir.join(collection_id.as_str());

        let mut document_ids = vec![];
        let index_ids = collection.get_index_ids().await;
        for index in index_ids {
            let Some(index) = collection.get_index(index).await else {
                continue;
            };
            document_ids.extend(index.get_document_ids().await);
        }

        collection.remove_from_fs(collection_dir).await;

        if let Err(error) = self
            .context
            .op_sender
            .send(WriteOperation::DeleteCollection(collection_id))
            .await
            .context("Cannot send delete collection")
        {
            error!(error = ?error, "Error sending delete collection");
        }

        Some(document_ids)
    }

    pub async fn cleanup_expired_temp_indexes(&self, max_age: Duration) -> Result<usize> {
        info!(
            "Starting cleanup of temp indexes older than {} seconds",
            max_age.as_secs()
        );

        // PHASE 1: Collect expired temp indexes (with collections lock)
        let expired_temp_indexes = {
            let collections = self.collections.read("cleanup_expired_temp_indexes").await;
            let mut expired_indexes = Vec::new();
            let now = Utc::now();

            for (collection_id, collection) in collections.iter() {
                // Get temp indexes for this collection
                let temp_indexes = collection.get_temp_index_ids().await;

                for temp_index_id in temp_indexes {
                    // Get the temp index to check its creation time
                    if let Some(temp_index) = collection.get_temporary_index(temp_index_id).await {
                        let age = now.signed_duration_since(temp_index.created_at());

                        if age.to_std().unwrap_or(Duration::from_secs(0)) > max_age {
                            info!(
                                "Found expired temp index {} from collection {} (age: {} seconds)",
                                temp_index_id,
                                collection_id,
                                age.num_seconds()
                            );
                            expired_indexes.push((*collection_id, temp_index_id));
                        }
                    }
                }
            }
            expired_indexes
        }; // collections lock is released here

        // PHASE 2: Delete expired temp indexes (without collections lock)
        let mut cleanup_count = 0;

        for (collection_id, temp_index_id) in expired_temp_indexes {
            let collections = self.collections.read("cleanup_expired_temp_indexes").await;
            let collection = collections.get(&collection_id);

            if let Some(collection) = collection {
                // Delete the expired temp index
                match collection.delete_temp_index(temp_index_id).await {
                    Ok(()) => {
                        info!(
                            "Successfully cleaned up expired temp index {} from collection {}",
                            temp_index_id, collection_id
                        );
                        cleanup_count += 1;
                    }
                    Err(e) => {
                        error!(
                            "Failed to delete expired temp index {} from collection {}: {:?}",
                            temp_index_id, collection_id, e
                        );
                    }
                }
            } else {
                // Collection was deleted between phase 1 and 2, which is fine
                info!(
                    "Collection {} was deleted during cleanup, skipping temp index {}",
                    collection_id, temp_index_id
                );
            }
        }

        info!("Cleaned up {} expired temp indexes", cleanup_count);
        Ok(cleanup_count)
    }
}

pub struct CollectionReadLock<'guard> {
    lock: OramaAsyncLockReadGuard<'guard, HashMap<CollectionId, CollectionWriter>>,
    pub id: CollectionId,
}

impl<'guard> CollectionReadLock<'guard> {
    pub fn try_new(
        lock: OramaAsyncLockReadGuard<'guard, HashMap<CollectionId, CollectionWriter>>,
        id: CollectionId,
    ) -> Option<Self> {
        let guard = lock.get(&id);
        match &guard {
            Some(_) => {
                let _ = guard;
                Some(CollectionReadLock { lock, id })
            }
            None => None,
        }
    }
}

impl Deref for CollectionReadLock<'_> {
    type Target = CollectionWriter;

    fn deref(&self) -> &Self::Target {
        // safety: the collection contains the id because we checked it before
        // no one can remove the collection from the map because we hold a read lock
        self.lock.get(&self.id).unwrap()
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct CollectionsInfoV1 {
    collection_ids: Vec<CollectionId>,
}

#[derive(Debug, Serialize, Deserialize)]
enum CollectionsInfo {
    V1(CollectionsInfoV1),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    async fn test_writer_sync_send() {
        fn assert_sync_send<T: Sync + Send>() {}
        assert_sync_send::<CollectionsWriter>();
    }
}
