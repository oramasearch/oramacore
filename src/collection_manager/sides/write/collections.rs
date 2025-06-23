use std::collections::HashMap;
use std::ops::Deref;
use std::sync::Arc;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, RwLockReadGuard};
use tracing::{error, info};

use crate::ai::automatic_embeddings_selector::AutomaticEmbeddingsSelector;
use crate::ai::OramaModel;
use crate::collection_manager::sides::hooks::HooksRuntime;
use crate::collection_manager::sides::write::WriteError;
use crate::collection_manager::sides::{OperationSender, WriteOperation};
use crate::file_utils::{create_if_not_exists, BufferedFile};
use crate::metrics::commit::COMMIT_CALCULATION_TIME;
use crate::metrics::Empty;
use crate::nlp::locales::Locale;
use crate::nlp::NLPService;
use crate::types::{CollectionId, DocumentId};
use crate::types::{CreateCollection, DescribeCollectionResponse, LanguageDTO};

use super::collection::CollectionWriter;
use super::embedding::MultiEmbeddingCalculationRequest;
use super::CollectionsWriterConfig;

pub struct CollectionsWriter {
    collections: RwLock<HashMap<CollectionId, CollectionWriter>>,
    config: CollectionsWriterConfig,
    embedding_sender: tokio::sync::mpsc::Sender<MultiEmbeddingCalculationRequest>,
    op_sender: OperationSender,
    hooks_runtime: Arc<HooksRuntime>,
    nlp_service: Arc<NLPService>,
    automatic_embeddings_selector: Arc<AutomaticEmbeddingsSelector>,
    default_model: OramaModel,
}

impl CollectionsWriter {
    pub async fn try_load(
        config: CollectionsWriterConfig,
        embedding_sender: tokio::sync::mpsc::Sender<MultiEmbeddingCalculationRequest>,
        hooks_runtime: Arc<HooksRuntime>,
        nlp_service: Arc<NLPService>,
        op_sender: OperationSender,
        automatic_embeddings_selector: Arc<AutomaticEmbeddingsSelector>,
    ) -> Result<Self> {
        let mut collections: HashMap<CollectionId, CollectionWriter> = Default::default();
        let default_model = config.default_embedding_model.0;

        let data_dir = &config.data_dir.join("collections");
        create_if_not_exists(data_dir).context("Cannot create data directory")?;

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
                    collections: Default::default(),
                    config,
                    embedding_sender,
                    op_sender,
                    hooks_runtime,
                    nlp_service,
                    automatic_embeddings_selector,
                    default_model,
                });
            }
        };

        for collection_id in collection_info.collection_ids {
            let collection_dir = data_dir.join(collection_id.as_str());

            // If the collection is not loaded correctly, we bail out the error
            // and we abort the start up process
            // Should we instead ignore it?
            // TODO: think about it
            let collection = CollectionWriter::try_load(
                collection_dir,
                hooks_runtime.clone(),
                nlp_service.clone(),
                embedding_sender.clone(),
                hooks_runtime.clone(),
                op_sender.clone(),
                automatic_embeddings_selector.clone(),
            )
            .await?;
            collections.insert(collection_id, collection);
        }

        let writer = CollectionsWriter {
            collections: RwLock::new(collections),
            config,
            embedding_sender,
            op_sender,
            hooks_runtime,
            nlp_service,
            automatic_embeddings_selector,
            default_model,
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
        let r = self.collections.read().await;
        CollectionReadLock::try_new(r, id)
    }

    pub async fn create_collection(
        &self,
        collection_option: CreateCollection,
        sender: OperationSender,
    ) -> Result<(), WriteError> {
        let CreateCollection {
            id,
            description,
            language,
            embeddings_model,
            write_api_key,
            read_api_key,
        } = collection_option;

        info!("Creating collection {:?}", id);

        let language = language.unwrap_or(LanguageDTO::English);
        let default_locale: Locale = language.into();

        let collection = CollectionWriter::empty(
            id,
            description.clone(),
            write_api_key,
            read_api_key,
            default_locale,
            embeddings_model.map(|m| m.0).unwrap_or(self.default_model),
            self.embedding_sender.clone(),
            self.hooks_runtime.clone(),
            self.op_sender.clone(),
            self.nlp_service.clone(),
            self.automatic_embeddings_selector.clone(),
        );

        let mut collections = self.collections.write().await;

        if collections.contains_key(&id) {
            return Err(WriteError::CollectionAlreadyExists(id));
        }

        collections.insert(id, collection);
        drop(collections);

        sender
            .send(WriteOperation::CreateCollection {
                id,
                read_api_key,
                description,
                default_locale,
            })
            .await
            .context("Cannot send create collection")?;

        Ok(())
    }

    pub async fn list(&self) -> Vec<DescribeCollectionResponse> {
        let collections = self.collections.read().await;

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
        let mut collections = self.collections.write().await;

        let data_dir = &self.config.data_dir.join("collections");
        create_if_not_exists(data_dir).context("Cannot create data directory")?;

        let m = COMMIT_CALCULATION_TIME.create(Empty);
        for (collection_id, collection) in collections.iter_mut() {
            let collection_dir = data_dir.join(collection_id.as_str());

            collection.commit(collection_dir).await?;
        }

        let info_path = data_dir.join("info.json");
        info!("Committing info at {:?}", info_path);
        BufferedFile::create_or_overwrite(info_path)
            .context("Cannot create info.json")?
            .write_json_data(&CollectionsInfo::V1(CollectionsInfoV1 {
                collection_ids: collections.keys().cloned().collect::<Vec<_>>(),
            }))
            .context("Cannot write info.json")?;

        drop(m);

        // Now it is safe to drop the lock
        // because we save everything to disk
        drop(collections);
        info!("Collections committed");

        Ok(())
    }

    pub async fn delete_collection(&self, collection_id: CollectionId) -> Option<Vec<DocumentId>> {
        let mut collections = self.collections.write().await;
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
            .op_sender
            .send(WriteOperation::DeleteCollection(collection_id))
            .await
            .context("Cannot send delete collection")
        {
            error!(error = ?error, "Error sending delete collection");
        }

        Some(document_ids)
    }
}

pub struct CollectionReadLock<'guard> {
    lock: RwLockReadGuard<'guard, HashMap<CollectionId, CollectionWriter>>,
    pub id: CollectionId,
}

impl<'guard> CollectionReadLock<'guard> {
    pub fn try_new(
        lock: RwLockReadGuard<'guard, HashMap<CollectionId, CollectionWriter>>,
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
