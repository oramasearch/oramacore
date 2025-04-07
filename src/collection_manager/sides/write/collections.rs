use std::collections::HashMap;
use std::ops::Deref;
use std::sync::Arc;

use anyhow::{anyhow, Context, Ok, Result};
use dashmap::DashMap;
use redact::Secret;
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, RwLockReadGuard};
use tracing::info;

use crate::collection_manager::dto::{
    ApiKey, CreateCollection, DocumentFields, EmbeddingTypedField, LanguageDTO, TypedField,
};
use crate::collection_manager::sides::hooks::HooksRuntime;
use crate::collection_manager::sides::write::collection::DEFAULT_EMBEDDING_FIELD_NAME;
use crate::collection_manager::sides::{OperationSender, WriteOperation};
use crate::file_utils::{create_if_not_exists, BufferedFile};
use crate::metrics::commit::COMMIT_CALCULATION_TIME;
use crate::metrics::CollectionCommitLabels;
use crate::{collection_manager::dto::CollectionDTO, types::CollectionId};
use nlp::NLPService;
use types::OramaModelSerializable;

use super::CollectionsWriterConfig;
use super::{collection::CollectionWriter, embedding::EmbeddingCalculationRequest};

pub struct CollectionsWriter {
    collections: RwLock<HashMap<CollectionId, CollectionWriter>>,
    config: CollectionsWriterConfig,
    embedding_sender: tokio::sync::mpsc::Sender<EmbeddingCalculationRequest>,
    pub(super) collection_options: DashMap<CollectionId, CreateCollection>,
}

impl CollectionsWriter {
    pub async fn try_load(
        config: CollectionsWriterConfig,
        embedding_sender: tokio::sync::mpsc::Sender<EmbeddingCalculationRequest>,
        hooks_runtime: Arc<HooksRuntime>,
        nlp_service: Arc<NLPService>,
    ) -> Result<Self> {
        let mut collections: HashMap<CollectionId, CollectionWriter> = Default::default();

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
                    collection_options: Default::default(),
                    collections: Default::default(),
                    config,
                    embedding_sender,
                });
            }
        };

        for collection_id in collection_info.collection_ids {
            let collection_dir = data_dir.join(collection_id.0);

            // All those values are replaced inside `load` method
            let mut collection = CollectionWriter::new(
                collection_id,
                None,
                ApiKey(Secret::new("".to_string())),
                LanguageDTO::English,
                embedding_sender.clone(),
            );
            collection
                .load(collection_dir, hooks_runtime.clone(), nlp_service.clone())
                .await?;

            collections.insert(collection_id, collection);
        }

        let writer = CollectionsWriter {
            collections: RwLock::new(collections),
            config,
            embedding_sender,
            collection_options: collection_info.collection_options.into_iter().collect(),
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
        hooks_runtime: Arc<HooksRuntime>,
    ) -> Result<()> {
        let CreateCollection {
            id,
            description,
            language,
            embeddings,
            write_api_key,
            read_api_key,
        } = collection_option.clone();

        info!("Creating collection {:?}", id);

        let default_language = language.unwrap_or(LanguageDTO::English);

        let collection = CollectionWriter::new(
            id,
            description.clone(),
            write_api_key,
            default_language,
            self.embedding_sender.clone(),
        );

        let typed_fields = if !cfg!(feature = "no_auto_embedding_field_on_creation") {
            let model = embeddings
                .as_ref()
                .and_then(|embeddings| embeddings.model.as_ref())
                .unwrap_or(&self.config.default_embedding_model);
            let model = model.0;
            let document_fields = embeddings
                .map(|embeddings| {
                    // Empty array means all string properties
                    if embeddings.document_fields.is_empty() {
                        DocumentFields::AllStringProperties
                    } else {
                        DocumentFields::Properties(embeddings.document_fields)
                    }
                })
                .unwrap_or(DocumentFields::AllStringProperties);
            let typed_field = TypedField::Embedding(EmbeddingTypedField {
                model: OramaModelSerializable(model),
                document_fields,
            });
            HashMap::from_iter([(DEFAULT_EMBEDDING_FIELD_NAME.to_string(), typed_field)])
        } else {
            HashMap::new()
        };

        let mut collections = self.collections.write().await;

        if collections.contains_key(&id) {
            // This error should be typed.
            // TODO: create a custom error type
            return Err(anyhow!(format!("Collection \"{}\" already exists", id.0)));
        }

        // Send event & Register field should be inside the lock transaction
        sender
            .send(WriteOperation::CreateCollection {
                id,
                read_api_key,
                description,
                default_language,
            })
            .await
            .context("Cannot send create collection")?;
        collection
            .register_fields(typed_fields, sender.clone(), hooks_runtime)
            .await
            .context("Cannot register fields")?;

        collections.insert(id, collection);
        drop(collections);

        self.collection_options.insert(id, collection_option);

        Ok(())
    }

    pub async fn replace(
        &self,
        collection_id_tmp: CollectionId,
        collection_id: CollectionId,
    ) -> Result<()> {
        let mut collections = self.collections.write().await;
        let mut collection_tmp = collections
            .remove(&collection_id_tmp)
            .ok_or_else(|| anyhow!("Collection not found"))?;
        collection_tmp.id = collection_id;
        collections.insert(collection_id, collection_tmp);
        drop(collections);
        Ok(())
    }

    pub async fn list(&self) -> Vec<CollectionDTO> {
        let collections = self.collections.read().await;

        let mut r = vec![];
        for collection in collections.values() {
            r.push(collection.as_dto().await);
        }
        r
    }

    pub async fn commit(&self) -> Result<()> {
        // During the commit, we don't accept any new write operation
        let collections = self.collections.write().await;

        let data_dir = &self.config.data_dir.join("collections");
        create_if_not_exists(data_dir).context("Cannot create data directory")?;

        for (collection_id, collection) in collections.iter() {
            let collection_dir = data_dir.join(collection_id);

            let m = COMMIT_CALCULATION_TIME.create(CollectionCommitLabels {
                collection: collection_id.to_string(),
                side: "write",
            });
            collection.commit(collection_dir).await?;
            drop(m);
        }

        let info_path = data_dir.join("info.json");
        info!("Committing info at {:?}", info_path);
        BufferedFile::create_or_overwrite(info_path)
            .context("Cannot create info.json")?
            .write_json_data(&CollectionsInfo::V1(CollectionsInfoV1 {
                collection_ids: collections.keys().cloned().collect::<Vec<_>>(),
                collection_options: self
                    .collection_options
                    .iter()
                    .map(|e| (*e.key(), e.value().clone()))
                    .collect(),
            }))
            .context("Cannot write info.json")?;

        // Now it is safe to drop the lock
        // because we safe everything to disk
        drop(collections);
        info!("Commit done");

        Ok(())
    }

    pub async fn delete_collection(&self, collection_id: CollectionId) -> bool {
        let mut collections = self.collections.write().await;
        let collection = collections.remove(&collection_id);

        let collection = match collection {
            None => return false,
            Some(coll) => coll,
        };

        let data_dir = &self.config.data_dir.join("collections");
        let collection_dir = data_dir.join(collection_id.0);

        collection.remove_from_fs(collection_dir).await;

        true
    }
}

pub struct CollectionReadLock<'guard> {
    lock: RwLockReadGuard<'guard, HashMap<CollectionId, CollectionWriter>>,
    id: CollectionId,
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
    collection_options: HashMap<CollectionId, CreateCollection>,
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
