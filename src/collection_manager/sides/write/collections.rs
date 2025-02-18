use std::collections::HashMap;
use std::ops::Deref;
use std::sync::Arc;

use anyhow::{anyhow, Context, Ok, Result};
use redact::Secret;
use tokio::sync::{RwLock, RwLockReadGuard};
use tracing::info;

use crate::collection_manager::sides::generic_kv::KV;
use crate::collection_manager::sides::hooks::HooksRuntime;
use crate::collection_manager::sides::write::collection::DEFAULT_EMBEDDING_FIELD_NAME;
use crate::collection_manager::sides::{OperationSender, OramaModelSerializable, WriteOperation};
use crate::metrics::commit::COMMIT_CALCULATION_TIME;
use crate::metrics::CollectionCommitLabels;
use crate::nlp::NLPService;
use crate::{
    collection_manager::dto::CollectionDTO, file_utils::list_directory_in_path, types::CollectionId,
};

use crate::collection_manager::dto::{
    ApiKey, CreateCollection, DocumentFields, EmbeddingTypedField, LanguageDTO, TypedField,
};

use super::CollectionsWriterConfig;
use super::{collection::CollectionWriter, embedding::EmbeddingCalculationRequest};

pub struct CollectionsWriter {
    collections: RwLock<HashMap<CollectionId, CollectionWriter>>,
    config: CollectionsWriterConfig,
    embedding_sender: tokio::sync::mpsc::Sender<EmbeddingCalculationRequest>,
}

impl CollectionsWriter {
    pub async fn try_load(
        config: CollectionsWriterConfig,
        embedding_sender: tokio::sync::mpsc::Sender<EmbeddingCalculationRequest>,
        hooks_runtime: Arc<HooksRuntime>,
        nlp_service: Arc<NLPService>,
        kv: Arc<KV>,
    ) -> Result<Self> {
        let mut collections: HashMap<CollectionId, CollectionWriter> = Default::default();

        let data_dir = &config.data_dir;

        info!("Loading collections from disk from {:?}", data_dir);

        let collection_dirs =
            list_directory_in_path(data_dir).context("Cannot read collection list from disk")?;

        let collection_dirs = match collection_dirs {
            Some(collection_dirs) => collection_dirs,
            None => {
                info!(
                    "No collections found in data directory {:?}. Skipping load.",
                    data_dir
                );
                return Ok(CollectionsWriter {
                    collections: Default::default(),
                    config,
                    embedding_sender,
                });
            }
        };

        for collection_dir in collection_dirs {
            let file_name = collection_dir
                .file_name()
                .expect("File name is always given at this point");
            let file_name: String = file_name.to_string_lossy().into();

            let collection_id = CollectionId(file_name);

            // All those values are replaced inside `load` method
            let mut collection = CollectionWriter::new(
                collection_id.clone(),
                None,
                ApiKey(Secret::new("".to_string())),
                LanguageDTO::English,
                embedding_sender.clone(),
            );
            collection
                .load(
                    collection_dir,
                    hooks_runtime.clone(),
                    nlp_service.clone(),
                    kv.clone(),
                )
                .await?;

            collections.insert(collection_id, collection);
        }

        let writer = CollectionsWriter {
            collections: RwLock::new(collections),
            config,
            embedding_sender,
        };

        Ok(writer)
    }

    pub async fn get_collection<'s, 'coll>(
        &'s self,
        id: CollectionId,
    ) -> Option<CollectionWriteLock<'coll>>
    where
        's: 'coll,
    {
        let r = self.collections.read().await;
        CollectionWriteLock::try_new(r, id)
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
        } = collection_option;

        info!("Creating collection {:?}", id);

        let collection = CollectionWriter::new(
            id.clone(),
            description,
            write_api_key,
            language.unwrap_or(LanguageDTO::English),
            self.embedding_sender.clone(),
        );

        let typed_fields = if !cfg!(feature = "no_auto_embedding_field_on_creation") {
            let model = embeddings
                .as_ref()
                .and_then(|embeddings| embeddings.model.as_ref())
                .unwrap_or(&self.config.default_embedding_model);
            let model = model.0;
            let document_fields = embeddings
                .map(|embeddings| embeddings.document_fields)
                .map(DocumentFields::Properties)
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
                id: id.clone(),
                read_api_key,
            })
            .await
            .context("Cannot send create collection")?;
        collection
            .register_fields(typed_fields, sender.clone(), hooks_runtime)
            .await
            .context("Cannot register fields")?;

        collections.insert(id, collection);
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
        let data_dir = &self.config.data_dir;

        // During the commit, we don't accept any new write operation
        let collections = self.collections.write().await;

        std::fs::create_dir_all(data_dir).context("Cannot create data directory")?;

        for (collection_id, collection) in collections.iter() {
            let collection_dir = data_dir.join(collection_id.0.clone());

            let m = COMMIT_CALCULATION_TIME.create(CollectionCommitLabels {
                collection: collection_id.0.clone(),
                side: "write",
            });
            collection.commit(collection_dir).await?;
            drop(m);
        }

        // Now it is safe to drop the lock
        // because we safe everything to disk
        drop(collections);

        Ok(())
    }
}

pub struct CollectionWriteLock<'guard> {
    lock: RwLockReadGuard<'guard, HashMap<CollectionId, CollectionWriter>>,
    id: CollectionId,
}

impl<'guard> CollectionWriteLock<'guard> {
    pub fn try_new(
        lock: RwLockReadGuard<'guard, HashMap<CollectionId, CollectionWriter>>,
        id: CollectionId,
    ) -> Option<Self> {
        let guard = lock.get(&id);
        match &guard {
            Some(_) => {
                let _ = guard;
                Some(CollectionWriteLock { lock, id })
            }
            None => None,
        }
    }
}

impl Deref for CollectionWriteLock<'_> {
    type Target = CollectionWriter;

    fn deref(&self) -> &Self::Target {
        // safety: the collection contains the id because we checked it before
        // no one can remove the collection from the map because we hold a read lock
        self.lock.get(&self.id).unwrap()
    }
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
