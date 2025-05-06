use std::{collections::HashMap, ops::Deref, path::PathBuf, sync::Arc};

use anyhow::{anyhow, bail, Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc::Sender, RwLock, RwLockReadGuard};
use tracing::{info, warn};

use crate::{
    ai::automatic_embeddings_selector::AutomaticEmbeddingsSelector,
    ai::OramaModel,
    collection_manager::sides::{
        field_name_to_path, hooks::HooksRuntime, CollectionWriteOperation,
        DocumentStorageWriteOperation, OperationSender, OramaModelSerializable, ReplaceIndexReason,
        WriteOperation,
    },
    file_utils::BufferedFile,
    nlp::{locales::Locale, NLPService, TextParser},
    types::{
        ApiKey, CollectionId, DescribeCollectionResponse, DocumentId, IndexEmbeddingsCalculation,
        IndexId,
    },
};

use super::{embedding::MultiEmbeddingCalculationRequest, index::Index};

pub const DEFAULT_EMBEDDING_FIELD_NAME: &str = "___orama_auto_embedding";

struct CollectionRuntimeConfig {
    default_locale: Locale,
    embeddings_model: OramaModel,
}

pub struct CollectionWriter {
    pub(super) id: CollectionId,
    description: Option<String>,
    runtime_config: RwLock<CollectionRuntimeConfig>,
    write_api_key: ApiKey,
    read_api_key: ApiKey,
    embedding_sender: Sender<MultiEmbeddingCalculationRequest>,
    indexes: RwLock<HashMap<IndexId, Index>>,
    temp_indexes: RwLock<HashMap<IndexId, Index>>,
    op_sender: OperationSender,
    hook_runtime: Arc<HooksRuntime>,
    nlp_service: Arc<NLPService>,
    automatic_embeddings_selector: Arc<AutomaticEmbeddingsSelector>,

    created_at: DateTime<Utc>,
}

impl CollectionWriter {
    pub fn empty(
        id: CollectionId,
        description: Option<String>,
        write_api_key: ApiKey,
        read_api_key: ApiKey,
        default_locale: Locale,
        embeddings_model: OramaModel,
        embedding_sender: Sender<MultiEmbeddingCalculationRequest>,
        hook_runtime: Arc<HooksRuntime>,
        op_sender: OperationSender,
        nlp_service: Arc<NLPService>,
        automatic_embeddings_selector: Arc<AutomaticEmbeddingsSelector>,
    ) -> Self {
        Self {
            id,
            description,
            write_api_key,
            read_api_key,
            runtime_config: RwLock::new(CollectionRuntimeConfig {
                default_locale,
                embeddings_model,
            }),
            embedding_sender,
            indexes: Default::default(),
            temp_indexes: Default::default(),
            op_sender,
            hook_runtime,
            nlp_service,
            automatic_embeddings_selector,

            created_at: Utc::now(),
        }
    }

    pub async fn try_load(
        path: PathBuf,
        hooks_runtime: Arc<HooksRuntime>,
        nlp_service: Arc<NLPService>,
        embedding_sender: Sender<MultiEmbeddingCalculationRequest>,
        hook_runtime: Arc<HooksRuntime>,
        op_sender: OperationSender,
        automatic_embeddings_selector: Arc<AutomaticEmbeddingsSelector>,
    ) -> Result<Self> {
        let dump: CollectionDump = BufferedFile::open(path.join("info.json"))
            .context("Cannot open info.json file")?
            .read_json_data()
            .context("Cannot deserialize collection info")?;

        let CollectionDump::V1(dump) = dump;

        let id = dump.id;
        let description = dump.description;
        let write_api_key =
            ApiKey::try_new(dump.write_api_key).context("Cannot create write api key")?;
        let read_api_key =
            ApiKey::try_new(dump.read_api_key).context("Cannot create read api key")?;
        let default_locale = dump.default_locale;

        let mut indexes = HashMap::new();
        for index_id in dump.indexes {
            let index = Index::try_load(
                id,
                index_id,
                path.join("indexes").join(index_id.as_str()),
                op_sender.clone(),
                hooks_runtime.clone(),
                embedding_sender.clone(),
                hooks_runtime.clone(),
                automatic_embeddings_selector.clone(),
            )
            .context("Cannot load index")?;
            indexes.insert(index_id, index);
        }
        let mut temp_indexes = HashMap::new();
        for temp_index_id in dump.temporary_indexes {
            let index = Index::try_load(
                id,
                temp_index_id,
                path.join("temp_indexes").join(temp_index_id.as_str()),
                op_sender.clone(),
                hooks_runtime.clone(),
                embedding_sender.clone(),
                hooks_runtime.clone(),
                automatic_embeddings_selector.clone(),
            )
            .context("Cannot load index")?;
            temp_indexes.insert(temp_index_id, index);
        }

        Ok(Self {
            id,
            description,
            write_api_key,
            read_api_key,
            runtime_config: RwLock::new(CollectionRuntimeConfig {
                default_locale,
                embeddings_model: dump.embeddings_model.0,
            }),
            embedding_sender,
            indexes: RwLock::new(indexes),
            temp_indexes: RwLock::new(temp_indexes),
            op_sender,
            hook_runtime,
            nlp_service,
            automatic_embeddings_selector,

            created_at: dump.created_at,
        })
    }

    pub async fn commit(&mut self, data_dir: PathBuf) -> Result<()> {
        info!(coll_id= ?self.id, "Committing collection");

        std::fs::create_dir_all(&data_dir).context("Cannot create collection directory")?;

        let indexes_lock = self.indexes.get_mut();
        let temp_indexes_lock = self.temp_indexes.get_mut();

        let indexes = indexes_lock.keys().copied().collect::<Vec<_>>();
        let indexes_path = data_dir.join("indexes");
        for (id, index) in indexes_lock.iter_mut() {
            index.commit(indexes_path.join(id.as_str())).await?;
        }

        let temporary_indexes = temp_indexes_lock.keys().copied().collect::<Vec<_>>();
        let indexes_path = data_dir.join("temp_indexes");
        for (id, index) in temp_indexes_lock.iter_mut() {
            index.commit(indexes_path.join(id.as_str())).await?;
        }

        let runtime_config = self.runtime_config.read().await;
        let default_locale = runtime_config.default_locale;
        let embeddings_model = runtime_config.embeddings_model;
        drop(runtime_config);

        let dump = CollectionDump::V1(CollectionDumpV1 {
            id: self.id,
            description: self.description.clone(),
            write_api_key: self.write_api_key.expose().to_string(),
            read_api_key: self.read_api_key.expose().to_string(),
            default_locale,
            embeddings_model: OramaModelSerializable(embeddings_model),
            indexes,
            temporary_indexes,
            created_at: self.created_at,
        });

        BufferedFile::create_or_overwrite(data_dir.join("info.json"))
            .context("Cannot create info.json file")?
            .write_json_data(&dump)
            .context("Cannot serialize collection info")?;

        Ok(())
    }

    pub async fn get_index_ids(&self) -> Vec<IndexId> {
        let indexes = self.indexes.read().await;
        indexes.keys().copied().collect()
    }

    pub async fn create_index(
        &self,
        index_id: IndexId,
        embedding: IndexEmbeddingsCalculation,
    ) -> Result<()> {
        let mut indexes = self.indexes.write().await;
        if indexes.contains_key(&index_id) {
            bail!("Index with id {} already exists", index_id);
        }

        let runtime_config = self.runtime_config.read().await;
        let default_locale = runtime_config.default_locale;
        let embeddings_model = runtime_config.embeddings_model;
        drop(runtime_config);

        let index = Index::empty(
            index_id,
            self.id,
            self.embedding_sender.clone(),
            self.get_text_parser(default_locale),
            self.op_sender.clone(),
            self.hook_runtime.clone(),
            self.automatic_embeddings_selector.clone(),
        )
        .await
        .context("Cannot create index")?;

        self.op_sender
            .send(WriteOperation::Collection(
                self.id,
                CollectionWriteOperation::CreateIndex2 {
                    index_id,
                    locale: default_locale,
                },
            ))
            .await
            .context("Cannot send create index operation")?;

        index
            .add_embedding_field(
                field_name_to_path(DEFAULT_EMBEDDING_FIELD_NAME),
                embeddings_model,
                embedding,
            )
            .await?;

        indexes.insert(index_id, index);
        drop(indexes);

        Ok(())
    }

    pub async fn change_runtime_config(
        &self,
        new_default_locale: Locale,
        new_embeddings_model: OramaModel,
    ) {
        let mut runtime_config = self.runtime_config.write().await;
        runtime_config.default_locale = new_default_locale;
        runtime_config.embeddings_model = new_embeddings_model;
    }

    pub async fn create_temp_index(
        &self,
        copy_from: IndexId,
        new_index_id: IndexId,
        embedding: Option<IndexEmbeddingsCalculation>,
    ) -> Result<()> {
        let indexes_lock = self.indexes.write().await;
        let Some(copy_from_index) = indexes_lock.get(&copy_from) else {
            bail!("Index with id {} not found", copy_from);
        };
        if indexes_lock.contains_key(&new_index_id) {
            bail!("Index with id {} already exists", new_index_id);
        }

        // Use "copy_from" index embedding calculation as default
        let copy_from_index_embedding_calculation = copy_from_index
            .get_embedding_field(field_name_to_path(DEFAULT_EMBEDDING_FIELD_NAME))
            .await?;
        let embedding = embedding.unwrap_or(copy_from_index_embedding_calculation);

        let mut temp_indexes_lock = self.temp_indexes.write().await;

        let runtime_config = self.runtime_config.read().await;
        let default_locale = runtime_config.default_locale;
        let embeddings_model = runtime_config.embeddings_model;
        drop(runtime_config);

        let index = Index::empty(
            new_index_id,
            self.id,
            self.embedding_sender.clone(),
            self.get_text_parser(default_locale),
            self.op_sender.clone(),
            self.hook_runtime.clone(),
            self.automatic_embeddings_selector.clone(),
        )
        .await
        .context("Cannot create index")?;

        self.op_sender
            .send(WriteOperation::Collection(
                self.id,
                CollectionWriteOperation::CreateTemporaryIndex2 {
                    index_id: new_index_id,
                    locale: default_locale,
                },
            ))
            .await
            .context("Cannot send create index operation")?;

        index
            .add_embedding_field(
                field_name_to_path(DEFAULT_EMBEDDING_FIELD_NAME),
                embeddings_model,
                embedding,
            )
            .await?;

        temp_indexes_lock.insert(new_index_id, index);

        Ok(())
    }

    pub async fn delete_index(&self, index_id: IndexId) -> Result<Vec<DocumentId>> {
        let mut indexes = self.indexes.write().await;
        let index = match indexes.remove(&index_id) {
            Some(index) => index,
            None => {
                warn!(collection_id= ?self.id, "Index not found. Ignored");
                return Ok(vec![]);
            }
        };
        drop(indexes);

        self.op_sender
            .send(WriteOperation::Collection(
                self.id,
                CollectionWriteOperation::DeleteIndex2 { index_id },
            ))
            .await
            .context("Cannot send delete index operation")?;

        let doc_ids = index.get_document_ids().await;
        self.op_sender
            .send(WriteOperation::DocumentStorage(
                DocumentStorageWriteOperation::DeleteDocuments {
                    doc_ids: doc_ids.clone(),
                },
            ))
            .await
            .context("Cannot send delete documents operation")?;

        Ok(doc_ids)
    }

    pub async fn get_temporary_index<'s, 'index>(
        &'s self,
        id: IndexId,
    ) -> Option<IndexReadLock<'index>>
    where
        's: 'index,
    {
        let lock = self.temp_indexes.read().await;
        IndexReadLock::try_new(lock, id)
    }

    pub async fn promote_temp_index(&self, index_id: IndexId, temp_index: IndexId) -> Result<()> {
        let mut temp_index_lock = self.indexes.write().await;
        let mut index_lock = self.indexes.write().await;

        let temp_index = match temp_index_lock.remove(&temp_index) {
            Some(index) => index,
            None => bail!("Temporary index not found"),
        };
        match index_lock.insert(index_id, temp_index) {
            Some(_) => {
                info!(coll_id= ?self.id, "Index replaced");
            }
            None => {
                warn!(coll_id= ?self.id, "Index replaced but before it was not found. This is just a warning");
            }
        }
        drop(temp_index_lock);
        drop(index_lock);

        panic!();
    }

    pub async fn get_document_ids(&self) -> Vec<DocumentId> {
        let mut doc_id = vec![];
        let indexes = self.indexes.read().await;
        for (_, index) in indexes.iter() {
            let index_doc_ids = index.get_document_ids().await;
            doc_id.extend(index_doc_ids);
        }

        doc_id
    }

    pub fn check_write_api_key(&self, api_key: ApiKey) -> Result<()> {
        if self.write_api_key == api_key {
            Ok(())
        } else {
            Err(anyhow!("Invalid write api key"))
        }
    }

    pub async fn as_dto(&self) -> DescribeCollectionResponse {
        let mut indexes_desc = vec![];
        let mut document_count = 0_usize;
        let indexes = self.indexes.read().await;
        for index in indexes.values() {
            let index_desc = index.describe().await;
            document_count += index_desc.document_count;
            indexes_desc.push(index_desc);
        }
        drop(indexes);

        DescribeCollectionResponse {
            id: self.id,
            description: self.description.clone(),
            document_count,
            indexes: indexes_desc,
            created_at: self.created_at,
        }
    }

    pub async fn get_index<'s, 'index>(&'s self, id: IndexId) -> Option<IndexReadLock<'index>>
    where
        's: 'index,
    {
        let lock = self.indexes.read().await;
        if lock.contains_key(&id) {
            return IndexReadLock::try_new(lock, id);
        }
        drop(lock);
        let lock = self.temp_indexes.read().await;
        if lock.contains_key(&id) {
            return IndexReadLock::try_new(lock, id);
        }
        drop(lock);

        None
    }

    fn get_text_parser(&self, locale: Locale) -> Arc<TextParser> {
        self.nlp_service.get(locale)
    }

    pub async fn remove_from_fs(self, path: PathBuf) {
        match std::fs::remove_dir_all(&path) {
            Ok(_) => {}
            Err(e) => {
                warn!(coll_id= ?self.id, "Cannot remove collection directory. Ignored: {:?}", e);
            }
        };
    }

    pub async fn replace_index(
        &self,
        runtime_index_id: IndexId,
        temp_index_id: IndexId,
        reason: ReplaceIndexReason,
        reference: Option<String>,
    ) -> Result<()> {
        self.op_sender
            .send(WriteOperation::Collection(
                self.id,
                CollectionWriteOperation::ReplaceIndex {
                    reference,
                    runtime_index_id,
                    temp_index_id,
                    reason,
                },
            ))
            .await
            .context("Cannot send operation")?;

        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "version")]
enum CollectionDump {
    #[serde(rename = "1")]
    V1(CollectionDumpV1),
}

#[derive(Debug, Serialize, Deserialize)]
struct CollectionDumpV1 {
    id: CollectionId,
    description: Option<String>,
    write_api_key: String,
    read_api_key: String,
    default_locale: Locale,
    embeddings_model: OramaModelSerializable,
    indexes: Vec<IndexId>,
    temporary_indexes: Vec<IndexId>,
    created_at: DateTime<Utc>,
}

pub struct IndexReadLock<'guard> {
    lock: RwLockReadGuard<'guard, HashMap<IndexId, Index>>,
    id: IndexId,
}

impl<'guard> IndexReadLock<'guard> {
    pub fn try_new(
        lock: RwLockReadGuard<'guard, HashMap<IndexId, Index>>,
        id: IndexId,
    ) -> Option<Self> {
        let guard = lock.get(&id);
        match &guard {
            Some(_) => {
                let _ = guard;
                Some(Self { lock, id })
            }
            None => None,
        }
    }
}

impl Deref for IndexReadLock<'_> {
    type Target = Index;

    fn deref(&self) -> &Self::Target {
        // safety: the collection contains the id because we checked it before
        // no one can remove the collection from the map because we hold a read lock
        self.lock.get(&self.id).unwrap()
    }
}
