use std::{collections::HashMap, ops::Deref, path::PathBuf, sync::Arc};

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Utc};
use futures::FutureExt;
use oramacore_lib::hook_storage::HookWriter;
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, RwLockReadGuard};
use tracing::{info, warn};

use crate::{
    ai::OramaModel,
    collection_manager::sides::{
        field_name_to_path,
        write::{
            context::WriteSideContext, index::EnumStrategy, OramaModelSerializable, WriteError,
        },
        CollectionWriteOperation, DocumentStorageWriteOperation, ReplaceIndexReason,
        WriteOperation,
    },
    types::{
        ApiKey, CollectionId, DescribeCollectionResponse, DocumentId, IndexEmbeddingsCalculation,
        IndexId, WriteApiKey,
    },
};
use oramacore_lib::fs::BufferedFile;

use super::index::Index;
use crate::collection_manager::sides::IndexWriteOperation;
use oramacore_lib::nlp::{locales::Locale, TextParser};
use oramacore_lib::pin_rules::{PinRule, PinRuleOperation};

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

    context: WriteSideContext,

    indexes: RwLock<HashMap<IndexId, Index>>,
    temp_indexes: RwLock<HashMap<IndexId, Index>>,

    created_at: DateTime<Utc>,

    hook: HookWriter,
}

impl CollectionWriter {
    pub fn empty(
        data_dir: PathBuf,
        req: CreateEmptyCollection,
        context: WriteSideContext,
    ) -> Result<Self> {
        let id = req.id;

        let send_operation_cb_op_sender = context.op_sender.clone();
        let send_hook_operation_cb = Box::new(move |op| {
            let op_sender = send_operation_cb_op_sender.clone();
            async move {
                let _ = op_sender
                    .send(WriteOperation::Collection(
                        id,
                        CollectionWriteOperation::Hook(op),
                    ))
                    .await;
            }
            .boxed()
        });

        Ok(Self {
            id,
            description: req.description,
            write_api_key: req.write_api_key,
            read_api_key: req.read_api_key,
            runtime_config: RwLock::new(CollectionRuntimeConfig {
                default_locale: req.default_locale,
                embeddings_model: req.embeddings_model,
            }),
            context,

            indexes: Default::default(),
            temp_indexes: Default::default(),

            created_at: Utc::now(),

            hook: HookWriter::try_new(data_dir.join("hooks"), send_hook_operation_cb)
                .context("Cannot create hook writer")?,
        })
    }

    pub async fn try_load(data_dir: PathBuf, context: WriteSideContext) -> Result<Self> {
        let dump: CollectionDump = BufferedFile::open(data_dir.join("info.json"))
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
                data_dir.join("indexes").join(index_id.as_str()),
                context.clone(),
            )
            .context("Cannot load index")?;
            indexes.insert(index_id, index);
        }
        let mut temp_indexes = HashMap::new();
        for temp_index_id in dump.temporary_indexes {
            let index = Index::try_load(
                id,
                temp_index_id,
                data_dir.join("temp_indexes").join(temp_index_id.as_str()),
                context.clone(),
            )
            .context("Cannot load index")?;
            temp_indexes.insert(temp_index_id, index);
        }

        let hook_op_sender = context.op_sender.clone();

        Ok(Self {
            id,
            description,
            write_api_key,
            read_api_key,
            runtime_config: RwLock::new(CollectionRuntimeConfig {
                default_locale,
                embeddings_model: dump.embeddings_model.0,
            }),
            context,
            indexes: RwLock::new(indexes),
            temp_indexes: RwLock::new(temp_indexes),

            created_at: dump.created_at,

            hook: HookWriter::try_new(
                data_dir.join("hooks"),
                Box::new(move |op| {
                    let hook_op_sender = hook_op_sender.clone();
                    async move {
                        let _ = hook_op_sender
                            .send(WriteOperation::Collection(
                                id,
                                CollectionWriteOperation::Hook(op),
                            ))
                            .await;
                    }
                    .boxed()
                }),
            )
            .context("Cannot create hook writer")?,
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

    pub async fn get_temp_index_ids(&self) -> Vec<IndexId> {
        let temp_indexes = self.temp_indexes.read().await;
        temp_indexes.keys().copied().collect()
    }

    pub async fn create_index(
        &self,
        index_id: IndexId,
        embedding: IndexEmbeddingsCalculation,
        enum_strategy: EnumStrategy,
    ) -> Result<(), WriteError> {
        let mut indexes = self.indexes.write().await;
        if indexes.contains_key(&index_id) {
            return Err(WriteError::IndexAlreadyExists(self.id, index_id));
        }

        let runtime_config = self.runtime_config.read().await;
        let default_locale = runtime_config.default_locale;
        let embeddings_model = runtime_config.embeddings_model;
        drop(runtime_config);

        let index = Index::empty(
            index_id,
            self.id,
            None,
            self.get_text_parser(default_locale),
            self.context.clone(),
            enum_strategy,
        )
        .await
        .context("Cannot create index")?;

        self.context
            .op_sender
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

    pub async fn create_temp_index(
        &self,
        copy_from: IndexId,
        new_index_id: IndexId,
        embedding: Option<IndexEmbeddingsCalculation>,
    ) -> Result<(), WriteError> {
        let indexes_lock = self.indexes.write().await;
        let Some(copy_from_index) = indexes_lock.get(&copy_from) else {
            return Err(WriteError::IndexNotFound(self.id, copy_from));
        };
        if indexes_lock.contains_key(&new_index_id) {
            return Err(WriteError::IndexAlreadyExists(self.id, new_index_id));
        }

        // Use "copy_from" index embedding calculation as default
        let copy_from_index_embedding_calculation = copy_from_index
            .get_embedding_field(field_name_to_path(DEFAULT_EMBEDDING_FIELD_NAME))
            .await?;
        let embedding = embedding.unwrap_or(copy_from_index_embedding_calculation);

        let mut temp_indexes_lock = self.temp_indexes.write().await;

        if temp_indexes_lock.contains_key(&new_index_id) {
            return Err(WriteError::IndexAlreadyExists(self.id, new_index_id));
        }

        let runtime_config = self.runtime_config.read().await;
        let default_locale = runtime_config.default_locale;
        let embeddings_model = runtime_config.embeddings_model;
        drop(runtime_config);

        let index = Index::empty(
            new_index_id,
            self.id,
            Some(copy_from),
            self.get_text_parser(default_locale),
            self.context.clone(),
            copy_from_index.get_enum_strategy(),
        )
        .await
        .context("Cannot create index")?;

        self.context
            .op_sender
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

    pub async fn change_runtime_config(
        &self,
        new_default_locale: Locale,
        new_embeddings_model: OramaModel,
    ) {
        let mut runtime_config = self.runtime_config.write().await;
        runtime_config.default_locale = new_default_locale;
        runtime_config.embeddings_model = new_embeddings_model;
    }

    pub async fn delete_index(&self, index_id: IndexId) -> Result<Vec<DocumentId>, WriteError> {
        let mut indexes = self.indexes.write().await;
        let index = match indexes.remove(&index_id) {
            Some(index) => index,
            None => {
                warn!(collection_id= ?self.id, "Index not found. Ignored");
                return Ok(vec![]);
            }
        };
        drop(indexes);

        let doc_ids = index.get_document_ids().await;
        self.context
            .op_sender
            .send_batch(vec![
                WriteOperation::Collection(
                    self.id,
                    CollectionWriteOperation::DeleteIndex2 { index_id },
                ),
                WriteOperation::DocumentStorage(DocumentStorageWriteOperation::DeleteDocuments {
                    doc_ids: doc_ids.clone(),
                }),
            ])
            .await
            .context("Cannot send delete index operation")?;

        Ok(doc_ids)
    }

    pub async fn delete_temp_index(&self, temp_index_id: IndexId) -> Result<(), WriteError> {
        let mut temp_indexes = self.temp_indexes.write().await;
        let temp_index = match temp_indexes.remove(&temp_index_id) {
            Some(index) => index,
            None => {
                warn!(collection_id= ?self.id, "Temp index not found. Ignored");
                return Ok(());
            }
        };
        drop(temp_indexes);

        let doc_ids = temp_index.get_document_ids().await;
        self.context
            .op_sender
            .send_batch(vec![
                WriteOperation::Collection(
                    self.id,
                    CollectionWriteOperation::DeleteTempIndex { temp_index_id },
                ),
                WriteOperation::DocumentStorage(DocumentStorageWriteOperation::DeleteDocuments {
                    doc_ids,
                }),
            ])
            .await
            .context("Cannot send delete temp index operation")?;

        Ok(())
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

    pub async fn check_write_api_key(&self, api_key: WriteApiKey) -> Result<(), WriteError> {
        match api_key {
            WriteApiKey::ApiKey(api_key) => {
                if self.write_api_key != api_key {
                    return Err(WriteError::InvalidWriteApiKey(self.id));
                }
            }
            WriteApiKey::Claims(claim) => {
                if claim.sub != self.id {
                    return Err(WriteError::JwtBelongToAnotherCollection(self.id));
                }
            }
        };

        Ok(())
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

        let temp_indexs = self.temp_indexes.read().await;
        for index in temp_indexs.values() {
            let index_desc = index.describe().await;
            document_count += index_desc.document_count;
            indexes_desc.push(index_desc);
        }
        drop(temp_indexs);

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
        self.context.nlp_service.get(locale)
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
    ) -> Result<(), WriteError> {
        let indexes_lock = self.indexes.read().await;
        if !indexes_lock.contains_key(&runtime_index_id) {
            return Err(WriteError::IndexNotFound(self.id, runtime_index_id));
        }
        drop(indexes_lock);

        info!(coll_id= ?self.id, "Replacing index {} with temporary index {}", runtime_index_id, temp_index_id);

        let mut temp_indexes_lock = self.temp_indexes.write().await;
        let mut temp_index = match temp_indexes_lock.remove(&temp_index_id) {
            Some(index) => index,
            None => return Err(WriteError::TempIndexNotFound(self.id, temp_index_id)),
        };
        temp_index.set_index_id(runtime_index_id);
        let mut indexes_lock = self.indexes.write().await;
        match indexes_lock.insert(runtime_index_id, temp_index) {
            Some(_) => {
                info!(coll_id= ?self.id, "Index {} replaced with temporary index {}", runtime_index_id, temp_index_id);
            }
            None => {
                warn!(coll_id= ?self.id, "Index {} replaced with temporary index {} but before it was not found. This is just a warning", runtime_index_id, temp_index_id);
            }
        }
        drop(temp_indexes_lock);

        self.context
            .op_sender
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

    pub fn get_hook_storage(&self) -> &HookWriter {
        &self.hook
    }

    pub async fn insert_pin_rule(
        &self,
        index_id: IndexId,
        rule: PinRule<String>,
    ) -> Result<(), WriteError> {
        let Some(index) = self.get_index(index_id).await else {
            return Err(WriteError::IndexNotFound(self.id, index_id));
        };
        let mut pin_rule_writer = index.get_write_pin_rule_writer().await;
        pin_rule_writer.insert_pin_rule(rule.clone()).await?;

        let storage = index.get_document_id_storage().await;
        let new_rule = rule
            .convert_ids(|id| {
                // DocumentId(u64::MAX) is never used, so this is equal to say
                // "ignore this rules"
                storage.get(&id).unwrap_or(DocumentId(u64::MAX))
            })
            .await;

        self.context
            .op_sender
            .send(WriteOperation::Collection(
                self.id,
                CollectionWriteOperation::IndexWriteOperation(
                    index_id,
                    IndexWriteOperation::PinRule(PinRuleOperation::Insert(new_rule)),
                ),
            ))
            .await?;

        Ok(())
    }

    pub async fn delete_pin_rule(
        &self,
        index_id: IndexId,
        rule_id: String,
    ) -> Result<(), WriteError> {
        let Some(index) = self.get_index(index_id).await else {
            return Err(WriteError::IndexNotFound(self.id, index_id));
        };
        let mut pin_rule_writer = index.get_write_pin_rule_writer().await;
        pin_rule_writer.delete_pin_rule(&rule_id).await?;

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

pub struct CreateEmptyCollection {
    pub id: CollectionId,
    pub description: Option<String>,
    pub write_api_key: ApiKey,
    pub read_api_key: ApiKey,
    pub default_locale: Locale,
    pub embeddings_model: OramaModel,
}
