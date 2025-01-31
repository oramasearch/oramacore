mod collection;
mod collections;
mod embedding;
mod fields;
mod operation;

use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

use super::hooks::{HookName, HooksRuntime};

use anyhow::{Context, Result};
use duration_str::deserialize_duration;
use serde::{Deserialize, Serialize};
use tokio::{sync::RwLock, time::MissedTickBehavior};
use tracing::{info, instrument, trace, warn};

use collections::CollectionsWriter;
use embedding::{start_calculate_embedding_loop, EmbeddingCalculationRequest};
pub use operation::*;

#[cfg(any(test, feature = "benchmarking"))]
pub use fields::*;

use crate::{
    ai::AIService,
    collection_manager::dto::{ApiKey, CollectionDTO, CreateCollection, DeleteDocuments},
    file_utils::BufferedFile,
    metrics::{
        AddedDocumentsLabels, DocumentProcessLabels, ADDED_DOCUMENTS_COUNTER,
        DOCUMENT_PROCESS_METRIC,
    },
    nlp::NLPService,
    types::{CollectionId, DocumentId, DocumentList},
    SideChannelType,
};

#[derive(Debug, Deserialize, Clone)]
pub struct CollectionsWriterConfig {
    pub data_dir: PathBuf,
    #[serde(default = "embedding_queue_limit_default")]
    pub embedding_queue_limit: u32,
    #[serde(default = "embedding_model_default")]
    pub default_embedding_model: OramaModelSerializable,
    #[serde(default = "default_insert_batch_commit_size")]
    pub insert_batch_commit_size: u64,
    #[serde(default = "javascript_queue_limit_default")]
    pub javascript_queue_limit: u32,
    #[serde(deserialize_with = "deserialize_duration")]
    pub commit_interval: Duration,
}

#[derive(Debug, Deserialize, Clone)]
pub struct WriteSideConfig {
    pub master_api_key: ApiKey,
    pub output: SideChannelType,
    pub config: CollectionsWriterConfig,
}

pub struct WriteSide {
    sender: OperationSender,
    collections: CollectionsWriter,
    document_count: AtomicU64,
    data_dir: PathBuf,
    hook_runtime: Arc<HooksRuntime>,
    nlp_service: Arc<NLPService>,

    operation_counter: RwLock<u64>,
    insert_batch_commit_size: u64,

    master_api_key: ApiKey,

    commit_interval: Duration,
}

impl WriteSide {
    pub fn new(
        sender: OperationSender,
        config: WriteSideConfig,
        ai_service: Arc<AIService>,
        hook_runtime: Arc<HooksRuntime>,
        nlp_service: Arc<NLPService>,
    ) -> WriteSide {
        let master_api_key = config.master_api_key;
        let collections_writer_config = config.config;
        let data_dir = collections_writer_config.data_dir.clone();

        let insert_batch_commit_size = collections_writer_config.insert_batch_commit_size;

        let (sx, rx) = tokio::sync::mpsc::channel::<EmbeddingCalculationRequest>(
            collections_writer_config.embedding_queue_limit as usize,
        );

        start_calculate_embedding_loop(
            ai_service,
            rx,
            collections_writer_config.embedding_queue_limit,
        );

        let commit_interval = collections_writer_config.commit_interval;

        WriteSide {
            sender,
            collections: CollectionsWriter::new(collections_writer_config, sx),
            document_count: AtomicU64::new(0),
            data_dir,
            hook_runtime,
            nlp_service,

            operation_counter: Default::default(),
            insert_batch_commit_size,

            master_api_key,

            commit_interval,
        }
    }

    pub async fn load(mut self) -> Result<Arc<Self>> {
        self.collections
            .load(self.hook_runtime.clone(), self.nlp_service.clone())
            .await?;

        let info: WriteSideInfo = match BufferedFile::open(self.data_dir.join("info.json"))
            .and_then(|f| f.read_json_data())
            .context("Cannot read info file")
        {
            Ok(info) => info,
            Err(err) => {
                warn!("Cannot read info file: {}. Skip loading", err);
                let s = Arc::new(self);

                s.clone().start_commit_loop(s.commit_interval);

                return Ok(s);
            }
        };
        let WriteSideInfo::V1(info) = info;

        self.document_count
            .store(info.document_count, Ordering::Relaxed);
        self.sender.set_offset(info.offset);

        let s = Arc::new(self);

        s.clone().start_commit_loop(s.commit_interval);

        Ok(s)
    }

    fn start_commit_loop(self: Arc<Self>, insert_batch_commit_size: Duration) {
        tokio::task::spawn(async move {
            let mut interval = tokio::time::interval(insert_batch_commit_size);

            // If for some reason we miss a tick, we skip it.
            // In fact, the commit is blocked only by `update` method.
            // If the collection is under heavy load,
            // the commit will be run due to the `insert_batch_commit_size` config.
            interval.set_missed_tick_behavior(MissedTickBehavior::Skip);

            loop {
                interval.tick().await;
                info!(
                    "{:?} time reached. Committing write side",
                    insert_batch_commit_size.clone()
                );
                if let Err(e) = self.commit().await {
                    tracing::error!(?e, "Cannot commit write side");
                }
            }
        });
    }

    #[instrument(skip(self))]
    pub async fn commit(&self) -> Result<()> {
        info!("Committing write side");

        let offset = self.sender.offset();

        self.collections.commit().await?;

        // This load is not atomic with the commit.
        // This means, we save a document count possible higher.
        // Anyway it is not a problem, because the document count is only used for the document id generation
        // So, if something goes wrong, we save an higher number, and this is ok.
        let document_count = self.document_count.load(Ordering::Relaxed);
        let info = WriteSideInfo::V1(WriteSideInfoV1 {
            document_count,
            offset,
        });
        BufferedFile::create_or_overwrite(self.data_dir.join("info.json"))
            .context("Cannot create info file")?
            .write_json_data(&info)
            .context("Cannot write info file")?;

        Ok(())
    }

    pub async fn create_collection(
        &self,
        master_api_key: ApiKey,
        option: CreateCollection,
    ) -> Result<()> {
        if self.master_api_key != master_api_key {
            return Err(anyhow::anyhow!("Invalid master api key"));
        }

        self.collections
            .create_collection(option, self.sender.clone(), self.hook_runtime.clone())
            .await?;

        Ok(())
    }

    pub async fn write(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
        document_list: DocumentList,
    ) -> Result<()> {
        let document_count = document_list.len();
        info!(?document_count, "Inserting batch of documents");

        // This counter is not atomic with the insert operation.
        // This means we increment the counter even if the insert operation fails.
        // TODO: think better
        ADDED_DOCUMENTS_COUNTER
            .create(AddedDocumentsLabels {
                collection: collection_id.0.clone(),
            })
            .increment_by(document_list.len());

        let collection = self
            .collections
            .get_collection(collection_id.clone())
            .await
            .ok_or_else(|| anyhow::anyhow!("Collection not found"))?;

        collection.check_write_api_key(write_api_key)?;

        let sender = self.sender.clone();

        for mut doc in document_list {
            info!("Insert doc");
            let m = DOCUMENT_PROCESS_METRIC.create(DocumentProcessLabels {
                collection: collection_id.0.clone(),
            });

            let doc_id = self.document_count.fetch_add(1, Ordering::Relaxed);

            let doc_id_value = doc.get("id");
            // Forces the id to be set, if not set
            if doc_id_value.is_none() {
                doc.inner.insert(
                    "id".to_string(),
                    serde_json::Value::String(cuid2::create_id()),
                );
            } else if let Some(doc_id_value) = doc_id_value {
                if !doc_id_value.is_string() {
                    // The search result contains the document id and it is defined as a string.
                    // So, if the original document id is not a string, we should overwrite it with a new one
                    // Anyway, this implies the loss of the original document id. For instance we could support number as well
                    // TODO: think better
                    warn!("Document id is not a string, overwriting it with new one");
                    doc.inner.insert(
                        "id".to_string(),
                        serde_json::Value::String(cuid2::create_id()),
                    );
                }
            }

            let doc_id = DocumentId(doc_id);
            info!(?doc_id, "Inserting document");
            collection
                .process_new_document(doc_id, doc, sender.clone(), self.hook_runtime.clone())
                .await
                .context("Cannot process document")?;
            info!("Document inserted");

            drop(m);

            info!("Doc inserted");
        }

        let mut lock = self.operation_counter.write().await;
        *lock += document_count as u64;
        let should_commit = if *lock >= self.insert_batch_commit_size {
            *lock = 0;
            true
        } else {
            false
        };
        drop(lock);

        if should_commit {
            info!(insert_batch_commit_size=?self.insert_batch_commit_size, "insert_batch_commit_size reached, committing");
            self.commit().await?;
        } else {
            trace!(insert_batch_commit_size=?self.insert_batch_commit_size, "insert_batch_commit_size not reached, not committing");
        }

        info!("Batch of documents inserted");

        Ok(())
    }

    pub async fn delete_documents(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
        document_ids_to_delete: DeleteDocuments,
    ) -> Result<()> {
        let collection = self
            .collections
            .get_collection(collection_id.clone())
            .await
            .context("Collection not found")?;

        collection.check_write_api_key(write_api_key)?;

        collection
            .delete_documents(document_ids_to_delete, self.sender.clone())
            .await?;

        Ok(())
    }

    pub async fn insert_javascript_hook(
        &self,
        collection_id: CollectionId,
        name: HookName,
        code: String,
    ) -> Result<()> {
        self.hook_runtime
            .insert_hook(collection_id.clone(), name.clone(), code)
            .context("Cannot insert hook")?;

        let collection = self
            .collections
            .get_collection(collection_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("Collection not found"))?;

        collection
            .set_embedding_hook(name)
            .await
            .context("Cannot set embedding hook")?;

        Ok(())
    }

    pub async fn list_collections(&self) -> Vec<CollectionDTO> {
        self.collections.list().await
    }

    pub async fn get_collection_dto(&self, collection_id: CollectionId) -> Option<CollectionDTO> {
        let collection = self.collections.get_collection(collection_id).await?;
        Some(collection.as_dto().await)
    }

    pub fn get_javascript_hook(
        &self,
        collection_id: CollectionId,
        name: HookName,
    ) -> Option<String> {
        self.hook_runtime
            .get_hook(collection_id, name)
            .map(|hook| hook.code)
    }

    pub fn delete_javascript_hook(
        &self,
        _collection_id: CollectionId,
        _name: HookName,
    ) -> Option<String> {
        None // @todo: implement delete hook in HooksRuntime and CollectionsWriter
    }

    pub fn list_javascript_hooks(&self, collection_id: CollectionId) -> HashMap<HookName, String> {
        self.hook_runtime
            .list_hooks(collection_id)
            .into_iter()
            .map(|(name, hook)| (name, hook.code))
            .collect()
    }
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "version")]
enum WriteSideInfo {
    #[serde(rename = "1")]
    V1(WriteSideInfoV1),
}

#[derive(Serialize, Deserialize)]
struct WriteSideInfoV1 {
    document_count: u64,
    offset: Offset,
}

fn embedding_queue_limit_default() -> u32 {
    50
}
fn javascript_queue_limit_default() -> u32 {
    50
}

fn embedding_model_default() -> OramaModelSerializable {
    OramaModelSerializable(crate::ai::OramaModel::BgeSmall)
}

fn default_insert_batch_commit_size() -> u64 {
    1_000
}
