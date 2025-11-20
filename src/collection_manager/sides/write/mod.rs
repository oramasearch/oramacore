pub mod collection;
mod collections;
pub mod document_storage;
mod embedding;
pub mod index;
use oramacore_lib::hook_storage::{HookWriter, HookWriterError};
use oramacore_lib::nlp::NLPService;
use thiserror::Error;
mod context;
pub mod jwt_manager;

use std::borrow::Cow;
use std::collections::HashSet;
use std::{
    collections::HashMap,
    ops::Deref,
    path::PathBuf,
    sync::{
        atomic::{AtomicU32, AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

use super::{
    system_prompts::SystemPromptInterface, Offset, OperationSender, OperationSenderCreator,
    OutputSideChannelType,
};

use anyhow::{Context, Result};
use document_storage::DocumentStorage;
use duration_str::deserialize_duration;
use futures::FutureExt;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio::time::{sleep, Instant, MissedTickBehavior};
use tokio_stream::StreamExt;
use tracing::{debug, error, info, trace, warn};

pub use collections::CollectionReadLock;
use collections::CollectionsWriter;
use embedding::{start_calculate_embedding_loop, MultiEmbeddingCalculationRequest};

pub use context::WriteSideContext;

use crate::collection_manager::sides::write::document_storage::ZeboDocument;
use crate::lock::OramaAsyncLock;
use crate::metrics::CollectionLabels;
use crate::python::embeddings::Model;
use crate::python::PythonService;
use crate::{
    ai::{
        automatic_embeddings_selector::AutomaticEmbeddingsSelector,
        llms::LLMService,
        tools::{CollectionToolsRuntime, ToolsRuntime},
        training_sets::TrainingSetInterface,
    },
    collection_manager::sides::{
        system_prompts::CollectionSystemPromptsInterface,
        write::jwt_manager::{JwtConfig, JwtManager},
        DocumentStorageWriteOperation, DocumentToInsert, ReplaceIndexReason, WriteOperation,
    },
    metrics::document_insertion::DOCUMENTS_INSERTION_TIME,
    types::{
        ApiKey, CollectionCreated, CollectionId, CreateCollection, CreateIndexRequest,
        DeleteDocuments, DescribeCollectionResponse, Document, DocumentId, DocumentList,
        IndexEmbeddingsCalculation, IndexId, InsertDocumentsResult, LanguageDTO,
        ReplaceIndexRequest, UpdateDocumentRequest, UpdateDocumentsResult, WriteApiKey,
    },
};
use oramacore_lib::fs::BufferedFile;
use oramacore_lib::generic_kv::{KVConfig, KVWriteOperation, KV};
use oramacore_lib::pin_rules::PinRulesWriterError;

#[derive(Error, Debug)]
pub enum WriteError {
    #[error("Invalid master API key")]
    InvalidMasterApiKey,
    #[error("Generic error: {0}")]
    Generic(#[from] anyhow::Error),
    #[error("Collection already exists: {0}")]
    CollectionAlreadyExists(CollectionId),
    #[error("Invalid write api key for: {0}")]
    InvalidWriteApiKey(CollectionId),
    #[error("JWT belong to another collection. Wanted: {0}")]
    JwtBelongToAnotherCollection(CollectionId),
    #[error("Collection not found: {0}")]
    CollectionNotFound(CollectionId),
    #[error("Index {1} already exists in collection {0}")]
    IndexAlreadyExists(CollectionId, IndexId),
    #[error("Index {1} doesn't exist in collection {0}")]
    IndexNotFound(CollectionId, IndexId),
    #[error("Temp index {1} doesn't exist in collection {0}")]
    TempIndexNotFound(CollectionId, IndexId),
    #[error("Error in hook")]
    HookWriterError(#[from] HookWriterError),
    #[error("Error in pin rule")]
    PinRulesError(#[from] PinRulesWriterError),
    #[error("Document limit exceeded for collection {0}. Limit: {1}")]
    DocumentLimitExceeded(CollectionId, usize),
}

#[derive(Debug, Deserialize, Clone)]
pub struct TempIndexCleanupConfig {
    #[serde(deserialize_with = "deserialize_duration")]
    pub cleanup_interval: Duration,
    #[serde(deserialize_with = "deserialize_duration")]
    pub max_age: Duration,
}

#[derive(Debug, Deserialize, Clone)]
pub struct CollectionsWriterConfig {
    pub data_dir: PathBuf,
    #[serde(default = "embedding_queue_limit_default")]
    pub embedding_queue_limit: u32,
    #[serde(default = "embedding_model_default")]
    pub default_embedding_model: Model,
    #[serde(default = "default_insert_batch_commit_size")]
    pub insert_batch_commit_size: u64,
    #[serde(default = "javascript_queue_limit_default")]
    pub javascript_queue_limit: usize,
    #[serde(deserialize_with = "deserialize_duration")]
    pub commit_interval: Duration,
    #[serde(default = "default_temp_index_cleanup_config")]
    pub temp_index_cleanup: TempIndexCleanupConfig,
}

#[derive(Deserialize, Clone)]
pub struct WriteSideConfig {
    pub master_api_key: ApiKey,
    // pub hooks: HooksRuntimeConfig,
    pub output: OutputSideChannelType,
    pub config: CollectionsWriterConfig,
    pub jwt: Option<JwtConfig>,
}

pub struct WriteSide {
    op_sender: OperationSender,
    collections: CollectionsWriter,
    document_count: AtomicU64,
    data_dir: PathBuf,
    operation_counter: OramaAsyncLock<u64>,
    insert_batch_commit_size: u64,

    document_storage: DocumentStorage,
    system_prompts: SystemPromptInterface,
    training_sets: TrainingSetInterface,
    tools: ToolsRuntime,
    kv: Arc<KV>,
    master_api_key: ApiKey,

    context: WriteSideContext,

    stop_sender: tokio::sync::broadcast::Sender<()>,
    stop_done_receiver: OramaAsyncLock<tokio::sync::mpsc::Receiver<()>>,

    // This counter is incremented each time we need to
    // change the collections hashmap:
    // - when we create a new collection
    // - when we delete a collection
    // After the operation, we decrement the counter
    // DUring the documents insertion, we can use this counter
    // to know if there're any changes that require to have
    // write lock on collections hashmap
    // In that case, we can pause the insertion for a while,
    // allowing other operations to obtain the write lock,
    // and then we can continue the insertion.
    write_operation_counter: AtomicU32,

    jwt_manager: JwtManager,

    python_service: Arc<PythonService>,
}

impl WriteSide {
    pub async fn try_load(
        op_sender_creator: OperationSenderCreator,
        config: WriteSideConfig,
        nlp_service: Arc<NLPService>,
        llm_service: Arc<LLMService>,
        automatic_embeddings_selector: Arc<AutomaticEmbeddingsSelector>,
        python_service: Arc<PythonService>,
    ) -> Result<Arc<Self>> {
        let master_api_key = config.master_api_key;
        let collections_writer_config = config.config;
        let data_dir = collections_writer_config.data_dir.clone();

        let insert_batch_commit_size = collections_writer_config.insert_batch_commit_size;
        let temp_index_cleanup_config = collections_writer_config.temp_index_cleanup.clone();

        let commit_interval = collections_writer_config.commit_interval;
        let embedding_queue_limit = collections_writer_config.embedding_queue_limit;

        let (sx, rx) = tokio::sync::mpsc::channel::<MultiEmbeddingCalculationRequest>(
            embedding_queue_limit as usize,
        );

        let document_count = AtomicU64::new(0);

        let write_side_info_path = data_dir.join("info.json");
        let r = BufferedFile::open(write_side_info_path)
            .and_then(|f| f.read_json_data::<WriteSideInfo>());

        let op_sender = if let Ok(info) = r {
            let WriteSideInfo::V1(info) = info;
            document_count.store(info.document_count, Ordering::Relaxed);
            op_sender_creator
                .create(info.offset)
                .await
                .context("Cannot create sender")?
        } else {
            op_sender_creator
                .create(Offset(0))
                .await
                .context("Cannot create sender")?
        };

        let context = WriteSideContext {
            python_service: python_service.clone(),
            embedding_sender: sx,
            op_sender: op_sender.clone(),
            nlp_service,
            automatic_embeddings_selector,
            llm_service,
        };

        let kv_op_sender = context.op_sender.clone();
        let kv_operation_cb = Box::new(move |op: KVWriteOperation| {
            let op_sender = kv_op_sender.clone();
            async move {
                let _ = op_sender.send(WriteOperation::KV(op)).await;
            }
            .boxed()
        });
        let kv = KV::try_load(KVConfig {
            data_dir: data_dir.join("kv"),
            sender: Some(kv_operation_cb),
        })
        .context("Cannot load KV")?;
        let kv = Arc::new(kv);
        let system_prompts = SystemPromptInterface::new(kv.clone(), context.llm_service.clone());
        let tools = ToolsRuntime::new(kv.clone(), context.llm_service.clone());

        let collections_writer =
            CollectionsWriter::try_load(collections_writer_config, context.clone())
                .await
                .context("Cannot load collections")?;

        let document_storage = DocumentStorage::try_new(data_dir.join("documents"))
            .await
            .context("Cannot create document storage")?;

        let (stop_done_sender, stop_done_receiver) = tokio::sync::mpsc::channel(1);
        let (stop_sender, _) = tokio::sync::broadcast::channel(1);
        let commit_loop_receiver = stop_sender.subscribe();
        let temp_index_cleanup_receiver = stop_sender.subscribe();
        let receive_operation_loop_receiver = stop_sender.subscribe();

        let jwt_manager = JwtManager::new(config.jwt)
            .await
            .context("Cannot create jwt_manager")?;

        let training_sets = TrainingSetInterface::new(kv.clone());

        let write_side = Self {
            document_count,
            collections: collections_writer,
            document_storage,
            data_dir,
            insert_batch_commit_size,
            master_api_key,
            operation_counter: OramaAsyncLock::new("operation_counter", Default::default()),
            op_sender: op_sender.clone(),
            system_prompts,
            training_sets,
            tools,
            kv,
            context: context.clone(),
            stop_sender,
            stop_done_receiver: OramaAsyncLock::new("stop_done_receiver", stop_done_receiver),
            write_operation_counter: AtomicU32::new(0),
            jwt_manager,
            python_service,
        };

        let write_side = Arc::new(write_side);

        start_commit_loop(
            write_side.clone(),
            commit_interval,
            commit_loop_receiver,
            stop_done_sender.clone(),
        );

        start_temp_index_cleanup_loop(
            write_side.clone(),
            temp_index_cleanup_config,
            temp_index_cleanup_receiver,
            stop_done_sender.clone(),
        );
        start_calculate_embedding_loop(
            context.python_service.clone(),
            rx,
            op_sender,
            embedding_queue_limit,
            receive_operation_loop_receiver,
            stop_done_sender.clone(),
        );

        Ok(write_side)
    }

    pub async fn cleanup_expired_temp_indexes(&self, max_age: Duration) -> Result<usize> {
        self.collections.cleanup_expired_temp_indexes(max_age).await
    }

    pub async fn stop(&self) -> Result<()> {
        info!("Stopping writer side");
        // Broadcast
        self.stop_sender
            .send(())
            .context("Cannot send stop signal")?;
        let mut stop_done_receiver = self.stop_done_receiver.write("stop").await;
        // Commit loop
        stop_done_receiver
            .recv()
            .await
            .context("Cannot send stop signal")?;
        // embedding calulcation loop
        stop_done_receiver
            .recv()
            .await
            .context("Cannot send stop signal")?;
        info!("Writer side stopped");

        Ok(())
    }

    pub async fn commit(&self) -> Result<()> {
        info!("Committing write side");

        let output = tokio::join!(self.collections.commit(), self.kv.commit());
        output.0.context("Cannot commit collections")?;
        output.1.context("Cannot commit kv")?;

        let offset = self.op_sender.get_offset();
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
    ) -> Result<CollectionCreated, WriteError> {
        self.check_master_api_key(master_api_key)?;

        let collection_id = option.id;

        self.write_operation_counter.fetch_add(1, Ordering::Relaxed);
        let res = self
            .collections
            .create_collection(option, self.op_sender.clone())
            .await;
        self.write_operation_counter.fetch_sub(1, Ordering::Relaxed);

        res?;

        Ok(CollectionCreated { collection_id })
    }

    pub async fn update_collection_mcp_description(
        &self,
        write_api_key: WriteApiKey,
        collection_id: CollectionId,
        mcp_description: Option<String>,
    ) -> Result<(), WriteError> {
        // Verify the collection exists and we have write access
        let _collection = self.get_collection(collection_id, write_api_key).await?;

        self.write_operation_counter.fetch_add(1, Ordering::Relaxed);
        let res = self
            .collections
            .update_collection_mcp_description(
                collection_id,
                mcp_description,
                self.op_sender.clone(),
            )
            .await;
        self.write_operation_counter.fetch_sub(1, Ordering::Relaxed);

        res
    }

    pub async fn create_index(
        &self,
        write_api_key: WriteApiKey,
        collection_id: CollectionId,
        req: CreateIndexRequest,
    ) -> Result<(), WriteError> {
        let collection = self.get_collection(collection_id, write_api_key).await?;

        let CreateIndexRequest {
            index_id,
            embedding,
            type_strategy,
        } = req;

        let default_string_calculation = if cfg!(test) {
            IndexEmbeddingsCalculation::AllProperties
        } else {
            IndexEmbeddingsCalculation::Automatic
        };
        let embedding: IndexEmbeddingsCalculation = embedding.unwrap_or(default_string_calculation);

        collection
            .create_index(index_id, embedding, type_strategy.enum_strategy)
            .await?;

        Ok(())
    }

    pub async fn reindex(
        &self,
        write_api_key: WriteApiKey,
        collection_id: CollectionId,
        language: LanguageDTO,
        model: Model,
        reference: Option<String>,
    ) -> Result<(), WriteError> {
        // Get initial collection state and create temporary index
        let mut collection = self.get_collection(collection_id, write_api_key).await?;

        collection
            .change_runtime_config(language.into(), model)
            .await;

        info!("Reindexing collection {}", collection_id);

        let index_ids = collection.get_index_ids().await;
        for copy_from in index_ids {
            info!("Reindexing index {}", copy_from);
            let copy_from_index = collection.get_index(copy_from).await.ok_or_else(|| {
                // The data doesn't come from the user:
                // `copy_from` is an index id that we already have
                // If this happen, we have a bug in the code but the user cannot do anything
                // No typed error
                anyhow::anyhow!("Index not found")
            })?;

            let new_index_id = cuid2::create_id();
            let new_index_id = IndexId::try_new(new_index_id)
                // This is safe because cuid2 is less than 64 bytes
                .expect("Cannot create new index id");
            let document_ids = copy_from_index.get_document_ids().await;
            let document_count = document_ids.len();

            // Free the read lock, so we can add new index
            drop(copy_from_index);

            collection
                .create_temp_index(copy_from, new_index_id, None)
                .await
                .context("Cannot create temporary index")?;
            let mut new_index = collection
                .get_temporary_index(new_index_id)
                .await
                .expect("Temporary index not found but it should be there because already created");

            let mut index_operation_batch = Vec::with_capacity(document_count * 10);
            let mut processed_count = -1;

            let mut stream = self.document_storage.stream_documents(document_ids).await;
            while let Some((doc_id, doc)) = stream.next().await {
                processed_count += 1;
                if processed_count % 100 == 0 {
                    trace!("Processing document {}/{}", processed_count, document_count);
                }

                if processed_count % 50 == 0 && !index_operation_batch.is_empty() {
                    trace!("Sending operations before yielding");
                    self.op_sender
                        .send_batch(index_operation_batch)
                        .await
                        .context("Cannot send batch of operations")?;
                    index_operation_batch = Vec::with_capacity(document_count * 10);
                }

                // Check for pending write operations and yield if needed
                if self.write_operation_counter.load(Ordering::Relaxed) > 0 {
                    // Free resources...
                    drop(new_index);
                    drop(collection);

                    self.wait_for_pending_write_operations().await?;

                    collection = match self.collections.get_collection(collection_id).await {
                        Some(collection) => collection,
                        None => return Err(WriteError::CollectionNotFound(collection_id)),
                    };
                    new_index = match collection.get_index(new_index_id).await {
                        Some(index) => index,
                        None => return Err(WriteError::IndexNotFound(collection_id, new_index_id)),
                    };
                }

                let inner = doc.inner;
                let inner: Map<String, Value> = serde_json::from_str(inner.get())
                    // This is safe because the document is a valid JSON
                    .context("Cannot deserialize document")?;
                let doc = Document { inner };

                new_index
                    .add_fields_if_needed(&DocumentList(vec![doc.clone()]))
                    .await
                    .context("Cannot add fields if needed")?;

                if index_operation_batch.capacity() * 4 / 5 < index_operation_batch.len() {
                    trace!("Sending operations");
                    self.op_sender
                        .send_batch(index_operation_batch)
                        .await
                        .context("Cannot send batch of operations")?;
                    trace!("Operations sent");
                    index_operation_batch = Vec::with_capacity(document_count * 10);
                }

                match new_index
                    .reindex_document(doc_id, doc, &mut index_operation_batch)
                    .await
                    .context("Cannot process document during the reindexing. Ignore the error and continue")
                {
                    Ok(_) => {}
                    Err(e) => {
                        // Ignore the error and continue to process the rest of the documents
                        error!(error = ?e, "Cannot process document during the reindexing. Ignore the error and continue");
                    }
                };
            }

            if !index_operation_batch.is_empty() {
                trace!("Sending final operations");
                self.op_sender
                    .send_batch(index_operation_batch)
                    .await
                    .context("Cannot send batch of operations")?;
                trace!("Final operations sent");
            }

            let req = ReplaceIndexRequest {
                runtime_index_id: copy_from,
                temp_index_id: new_index_id,
                reference: reference.clone(),
            };

            drop(new_index);

            self.replace_index(
                write_api_key,
                collection_id,
                req,
                ReplaceIndexReason::CollectionReindexed,
            )
            .await
            .context("Cannot substitute index")?;

            info!("Index reindexed {}", copy_from);
        }

        Ok(())
    }

    pub async fn replace_index(
        &self,
        write_api_key: WriteApiKey,
        collection_id: CollectionId,
        req: ReplaceIndexRequest,
        reason: ReplaceIndexReason,
    ) -> Result<(), WriteError> {
        let collection = self.get_collection(collection_id, write_api_key).await?;

        let temp_index = match collection.get_temporary_index(req.temp_index_id).await {
            Some(temp_index) => temp_index,
            None => {
                return Err(WriteError::TempIndexNotFound(
                    collection_id,
                    req.temp_index_id,
                ))
            }
        };

        let index = match collection.get_index(req.runtime_index_id).await {
            Some(index) => index,
            None => {
                return Err(WriteError::IndexNotFound(
                    collection_id,
                    req.runtime_index_id,
                ))
            }
        };

        let temp_index_count = temp_index
            .get_document_count("replace_index_check_limits")
            .await;
        let prev_index_count = index.get_document_count("replace_index_check_limits").await;

        collection
            .check_claim_limitations(write_api_key, temp_index_count, prev_index_count)
            .await?;
        drop(temp_index);
        drop(index);

        let old = collection
            .replace_index(
                req.runtime_index_id,
                req.temp_index_id,
                reason,
                req.reference,
            )
            .await?;

        if let Some(old_index) = old {
            let document_ids = old_index.get_document_ids().await;
            self.document_storage.remove(document_ids).await;

            // Forward document deletions to the reader side
            let old_doc_ids = old_index.get_document_ids().await;
            self.context
                .op_sender
                .send(WriteOperation::DocumentStorage(
                    DocumentStorageWriteOperation::DeleteDocuments {
                        doc_ids: old_doc_ids,
                    },
                ))
                .await
                .context("Cannot send operation to delete old index documents")?;
        }

        Ok(())
    }

    pub async fn create_temp_index(
        &self,
        write_api_key: WriteApiKey,
        collection_id: CollectionId,
        copy_from: IndexId,
        req: CreateIndexRequest,
    ) -> Result<(), WriteError> {
        let collection = self.get_collection(collection_id, write_api_key).await?;

        let CreateIndexRequest {
            index_id: new_index_id,
            embedding,
            ..
        } = req;

        collection
            .create_temp_index(copy_from, new_index_id, embedding)
            .await?;

        Ok(())
    }

    pub async fn delete_index(
        &self,
        write_api_key: WriteApiKey,
        collection_id: CollectionId,
        index_id: IndexId,
    ) -> Result<(), WriteError> {
        let collection = self.get_collection(collection_id, write_api_key).await?;

        let document_to_remove = collection.delete_index(index_id).await?;

        self.document_storage.remove(document_to_remove).await;

        Ok(())
    }

    pub async fn insert_documents(
        &self,
        write_api_key: WriteApiKey,
        collection_id: CollectionId,
        index_id: IndexId,
        mut document_list: DocumentList,
    ) -> Result<InsertDocumentsResult, WriteError> {
        let document_count = document_list.len();
        info!(?document_count, "Inserting batch of documents");

        let collection = self.get_collection(collection_id, write_api_key).await?;

        let index = collection
            .get_index(index_id)
            .await
            .ok_or_else(|| WriteError::IndexNotFound(collection_id, index_id))?;

        // Constraint checks
        if index.is_runtime() {
            collection
                .check_claim_limitations(write_api_key, document_list.len(), 0)
                .await?;
        }

        // The doc_id_str is the composition of index_id + document_id
        // Anyway, for temp indexes, instead of using the temp index id,
        // we use the original index id
        let target_index_id = index.get_runtime_index_id().unwrap_or(index_id);

        let metric = DOCUMENTS_INSERTION_TIME.create(CollectionLabels {
            collection: collection_id.to_string(),
        });

        debug!("Inserting documents {}", document_count);
        let doc_ids = self
            .add_documents_to_storage(&mut document_list, target_index_id)
            .await
            .context("Cannot insert documents into document storage")?;
        debug!("Document inserted");

        debug!("Looking for new fields...");
        index
            .add_fields_if_needed(&document_list)
            .await
            .context("Cannot add fields if needed")?;
        debug!("Done");

        drop(index);
        drop(collection);

        debug!("Processing documents {}", document_count);
        let result = self
            .inner_process_documents(collection_id, index_id, document_list, doc_ids)
            .await
            .context("Cannot process documents")?;
        info!("All documents are inserted");

        drop(metric);

        let mut lock = self.operation_counter.write("insert_doc").await;
        **lock += document_count as u64;
        let should_commit = if **lock >= self.insert_batch_commit_size {
            **lock = 0;
            true
        } else {
            false
        };
        drop(lock);

        if should_commit {
            info!(insert_batch_commit_size=?self.insert_batch_commit_size, "insert_batch_commit_size reached, committing");
            self.commit().await?;
        }

        Ok(result)
    }

    pub async fn delete_documents(
        &self,
        write_api_key: WriteApiKey,
        collection_id: CollectionId,
        index_id: IndexId,
        document_ids_to_delete: DeleteDocuments,
    ) -> Result<(), WriteError> {
        let collection = self.get_collection(collection_id, write_api_key).await?;
        let index = collection
            .get_index(index_id)
            .await
            .ok_or_else(|| WriteError::IndexNotFound(collection_id, index_id))?;

        let doc_ids = index.delete_documents(document_ids_to_delete).await?;

        self.document_storage.remove(doc_ids).await;

        Ok(())
    }

    pub async fn update_documents(
        &self,
        write_api_key: WriteApiKey,
        collection_id: CollectionId,
        index_id: IndexId,
        update_document_request: UpdateDocumentRequest,
    ) -> Result<UpdateDocumentsResult, WriteError> {
        let mut collection = self.get_collection(collection_id, write_api_key).await?;
        let document_count = update_document_request.documents.len();

        let mut index = collection
            .get_index(index_id)
            .await
            .ok_or_else(|| WriteError::IndexNotFound(collection_id, index_id))?;

        // Constraint checks
        if index.is_runtime() {
            collection
                .check_claim_limitations(write_api_key, document_count, 0)
                .await?;
        }

        let target_index_id = index.get_runtime_index_id().unwrap_or(index_id);

        // Prepare document ID mapping
        let document_id_storage = index.get_document_id_storage().await;
        let document_ids_map: HashMap<_, _> = update_document_request
            .documents
            .0
            .iter()
            .filter_map(|d| {
                d.get("id")
                    .and_then(|v| v.as_str())
                    .and_then(|s| document_id_storage.get(s).map(|id| (id, s.to_string())))
            })
            .collect();
        let document_ids: Vec<_> = document_ids_map.keys().copied().collect();
        drop(document_id_storage);

        // Prepare documents with IDs
        let mut documents: HashMap<String, Document> = update_document_request
            .documents
            .into_iter()
            .map(|mut doc| {
                let doc_id_str = doc.get("id").and_then(|v| v.as_str());
                let doc_id_str = if let Some(doc_id_str) = doc_id_str {
                    if doc_id_str.is_empty() {
                        cuid2::create_id()
                    } else {
                        doc_id_str.to_string()
                    }
                } else {
                    cuid2::create_id()
                };
                // Ensure the document contains a valid id
                // NB: this overwrite the previous id if it is not a string
                // TODO: is it correct?
                doc.inner.insert(
                    "id".to_string(),
                    serde_json::Value::String(doc_id_str.clone()),
                );

                (doc_id_str, doc)
            })
            .collect();

        let mut pin_rules_writer = collection.get_pin_rule_writer("update_documents").await;
        let mut pin_rules_touched = HashSet::new();

        let mut result = UpdateDocumentsResult {
            inserted: 0,
            updated: 0,
            failed: 0,
        };

        let mut index_operation_batch = Vec::with_capacity(document_count * 10);
        let mut docs_to_remove = Vec::with_capacity(document_count);

        let mut doc_stream = self.document_storage.stream_documents(document_ids).await;
        let mut processed_count = -1;

        while let Some((doc_id, doc)) = doc_stream.next().await {
            processed_count += 1;
            if processed_count % 100 == 0 {
                trace!("Processing document {}/{}", processed_count, document_count);
            }

            if index_operation_batch.capacity() * 4 / 5 < index_operation_batch.len() {
                trace!("Sending operations");
                self.op_sender
                    .send_batch(index_operation_batch)
                    .await
                    .context("Cannot send index operation")?;
                trace!("Operations sent");
                index_operation_batch = Vec::with_capacity(document_count * 10);
            }

            // Check for pending write operations and yield if needed
            if self.write_operation_counter.load(Ordering::Relaxed) > 0 {
                drop(pin_rules_writer);
                drop(index);
                drop(collection);

                self.wait_for_pending_write_operations().await?;

                collection = match self.collections.get_collection(collection_id).await {
                    Some(collection) => collection,
                    None => return Err(WriteError::CollectionNotFound(collection_id)),
                };
                index = match collection.get_index(index_id).await {
                    Some(index) => index,
                    None => return Err(WriteError::IndexNotFound(collection_id, index_id)),
                };
                pin_rules_writer = collection.get_pin_rule_writer("update_documents").await;
            }

            if let Some(doc_id_str) = document_ids_map.get(&doc_id) {
                if let Ok(v) = serde_json::from_str(doc.inner.get()) {
                    if let Some(delta) = documents.remove(doc_id_str) {
                        let new_document = merge(v, delta);

                        let doc_id = self.document_count.fetch_add(1, Ordering::Relaxed);
                        let doc_id = DocumentId(doc_id);

                        let doc_str = serde_json::to_string(&doc.inner)
                            .context("Cannot serialize document")?;
                        self.document_storage
                            .insert_many(&[(
                                doc_id,
                                ZeboDocument::new(Cow::Borrowed(doc_id_str), Cow::Owned(doc_str)),
                            )])
                            .await
                            .context("Cannot insert document into document storage")?;

                        self.op_sender
                            .send(WriteOperation::DocumentStorage(
                                DocumentStorageWriteOperation::InsertDocument {
                                    doc_id,
                                    doc: DocumentToInsert(
                                        new_document
                                            .clone()
                                            .into_raw(format!("{target_index_id}:{doc_id_str}"))
                                            .expect("Cannot get raw document"),
                                    ),
                                },
                            ))
                            .await
                            .context("Cannot send document storage operation")?;

                        let doc_id_str = new_document
                            .inner
                            .get("id")
                            .context("Document does not have an id")?
                            .as_str()
                            .context("Document id is not a string")?
                            .to_string();

                        match index
                            .process_new_document(
                                doc_id,
                                doc_id_str,
                                new_document,
                                &mut index_operation_batch,
                            )
                            .await
                            .context("Cannot process document")
                        {
                            Ok(Some(old_doc_id)) => {
                                docs_to_remove.push(old_doc_id);
                                result.updated += 1;
                            }
                            Ok(None) => {
                                result.inserted += 1;
                            }
                            Err(e) => {
                                // If the document cannot be processed, we should remove it from the document storage
                                // and from the read side
                                // NB: check if the error handling is correct
                                self.document_storage.remove(vec![doc_id]).await;

                                tracing::error!(error = ?e, "Cannot process document");
                                result.failed += 1;
                            }
                        };
                    }
                }

                pin_rules_touched.extend(pin_rules_writer.get_matching_rules_ids(doc_id_str));
            }
        }

        collection
            .update_pin_rules(pin_rules_touched, &mut index_operation_batch)
            .await;

        if !index_operation_batch.is_empty() {
            trace!("Sending operations");
            self.op_sender
                .send_batch(index_operation_batch)
                .await
                .context("Cannot send index operation")?;
            trace!("Operations sent");
        }

        self.document_storage.remove(docs_to_remove).await;

        debug!("All documents");

        Ok(result)
    }

    pub async fn delete_collection(
        &self,
        master_api_key: ApiKey,
        collection_id: CollectionId,
    ) -> Result<(), WriteError> {
        self.check_master_api_key(master_api_key)?;

        self.write_operation_counter.fetch_add(1, Ordering::Relaxed);
        let document_ids = self.collections.delete_collection(collection_id).await;
        self.write_operation_counter.fetch_sub(1, Ordering::Relaxed);

        if let Some(document_ids) = document_ids {
            self.commit()
                .await
                .context("Cannot commit collections after collection deletion")?;

            self.op_sender
                .send(WriteOperation::DeleteCollection(collection_id))
                .await
                .context("Cannot send delete collection operation")?;

            self.kv
                .delete_with_prefix(collection_id.as_str())
                .await
                .context("Cannot delete collection from KV")?;

            self.document_storage.remove(document_ids).await;
        }

        Ok(())
    }

    pub async fn list_document(
        &self,
        write_api_key: WriteApiKey,
        collection_id: CollectionId,
    ) -> Result<Vec<Document>, WriteError> {
        let collection = self.get_collection(collection_id, write_api_key).await?;

        let document_ids = collection.get_document_ids().await;

        let stream = self.document_storage.stream_documents(document_ids).await;
        let docs = stream
            .filter_map(|(_, doc)| {
                let inner = doc.inner;
                let inner: Map<String, Value> = match serde_json::from_str(inner.get()) {
                    Ok(inner) => inner,
                    Err(e) => {
                        error!(error = ?e, "Cannot deserialize document");
                        return None;
                    }
                };
                let doc = Document { inner };
                Some(doc)
            })
            .collect::<Vec<_>>()
            .await;

        Ok(docs)
    }

    pub async fn list_collections(
        &self,
        master_api_key: ApiKey,
    ) -> Result<Vec<DescribeCollectionResponse>, WriteError> {
        self.check_master_api_key(master_api_key)?;

        Ok(self.collections.list().await)
    }

    pub async fn get_collection_dto(
        &self,
        master_api_key: ApiKey,
        collection_id: CollectionId,
    ) -> Result<DescribeCollectionResponse, WriteError> {
        self.check_master_api_key(master_api_key)?;
        let collection = match self.collections.get_collection(collection_id).await {
            Some(collection) => collection,
            None => return Err(WriteError::CollectionNotFound(collection_id)),
        };
        Ok(collection.as_dto().await)
    }

    async fn add_documents_to_storage(
        self: &WriteSide,
        document_list: &mut DocumentList,
        target_index_id: IndexId,
    ) -> Result<Vec<DocumentId>> {
        let document_count = document_list.len();

        let batch_size = document_list.0.len().min(200);
        let mut batch = Vec::with_capacity(batch_size);

        let mut insert_document_batch = Vec::with_capacity(document_count);
        let mut doc_ids = Vec::with_capacity(document_count);
        let mut docs = Vec::with_capacity(batch_size);
        for (index, doc) in document_list.0.iter_mut().enumerate() {
            if index % 100 == 0 {
                trace!("Processing document {}/{}", index, document_count);
            }

            if index % batch_size == 0 && !batch.is_empty() {
                self.document_storage
                    .insert_many(&docs)
                    .await
                    .context("Cannot inser document into document storage")?;
                docs.clear();

                insert_document_batch.push(WriteOperation::DocumentStorage(
                    DocumentStorageWriteOperation::InsertDocuments(batch),
                ));
                batch = Vec::with_capacity(batch_size);
            }

            let doc_id = self.document_count.fetch_add(1, Ordering::Relaxed);

            // We try to guess the document id from the document
            // Eventually, we generate a new one
            let doc_id_str = doc.get("id").and_then(|v| v.as_str());
            let doc_id_str = if let Some(doc_id_str) = doc_id_str {
                if doc_id_str.is_empty() {
                    cuid2::create_id()
                } else {
                    doc_id_str.to_string()
                }
            } else {
                cuid2::create_id()
            };
            // Ensure the document contains a valid id
            // NB: this overwrite the previous id if it is not a string
            // TODO: is it correct?
            doc.inner.insert(
                "id".to_string(),
                serde_json::Value::String(doc_id_str.clone()),
            );

            let doc_id = DocumentId(doc_id);
            doc_ids.push(doc_id);

            batch.push((
                doc_id,
                DocumentToInsert(
                    doc.clone()
                        .into_raw(format!("{target_index_id}:{doc_id_str}"))
                        .expect("Cannot get raw document"),
                ),
            ));

            let doc_str = serde_json::to_string(&doc.inner).context("Cannot serialize document")?;
            docs.push((
                doc_id,
                ZeboDocument::new(Cow::Owned(doc_id_str), Cow::Owned(doc_str)),
            ));
        }

        if !batch.is_empty() {
            insert_document_batch.push(WriteOperation::DocumentStorage(
                DocumentStorageWriteOperation::InsertDocuments(batch),
            ));

            self.document_storage
                .insert_many(&docs)
                .await
                .context("Cannot inser document into document storage")?;
        }

        trace!("Sending documents");
        self.op_sender
            .send_batch(insert_document_batch)
            .await
            .context("Cannot send document storage operation")?;
        trace!("Documents sent");

        Ok(doc_ids)
    }

    async fn inner_process_documents(
        &self,
        collection_id: CollectionId,
        index_id: IndexId,
        document_list: DocumentList,
        doc_ids: Vec<DocumentId>,
    ) -> Result<InsertDocumentsResult, WriteError> {
        let mut result = InsertDocumentsResult {
            inserted: 0,
            replaced: 0,
            failed: 0,
        };

        let mut collection = match self.collections.get_collection(collection_id).await {
            Some(collection) => collection,
            None => return Err(WriteError::CollectionNotFound(collection_id)),
        };
        let mut index = match collection.get_index(index_id).await {
            Some(index) => index,
            None => return Err(WriteError::IndexNotFound(collection_id, index_id)),
        };

        let mut pin_rules_writer = collection.get_pin_rule_writer("process_documents").await;
        let mut pin_rules_touched = HashSet::new();

        let document_count = document_list.len();

        let mut index_operation_batch = Vec::with_capacity(document_count * 10);
        let mut docs_to_remove = Vec::with_capacity(document_count);
        for (i, doc) in document_list.0.into_iter().enumerate() {
            if i % 100 == 0 {
                info!("Processing document {}/{}", i, document_count);
            }

            let doc_id = doc_ids[i];
            if index_operation_batch.capacity() * 4 / 5 < index_operation_batch.len() {
                trace!("Sending operations");
                self.op_sender
                    .send_batch(index_operation_batch)
                    .await
                    .context("Cannot send index operation")?;
                trace!("Operations sent");
                index_operation_batch = Vec::with_capacity(document_count * 10);
            }

            // Check for pending write operations and yield if needed
            if self.write_operation_counter.load(Ordering::Relaxed) > 0 {
                // We need to drop the index lock before waiting
                drop(pin_rules_writer);
                drop(index);
                drop(collection);

                self.wait_for_pending_write_operations().await?;

                // Reacquire the index lock
                collection = match self.collections.get_collection(collection_id).await {
                    Some(collection) => collection,
                    None => return Err(WriteError::CollectionNotFound(collection_id)),
                };
                index = match collection.get_index(index_id).await {
                    Some(index) => index,
                    None => return Err(WriteError::IndexNotFound(collection_id, index_id)),
                };
                pin_rules_writer = collection.get_pin_rule_writer("process_documents").await;
            }

            let doc_id_str = doc
                .inner
                .get("id")
                .context("Document does not have an id")?
                .as_str()
                .context("Document id is not a string")?
                .to_string();

            match index
                .process_new_document(doc_id, doc_id_str.clone(), doc, &mut index_operation_batch)
                .await
                .context("Cannot process document")
            {
                Ok(Some(old_doc_id)) => {
                    docs_to_remove.push(old_doc_id);
                    result.inserted += 1;
                }
                Ok(None) => {
                    result.inserted += 1;
                }
                Err(e) => {
                    // If the document cannot be processed, we should remove it from the document storage
                    // and from the read side
                    // NB: check if the error handling is correct
                    self.document_storage.remove(vec![doc_id]).await;

                    tracing::error!(error = ?e, "Cannot process document");
                    result.failed += 1;
                    continue;
                }
            };

            pin_rules_touched.extend(pin_rules_writer.get_matching_rules_ids(&doc_id_str));
        }
        debug!("All documents processed {}", document_count);

        collection
            .update_pin_rules(pin_rules_touched, &mut index_operation_batch)
            .await;

        if !index_operation_batch.is_empty() {
            trace!("Sending operations");
            self.op_sender
                .send_batch(index_operation_batch)
                .await
                .context("Cannot send index operation")?;
            trace!("Operations sent");
        }

        self.document_storage.remove(docs_to_remove).await;

        Ok(result)
    }

    pub async fn get_collection(
        &self,
        collection_id: CollectionId,
        write_api_key: WriteApiKey,
    ) -> Result<CollectionReadLock<'_>, WriteError> {
        let collection = self
            .collections
            .get_collection(collection_id)
            .await
            .ok_or_else(|| WriteError::CollectionNotFound(collection_id))?;

        if let WriteApiKey::ApiKey(p) = write_api_key {
            // Allow master api key usage as a write api key
            if p == self.master_api_key {
                return Ok(collection);
            }
        }
        collection.check_write_api_key(write_api_key).await?;

        Ok(collection)
    }

    #[allow(clippy::result_large_err)]
    fn check_master_api_key(&self, master_api_key: ApiKey) -> Result<(), WriteError> {
        if self.master_api_key != master_api_key {
            return Err(WriteError::InvalidMasterApiKey);
        }

        Ok(())
    }

    async fn check_write_api_key(
        &self,
        collection_id: CollectionId,
        write_api_key: WriteApiKey,
    ) -> Result<()> {
        self.get_collection(collection_id, write_api_key).await?;
        Ok(())
    }

    pub fn llm_service(&self) -> &LLMService {
        &self.context.llm_service
    }

    pub async fn get_system_prompts_manager(
        &self,
        write_api_key: WriteApiKey,
        collection_id: CollectionId,
    ) -> Result<CollectionSystemPromptsInterface, WriteError> {
        self.check_write_api_key(collection_id, write_api_key)
            .await?;
        Ok(CollectionSystemPromptsInterface::new(
            self.system_prompts.clone(),
            collection_id,
        ))
    }

    pub async fn get_hooks_storage<'s>(
        &'s self,
        write_api_key: WriteApiKey,
        collection_id: CollectionId,
    ) -> Result<HookWriterLock<'s>, WriteError> {
        let collection = self.get_collection(collection_id, write_api_key).await?;
        Ok(HookWriterLock { collection })
    }

    pub async fn get_tools_manager(
        &self,
        write_api_key: WriteApiKey,
        collection_id: CollectionId,
    ) -> Result<CollectionToolsRuntime, WriteError> {
        self.check_write_api_key(collection_id, write_api_key)
            .await?;
        Ok(CollectionToolsRuntime::new(
            self.tools.clone(),
            collection_id,
        ))
    }

    pub async fn get_training_set_interface(
        &self,
        write_api_key: WriteApiKey,
        collection_id: CollectionId,
    ) -> Result<TrainingSetInterface, WriteError> {
        self.check_write_api_key(collection_id, write_api_key)
            .await?;

        Ok(self.training_sets.clone())
    }

    pub fn get_jwt_manager(&self) -> JwtManager {
        self.jwt_manager.clone()
    }

    async fn wait_for_pending_write_operations(&self) -> Result<()> {
        let timeout = Duration::from_secs(5);
        let start = Instant::now();

        // Initialize backoff parameters
        let mut backoff = Duration::from_millis(10); // Start at 10ms
        let max_backoff = Duration::from_millis(500); // Cap at 500ms to allow multiple retries within 5s

        while self.write_operation_counter.load(Ordering::Relaxed) > 0 {
            if start.elapsed() > timeout {
                warn!("Timeout waiting for write operations to complete");
                break;
            }

            // If there's some write pending operation we yield to let the write operation be processed
            tokio::task::yield_now().await;

            // Anyway, `yield_now` doens't guarantee that the other task will be executed
            // If the scheduler will process this task without waiting the other task,
            // we propose a sleep of 10ms to be sure that the other task will be executed.
            // Anyway, this is not guaranteed again, it is just an hope.
            if self.write_operation_counter.load(Ordering::Relaxed) > 0 {
                sleep(backoff).await;
                // Triple the backoff time, but cap it at max_backoff
                backoff = (backoff * 3).min(max_backoff);
            } else {
                break;
            }
        }

        Ok(())
    }
}

fn start_commit_loop(
    write_side: Arc<WriteSide>,
    insert_batch_commit_size: Duration,
    mut stop_receiver: tokio::sync::broadcast::Receiver<()>,
    stop_done_sender: tokio::sync::mpsc::Sender<()>,
) {
    tokio::task::spawn(async move {
        let start = tokio::time::Instant::now() + insert_batch_commit_size;
        let mut interval = tokio::time::interval_at(start, insert_batch_commit_size);

        // If for some reason we miss a tick, we skip it.
        // In fact, the commit is blocked only by `update` method.
        // If the collection is under heavy load,
        // the commit will be run due to the `insert_batch_commit_size` config.
        interval.set_missed_tick_behavior(MissedTickBehavior::Skip);

        'outer: loop {
            tokio::select! {
                _ = stop_receiver.recv() => {
                    info!("Stopping commit loop");
                    break 'outer;
                }
                _ = interval.tick() => {

                }
            };
            info!(
                "{:?} time reached. Committing write side",
                insert_batch_commit_size.clone()
            );
            if let Err(e) = write_side.commit().await {
                error!(error = ?e, "Cannot commit write side");
            }
        }

        if let Err(e) = stop_done_sender.send(()).await {
            error!(error = ?e, "Cannot send stop signal to writer side");
        }
    });
}

fn start_temp_index_cleanup_loop(
    write_side: Arc<WriteSide>,
    cleanup_config: TempIndexCleanupConfig,
    mut stop_receiver: tokio::sync::broadcast::Receiver<()>,
    stop_done_sender: tokio::sync::mpsc::Sender<()>,
) {
    tokio::task::spawn(async move {
        let start = tokio::time::Instant::now() + cleanup_config.cleanup_interval;
        let mut interval = tokio::time::interval_at(start, cleanup_config.cleanup_interval);

        // If for some reason we miss a tick, we skip it.
        // Cleanup is not critical and we want to avoid overloading the system.
        interval.set_missed_tick_behavior(MissedTickBehavior::Skip);

        info!(
            "Started temp index cleanup loop with interval: {:?}, max_age: {:?}",
            cleanup_config.cleanup_interval, cleanup_config.max_age
        );

        'outer: loop {
            tokio::select! {
                _ = stop_receiver.recv() => {
                    info!("Stopping temp index cleanup loop");
                    break 'outer;
                }
                _ = interval.tick() => {
                    // Cleanup expired temp indexes
                    match write_side.cleanup_expired_temp_indexes(cleanup_config.max_age).await {
                        Ok(cleaned_count) => {
                            if cleaned_count > 0 {
                                info!("Cleaned up {} expired temp indexes", cleaned_count);
                            }
                        }
                        Err(e) => {
                            error!(error = ?e, "Error during temp index cleanup");
                        }
                    }
                }
            }
        }

        if let Err(e) = stop_done_sender.send(()).await {
            error!(error = ?e, "Cannot send stop signal from temp index cleanup loop");
        }
    });
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "version")]
enum WriteSideInfo {
    #[serde(rename = "1")]
    V1(WriteSideInfoV1),
}

#[derive(Serialize, Deserialize, Debug)]
struct WriteSideInfoV1 {
    document_count: u64,
    offset: Offset,
}

fn embedding_queue_limit_default() -> u32 {
    50
}
fn javascript_queue_limit_default() -> usize {
    50
}

fn embedding_model_default() -> Model {
    Model::BGESmall
}

fn default_insert_batch_commit_size() -> u64 {
    1_000
}

fn default_temp_index_cleanup_config() -> TempIndexCleanupConfig {
    TempIndexCleanupConfig {
        cleanup_interval: Duration::from_secs(3600), // 1 hour
        max_age: Duration::from_secs(43200),         // 12 hours
    }
}

pub struct HookWriterLock<'guard> {
    collection: CollectionReadLock<'guard>,
}
impl Deref for HookWriterLock<'_> {
    type Target = HookWriter;

    fn deref(&self) -> &Self::Target {
        self.collection.get_hook_storage()
    }
}

fn merge(mut old: serde_json::value::Map<String, serde_json::Value>, delta: Document) -> Document {
    let delta = delta.inner;
    for (k, v) in delta {
        if k.contains(".") {
            let mut path = k.split(".").peekable();
            let mut nested_doc = Some(&mut old);
            let k = loop {
                let k = path.next();
                let f = path.peek();
                nested_doc = match (k, f) {
                    (None, _) => break None,
                    (Some(k), None) => break Some(k),
                    (Some(k), _) => {
                        nested_doc.and_then(|v| v.get_mut(k).and_then(|v| v.as_object_mut()))
                    }
                }
            };
            if let Some(nested_doc) = nested_doc {
                if let Some(k) = k {
                    if v.is_null() {
                        // Null removes the key from an object
                        nested_doc.remove(k);
                    } else {
                        nested_doc.insert(k.to_string(), v);
                    }
                }
            }
        } else {
            // Fast path
            if v.is_null() {
                // Null removes the key from an object
                old.remove(&k);
            } else {
                old.insert(k, v);
            }
        }
    }
    Document { inner: old }
}

#[allow(clippy::approx_constant)]
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_merge_documents_simple() {
        let old = json!({
            "id": "1",
            "text": "foo",
            "name": "Tommaso",
        });
        let old = old.as_object().unwrap();
        let serde_json::Value::Object(delta) = json!({
            "id": "1",
            "text": "bar",
        }) else {
            panic!("");
        };
        let delta = Document { inner: delta };
        let new = merge(old.clone(), delta);

        assert_eq!(
            serde_json::Value::Object(new.inner),
            json!({
                "id": "1",
                "text": "bar",
                "name": "Tommaso",
            })
        );
    }

    #[test]
    fn test_merge_documents_all_types() {
        let old = json!({
            "id": "1",
            "name": "Tommaso",
            "age": 34,
            "amazing": false,
            "another_float": 3.14,
            "untouched_str": "hello",
            "untouched_number": 42,
            "untouched_bool": true,
        });
        let old = old.as_object().unwrap();
        let serde_json::Value::Object(delta) = json!({
            "id": "1",
            "name": "Michele",
            "age": 30,
            "amazing": true,
            "another_float": 2.71,
        }) else {
            panic!("");
        };
        let delta = Document { inner: delta };
        let new = merge(old.clone(), delta);

        assert_eq!(
            serde_json::Value::Object(new.inner),
            json!({
                "id": "1",
                "name": "Michele",
                "age": 30,
                "amazing": true,
                "another_float": 2.71,
                "untouched_str": "hello",
                "untouched_number": 42,
                "untouched_bool": true,
            })
        );
    }

    #[test]
    fn test_merge_documents_null_removes_key() {
        let old = json!({
            "id": "1",
            "name": "Tommaso",
            "age": 34,
            "amazing": false,
            "another_float": 3.14,
        });
        let old = old.as_object().unwrap();
        let serde_json::Value::Object(delta) = json!({
            "id": "1",
            "amazing": null,
        }) else {
            panic!("");
        };
        let delta = Document { inner: delta };
        let new = merge(old.clone(), delta);

        assert_eq!(
            serde_json::Value::Object(new.inner),
            json!({
                "id": "1",
                "name": "Tommaso",
                "age": 34,
                "another_float": 3.14,
            })
        );
    }

    #[test]
    fn test_merge_documents_replace_whole_object() {
        let old = json!({
            "id": "1",
            "person": {
                "name": "Tommaso",
                "age": 34,
            }
        });
        let old = old.as_object().unwrap();
        let serde_json::Value::Object(delta) = json!({
            "id": "1",
            "person": {
                "name": "Michele",
                "age": 30,
            }
        }) else {
            panic!("");
        };
        let delta = Document { inner: delta };
        let new = merge(old.clone(), delta);

        assert_eq!(
            serde_json::Value::Object(new.inner),
            json!({
                "id": "1",
                "person": {
                    "name": "Michele",
                    "age": 30,
                }
            })
        );
    }

    #[test]
    fn test_merge_documents_update_nested_property() {
        let old = json!({
            "id": "1",
            "person": {
                "name": "Tommaso",
                "age": 34,
                "amazing": true,
            }
        });
        let old = old.as_object().unwrap();
        let serde_json::Value::Object(delta) = json!({
            "id": "1",
            "person.name": "Michele",
            "person.age": 30,
        }) else {
            panic!("");
        };
        let delta = Document { inner: delta };
        let new = merge(old.clone(), delta);

        assert_eq!(
            serde_json::Value::Object(new.inner),
            json!({
                "id": "1",
                "person": {
                    "name": "Michele",
                    "age": 30,
                    "amazing": true,
                }
            })
        );
    }

    #[test]
    fn test_merge_documents_update_very_nested_property() {
        let old = json!({
            "id": "1",
            "l1": {
                "l2" : {
                    "l3": {
                        "l4": {
                            "l5": {
                                "l6": {
                                    "l7": {
                                        "l8": {
                                            "l9": {
                                                "value": 5,
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
        let old = old.as_object().unwrap();
        let serde_json::Value::Object(delta) = json!({
            "id": "1",
            "l1.l2.l3.l4.l5.l6.l7.l8.l9.value": 42,
        }) else {
            panic!("");
        };
        let delta = Document { inner: delta };
        let new = merge(old.clone(), delta);

        assert_eq!(
            serde_json::Value::Object(new.inner),
            json!({
                "id": "1",
                "l1": {
                "l2" : {
                    "l3": {
                        "l4": {
                            "l5": {
                                "l6": {
                                    "l7": {
                                        "l8": {
                                            "l9": {
                                                "value": 42,
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            })
        );
    }
}
