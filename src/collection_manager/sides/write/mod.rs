pub mod collection;
mod collections;
pub mod document_storage;
mod embedding;
pub mod index;
pub use index::OramaModelSerializable;

use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{
        atomic::{AtomicU32, AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

use super::{
    generic_kv::{KVConfig, KV},
    hooks::{HookName, HooksRuntime, HooksRuntimeConfig},
    segments::{Segment, SegmentInterface},
    system_prompts::{SystemPrompt, SystemPromptInterface, SystemPromptValidationResponse},
    triggers::{get_trigger_key, parse_trigger_id, Trigger, TriggerInterface},
    Offset, OperationSender, OperationSenderCreator, OutputSideChannelType,
};

use anyhow::{bail, Context, Result};
use document_storage::DocumentStorage;
use duration_str::deserialize_duration;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio::{
    sync::RwLock,
    time::{sleep, Instant, MissedTickBehavior},
};
use tokio_stream::StreamExt;
use tracing::{debug, error, info, trace};

use collections::CollectionsWriter;
use embedding::{start_calculate_embedding_loop, MultiEmbeddingCalculationRequest};

use crate::{
    ai::{
        automatic_embeddings_selector::AutomaticEmbeddingsSelector,
        gpu::LocalGPUManager,
        llms::LLMService,
        tools::{Tool, ToolsRuntime},
        AIService, OramaModel, RemoteLLMProvider,
    },
    collection_manager::sides::{
        DocumentStorageWriteOperation, DocumentToInsert, ReplaceIndexReason, WriteOperation,
    },
    file_utils::BufferedFile,
    metrics::{document_insertion::DOCUMENTS_INSERTION_TIME, Empty},
    nlp::NLPService,
    types::{
        ApiKey, CollectionId, CreateCollection, CreateIndexRequest, DeleteDocuments,
        DescribeCollectionResponse, Document, DocumentId, DocumentList, IndexEmbeddingsCalculation,
        IndexId, InsertDocumentsResult, InteractionLLMConfig, LanguageDTO, ReplaceIndexRequest,
    },
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
    pub javascript_queue_limit: usize,
    #[serde(deserialize_with = "deserialize_duration")]
    pub commit_interval: Duration,
}

#[derive(Deserialize, Clone)]
pub struct WriteSideConfig {
    pub master_api_key: ApiKey,
    pub hooks: HooksRuntimeConfig,
    pub output: OutputSideChannelType,
    pub config: CollectionsWriterConfig,
}

pub struct WriteSide {
    op_sender: OperationSender,
    collections: CollectionsWriter,
    document_count: AtomicU64,
    data_dir: PathBuf,
    hook_runtime: Arc<HooksRuntime>,
    operation_counter: RwLock<u64>,
    insert_batch_commit_size: u64,

    document_storage: DocumentStorage,
    segments: SegmentInterface,
    triggers: TriggerInterface,
    system_prompts: SystemPromptInterface,
    tools: ToolsRuntime,
    kv: Arc<KV>,
    local_gpu_manager: Arc<LocalGPUManager>,
    master_api_key: ApiKey,
    llm_service: Arc<LLMService>,

    stop_sender: tokio::sync::broadcast::Sender<()>,
    stop_done_receiver: RwLock<tokio::sync::mpsc::Receiver<()>>,

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
}

impl WriteSide {
    pub async fn try_load(
        op_sender_creator: OperationSenderCreator,
        config: WriteSideConfig,
        ai_service: Arc<AIService>,
        nlp_service: Arc<NLPService>,
        llm_service: Arc<LLMService>,
        local_gpu_manager: Arc<LocalGPUManager>,
        automatic_embeddings_selector: Arc<AutomaticEmbeddingsSelector>,
    ) -> Result<Arc<Self>> {
        let master_api_key = config.master_api_key;
        let collections_writer_config = config.config;
        let data_dir = collections_writer_config.data_dir.clone();

        let insert_batch_commit_size = collections_writer_config.insert_batch_commit_size;

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

        let kv = KV::try_load(KVConfig {
            data_dir: data_dir.join("kv"),
            sender: Some(op_sender.clone()),
        })
        .context("Cannot load KV")?;
        let kv = Arc::new(kv);
        let segments = SegmentInterface::new(kv.clone(), llm_service.clone());
        let triggers = TriggerInterface::new(kv.clone(), llm_service.clone());
        let system_prompts = SystemPromptInterface::new(kv.clone(), llm_service.clone());
        let tools = ToolsRuntime::new(kv.clone(), llm_service.clone());
        let hook = HooksRuntime::new(kv.clone(), config.hooks).await;
        let hook_runtime = Arc::new(hook);

        let collections_writer = CollectionsWriter::try_load(
            collections_writer_config,
            sx,
            hook_runtime.clone(),
            nlp_service.clone(),
            op_sender.clone(),
            automatic_embeddings_selector.clone(),
        )
        .await
        .context("Cannot load collections")?;

        let document_storage = DocumentStorage::try_new(data_dir.join("documents"))
            .context("Cannot create document storage")?;

        let (stop_done_sender, stop_done_receiver) = tokio::sync::mpsc::channel(1);
        let (stop_sender, _) = tokio::sync::broadcast::channel(1);
        let commit_loop_receiver = stop_sender.subscribe();
        let receive_operation_loop_receiver = stop_sender.subscribe();

        let write_side = Self {
            document_count,
            collections: collections_writer,
            document_storage,
            data_dir,
            hook_runtime,
            insert_batch_commit_size,
            master_api_key,
            operation_counter: Default::default(),
            op_sender: op_sender.clone(),
            segments,
            triggers,
            system_prompts,
            tools,
            kv,
            local_gpu_manager,
            llm_service,
            stop_sender,
            stop_done_receiver: RwLock::new(stop_done_receiver),
            write_operation_counter: AtomicU32::new(0),
        };

        let write_side = Arc::new(write_side);

        start_commit_loop(
            write_side.clone(),
            commit_interval,
            commit_loop_receiver,
            stop_done_sender.clone(),
        );
        start_calculate_embedding_loop(
            ai_service,
            rx,
            op_sender,
            embedding_queue_limit,
            receive_operation_loop_receiver,
            stop_done_sender.clone(),
        );

        Ok(write_side)
    }

    pub async fn stop(&self) -> Result<()> {
        info!("Stopping writer side");
        // Broadcast
        self.stop_sender
            .send(())
            .context("Cannot send stop signal")?;
        let mut stop_done_receiver = self.stop_done_receiver.write().await;
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

        self.collections.commit().await?;

        self.kv.commit().await?;

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
    ) -> Result<()> {
        self.check_master_api_key(master_api_key)?;

        self.write_operation_counter.fetch_add(1, Ordering::Relaxed);
        let res = self
            .collections
            .create_collection(option, self.op_sender.clone())
            .await;
        self.write_operation_counter.fetch_sub(1, Ordering::Relaxed);

        res?;

        Ok(())
    }

    pub async fn create_index(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
        req: CreateIndexRequest,
    ) -> Result<()> {
        let collection = self
            .collections
            .get_collection(collection_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("Collection not found"))?;
        collection.check_write_api_key(write_api_key)?;

        let CreateIndexRequest {
            index_id,
            embedding,
        } = req;

        let default_string_calculation = if cfg!(test) {
            IndexEmbeddingsCalculation::AllProperties
        } else {
            IndexEmbeddingsCalculation::Automatic
        };
        let embedding: IndexEmbeddingsCalculation = embedding.unwrap_or(default_string_calculation);

        collection.create_index(index_id, embedding).await?;

        Ok(())
    }

    pub async fn reindex(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
        language: LanguageDTO,
        model: OramaModel,
        reference: Option<String>,
    ) -> Result<()> {
        let mut collection = self
            .collections
            .get_collection(collection_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("Collection not found"))?;
        collection.check_write_api_key(write_api_key)?;

        collection
            .change_runtime_config(language.into(), model)
            .await;

        // We should lock the collection during the reindex.
        // Adding documents during the reindex could create problems
        // TODO: check if we should lock the collection

        info!("Reindexing collection {}", collection_id);

        let index_ids = collection.get_index_ids().await;
        for copy_from in index_ids {
            info!("Reindexing index {}", copy_from);
            let copy_from_index = collection
                .get_index(copy_from)
                .await
                .ok_or_else(|| anyhow::anyhow!("Index not found"))?;

            let new_index_id = cuid2::create_id();
            let new_index_id = IndexId::try_new(new_index_id)
                // This is safe because cuid2 is less than 64 bytes
                .expect("Cannot create new index id");
            let document_ids = copy_from_index.get_document_ids().await;

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

            let mut i = 0;

            let mut stream = self.document_storage.stream_documents(document_ids).await;
            while let Some((doc_id, doc)) = stream.next().await {
                i += 1;

                // We want to check if there's some pending write operation on collections hashmap
                // If so, we need to yield the task to allow the other operations to acquire the write lock
                if i % 50 == 0 && self.write_operation_counter.load(Ordering::Relaxed) > 0 {
                    drop(new_index);
                    drop(collection);

                    tokio::task::yield_now().await;
                    // `yield_now` doens't guarantee that the other task will be executed
                    // If the scheduler will process this task without waiting the other task,
                    // we propose a sleep of 10ms to be sure that the other task will be executed.
                    // Anyway, this is not guaranteed again, it is just an hope.
                    if self.write_operation_counter.load(Ordering::Relaxed) > 0 {
                        sleep(Duration::from_millis(10)).await;
                    }

                    collection = self
                        .collections
                        .get_collection(collection_id)
                        .await
                        .ok_or_else(|| anyhow::anyhow!("Collection not found"))?;
                    new_index = collection
                        .get_index(new_index_id)
                        .await
                        .ok_or_else(|| anyhow::anyhow!("Index not found"))?;
                }

                let inner = doc.inner;
                let inner: Map<String, Value> = serde_json::from_str(inner.get())
                    // This is safe because the document is a valid JSON
                    .context("Cannot deserialize document")?;
                let doc = Document { inner };
                match new_index
                    .reindex_document(doc_id, doc)
                    .await
                    .context("Cannot process document")
                {
                    Ok(_) => {}
                    Err(e) => {
                        // Ignore the error and continue to process the rest of the documents
                        error!(error = ?e, "Cannot process document during the reindexing. Ignore the error and continue");
                    }
                };
            }

            let req = ReplaceIndexRequest {
                runtime_index_id: copy_from,
                temp_index_id: new_index_id,
                reference: reference.clone(),
            };
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
        write_api_key: ApiKey,
        collection_id: CollectionId,
        req: ReplaceIndexRequest,
        reason: ReplaceIndexReason,
    ) -> Result<()> {
        let collection = self
            .collections
            .get_collection(collection_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("Collection not found"))?;
        collection.check_write_api_key(write_api_key)?;

        collection
            .replace_index(
                req.runtime_index_id,
                req.temp_index_id,
                reason,
                req.reference,
            )
            .await?;

        Ok(())
    }

    pub async fn create_temp_index(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
        copy_from: IndexId,
        req: CreateIndexRequest,
    ) -> Result<()> {
        let collection = self
            .collections
            .get_collection(collection_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("Collection not found"))?;
        collection.check_write_api_key(write_api_key)?;

        let CreateIndexRequest {
            index_id: new_index_id,
            embedding,
        } = req;

        collection
            .create_temp_index(copy_from, new_index_id, embedding)
            .await?;

        Ok(())
    }

    pub async fn delete_index(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
        index_id: IndexId,
    ) -> Result<()> {
        let collection = self
            .collections
            .get_collection(collection_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("Collection not found"))?;
        collection.check_write_api_key(write_api_key)?;

        let document_to_remove = collection.delete_index(index_id).await?;

        self.document_storage.remove(document_to_remove).await;

        Ok(())
    }

    pub async fn insert_documents(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
        index_id: IndexId,
        document_list: DocumentList,
    ) -> Result<InsertDocumentsResult> {
        let document_count = document_list.len();
        info!(?document_count, "Inserting batch of documents");

        let mut collection = self
            .collections
            .get_collection(collection_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("Collection not found"))?;

        collection.check_write_api_key(write_api_key)?;

        let mut index = collection
            .get_index(index_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("Index not found"))?;

        let mut result = InsertDocumentsResult {
            inserted: 0,
            replaced: 0,
            failed: 0,
        };
        let metric = DOCUMENTS_INSERTION_TIME.create(Empty);

        let mut i = 0;

        for mut doc in document_list {
            i += 1;

            // We want to check if there's some pending write operation on collections hashmap
            // If so, we need to yield the task to allow the other operations to acquire the write lock
            if i % 50 == 0 && self.write_operation_counter.load(Ordering::Relaxed) > 0 {
                drop(index);
                drop(collection);

                tokio::task::yield_now().await;
                // `yield_now` doens't guarantee that the other task will be executed
                // If the scheduler will process this task without waiting the other task,
                // we propose a sleep of 10ms to be sure that the other task will be executed.
                // Anyway, this is not guaranteed again, it is just an hope.
                if self.write_operation_counter.load(Ordering::Relaxed) > 0 {
                    sleep(Duration::from_millis(10)).await;
                }

                collection = self
                    .collections
                    .get_collection(collection_id)
                    .await
                    .ok_or_else(|| anyhow::anyhow!("Collection not found"))?;
                index = collection
                    .get_index(index_id)
                    .await
                    .ok_or_else(|| anyhow::anyhow!("Index not found"))?;
            }

            debug!("Inserting doc");
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

            self.document_storage
                .insert(doc_id, doc_id_str.clone(), doc.clone())
                .await
                .context("Cannot inser document into document storage")?;

            // High inefficiency here: we send a message for each document
            // We could send a unique message with a batch of documents
            // Anyway, for now, we keep it simple
            // TODO: check if we can send a batch of documents
            self.op_sender
                .send(WriteOperation::DocumentStorage(
                    DocumentStorageWriteOperation::InsertDocument {
                        doc_id,
                        doc: DocumentToInsert(
                            doc.clone()
                                .into_raw(format!("{}:{}", index_id, doc_id_str))
                                .expect("Cannot get raw document"),
                        ),
                    },
                ))
                .await
                .context("Cannot send document storage operation")?;

            trace!(?doc_id, "Inserting document");
            match index
                .process_new_document(doc_id, doc)
                .await
                .context("Cannot process document")
            {
                Ok(Some(old_doc_id)) => {
                    self.document_storage.remove(vec![old_doc_id]).await;
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
                }
            };

            debug!("Document inserted");
        }

        info!("All documents are inserted");

        drop(index);
        drop(collection);
        drop(metric);

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
        }

        Ok(result)
    }

    pub async fn delete_documents(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
        index_id: IndexId,
        document_ids_to_delete: DeleteDocuments,
    ) -> Result<()> {
        let collection = self
            .collections
            .get_collection(collection_id)
            .await
            .context("Collection not found")?;
        collection.check_write_api_key(write_api_key)?;
        let index = collection
            .get_index(index_id)
            .await
            .context("Cannot get index")?;
        let doc_ids = index.delete_documents(document_ids_to_delete).await?;

        self.document_storage.remove(doc_ids).await;

        Ok(())
    }

    pub async fn delete_collection(
        &self,
        master_api_key: ApiKey,
        collection_id: CollectionId,
    ) -> Result<()> {
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
        write_api_key: ApiKey,
        collection_id: CollectionId,
    ) -> Result<Vec<Document>> {
        let collection = self
            .collections
            .get_collection(collection_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("Collection not found"))?;

        collection.check_write_api_key(write_api_key)?;

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

    pub async fn insert_javascript_hook(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
        index_id: IndexId,
        name: HookName,
        code: String,
    ) -> Result<()> {
        let collection = self
            .collections
            .get_collection(collection_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("Collection not found"))?;

        collection.check_write_api_key(write_api_key)?;

        let index = collection
            .get_index(index_id)
            .await
            .context("Cannot get index")?;
        index
            .switch_to_embedding_hook(self.hook_runtime.clone())
            .await
            .context("Cannot set embedding hook")?;

        self.hook_runtime
            .insert_hook(collection_id, index_id, name.clone(), code)
            .await
            .context("Cannot insert hook")?;

        Ok(())
    }

    pub async fn list_collections(
        &self,
        master_api_key: ApiKey,
    ) -> Result<Vec<DescribeCollectionResponse>> {
        self.check_master_api_key(master_api_key)?;

        Ok(self.collections.list().await)
    }

    pub async fn get_collection_dto(
        &self,
        master_api_key: ApiKey,
        collection_id: CollectionId,
    ) -> Result<Option<DescribeCollectionResponse>> {
        self.check_master_api_key(master_api_key)?;
        let collection = match self.collections.get_collection(collection_id).await {
            Some(collection) => collection,
            None => return Ok(None),
        };
        Ok(Some(collection.as_dto().await))
    }

    pub async fn get_javascript_hook(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
        index_id: IndexId,
        name: HookName,
    ) -> Result<Option<String>> {
        let collection = self
            .collections
            .get_collection(collection_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("Collection not found"))?;
        collection.check_write_api_key(write_api_key)?;

        Ok(self
            .hook_runtime
            .get_hook(collection_id, index_id, name)
            .await
            .map(|hook| hook.code))
    }

    pub async fn delete_javascript_hook(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
        _name: HookName,
    ) -> Result<Option<String>> {
        let collection = self
            .collections
            .get_collection(collection_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("Collection not found"))?;
        collection.check_write_api_key(write_api_key)?;

        bail!("Not implemented yet.") // @todo: implement delete hook in HooksRuntime and CollectionsWriter
    }

    pub async fn list_javascript_hooks(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
    ) -> Result<HashMap<HookName, String>> {
        let collection = self
            .collections
            .get_collection(collection_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("Collection not found"))?;
        collection.check_write_api_key(write_api_key)?;

        Ok(self
            .hook_runtime
            .list_hooks(collection_id)
            .await
            .context("Cannot list hooks")?
            .into_iter()
            .map(|(name, hook)| (name, hook.code))
            .collect())
    }

    async fn check_write_api_key(
        &self,
        collection_id: CollectionId,
        write_api_key: ApiKey,
    ) -> Result<()> {
        let collection = self
            .collections
            .get_collection(collection_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("Collection not found"))?;

        collection.check_write_api_key(write_api_key)?;

        Ok(())
    }

    fn check_master_api_key(&self, master_api_key: ApiKey) -> Result<()> {
        if self.master_api_key != master_api_key {
            return Err(anyhow::anyhow!("Invalid master api key"));
        }

        Ok(())
    }

    pub async fn validate_system_prompt(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
        system_prompt: SystemPrompt,
        llm_config: Option<InteractionLLMConfig>,
    ) -> Result<SystemPromptValidationResponse> {
        self.check_write_api_key(collection_id, write_api_key)
            .await?;

        self.system_prompts
            .validate_prompt(system_prompt, llm_config)
            .await
    }

    pub async fn insert_system_prompt(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
        system_prompt: SystemPrompt,
    ) -> Result<()> {
        self.check_write_api_key(collection_id, write_api_key)
            .await?;

        self.system_prompts
            .insert(collection_id, system_prompt.clone())
            .await
            .context("Cannot insert system prompt")?;

        Ok(())
    }

    pub async fn delete_system_prompt(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
        system_prompt_id: String,
    ) -> Result<Option<SystemPrompt>> {
        self.check_write_api_key(collection_id, write_api_key)
            .await?;

        self.system_prompts
            .delete(collection_id, system_prompt_id.clone())
            .await
            .context("Cannot delete system prompt")
    }

    pub async fn update_system_prompt(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
        system_prompt: SystemPrompt,
    ) -> Result<()> {
        self.check_write_api_key(collection_id, write_api_key)
            .await?;

        self.system_prompts
            .delete(collection_id, system_prompt.id.clone())
            .await
            .context("Cannot delete system prompt")?;
        self.system_prompts
            .insert(collection_id, system_prompt)
            .await
            .context("Cannot insert system prompt")?;

        Ok(())
    }

    pub async fn insert_segment(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
        segment: Segment,
    ) -> Result<()> {
        self.check_write_api_key(collection_id, write_api_key)
            .await?;

        self.segments
            .insert(collection_id, segment.clone())
            .await
            .context("Cannot insert segment")?;

        Ok(())
    }

    pub async fn delete_segment(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
        segment_id: String,
    ) -> Result<Option<Segment>> {
        self.check_write_api_key(collection_id, write_api_key)
            .await?;

        self.segments
            .delete(collection_id, segment_id.clone())
            .await
            .context("Cannot delete segment")
    }

    pub async fn update_segment(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
        segment: Segment,
    ) -> Result<()> {
        self.check_write_api_key(collection_id, write_api_key)
            .await?;

        self.segments
            .delete(collection_id, segment.id.clone())
            .await
            .context("Cannot delete segment")?;
        self.segments
            .insert(collection_id, segment)
            .await
            .context("Cannot insert segment")?;

        Ok(())
    }

    pub async fn insert_trigger(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
        trigger: Trigger,
        trigger_id: Option<String>,
    ) -> Result<Trigger> {
        self.check_write_api_key(collection_id, write_api_key)
            .await?;

        let final_trigger_id = match trigger_id {
            Some(mut id) => {
                let required_prefix = format!("{}:trigger:", collection_id.as_str());

                if !id.starts_with(&required_prefix) {
                    id = get_trigger_key(collection_id, id, trigger.segment_id.clone());
                }

                id
            }
            None => {
                let cuid = cuid2::create_id();
                get_trigger_key(collection_id, cuid, trigger.segment_id.clone())
            }
        };

        let trigger = Trigger {
            id: final_trigger_id,
            name: trigger.name,
            description: trigger.description,
            response: trigger.response,
            segment_id: trigger.segment_id,
        };

        self.triggers
            .insert(trigger.clone())
            .await
            .context("Cannot insert trigger")?;

        Ok(trigger)
    }

    pub async fn get_trigger(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
        trigger_id: String,
    ) -> Result<Trigger> {
        self.check_write_api_key(collection_id, write_api_key)
            .await?;

        let trigger = self
            .triggers
            .get(collection_id, trigger_id)
            .await
            .context("Cannot insert trigger")?;
        let trigger = match trigger {
            Some(trigger) => trigger,
            None => bail!("Trigger not found"),
        };

        Ok(trigger)
    }

    pub async fn delete_trigger(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
        trigger_id: String,
    ) -> Result<Option<Trigger>> {
        self.check_write_api_key(collection_id, write_api_key)
            .await?;

        self.triggers
            .delete(collection_id, trigger_id)
            .await
            .context("Cannot delete trigger")
    }

    pub async fn update_trigger(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
        trigger: Trigger,
    ) -> Result<Option<Trigger>> {
        let trigger_key = get_trigger_key(
            collection_id,
            trigger.id.clone(),
            trigger.segment_id.clone(),
        );

        let new_trigger = Trigger {
            id: trigger_key.clone(),
            ..trigger
        };

        self.insert_trigger(write_api_key, collection_id, new_trigger, Some(trigger.id))
            .await
            .context("Cannot insert updated trigger")?;

        match parse_trigger_id(trigger_key.clone()) {
            Some(key_content) => {
                let updated_trigger = self
                    .triggers
                    .get(collection_id, key_content.trigger_id.clone())
                    .await
                    .context("Cannot get updated trigger")?;

                match updated_trigger {
                    Some(trigger) => Ok(Some(Trigger {
                        id: key_content.trigger_id,
                        ..trigger
                    })),
                    None => bail!("Cannot get updated trigger"),
                }
            }
            None => {
                bail!("Cannot parse trigger id")
            }
        }
    }

    pub async fn insert_tool(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
        tool: Tool,
    ) -> Result<()> {
        self.check_write_api_key(collection_id, write_api_key)
            .await?;

        self.tools
            .insert(collection_id, tool.clone())
            .await
            .context("Cannot insert tool")?;

        Ok(())
    }

    pub async fn delete_tool(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
        tool_id: String,
    ) -> Result<()> {
        self.check_write_api_key(collection_id, write_api_key)
            .await?;

        self.tools
            .delete(collection_id, tool_id.clone())
            .await
            .context("Cannot delete tool")?;

        Ok(())
    }

    pub async fn update_tool(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
        tool: Tool,
    ) -> Result<()> {
        self.delete_tool(write_api_key, collection_id, tool.id.clone())
            .await?;
        self.insert_tool(write_api_key, collection_id, tool).await?;

        Ok(())
    }

    pub fn is_gpu_overloaded(&self) -> bool {
        match self.local_gpu_manager.is_overloaded() {
            Ok(overloaded) => overloaded,
            Err(e) => {
                error!(error = ?e, "Cannot check if GPU is overloaded. This may be due to GPU malfunction. Forcing inference on remote LLMs for safety.");
                true
            }
        }
    }

    pub fn get_available_remote_llm_services(&self) -> Option<HashMap<RemoteLLMProvider, String>> {
        self.llm_service.default_remote_models.clone()
    }

    pub fn select_random_remote_llm_service(&self) -> Option<(RemoteLLMProvider, String)> {
        match self.get_available_remote_llm_services() {
            Some(services) => {
                let mut rng = rand::rng();
                let random_index = rand::Rng::random_range(&mut rng, 0..services.len());
                services.into_iter().nth(random_index)
            }
            None => {
                error!("No remote LLM services available. Unable to select a random one for handling a offloading request.");
                None
            }
        }
    }
}

fn start_commit_loop(
    write_side: Arc<WriteSide>,
    insert_batch_commit_size: Duration,
    mut stop_receiver: tokio::sync::broadcast::Receiver<()>,
    stop_done_sender: tokio::sync::mpsc::Sender<()>,
) {
    tokio::task::spawn(async move {
        let start = Instant::now() + insert_batch_commit_size;
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

fn embedding_model_default() -> OramaModelSerializable {
    OramaModelSerializable(crate::ai::OramaModel::BgeSmall)
}

fn default_insert_batch_commit_size() -> u64 {
    1_000
}
