pub mod collection;
mod collections;
pub mod document_storage;
mod embedding;
mod fields;
pub mod index;

use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
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
use index::CreateIndexRequest;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tokio::{
    sync::RwLock,
    time::{Instant, MissedTickBehavior},
};
use tokio_stream::StreamExt;
use tracing::{debug, error, info, instrument, trace};

use collections::CollectionsWriter;
use embedding::{start_calculate_embedding_loop, MultiEmbeddingCalculationRequest};

pub use fields::*;

use crate::{
    ai::{gpu::LocalGPUManager, llms::LLMService, AIService, RemoteLLMProvider},
    collection_manager::sides::{
        CollectionWriteOperation, DocumentStorageWriteOperation, DocumentToInsert, WriteOperation,
    },
    file_utils::BufferedFile,
    metrics::{document_insertion::DOCUMENTS_INSERTION_TIME, Empty},
    nlp::NLPService,
    types::{
        ApiKey, CollectionId, CreateCollection, DeleteDocuments, DescribeCollectionResponse,
        Document, DocumentId, DocumentList, IndexId, InsertDocumentsResult, InteractionLLMConfig,
        ReindexConfig,
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
    kv: Arc<KV>,
    local_gpu_manager: Arc<LocalGPUManager>,
    master_api_key: ApiKey,
    llm_service: Arc<LLMService>,
}

impl WriteSide {
    pub async fn try_load(
        op_sender_creator: OperationSenderCreator,
        config: WriteSideConfig,
        ai_service: Arc<AIService>,
        nlp_service: Arc<NLPService>,
        llm_service: Arc<LLMService>,
        local_gpu_manager: Arc<LocalGPUManager>,
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
        let hook = HooksRuntime::new(kv.clone(), config.hooks).await;
        let hook_runtime = Arc::new(hook);

        let collections_writer = CollectionsWriter::try_load(
            collections_writer_config,
            sx,
            hook_runtime.clone(),
            nlp_service.clone(),
            op_sender.clone(),
        )
        .await
        .context("Cannot load collections")?;

        let document_storage = DocumentStorage::try_new(data_dir.join("documents"))
            .context("Cannot create document storage")?;

        let write_side = Self {
            document_count,
            collections: collections_writer,
            document_storage,
            data_dir,
            hook_runtime,
            insert_batch_commit_size,
            master_api_key,
            operation_counter: Default::default(),
            op_sender,
            segments,
            triggers,
            system_prompts,
            kv,
            local_gpu_manager,
            llm_service,
        };

        let write_side = Arc::new(write_side);

        start_commit_loop(write_side.clone(), commit_interval);
        start_calculate_embedding_loop(ai_service, rx, embedding_queue_limit);

        Ok(write_side)
    }

    #[instrument(skip(self))]
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

        self.collections
            .create_collection(option, self.op_sender.clone(), self.hook_runtime.clone())
            .await?;

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

        collection.create_index(req).await?;

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

        collection.delete_index(index_id).await?;

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

        let collection = self
            .collections
            .get_collection(collection_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("Collection not found"))?;

        collection.check_write_api_key(write_api_key)?;

        let index = collection
            .get_index(index_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("Index not found"))?;

        let mut result = InsertDocumentsResult {
            inserted: 0,
            replaced: 0,
            failed: 0,
        };
        let metric = DOCUMENTS_INSERTION_TIME.create(Empty);

        for mut doc in document_list {
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
        self.check_master_api_key(master_api_key).unwrap();

        let deleted = self.collections.delete_collection(collection_id).await;
        if deleted {
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

    /*
    pub async fn create_collection_from(
        &self,
        write_api_key: ApiKey,
        request: CreateCollectionFrom,
    ) -> Result<CollectionId> {
        info!("create temporary collection");
        self.check_write_api_key(request.from, write_api_key)
            .await
            .context("Check write api key fails")?;

        let collection_id_tmp = cuid2::create_id();
        let collection_id_tmp = CollectionId::from(collection_id_tmp);

        let mut option = self
            .collections
            .collection_options
            .get(&request.from)
            .ok_or_else(|| anyhow!("Collection options not found"))?
            .clone();
        option.language = request.language.or(option.language);
        option.embeddings = request.embeddings.or(option.embeddings);
        option.id = collection_id_tmp;

        self.collections
            .create_collection(option, self.sender.clone(), self.hook_runtime.clone())
            .await?;

        let hook = self
            .get_javascript_hook(
                write_api_key,
                request.from,
                HookName::SelectEmbeddingsProperties,
            )
            .await
            .context("Cannot get embedding hook")?;
        if let Some(hook) = hook {
            self.insert_javascript_hook(
                write_api_key,
                collection_id_tmp,
                HookName::SelectEmbeddingsProperties,
                hook,
            )
            .await
            .context("Cannot insert embedding hook to new collection")?;
        }

        Ok(collection_id_tmp)
    }

    pub async fn swap_collections(
        &self,
        write_api_key: ApiKey,
        request: SwapCollections,
    ) -> Result<()> {
        info!("Replacing collection");
        self.check_write_api_key(request.from, write_api_key)
            .await
            .context("Check write api key fails")?;
        self.check_write_api_key(request.to, write_api_key)
            .await
            .context("Check write api key fails")?;

        self.collections
            .replace(request.from, request.to)
            .await
            .context("Cannot replace collection")?;
        info!("Replaced");

        info!("Substitute collection");
        self.sender
            .send(WriteOperation::SubstituteCollection {
                subject_collection_id: request.from,
                target_collection_id: request.to,
                reference: request.reference,
            })
            .await?;

        self.commit()
            .await
            .context("Cannot commit collection after replace")?;

        Ok(())
    }
    */

    pub async fn reindex(
        &self,
        write_api_key: ApiKey,
        collection_id: CollectionId,
        index_id: IndexId,
        temp_index_id: IndexId,
        reindex_config: ReindexConfig,
    ) -> Result<()> {
        let collection = self
            .collections
            .get_collection(collection_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("Collection not found"))?;
        collection.check_write_api_key(write_api_key)?;

        collection
            .create_temporary_index_from(index_id, temp_index_id, reindex_config)
            .await
            .context("Cannot create temporary index")?;
        let temp_index = collection
            .get_temporary_index(temp_index_id)
            .await
            // This should never happen because we just created it above
            .expect("Temporary index not found but it should be there because already created");

        let index = collection
            .get_index(index_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("Index not found"))?;
        let document_ids = index.get_document_ids().await;
        drop(index);

        let mut stream = self.document_storage.stream_documents(document_ids).await;
        while let Some((doc_id, doc)) = stream.next().await {
            debug!(?doc_id, "Reindexing document");

            let inner = doc.inner;
            let inner: Map<String, Value> =
                serde_json::from_str(inner.get()).context("Cannot deserialize document")?;
            let doc = Document { inner };
            match temp_index
                .process_new_document(doc_id, doc)
                .await
                .context("Cannot process document")
            {
                Ok(_) => {}
                Err(e) => {
                    // If the document cannot be processed, we should remove it from the document storage
                    // and from the read side
                    // NB: check if the error handling is correct

                    self.op_sender
                        .send(WriteOperation::Collection(
                            collection_id,
                            CollectionWriteOperation::DeleteDocuments {
                                doc_ids: vec![doc_id],
                            },
                        ))
                        .await
                        .context("Cannot send delete document operation")?;

                    return Err(e);
                }
            };
            info!("Document reindexed");
        }
        drop(temp_index);

        collection
            .promote_temp_index(index_id, temp_index_id)
            .await
            .context("Cannot replace index")?;

        drop(collection);

        Ok(())
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

fn start_commit_loop(write_side: Arc<WriteSide>, insert_batch_commit_size: Duration) {
    tokio::task::spawn(async move {
        let start = Instant::now() + insert_batch_commit_size;
        let mut interval = tokio::time::interval_at(start, insert_batch_commit_size);

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
            if let Err(e) = write_side.commit().await {
                error!(error = ?e, "Cannot commit write side");
            }
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
