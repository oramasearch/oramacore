mod collection;
mod collections;
mod document_storage;
mod index;
pub mod notify;

use axum::extract::State;
use futures::Stream;
pub use index::*;

pub use collection::CollectionStats;
use duration_str::deserialize_duration;
use notify::NotifierConfig;
use std::sync::Arc;
use std::time::Duration;
use std::{collections::HashMap, path::PathBuf};
use tokio::time::{Instant, MissedTickBehavior};

use anyhow::{Context, Result};
pub use collection::IndexFieldStatsType;
use collections::CollectionsReader;
use document_storage::{DocumentStorage, DocumentStorageConfig};
use ordered_float::NotNan;
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, RwLock};
use tracing::{error, info, trace, warn};

use crate::ai::advanced_autoquery::AdvancedAutoQuerySteps;
use crate::ai::gpu::LocalGPUManager;
use crate::ai::llms::{self, LLMService};
use crate::ai::tools::{Tool, ToolExecutionReturnType, ToolsRuntime};
use crate::ai::RemoteLLMProvider;
use crate::collection_manager::sides::generic_kv::{KVConfig, KV};
use crate::collection_manager::sides::segments::SegmentInterface;
use crate::file_utils::BufferedFile;
use crate::metrics::operations::OPERATION_COUNT;
use crate::metrics::search::SEARCH_CALCULATION_TIME;
use crate::metrics::{Empty, SearchCollectionLabels};
use crate::types::{
    ApiKey, CollectionStatsRequest, InteractionLLMConfig, InteractionMessage, NLPSearchRequest,
    SearchMode, SearchModeResult, SearchParams, SearchResult, SearchResultHit, TokenScore,
};
use crate::{
    ai::AIService,
    capped_heap::CappedHeap,
    nlp::NLPService,
    types::{CollectionId, DocumentId},
};

use super::segments::{Segment, SelectedSegment};
use super::system_prompts::{SystemPrompt, SystemPromptInterface};
use super::triggers::{SelectedTrigger, Trigger, TriggerInterface};
use super::{
    InputSideChannelType, Offset, OperationReceiver, OperationReceiverCreator, WriteOperation,
};

#[derive(Deserialize, Clone)]
pub struct ReadSideConfig {
    pub input: InputSideChannelType,
    pub config: IndexesConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct IndexesConfig {
    pub data_dir: PathBuf,
    #[serde(default = "default_insert_batch_commit_size")]
    pub insert_batch_commit_size: u64,
    #[serde(deserialize_with = "deserialize_duration")]
    pub commit_interval: Duration,
    pub notifier: Option<NotifierConfig>,
}

pub struct ReadSide {
    pub collections: CollectionsReader,
    document_storage: DocumentStorage,
    operation_counter: RwLock<u64>,
    insert_batch_commit_size: u64,
    data_dir: PathBuf,
    live_offset: RwLock<Offset>,
    // This offset will update everytime a change is made to the read side.
    commit_insert_mutex: Mutex<Offset>,

    triggers: TriggerInterface,
    segments: SegmentInterface,
    system_prompts: SystemPromptInterface,
    tools: ToolsRuntime,
    kv: Arc<KV>,
    llm_service: Arc<LLMService>,
    local_gpu_manager: Arc<LocalGPUManager>,

    // Handle to stop the read side
    // This is used to stop the read side when the server is shutting down
    stop_sender: tokio::sync::broadcast::Sender<()>,
    stop_done_receiver: RwLock<tokio::sync::mpsc::Receiver<()>>,
}

impl ReadSide {
    pub async fn try_load(
        operation_receiver_creator: OperationReceiverCreator,
        ai_service: Arc<AIService>,
        nlp_service: Arc<NLPService>,
        llm_service: Arc<LLMService>,
        config: ReadSideConfig,
        local_gpu_manager: Arc<LocalGPUManager>,
    ) -> Result<Arc<Self>> {
        let mut document_storage = DocumentStorage::try_new(DocumentStorageConfig {
            data_dir: config.config.data_dir.join("docs"),
        })
        .await
        .context("Cannot create document storage")?;

        let insert_batch_commit_size = config.config.insert_batch_commit_size;
        let commit_interval = config.config.commit_interval;
        let data_dir = config.config.data_dir.clone();

        let collections_reader = CollectionsReader::try_load(
            ai_service.clone(),
            nlp_service,
            llm_service.clone(),
            config.config,
        )
        .await
        .context("Cannot load collections")?;
        document_storage
            .load()
            .context("Cannot load document storage")?;

        let read_info: Result<ReadInfo> = BufferedFile::open(data_dir.join("read.info"))
            .and_then(|f| f.read_json_data())
            .context("Cannot read offset file");
        let last_offset = match read_info {
            Ok(ReadInfo::V1(info)) => info.offset,
            Err(_) => {
                warn!("Cannot read 'read.info' file. Starting from 0");
                Offset(0)
            }
        };
        info!(offset=?last_offset, "Starting read side");

        let kv = KV::try_load(KVConfig {
            data_dir: data_dir.join("kv"),
            sender: None,
        })
        .context("Cannot load KV")?;
        let kv = Arc::new(kv);
        let segments = SegmentInterface::new(kv.clone(), llm_service.clone());
        let triggers = TriggerInterface::new(kv.clone(), llm_service.clone());
        let system_prompts = SystemPromptInterface::new(kv.clone(), llm_service.clone());
        let tools = ToolsRuntime::new(kv.clone(), llm_service.clone());

        let (stop_done_sender, stop_done_receiver) = tokio::sync::mpsc::channel(1);
        let (stop_sender, _) = tokio::sync::broadcast::channel(1);
        let commit_loop_receiver = stop_sender.subscribe();
        let receive_operation_loop_receiver = stop_sender.subscribe();

        let read_side = ReadSide {
            collections: collections_reader,
            document_storage,
            operation_counter: Default::default(),
            insert_batch_commit_size,
            data_dir,
            live_offset: RwLock::new(last_offset),
            commit_insert_mutex: Mutex::new(last_offset),
            segments,
            triggers,
            system_prompts,
            tools,
            kv,
            llm_service,
            local_gpu_manager,
            stop_sender,
            stop_done_receiver: RwLock::new(stop_done_receiver),
        };

        let operation_receiver = operation_receiver_creator.create(last_offset).await?;

        let read_side = Arc::new(read_side);

        start_commit_loop(
            read_side.clone(),
            commit_interval,
            commit_loop_receiver,
            stop_done_sender.clone(),
        );
        start_receive_operations(
            read_side.clone(),
            operation_receiver,
            receive_operation_loop_receiver,
            stop_done_sender,
        );

        Ok(read_side)
    }

    pub async fn stop(&self) -> Result<()> {
        info!("Stopping read side");
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
        // Operation receiver loop
        stop_done_receiver
            .recv()
            .await
            .context("Cannot send stop signal")?;
        info!("Read side stopped");

        Ok(())
    }

    pub async fn commit(&self) -> Result<()> {
        // We stop insertion operations while we are committing
        // This lock is needed to prevent any collection from being created, deleted or changed
        // ie, we stop to process any new events
        let mut commit_insert_mutex_lock = self.commit_insert_mutex.lock().await;

        let live_offset = self.live_offset.write().await;
        let offset = *live_offset;

        self.collections.commit(offset).await?;
        self.document_storage
            .commit()
            .await
            .context("Cannot commit document storage")?;
        self.kv.commit().await.context("Cannot commit KV")?;

        BufferedFile::create_or_overwrite(self.data_dir.join("read.info"))
            .context("Cannot create read.info file")?
            .write_json_data(&ReadInfo::V1(ReadInfoV1 { offset }))
            .context("Cannot write read.info file")?;

        *commit_insert_mutex_lock = offset;

        drop(commit_insert_mutex_lock);

        Ok(())
    }

    pub async fn collection_stats(
        &self,
        read_api_key: ApiKey,
        collection_id: CollectionId,
        req: CollectionStatsRequest,
    ) -> Result<CollectionStats> {
        let collection = self
            .collections
            .get_collection(collection_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("Collection not found"))?;
        collection.check_read_api_key(read_api_key)?;

        collection.stats(req).await
    }

    pub async fn update(&self, (offset, op): (Offset, WriteOperation)) -> Result<()> {
        trace!(offset=?offset, "Updating read side");

        let m = OPERATION_COUNT.create(Empty);

        // We stop commit operations while we are updating
        // The lock is released at the end of this function
        let commit_insert_mutex_lock = self.commit_insert_mutex.lock().await;

        // Already applied. We can skip this operation.
        if offset <= *commit_insert_mutex_lock && !commit_insert_mutex_lock.is_zero() {
            warn!(offset=?offset, "Operation already applied. Skipping");
            return Ok(());
        }

        let mut live_offset = self.live_offset.write().await;
        *live_offset = offset;
        drop(live_offset);

        match op {
            WriteOperation::CreateCollection {
                id,
                read_api_key,
                default_locale,
                description,
            } => {
                self.collections
                    .create_collection(id, description, default_locale, read_api_key)
                    .await?;
            }
            WriteOperation::DeleteCollection(collection_id) => {
                self.collections.delete_collection(collection_id).await?;
            }
            WriteOperation::Collection(collection_id, collection_operation) => {
                let collection = self
                    .collections
                    .get_collection(collection_id)
                    .await
                    .ok_or_else(|| anyhow::anyhow!("Collection not found"))?;
                collection.update(collection_operation).await?;
            }
            WriteOperation::DocumentStorage(op) => {
                self.document_storage
                    .update(op)
                    .await
                    .context("Cannot update document storage")?;
            }
            WriteOperation::KV(op) => {
                self.kv
                    .update(offset, op)
                    .await
                    .context("Cannot insert into KV")?;
            }
        }

        drop(m);

        let mut lock = self.operation_counter.write().await;
        *lock += 1;
        let should_commit = if *lock >= self.insert_batch_commit_size {
            *lock = 0;
            true
        } else {
            false
        };
        drop(lock);

        drop(commit_insert_mutex_lock);

        if should_commit {
            info!(insert_batch_commit_size=?self.insert_batch_commit_size, "insert_batch_commit_size reached, committing");
            self.commit().await?;
        }

        trace!(offset=?offset, "Updated");

        Ok(())
    }

    pub async fn search(
        &self,
        read_api_key: ApiKey,
        collection_id: CollectionId,
        search_params: SearchParams,
    ) -> Result<SearchResult> {
        let limit = search_params.limit;
        let offset = search_params.offset;

        let has_filter = !search_params.where_filter.is_empty();
        let has_facets = !search_params.facets.is_empty();

        let collection = self
            .collections
            .get_collection(collection_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("Collection not found"))?;
        collection.check_read_api_key(read_api_key)?;

        let m = SEARCH_CALCULATION_TIME.create(SearchCollectionLabels {
            collection: collection_id.to_string().into(),
            mode: search_params.mode.as_str(),
            has_filter: if has_filter { "true" } else { "false" },
            has_facet: if has_facets { "true" } else { "false" },
        });

        let token_scores = collection.search(&search_params).await?;

        let facets = if has_facets {
            Some(
                collection
                    .calculate_facets(&token_scores, &search_params)
                    .await?,
            )
        } else {
            None
        };

        let count = token_scores.len();

        let top_results: Vec<TokenScore> = top_n(token_scores, limit.0 + offset.0);
        trace!("Top results: {:?}", top_results);

        let result = top_results
            .into_iter()
            .skip(offset.0)
            .take(limit.0)
            .collect::<Vec<_>>();

        let docs = self
            .document_storage
            .get_documents_by_ids(result.iter().map(|m| m.document_id).collect())
            .await?;

        trace!("Calculates hits");
        let hits: Vec<_> = result
            .into_iter()
            .zip(docs)
            .map(|(token_score, document)| {
                let id = document
                    .as_ref()
                    .and_then(|d| d.id.clone())
                    .unwrap_or_default();
                SearchResultHit {
                    id,
                    score: token_score.score,
                    document,
                }
            })
            .collect();

        drop(m);

        Ok(SearchResult {
            count,
            hits,
            facets,
        })
    }

    pub async fn nlp_search(
        &self,
        read_side: State<Arc<ReadSide>>,
        read_api_key: ApiKey,
        collection_id: CollectionId,
        search_params: NLPSearchRequest,
    ) -> Result<Vec<SearchResult>> {
        let collection = self
            .collections
            .get_collection(collection_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("Collection not found"))?;
        collection.check_read_api_key(read_api_key)?;

        let collection_stats = self
            .collection_stats(
                read_api_key,
                collection_id,
                CollectionStatsRequest { with_keys: true },
            )
            .await?;

        let search_results = collection
            .nlp_search(
                read_side.clone(),
                read_api_key,
                collection_id,
                &search_params,
                collection_stats,
            )
            .await?;

        Ok(search_results)
    }

    pub async fn nlp_search_stream(
        &self,
        read_side: State<Arc<ReadSide>>,
        read_api_key: ApiKey,
        collection_id: CollectionId,
        search_params: NLPSearchRequest,
    ) -> Result<impl Stream<Item = Result<AdvancedAutoQuerySteps>>> {
        let collection = self
            .collections
            .get_collection(collection_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("Collection not found"))?;
        collection.check_read_api_key(read_api_key)?;

        let collection_stats = self
            .collection_stats(
                read_api_key,
                collection_id,
                CollectionStatsRequest { with_keys: true },
            )
            .await?;

        collection
            .nlp_search_stream(
                read_side.clone(),
                read_api_key,
                collection_id,
                &search_params,
                collection_stats,
            )
            .await
    }

    // This is wrong. We should not expose the ai service to the read side.
    // @todo: Remove this method.
    pub fn get_ai_service(&self) -> Arc<AIService> {
        self.collections.get_ai_service()
    }

    // This is wrong. We should not expose the vllm service to the read side.
    // @todo: Remove this method.
    pub fn get_llm_service(&self) -> Arc<LLMService> {
        self.llm_service.clone()
    }

    pub async fn get_system_prompt(
        &self,
        read_api_key: ApiKey,
        collection_id: CollectionId,
        system_prompt_id: String,
    ) -> Result<Option<SystemPrompt>> {
        self.check_read_api_key(collection_id, read_api_key).await?;

        self.system_prompts
            .get(collection_id, system_prompt_id)
            .await
    }

    pub async fn has_system_prompts(
        &self,
        read_api_key: ApiKey,
        collection_id: CollectionId,
    ) -> Result<bool> {
        self.check_read_api_key(collection_id, read_api_key).await?;

        self.system_prompts.has_system_prompts(collection_id).await
    }

    pub async fn perform_system_prompt_selection(
        &self,
        read_api_key: ApiKey,
        collection_id: CollectionId,
    ) -> Result<Option<SystemPrompt>> {
        self.check_read_api_key(collection_id, read_api_key).await?;

        self.system_prompts
            .perform_system_prompt_selection(collection_id)
            .await
    }

    pub async fn get_all_system_prompts_by_collection(
        &self,
        read_api_key: ApiKey,
        collection_id: CollectionId,
    ) -> Result<Vec<SystemPrompt>> {
        self.check_read_api_key(collection_id, read_api_key).await?;

        self.system_prompts.list_by_collection(collection_id).await
    }

    pub async fn get_segment(
        &self,
        read_api_key: ApiKey,
        collection_id: CollectionId,
        segment_id: String,
    ) -> Result<Option<Segment>> {
        self.check_read_api_key(collection_id, read_api_key).await?;
        self.segments.get(collection_id, segment_id).await
    }

    pub async fn get_all_segments_by_collection(
        &self,
        read_api_key: ApiKey,
        collection_id: CollectionId,
    ) -> Result<Vec<Segment>> {
        self.check_read_api_key(collection_id, read_api_key).await?;
        self.segments.list_by_collection(collection_id).await
    }

    pub async fn perform_segment_selection(
        &self,
        read_api_key: ApiKey,
        collection_id: CollectionId,
        conversation: Option<Vec<InteractionMessage>>,
        llm_config: Option<InteractionLLMConfig>,
    ) -> Result<Option<SelectedSegment>> {
        self.check_read_api_key(collection_id, read_api_key).await?;

        self.segments
            .perform_segment_selection(collection_id, conversation, llm_config)
            .await
    }

    pub async fn perform_trigger_selection(
        &self,
        read_api_key: ApiKey,
        collection_id: CollectionId,
        conversation: Option<Vec<InteractionMessage>>,
        triggers: Vec<Trigger>,
        llm_config: Option<InteractionLLMConfig>,
    ) -> Result<Option<SelectedTrigger>> {
        self.check_read_api_key(collection_id, read_api_key).await?;

        self.triggers
            .perform_trigger_selection(collection_id, conversation, triggers, llm_config)
            .await
    }

    pub async fn get_all_triggers_by_segment(
        &self,
        read_api_key: ApiKey,
        collection_id: CollectionId,
        segment_id: String,
    ) -> Result<Vec<Trigger>> {
        self.check_read_api_key(collection_id, read_api_key).await?;
        self.triggers
            .list_by_segment(collection_id, segment_id)
            .await
    }

    pub async fn get_trigger(
        &self,
        read_api_key: ApiKey,
        collection_id: CollectionId,
        trigger_id: String,
    ) -> Result<Option<Trigger>> {
        self.check_read_api_key(collection_id, read_api_key).await?;
        self.triggers.get(collection_id, trigger_id).await
    }

    pub async fn get_all_triggers_by_collection(
        &self,
        read_api_key: ApiKey,
        collection_id: CollectionId,
    ) -> Result<Vec<Trigger>> {
        self.check_read_api_key(collection_id, read_api_key).await?;

        self.triggers.list_by_collection(collection_id).await
    }

    pub async fn get_search_mode(
        &self,
        query: String,
        llm_config: Option<InteractionLLMConfig>,
    ) -> Result<SearchMode> {
        let search_mode: String = self
            .llm_service
            .run_known_prompt(
                llms::KnownPrompts::Autoquery,
                vec![("query".to_string(), query.clone())],
                llm_config,
            )
            .await?;
        let parsed_mode: SearchModeResult = serde_json::from_str(&search_mode)?;

        Ok(SearchMode::from_str(&parsed_mode.mode, query))
    }

    pub async fn check_read_api_key(
        &self,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    ) -> Result<()> {
        let collection = self
            .collections
            .get_collection(collection_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("Collection not found"))?;

        collection.check_read_api_key(read_api_key)
    }

    pub fn is_gpu_overloaded(&self) -> bool {
        match self.local_gpu_manager.is_overloaded() {
            Ok(overloaded) => overloaded,
            Err(e) => {
                error!(errpr = ?e, "Cannot check if GPU is overloaded. This may be due to GPU malfunction. Forcing inference on remote LLMs for safety.");
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

    pub fn get_default_llm_config(&self) -> (RemoteLLMProvider, String) {
        (RemoteLLMProvider::OramaCore, self.llm_service.model.clone())
    }

    pub async fn get_tool(
        &self,
        read_api_key: ApiKey,
        collection_id: CollectionId,
        tool_id: String,
    ) -> Result<Option<Tool>> {
        self.check_read_api_key(collection_id, read_api_key).await?;
        self.tools.get(collection_id, tool_id).await
    }

    pub async fn get_all_tools_by_collection(
        &self,
        read_api_key: ApiKey,
        collection_id: CollectionId,
    ) -> Result<Vec<Tool>> {
        self.check_read_api_key(collection_id, read_api_key).await?;
        self.tools.list_by_collection(collection_id).await
    }

    pub async fn execute_tools(
        &self,
        read_api_key: ApiKey,
        collection_id: CollectionId,
        messages: Vec<InteractionMessage>,
        tool_ids: Option<Vec<String>>,
        llm_config: Option<InteractionLLMConfig>,
    ) -> Result<Option<Vec<ToolExecutionReturnType>>> {
        self.check_read_api_key(collection_id, read_api_key).await?;
        self.tools
            .execute_tools(collection_id, messages, tool_ids, llm_config)
            .await
    }
}

fn top_n(map: HashMap<DocumentId, f32>, n: usize) -> Vec<TokenScore> {
    let mut capped_heap = CappedHeap::new(n);

    for (key, value) in map {
        let k = match NotNan::new(value) {
            Ok(k) => k,
            Err(_) => continue,
        };
        let v = key;
        capped_heap.insert(k, v);
    }

    let result: Vec<TokenScore> = capped_heap
        .into_top()
        .map(|(value, key)| TokenScore {
            document_id: key,
            score: value.into_inner(),
        })
        .collect();

    result
}

fn default_insert_batch_commit_size() -> u64 {
    300
}

fn start_commit_loop(
    read_side: Arc<ReadSide>,
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
                "{:?} time reached. Committing read side",
                insert_batch_commit_size.clone()
            );
            if let Err(e) = read_side.commit().await {
                error!(error = ?e, "Cannot commit read side");
            }
        }

        if let Err(e) = stop_done_sender.send(()).await {
            error!(error = ?e, "Cannot send stop signal to read side");
        }
    });
}

fn start_receive_operations(
    read_side: Arc<ReadSide>,
    mut operation_receiver: OperationReceiver,
    stop_receiver: tokio::sync::broadcast::Receiver<()>,
    stop_done_sender: tokio::sync::mpsc::Sender<()>,
) {
    tokio::task::spawn(async move {
        use backoff::future::retry;
        use backoff::ExponentialBackoff;
        use core::pin::pin;

        let mut stopped = false;

        let mut stop_receiver = pin!(stop_receiver);

        info!("Starting operation receiver");
        'outer: loop {
            loop {
                let op = tokio::select! {
                    _ = stop_receiver.recv() => {
                        stopped = true;
                        info!("Stopping operation receiver");
                        break 'outer;
                    }
                    op = operation_receiver.recv() => {
                        op
                    }
                };

                let op = match op {
                    None => {
                        warn!("Operation receiver is closed");
                        break;
                    }
                    Some(op) => op,
                };
                let op = match op {
                    Ok(op) => op,
                    Err(e) => {
                        // If there's a deserialization error, should we skip it or something different?
                        // TODO: think about it
                        error!(error = ?e, "Cannot receive operation");
                        continue;
                    }
                };
                trace!(?op, "Received operation");
                if let Err(e) = read_side.update(op).await {
                    error!(error = ?e, "Cannot update read side");
                    e.chain()
                        .skip(1)
                        .for_each(|cause| eprintln!("because: {}", cause));
                }

                trace!("Operation applied");
            }

            warn!("Operation receiver is closed.");

            if !operation_receiver.should_reconnect() {
                break;
            }

            warn!("Reconnecting...");

            let arc = Arc::new(RwLock::new(&mut operation_receiver));
            let op = || async {
                let arc = arc.clone();
                let mut operation_receiver = arc.write().await;
                operation_receiver
                    .reconnect()
                    .await
                    .map_err(|e| backoff::Error::Transient {
                        err: e,
                        retry_after: None,
                    })?;
                Ok::<(), backoff::Error<anyhow::Error>>(())
            };

            match retry(ExponentialBackoff::default(), op).await {
                Ok(_) => {}
                Err(e) => {
                    error!(error = ?e, "Cannot reconnect to operation receiver");
                    break;
                }
            };

            info!("Reconnected to operation receiver");
        }

        if !stopped {
            error!("Read side stopped to receive operations. It is disconnected and will not be able to update the read side");
        }

        if let Err(e) = stop_done_sender.send(()).await {
            error!(error = ?e, "Cannot send stop signal to read side");
        }
    });
}

#[derive(Deserialize, Serialize, Debug)]
struct ReadInfoV1 {
    offset: Offset,
}

#[derive(Deserialize, Serialize, Debug)]
enum ReadInfo {
    V1(ReadInfoV1),
}

#[cfg(test)]
mod tests {
    use crate::collection_manager::sides::read::collection::CollectionReader;

    use super::*;

    #[test]
    fn test_side_read_sync_send() {
        fn assert_sync_send<T: Sync + Send>() {}
        assert_sync_send::<CollectionsReader>();
        assert_sync_send::<CollectionReader>();
    }

    #[test]
    fn test_top_n() {
        let search_result = HashMap::from([
            (DocumentId(1), 0.1),
            (DocumentId(2), 0.2),
            (DocumentId(3), 0.3),
            (DocumentId(4), 0.4),
            (DocumentId(5), 0.5),
            (DocumentId(6), 0.6),
            (DocumentId(7), 0.7),
            (DocumentId(8), 0.8),
            (DocumentId(9), 0.9),
            (DocumentId(10), 1.0),
            (DocumentId(11), 1.1),
            (DocumentId(12), 1.2),
            (DocumentId(13), 1.3),
            (DocumentId(14), 1.4),
            (DocumentId(15), 1.5),
        ]);

        let r1 = top_n(search_result.clone(), 5);
        assert_eq!(r1.len(), 5);
        assert_eq!(
            vec![
                DocumentId(15),
                DocumentId(14),
                DocumentId(13),
                DocumentId(12),
                DocumentId(11),
            ],
            r1.iter().map(|x| x.document_id).collect::<Vec<_>>()
        );

        let r2 = top_n(search_result.clone(), 10);
        assert_eq!(r2.len(), 10);
        assert_eq!(
            vec![
                DocumentId(15),
                DocumentId(14),
                DocumentId(13),
                DocumentId(12),
                DocumentId(11),
                DocumentId(10),
                DocumentId(9),
                DocumentId(8),
                DocumentId(7),
                DocumentId(6),
            ],
            r2.iter().map(|x| x.document_id).collect::<Vec<_>>()
        );
    }
}
