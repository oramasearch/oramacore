mod analytics;
mod collection;
mod collections;
mod context;
pub mod document_storage;
mod index;
mod logs;
pub mod notify;
mod search;
pub mod sort;

use axum::extract::State;
use duration_string::DurationString;
use futures::Stream;
pub use index::*;
use oramacore_lib::hook_storage::{HookReader, HookReaderError};
pub use sort::SortContext;

pub use collection::CollectionStats;
use duration_str::deserialize_duration;
use notify::NotifierConfig;
use orama_js_pool::OutputChannel;
use std::ops::Deref;
use std::sync::Arc;
use std::time::Duration;
use std::{collections::HashMap, path::PathBuf};
use tokio::time::{Instant, MissedTickBehavior};

pub use analytics::{
    AnalyticsHolder, AnalyticsMetadataFromRequest, InteractionAnalyticEventV1,
    OramaCoreAnalyticConfig, OramaCoreAnalytics, SearchAnalyticEventOrigin,
};

use anyhow::{Context, Result};
pub use collection::{IndexFieldStats, IndexFieldStatsType};
use collections::CollectionsReader;
use document_storage::{DocumentStorage, DocumentStorageConfig};
use serde::{Deserialize, Serialize};
use tracing::{error, info, trace, warn};

use crate::ai::advanced_autoquery::{AdvancedAutoQuerySteps, QueryMappedSearchResult};
use crate::ai::gpu::LocalGPUManager;
use crate::ai::llms::{self, KnownPrompts, LLMService};
use crate::ai::tools::{CollectionToolsRuntime, ToolError, ToolsRuntime};
use crate::ai::training_sets::{TrainingDestination, TrainingSetInterface};
use crate::ai::RemoteLLMProvider;
pub use crate::collection_manager::sides::read::context::ReadSideContext;
use crate::collection_manager::sides::read::logs::HookLogs;
use crate::collection_manager::sides::read::notify::Notifier;
use crate::collection_manager::sides::read::search::Search;
use crate::lock::{OramaAsyncLock, OramaAsyncMutex};
use crate::metrics::operations::OPERATION_COUNT;
use crate::metrics::Empty;
use crate::python::embeddings::EmbeddingsService;
use crate::python::PythonService;
use crate::types::CollectionId;
use crate::types::{
    ApiKey, CollectionStatsRequest, InteractionLLMConfig, SearchMode, SearchModeResult,
    SearchResult,
};
use crate::types::{IndexId, NLPSearchRequest};
use oramacore_lib::fs::BufferedFile;
use oramacore_lib::generic_kv::{KVConfig, KV};
use oramacore_lib::nlp::NLPService;
pub use search::SearchRequest;

use super::system_prompts::{SystemPrompt, SystemPromptInterface};
use super::{
    InputSideChannelType, Offset, OperationReceiver, OperationReceiverCreator, WriteOperation,
};
pub use collections::CollectionReadLock;
use thiserror::Error;

#[derive(Deserialize, Clone)]
pub struct ReadSideConfig {
    pub master_api_key: Option<ApiKey>,
    pub analytics: Option<OramaCoreAnalyticConfig>,
    pub input: InputSideChannelType,
    pub config: IndexesConfig,
}

#[derive(Debug, Deserialize, Clone, Copy)]
pub struct OffloadFieldConfig {
    pub unload_window: DurationString,
    pub slot_count_exp: u8,
    pub slot_size_exp: u8,
}

#[derive(Debug, Deserialize, Clone)]
pub struct IndexesConfig {
    pub data_dir: PathBuf,
    #[serde(default = "default_insert_batch_commit_size")]
    pub insert_batch_commit_size: u64,
    #[serde(deserialize_with = "deserialize_duration")]
    pub commit_interval: Duration,
    pub notifier: Option<NotifierConfig>,
    #[serde(default = "default_offload_field")]
    pub offload_field: OffloadFieldConfig,
}

#[derive(Error, Debug)]
pub enum ReadError {
    #[error("Generic error: {0}")]
    Generic(#[from] anyhow::Error),
    #[error("Not found {0}")]
    NotFound(CollectionId),
    #[error("Index not found {0}")]
    IndexNotFound(CollectionId, IndexId),
    #[error("Hook error: {0:?}")]
    Hook(#[from] HookReaderError),
    #[error("Unknown field for filter: {0:?}")]
    FilterFieldNotFound(String),
    #[error("Unknown field for sort: {0:?}")]
    SortFieldNotFound(String),
    #[error("Cannot sort by {0:?}. Only number, date or boolean fields are supported for sorting, but got {1:?} for property {0:?}")]
    InvalidSortField(String, String),
}

pub struct ReadSide {
    collections: CollectionsReader,
    document_storage: DocumentStorage,
    operation_counter: OramaAsyncLock<u64>,
    insert_batch_commit_size: u64,
    data_dir: PathBuf,
    live_offset: OramaAsyncLock<Offset>,
    // This offset will update everytime a change is made to the read side.
    commit_insert_mutex: OramaAsyncMutex<Offset>,
    master_api_key: Option<ApiKey>,
    system_prompts: SystemPromptInterface,
    training_sets: TrainingSetInterface,
    tools: ToolsRuntime,
    kv: Arc<KV>,
    llm_service: Arc<LLMService>,
    local_gpu_manager: Arc<LocalGPUManager>,

    hook_logs: HookLogs,

    analytics_storage: Option<OramaCoreAnalytics>,

    // Handle to stop the read side
    // This is used to stop the read side when the server is shutting down
    stop_sender: tokio::sync::broadcast::Sender<()>,
    stop_done_receiver: OramaAsyncLock<tokio::sync::mpsc::Receiver<()>>,

    python_service: Arc<PythonService>,
}

impl ReadSide {
    pub async fn try_load(
        operation_receiver_creator: OperationReceiverCreator,
        nlp_service: Arc<NLPService>,
        llm_service: Arc<LLMService>,
        config: ReadSideConfig,
        local_gpu_manager: Arc<LocalGPUManager>,
        python_service: Arc<PythonService>,
    ) -> Result<Arc<Self>> {
        let mut document_storage = DocumentStorage::try_new(DocumentStorageConfig {
            data_dir: config.config.data_dir.join("docs"),
        })
        .await
        .context("Cannot create document storage")?;

        let insert_batch_commit_size = config.config.insert_batch_commit_size;
        let commit_interval = config.config.commit_interval;
        let data_dir = config.config.data_dir.clone();

        let mut notifier = None;
        if let Some(notifier_config) = &config.config.notifier {
            let n = Notifier::try_new(notifier_config).context("Cannot create notifier")?;
            notifier = Some(n);
        }

        let context = ReadSideContext {
            python_service: python_service.clone(),
            nlp_service: nlp_service.clone(),
            llm_service: llm_service.clone(),
            notifier,
        };

        let collections_reader = CollectionsReader::try_load(context, config.config)
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
        let system_prompts = SystemPromptInterface::new(kv.clone(), llm_service.clone());
        let training_sets = TrainingSetInterface::new(kv.clone());
        let tools = ToolsRuntime::new(kv.clone(), llm_service.clone());

        let (stop_done_sender, stop_done_receiver) = tokio::sync::mpsc::channel(1);
        let (stop_sender, _) = tokio::sync::broadcast::channel(1);
        let commit_loop_receiver = stop_sender.subscribe();
        let receive_operation_loop_receiver = stop_sender.subscribe();

        let analytics_storage = if let Some(config) = config.analytics {
            Some(
                OramaCoreAnalytics::try_new(data_dir.join("analytics_new"), config)
                    .context("Cannot create analytics storage")?,
            )
        } else {
            None
        };

        let read_side = ReadSide {
            collections: collections_reader,
            document_storage,
            operation_counter: OramaAsyncLock::new("operation_counter", Default::default()),
            insert_batch_commit_size,
            live_offset: OramaAsyncLock::new("live_offset", last_offset),
            commit_insert_mutex: OramaAsyncMutex::new("commit_insert_mutex", last_offset),
            master_api_key: config.master_api_key,
            system_prompts,
            training_sets,
            tools,
            kv,
            llm_service,
            local_gpu_manager,

            hook_logs: HookLogs::new(),
            analytics_storage,

            data_dir,

            stop_sender,
            stop_done_receiver: OramaAsyncLock::new("stop_done_receiver", stop_done_receiver),
            python_service,
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
        let mut stop_done_receiver = self.stop_done_receiver.write("stop").await;
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
        let mut commit_insert_mutex_lock = self.commit_insert_mutex.lock("commit").await;

        let live_offset = self.live_offset.write("commit").await;
        let offset = **live_offset;
        drop(live_offset);

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

        **commit_insert_mutex_lock = offset;

        drop(commit_insert_mutex_lock);

        Ok(())
    }

    pub async fn collection_stats(
        &self,
        read_api_key: ApiKey,
        collection_id: CollectionId,
        req: CollectionStatsRequest,
    ) -> Result<CollectionStats, ReadError> {
        let collection = self
            .collections
            .get_collection(collection_id)
            .await
            .ok_or_else(|| ReadError::NotFound(collection_id))?;
        collection.check_read_api_key(read_api_key, self.master_api_key)?;

        collection.stats(req).await
    }

    pub async fn update(&self, (offset, op): (Offset, WriteOperation)) -> Result<()> {
        trace!(offset=?offset, "Updating read side");

        let m = OPERATION_COUNT.create(Empty);

        // We stop commit operations while we are updating
        // The lock is released at the end of this function
        let commit_insert_mutex_lock = self.commit_insert_mutex.lock("update").await;

        // Already applied. We can skip this operation.
        if offset <= **commit_insert_mutex_lock && !commit_insert_mutex_lock.is_zero() {
            warn!(offset=?offset, "Operation already applied. Skipping");
            return Ok(());
        }

        let mut live_offset = self.live_offset.write("update").await;
        **live_offset = offset;
        drop(live_offset);

        match op {
            WriteOperation::CreateCollection {
                id,
                read_api_key,
                default_locale,
                description,
                mcp_description,
            } => {
                self.collections
                    .create_collection(
                        id,
                        description,
                        mcp_description,
                        default_locale,
                        read_api_key,
                        None,
                    )
                    .await?;
            }
            WriteOperation::CreateCollection2 {
                id,
                read_api_key,
                write_api_key,
                default_locale,
                description,
                mcp_description,
            } => {
                self.collections
                    .create_collection(
                        id,
                        description,
                        mcp_description,
                        default_locale,
                        read_api_key,
                        Some(write_api_key),
                    )
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
                    .update(offset.0, op)
                    .await
                    .context("Cannot insert into KV")?;
            }
        }

        drop(m);

        let mut lock = self.operation_counter.write("update").await;
        **lock += 1;
        let should_commit = if **lock >= self.insert_batch_commit_size {
            **lock = 0;
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
        request: SearchRequest,
    ) -> Result<SearchResult, ReadError> {
        let collection = self
            .collections
            .get_collection(collection_id)
            .await
            .ok_or_else(|| ReadError::NotFound(collection_id))?;
        collection.check_read_api_key(read_api_key, self.master_api_key)?;

        let search = Search::new(
            &collection,
            &self.document_storage,
            self.analytics_storage.as_ref(),
            request,
        );

        search.execute().await
    }

    pub async fn nlp_search(
        &self,
        read_side: State<Arc<ReadSide>>,
        read_api_key: ApiKey,
        collection_id: CollectionId,
        search_params: NLPSearchRequest,
        log_sender: Option<Arc<tokio::sync::broadcast::Sender<(OutputChannel, String)>>>,
    ) -> Result<Vec<QueryMappedSearchResult>, ReadError> {
        let collection = self
            .collections
            .get_collection(collection_id)
            .await
            .ok_or_else(|| ReadError::NotFound(collection_id))?;
        collection.check_read_api_key(read_api_key, self.master_api_key)?;

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
                log_sender,
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
        log_sender: Option<Arc<tokio::sync::broadcast::Sender<(OutputChannel, String)>>>,
    ) -> Result<impl Stream<Item = Result<AdvancedAutoQuerySteps>>, ReadError> {
        let collection = self
            .collections
            .get_collection(collection_id)
            .await
            .ok_or_else(|| ReadError::NotFound(collection_id))?;
        collection.check_read_api_key(read_api_key, self.master_api_key)?;

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
                log_sender,
            )
            .await
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

    pub async fn get_search_mode(
        &self,
        query: String,
        llm_config: Option<InteractionLLMConfig>,
    ) -> Result<SearchMode> {
        let search_mode: String = self
            .llm_service
            .run_known_prompt(
                llms::KnownPrompts::Autoquery,
                vec![],
                vec![("query".to_string(), query.clone())],
                None,
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
    ) -> Result<(), ReadError> {
        let collection = self
            .collections
            .get_collection(collection_id)
            .await
            .ok_or_else(|| ReadError::NotFound(collection_id))?;

        collection.check_read_api_key(read_api_key, self.master_api_key)
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

    pub async fn get_tools_interface(
        &self,
        read_api_key: ApiKey,
        collection_id: CollectionId,
    ) -> Result<CollectionToolsRuntime, ToolError> {
        self.check_read_api_key(collection_id, read_api_key).await?;

        Ok(CollectionToolsRuntime::new(
            self.tools.clone(),
            collection_id,
        ))
    }

    pub async fn get_hook_storage<'s>(
        &'s self,
        read_api_key: ApiKey,
        collection_id: CollectionId,
    ) -> Result<HookReaderLock<'s>, ReadError> {
        let collection = self
            .collections
            .get_collection(collection_id)
            .await
            .ok_or_else(|| ReadError::NotFound(collection_id))?;
        collection.check_read_api_key(read_api_key, self.master_api_key)?;

        Ok(HookReaderLock { collection })
    }

    pub fn get_hook_logs(&self) -> &HookLogs {
        &self.hook_logs
    }

    pub fn get_analytics_logs(&self) -> Option<OramaCoreAnalytics> {
        self.analytics_storage.clone()
    }

    pub async fn get_default_system_prompt(
        &self,
        collection_id: CollectionId,
        read_api_key: ApiKey,
        system_prompt_id: String,
    ) -> Result<String, ReadError> {
        let collection = self
            .collections
            .get_collection(collection_id)
            .await
            .ok_or_else(|| ReadError::NotFound(collection_id))?;
        collection.check_read_api_key(read_api_key, self.master_api_key)?;

        let known_prompt: KnownPrompts = system_prompt_id
            .as_str()
            .try_into()
            .map_err(|e| ReadError::Generic(anyhow::anyhow!("Unknown system prompt ID: {}", e)))?;

        let prompt = known_prompt.get_prompts().system;

        Ok(prompt)
    }

    pub async fn list_training_data_for(
        &self,
        collection_id: CollectionId,
        read_api_key: ApiKey,
        training_destination: TrainingDestination,
    ) -> Result<Vec<String>, ReadError> {
        let collection = self
            .collections
            .get_collection(collection_id)
            .await
            .ok_or_else(|| ReadError::NotFound(collection_id))?;
        collection.check_read_api_key(read_api_key, self.master_api_key)?;

        match self
            .training_sets
            .list_training_sets(collection_id, training_destination)
            .await
        {
            Ok(training_sets) => Ok(training_sets),
            Err(e) => Err(ReadError::Generic(anyhow::anyhow!(
                "Failed to list training data: {}",
                e
            ))),
        }
    }

    pub async fn get_training_data_for(
        &self,
        collection_id: CollectionId,
        read_api_key: ApiKey,
        training_destination: TrainingDestination,
    ) -> Option<Result<String, ReadError>> {
        let collection = match self.collections.get_collection(collection_id).await {
            Some(collection) => collection,
            None => return Some(Err(ReadError::NotFound(collection_id))),
        };

        if let Err(e) = collection.check_read_api_key(read_api_key, self.master_api_key) {
            return Some(Err(e));
        }

        match self
            .training_sets
            .get_training_set(collection_id, training_destination)
            .await
        {
            Some(training_data_result) => match training_data_result {
                Ok(training_data) => Some(Ok(training_data)),
                Err(e) => Some(Err(ReadError::Generic(anyhow::anyhow!(
                    "Failed to get training data: {}",
                    e
                )))),
            },
            None => None,
        }
    }

    pub async fn list_pin_rule_ids(
        &self,
        collection_id: CollectionId,
        index_id: IndexId,
        read_api_key: ApiKey,
    ) -> Result<Vec<String>, ReadError> {
        let collection = self
            .collections
            .get_collection(collection_id)
            .await
            .ok_or_else(|| ReadError::NotFound(collection_id))?;
        collection.check_read_api_key(read_api_key, self.master_api_key)?;

        let Some(index) = collection.get_index(index_id).await else {
            return Err(ReadError::IndexNotFound(collection_id, index_id));
        };

        let ids = index.get_pin_rule_ids().await;

        Ok(ids)
    }
}

fn default_insert_batch_commit_size() -> u64 {
    300
}

fn default_offload_field() -> OffloadFieldConfig {
    OffloadFieldConfig {
        unload_window: Duration::from_secs(30 * 60).into(),
        slot_count_exp: 8,
        slot_size_exp: 4,
    }
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
                let offset = op.0;
                if let Err(e) = read_side.update(op).await {
                    error!(error = ?e, "Cannot update read side");
                    e.chain()
                        .skip(1)
                        .for_each(|cause| eprintln!("because: {cause}"));
                }

                info!("Operation applied {:?}", offset);
            }

            warn!("Operation receiver is closed.");

            if !operation_receiver.should_reconnect() {
                break;
            }

            warn!("Reconnecting...");

            let arc = Arc::new(OramaAsyncLock::new(
                "operation_receiver",
                &mut operation_receiver,
            ));
            let op = || async {
                let arc = arc.clone();
                let mut operation_receiver = arc.write("reconnect").await;
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

pub struct HookReaderLock<'guard> {
    collection: CollectionReadLock<'guard>,
}

impl Deref for HookReaderLock<'_> {
    type Target = OramaAsyncLock<HookReader>;

    fn deref(&self) -> &Self::Target {
        self.collection.get_hook_storage()
    }
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
}
