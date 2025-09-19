use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use backoff::{backoff::Backoff, ExponentialBackoffBuilder};
use futures::future::{join_all, Future};
use llm_json::repair_json;
use orama_js_pool::ExecOption;
use regex::Regex;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, Mutex};
use tokio::time::sleep;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{error, info, warn};

use crate::ai::llms::{KnownPrompts, LLMService};
use crate::types::{
    ApiKey, CollectionId, IndexId, InteractionLLMConfig, InteractionMessage, SearchParams,
    SearchResult,
};

use crate::ai::run_hooks::run_before_retrieval;
use crate::collection_manager::sides::read::{
    CollectionStats, ReadSide, SearchAnalyticEventOrigin, SearchRequest,
};

// ==== SSE Event Types ====

#[derive(Debug, Clone, Serialize)]
pub enum AdvancedAutoqueryEvent {
    #[serde(rename = "state_changed")]
    StateChanged {
        state: String,
        message: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        data: Option<serde_json::Value>,
    },
    #[serde(rename = "error")]
    Error {
        error: String,
        state: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_terminal: Option<bool>,
    },
    #[serde(rename = "progress")]
    Progress {
        current_step: serde_json::Value,
        total_steps: usize,
        message: String,
    },
    #[serde(rename = "search_results")]
    SearchResults {
        results: Vec<QueryMappedSearchResult>,
    },
}

// ==== Data Models ====

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct SelectedProperty {
    pub property: String,
    #[serde(rename = "type")]
    pub field_type: String,
    pub get: bool,
}

#[derive(Serialize, Debug, Clone)]
pub struct CollectionSelectedProperties {
    pub selected_properties: Vec<SelectedProperty>,
}

#[derive(Serialize, Debug, Clone)]
pub struct QueryAndProperties {
    pub query: String,
    pub properties: HashMap<String, CollectionSelectedProperties>,
    pub filter_properties: HashMap<String, Vec<String>>,
}

#[derive(Serialize, Debug, Clone)]
pub struct TrackedQuery {
    pub index: usize,
    pub original_query: String,
    pub generated_query_text: String,
    pub search_params: SearchParams,
}

#[derive(Serialize, Debug, Clone)]
pub struct ProcessedTrackedQuery {
    pub index: usize,
    pub original_query: String,
    pub generated_query_text: String,
    pub original_search_params: SearchParams,
    pub processed_search_params: SearchParams,
}

#[derive(Serialize, Debug, Clone)]
pub struct QueryMappedSearchResult {
    pub original_query: String,
    pub generated_query: String,
    pub search_params: SearchParams,
    pub results: Vec<SearchResult>,
    pub query_index: usize,
}

#[derive(Serialize, Debug, Clone)]
pub struct GeneratedAnswer {
    pub answer: String,
    pub context: ProcessedAnswerContext,
}

#[derive(Serialize, Debug, Clone)]
pub struct ProcessedAnswerContext {
    pub conversation: Vec<InteractionMessage>,
    pub search_results: Vec<QueryMappedSearchResult>,
    pub original_system_prompt:
        Option<crate::collection_manager::sides::system_prompts::SystemPrompt>,
    pub processed_system_prompt:
        Option<crate::collection_manager::sides::system_prompts::SystemPrompt>,
}

// ==== Error Types ====

#[derive(Debug, thiserror::Error, Clone)]
pub enum AdvancedAutoqueryError {
    #[error("Failed to initialize: {0}")]
    InitializeError(String),
    #[error("Failed to analyze input: {0}")]
    AnalyzeInputError(String),
    #[error("Failed to select properties: {0}")]
    SelectPropertiesError(String),
    #[error("Failed to combine queries and properties: {0}")]
    CombineQueriesError(String),
    #[error("Failed to generate tracked queries: {0}")]
    GenerateTrackedQueriesError(String),
    #[error("Failed to execute before retrieval hook: {0}")]
    ExecuteBeforeRetrievalHookError(String),
    #[error("Failed to execute searches: {0}")]
    ExecuteSearchesError(String),
    #[error("LLM service error: {0}")]
    LLMServiceError(String),
    #[error("JSON parsing error: {0}")]
    JsonParsingError(String),
    #[error("Collection stats error: {0}")]
    CollectionStatsError(String),
}

// ==== State Machine States ====

#[derive(Debug, Clone)]
pub enum AdvancedAutoqueryFlow {
    Initialize {
        conversation: Vec<InteractionMessage>,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    },
    AnalyzeInput {
        conversation_json: String,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    },
    QueryOptimized {
        optimized_queries: Vec<String>,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    },
    SelectProperties {
        queries: Vec<String>,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    },
    PropertiesSelected {
        queries: Vec<String>,
        selected_properties: Vec<HashMap<String, CollectionSelectedProperties>>,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    },
    CombineQueriesAndProperties {
        queries: Vec<String>,
        properties: Vec<HashMap<String, CollectionSelectedProperties>>,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    },
    QueriesCombined {
        queries_and_properties: Vec<QueryAndProperties>,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    },
    GenerateTrackedQueries {
        queries_and_properties: Vec<QueryAndProperties>,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    },
    TrackedQueriesGenerated {
        tracked_queries: Vec<TrackedQuery>,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    },
    ExecuteBeforeRetrievalHook {
        tracked_queries: Vec<TrackedQuery>,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    },
    HooksExecuted {
        processed_queries: Vec<ProcessedTrackedQuery>,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    },
    ExecuteSearches {
        processed_queries: Vec<ProcessedTrackedQuery>,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    },
    SearchResults {
        results: Vec<QueryMappedSearchResult>,
    },
    Error(AdvancedAutoqueryError),
}

// ==== Configuration ====

#[derive(Debug, Clone)]
pub struct AdvancedAutoqueryConfig {
    pub max_retries: usize,
    pub initial_backoff: Duration,
    pub max_backoff: Duration,
    pub max_concurrent_operations: usize,
    pub timeout: Duration,
    pub llm_timeout: Duration,
}

impl Default for AdvancedAutoqueryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(30),
            max_concurrent_operations: 5,
            timeout: Duration::from_secs(60),
            llm_timeout: Duration::from_secs(30),
        }
    }
}

// ==== Main State Machine ====

pub struct AdvancedAutoqueryStateMachine {
    config: AdvancedAutoqueryConfig,
    state: Arc<Mutex<AdvancedAutoqueryFlow>>,
    retry_count: Arc<Mutex<HashMap<String, usize>>>,
    llm_service: Arc<LLMService>,
    llm_config: Option<InteractionLLMConfig>,
    collection_stats: CollectionStats,
    read_side: Arc<ReadSide>,
    event_sender: Option<mpsc::UnboundedSender<AdvancedAutoqueryEvent>>,
}

impl AdvancedAutoqueryStateMachine {
    pub fn new(
        config: AdvancedAutoqueryConfig,
        llm_service: Arc<LLMService>,
        llm_config: Option<InteractionLLMConfig>,
        collection_stats: CollectionStats,
        read_side: Arc<ReadSide>,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    ) -> Self {
        Self {
            config,
            state: Arc::new(Mutex::new(AdvancedAutoqueryFlow::Initialize {
                conversation: vec![],
                collection_id,
                read_api_key,
            })),
            retry_count: Arc::new(Mutex::new(HashMap::new())),
            llm_service,
            llm_config,
            collection_stats,
            read_side,
            event_sender: None,
        }
    }

    /// Send an event if the event sender is available
    async fn send_event(&self, event: AdvancedAutoqueryEvent) {
        if let Some(sender) = &self.event_sender {
            let _ = sender.send(event);
        }
    }

    /// Convert state to JSON for progress events
    fn state_to_json(&self, state: &AdvancedAutoqueryFlow) -> serde_json::Value {
        match state {
            AdvancedAutoqueryFlow::Initialize { conversation, .. } => {
                serde_json::json!({
                    "type": "Initialize",
                    "conversation_messages": conversation.len()
                })
            }
            AdvancedAutoqueryFlow::AnalyzeInput { .. } => {
                serde_json::json!({
                    "type": "AnalyzeInput"
                })
            }
            AdvancedAutoqueryFlow::QueryOptimized {
                optimized_queries, ..
            } => {
                serde_json::json!({
                    "type": "QueryOptimized",
                    "queries_count": optimized_queries.len()
                })
            }
            AdvancedAutoqueryFlow::SelectProperties { queries, .. } => {
                serde_json::json!({
                    "type": "SelectProperties",
                    "queries_count": queries.len()
                })
            }
            AdvancedAutoqueryFlow::PropertiesSelected {
                queries,
                selected_properties,
                ..
            } => {
                serde_json::json!({
                    "type": "PropertiesSelected",
                    "queries_count": queries.len(),
                    "properties_count": selected_properties.len()
                })
            }
            AdvancedAutoqueryFlow::CombineQueriesAndProperties {
                queries,
                properties,
                ..
            } => {
                serde_json::json!({
                    "type": "CombineQueriesAndProperties",
                    "queries_count": queries.len(),
                    "properties_count": properties.len()
                })
            }
            AdvancedAutoqueryFlow::QueriesCombined {
                queries_and_properties,
                ..
            } => {
                serde_json::json!({
                    "type": "QueriesCombined",
                    "combined_count": queries_and_properties.len()
                })
            }
            AdvancedAutoqueryFlow::GenerateTrackedQueries {
                queries_and_properties,
                ..
            } => {
                serde_json::json!({
                    "type": "GenerateTrackedQueries",
                    "queries_count": queries_and_properties.len()
                })
            }
            AdvancedAutoqueryFlow::TrackedQueriesGenerated {
                tracked_queries, ..
            } => {
                serde_json::json!({
                    "type": "TrackedQueriesGenerated",
                    "tracked_queries_count": tracked_queries.len()
                })
            }
            AdvancedAutoqueryFlow::ExecuteBeforeRetrievalHook {
                tracked_queries, ..
            } => {
                serde_json::json!({
                    "type": "ExecuteBeforeRetrievalHook",
                    "queries_count": tracked_queries.len()
                })
            }
            AdvancedAutoqueryFlow::HooksExecuted {
                processed_queries, ..
            } => {
                serde_json::json!({
                    "type": "HooksExecuted",
                    "processed_queries_count": processed_queries.len()
                })
            }
            AdvancedAutoqueryFlow::ExecuteSearches {
                processed_queries, ..
            } => {
                serde_json::json!({
                    "type": "ExecuteSearches",
                    "queries_count": processed_queries.len()
                })
            }
            AdvancedAutoqueryFlow::SearchResults { results } => {
                serde_json::json!({
                    "type": "SearchResults",
                    "results_count": results.len()
                })
            }
            AdvancedAutoqueryFlow::Error(error) => {
                serde_json::json!({
                    "type": "Error",
                    "error": error.to_string()
                })
            }
        }
    }

    /// Run the state machine with the given input
    pub async fn run(
        &self,
        conversation: Vec<InteractionMessage>,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    ) -> Result<GeneratedAnswer, AdvancedAutoqueryError> {
        info!("Starting advanced search for collection: {}", collection_id);

        // Send initial event
        self.send_event(AdvancedAutoqueryEvent::StateChanged {
            state: "initializing".to_string(),
            message: "Starting advanced autoquery".to_string(),
            data: None,
        })
        .await;

        // Initialize state
        {
            let mut state = self.state.lock().await;
            *state = AdvancedAutoqueryFlow::Initialize {
                conversation,
                collection_id,
                read_api_key,
            };
        }

        let total_steps = 8; // Total number of states in the flow
        let mut current_step = 0;

        loop {
            current_step += 1;
            let current_state = {
                let state = self.state.lock().await;
                state.clone()
            };

            // Send progress event
            let current_step_json = self.state_to_json(&current_state);
            info!("Current step {:?}", current_step_json);
            self.send_event(AdvancedAutoqueryEvent::Progress {
                current_step: current_step_json,
                total_steps,
                message: format!("Processing step {current_step}/{total_steps}"),
            })
            .await;

            match current_state {
                AdvancedAutoqueryFlow::Initialize {
                    conversation,
                    collection_id,
                    read_api_key,
                } => {
                    self.send_event(AdvancedAutoqueryEvent::StateChanged {
                        state: "analyzing_input".to_string(),
                        message: "Analyzing user input".to_string(),
                        data: Some(serde_json::json!({
                            "conversation_messages": conversation.len()
                        })),
                    })
                    .await;
                    self.transition_to_analyze_input(conversation, collection_id, read_api_key)
                        .await?;
                }
                AdvancedAutoqueryFlow::AnalyzeInput {
                    conversation_json,
                    collection_id,
                    read_api_key,
                } => {
                    self.send_event(AdvancedAutoqueryEvent::StateChanged {
                        state: "query_optimized".to_string(),
                        message: "Optimizing queries".to_string(),
                        data: Some(serde_json::json!({
                            "conversation": conversation_json,
                            "conversation_json_length": conversation_json.len()
                        })),
                    })
                    .await;
                    self.transition_to_query_optimized(
                        conversation_json,
                        collection_id,
                        read_api_key,
                    )
                    .await?;
                }
                AdvancedAutoqueryFlow::QueryOptimized {
                    optimized_queries,
                    collection_id,
                    read_api_key,
                } => {
                    self.send_event(AdvancedAutoqueryEvent::StateChanged {
                        state: "select_properties".to_string(),
                        message: "Selecting properties".to_string(),
                        data: Some(serde_json::json!({
                            "optimized_queries": optimized_queries.clone(),
                        })),
                    })
                    .await;
                    self.transition_to_select_properties(
                        optimized_queries,
                        collection_id,
                        read_api_key,
                    )
                    .await?;
                }
                AdvancedAutoqueryFlow::SelectProperties {
                    queries,
                    collection_id,
                    read_api_key,
                } => {
                    self.send_event(AdvancedAutoqueryEvent::StateChanged {
                        state: "properties_selected".to_string(),
                        message: "Properties selected".to_string(),
                        data: Some(serde_json::json!({
                            "queries_count": queries.len()
                        })),
                    })
                    .await;
                    self.transition_to_properties_selected(queries, collection_id, read_api_key)
                        .await?;
                }
                AdvancedAutoqueryFlow::PropertiesSelected {
                    queries,
                    selected_properties,
                    collection_id,
                    read_api_key,
                } => {
                    self.send_event(AdvancedAutoqueryEvent::StateChanged {
                        state: "combine_queries".to_string(),
                        message: "Combining queries and properties".to_string(),
                        data: Some(serde_json::json!({
                            "queries_count": queries.len(),
                            "selected_properties_count": selected_properties.len(),
                            "selected_properties": selected_properties
                        })),
                    })
                    .await;
                    self.transition_to_combine_queries(
                        queries,
                        selected_properties,
                        collection_id,
                        read_api_key,
                    )
                    .await?;
                }
                AdvancedAutoqueryFlow::CombineQueriesAndProperties {
                    queries,
                    properties,
                    collection_id,
                    read_api_key,
                } => {
                    self.send_event(AdvancedAutoqueryEvent::StateChanged {
                        state: "queries_combined".to_string(),
                        message: "Queries combined".to_string(),
                        data: Some(serde_json::json!({
                            "queries_count": queries.len(),
                            "properties_count": properties.len()
                        })),
                    })
                    .await;
                    self.transition_to_queries_combined(
                        queries,
                        properties,
                        collection_id,
                        read_api_key,
                    )
                    .await?;
                }
                AdvancedAutoqueryFlow::QueriesCombined {
                    queries_and_properties,
                    collection_id,
                    read_api_key,
                } => {
                    self.send_event(AdvancedAutoqueryEvent::StateChanged {
                        state: "generate_tracked_queries".to_string(),
                        message: "Generating tracked queries".to_string(),
                        data: Some(serde_json::json!({
                            "queries_and_properties": queries_and_properties
                        })),
                    })
                    .await;
                    self.transition_to_generate_tracked_queries(
                        queries_and_properties,
                        collection_id,
                        read_api_key,
                    )
                    .await?;
                }
                AdvancedAutoqueryFlow::GenerateTrackedQueries {
                    queries_and_properties,
                    collection_id,
                    read_api_key,
                } => {
                    self.send_event(AdvancedAutoqueryEvent::StateChanged {
                        state: "tracked_queries_generated".to_string(),
                        message: "Tracked queries generated".to_string(),
                        data: Some(serde_json::json!({
                            "queries_count": queries_and_properties.len()
                        })),
                    })
                    .await;
                    self.transition_to_tracked_queries_generated(
                        queries_and_properties,
                        collection_id,
                        read_api_key,
                    )
                    .await?;
                }
                AdvancedAutoqueryFlow::TrackedQueriesGenerated {
                    tracked_queries,
                    collection_id,
                    read_api_key,
                } => {
                    self.send_event(AdvancedAutoqueryEvent::StateChanged {
                        state: "execute_before_retrieval_hook".to_string(),
                        message: "Executing before retrieval hook".to_string(),
                        data: Some(serde_json::json!({
                            "tracked_queries": tracked_queries
                        })),
                    })
                    .await;
                    self.transition_to_execute_before_retrieval_hook(
                        tracked_queries,
                        collection_id,
                        read_api_key,
                    )
                    .await?;
                }
                AdvancedAutoqueryFlow::ExecuteBeforeRetrievalHook {
                    tracked_queries,
                    collection_id,
                    read_api_key,
                } => {
                    self.send_event(AdvancedAutoqueryEvent::StateChanged {
                        state: "hooks_executed".to_string(),
                        message: "Hooks executed".to_string(),
                        data: Some(serde_json::json!({
                            "tracked_queries_count": tracked_queries.len()
                        })),
                    })
                    .await;
                    self.transition_to_hooks_executed(tracked_queries, collection_id, read_api_key)
                        .await?;
                }
                AdvancedAutoqueryFlow::HooksExecuted {
                    processed_queries,
                    collection_id,
                    read_api_key,
                } => {
                    self.send_event(AdvancedAutoqueryEvent::StateChanged {
                        state: "execute_searches".to_string(),
                        message: "Executing searches".to_string(),
                        data: Some(serde_json::json!({
                            "processed_queries_count": processed_queries.len()
                        })),
                    })
                    .await;
                    self.transition_to_execute_searches(
                        processed_queries,
                        collection_id,
                        read_api_key,
                    )
                    .await?;
                }
                AdvancedAutoqueryFlow::ExecuteSearches {
                    processed_queries,
                    collection_id,
                    read_api_key,
                } => {
                    self.send_event(AdvancedAutoqueryEvent::StateChanged {
                        state: "search_results".to_string(),
                        message: "Search results obtained".to_string(),
                        data: Some(serde_json::json!({
                            "processed_queries_count": processed_queries.len()
                        })),
                    })
                    .await;
                    self.transition_to_search_results(
                        processed_queries,
                        collection_id,
                        read_api_key,
                    )
                    .await?;
                }
                AdvancedAutoqueryFlow::SearchResults { results } => {
                    info!(
                        "Advanced search completed successfully with {} results",
                        results.len()
                    );
                    // Send search results event
                    self.send_event(AdvancedAutoqueryEvent::SearchResults {
                        results: results.clone(),
                    })
                    .await;
                    // Send completion event
                    self.send_event(AdvancedAutoqueryEvent::StateChanged {
                        state: "completed".to_string(),
                        message: "Advanced autoquery completed".to_string(),
                        data: Some(serde_json::json!({
                            "results_count": results.len(),
                            "results": results
                        })),
                    })
                    .await;
                    // Return the search results instead of continuing to answer generation
                    return Ok(GeneratedAnswer {
                        answer: String::new(),
                        context: ProcessedAnswerContext {
                            conversation: vec![],
                            search_results: results,
                            original_system_prompt: None,
                            processed_system_prompt: None,
                        },
                    });
                }
                AdvancedAutoqueryFlow::Error(error) => {
                    error!("Advanced search failed: {:?}", error);
                    self.send_event(AdvancedAutoqueryEvent::Error {
                        error: error.to_string(),
                        state: format!("{error:?}"),
                        is_terminal: Some(true),
                    })
                    .await;
                    return Err(error);
                }
            }
        }
    }

    /// Run the state machine with streaming support
    pub async fn run_stream(
        mut self,
        conversation: Vec<InteractionMessage>,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    ) -> Result<UnboundedReceiverStream<AdvancedAutoqueryEvent>, AdvancedAutoqueryError> {
        let (event_sender, event_receiver) = mpsc::unbounded_channel();

        self.event_sender = Some(event_sender);

        // Spawn the state machine in a separate task
        let state_machine = self;
        tokio::spawn(async move {
            let result = state_machine
                .run(conversation, collection_id, read_api_key)
                .await;
            match result {
                Ok(_) => {
                    // Success - the final event will be sent by the state machine
                }
                Err(e) => {
                    // Error - send error event
                    if let Some(sender) = &state_machine.event_sender {
                        let _ = sender.send(AdvancedAutoqueryEvent::Error {
                            error: e.to_string(),
                            state: format!("{e:?}"),
                            is_terminal: Some(true),
                        });
                    }
                }
            }
        });

        Ok(UnboundedReceiverStream::new(event_receiver))
    }

    /// Transition with retry logic
    async fn transition_with_retry<F, Fut, T>(
        &self,
        operation_name: &str,
        operation: F,
    ) -> Result<T, AdvancedAutoqueryError>
    where
        F: Fn() -> Fut + Send + Sync,
        Fut: Future<Output = Result<T, AdvancedAutoqueryError>> + Send,
    {
        let mut backoff = ExponentialBackoffBuilder::new()
            .with_initial_interval(self.config.initial_backoff)
            .with_max_interval(self.config.max_backoff)
            .with_max_elapsed_time(Some(self.config.timeout))
            .build();

        let mut retry_count = 0;
        loop {
            match operation().await {
                Ok(result) => {
                    if retry_count > 0 {
                        info!(
                            "Operation {} succeeded after {} retries",
                            operation_name, retry_count
                        );
                    }
                    return Ok(result);
                }
                Err(e) => {
                    retry_count += 1;
                    if retry_count > self.config.max_retries {
                        error!(
                            "Operation {} failed after {} retries: {:?}",
                            operation_name, retry_count, e
                        );
                        return Err(e);
                    }

                    warn!(
                        "Operation {} failed (attempt {}/{}), retrying: {:?}",
                        operation_name, retry_count, self.config.max_retries, e
                    );

                    // Update retry count
                    {
                        let mut counts = self.retry_count.lock().await;
                        counts.insert(operation_name.to_string(), retry_count);
                    }

                    // Wait before retry
                    if let Some(duration) = Backoff::next_backoff(&mut backoff) {
                        sleep(duration).await;
                    }
                }
            }
        }
    }

    async fn transition_to_analyze_input(
        &self,
        conversation: Vec<InteractionMessage>,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    ) -> Result<(), AdvancedAutoqueryError> {
        let conversation_json = serde_json::to_string(&conversation)
            .map_err(|e| AdvancedAutoqueryError::JsonParsingError(e.to_string()))?;

        let mut state = self.state.lock().await;
        *state = AdvancedAutoqueryFlow::AnalyzeInput {
            conversation_json,
            collection_id,
            read_api_key,
        };
        Ok(())
    }

    async fn transition_to_query_optimized(
        &self,
        conversation_json: String,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    ) -> Result<Vec<String>, AdvancedAutoqueryError> {
        let optimized_queries = self
            .transition_with_retry("analyze_input", || {
                self.analyze_input(conversation_json.clone())
            })
            .await?;

        self.send_event(AdvancedAutoqueryEvent::StateChanged {
            state: "query_optimized".to_string(),
            message: "Optimizing queries".to_string(),
            data: Some(serde_json::json!({
                "optimized_queries": optimized_queries.clone(),
            })),
        })
        .await;

        let mut state = self.state.lock().await;
        *state = AdvancedAutoqueryFlow::QueryOptimized {
            optimized_queries: optimized_queries.clone(),
            collection_id,
            read_api_key,
        };

        Ok(optimized_queries)
    }

    async fn transition_to_select_properties(
        &self,
        optimized_queries: Vec<String>,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    ) -> Result<(), AdvancedAutoqueryError> {
        let selected_properties = self
            .transition_with_retry("select_properties", || {
                self.select_properties(optimized_queries.clone())
            })
            .await?;

        self.send_event(AdvancedAutoqueryEvent::StateChanged {
            state: "properties_selected".to_string(),
            message: "Properties selected".to_string(),
            data: Some(serde_json::json!({
                "queries_count": optimized_queries.len(),
                "selected_properties": selected_properties
            })),
        })
        .await;

        let mut state = self.state.lock().await;
        *state = AdvancedAutoqueryFlow::PropertiesSelected {
            queries: optimized_queries,
            selected_properties,
            collection_id,
            read_api_key,
        };
        Ok(())
    }

    async fn transition_to_properties_selected(
        &self,
        queries: Vec<String>,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    ) -> Result<(), AdvancedAutoqueryError> {
        let selected_properties = self
            .transition_with_retry("select_properties", || {
                self.select_properties(queries.clone())
            })
            .await?;

        self.send_event(AdvancedAutoqueryEvent::StateChanged {
            state: "properties_selected".to_string(),
            message: "Properties selected".to_string(),
            data: Some(serde_json::json!({
                "selected_properties": selected_properties
            })),
        })
        .await;

        let mut state = self.state.lock().await;
        *state = AdvancedAutoqueryFlow::PropertiesSelected {
            queries,
            selected_properties,
            collection_id,
            read_api_key,
        };
        Ok(())
    }

    async fn transition_to_combine_queries(
        &self,
        queries: Vec<String>,
        selected_properties: Vec<HashMap<String, CollectionSelectedProperties>>,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    ) -> Result<(), AdvancedAutoqueryError> {
        let queries_and_properties =
            self.combine_queries_and_properties(queries.clone(), selected_properties.clone());

        self.send_event(AdvancedAutoqueryEvent::StateChanged {
            state: "combine_queries".to_string(),
            message: "Combining queries and properties".to_string(),
            data: Some(serde_json::json!({
                "queries_count": queries.len(),
                "properties_count": selected_properties.len(),
                "queries_and_properties": queries_and_properties
            })),
        })
        .await;

        let mut state = self.state.lock().await;
        *state = AdvancedAutoqueryFlow::CombineQueriesAndProperties {
            queries,
            properties: selected_properties,
            collection_id,
            read_api_key,
        };
        Ok(())
    }

    async fn transition_to_queries_combined(
        &self,
        queries: Vec<String>,
        properties: Vec<HashMap<String, CollectionSelectedProperties>>,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    ) -> Result<(), AdvancedAutoqueryError> {
        let queries_and_properties =
            self.combine_queries_and_properties(queries.clone(), properties.clone());

        self.send_event(AdvancedAutoqueryEvent::StateChanged {
            state: "queries_combined".to_string(),
            message: "Queries combined".to_string(),
            data: Some(serde_json::json!({
                "queries_count": queries.len(),
                "properties_count": properties.len(),
                "queries_and_properties": queries_and_properties
            })),
        })
        .await;

        let mut state = self.state.lock().await;
        *state = AdvancedAutoqueryFlow::QueriesCombined {
            queries_and_properties,
            collection_id,
            read_api_key,
        };
        Ok(())
    }

    async fn transition_to_generate_tracked_queries(
        &self,
        queries_and_properties: Vec<QueryAndProperties>,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    ) -> Result<(), AdvancedAutoqueryError> {
        let mut state = self.state.lock().await;
        *state = AdvancedAutoqueryFlow::GenerateTrackedQueries {
            queries_and_properties,
            collection_id,
            read_api_key,
        };
        Ok(())
    }

    async fn transition_to_tracked_queries_generated(
        &self,
        queries_and_properties: Vec<QueryAndProperties>,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    ) -> Result<(), AdvancedAutoqueryError> {
        let tracked_queries = self
            .transition_with_retry("generate_tracked_queries", || {
                self.generate_tracked_search_queries(queries_and_properties.clone())
            })
            .await?;

        self.send_event(AdvancedAutoqueryEvent::StateChanged {
            state: "tracked_queries_generated".to_string(),
            message: "Tracked queries generated".to_string(),
            data: Some(serde_json::json!({
                "queries_count": tracked_queries.len(),
                "tracked_queries": tracked_queries
            })),
        })
        .await;

        let mut state = self.state.lock().await;
        *state = AdvancedAutoqueryFlow::TrackedQueriesGenerated {
            tracked_queries,
            collection_id,
            read_api_key,
        };
        Ok(())
    }

    async fn transition_to_execute_before_retrieval_hook(
        &self,
        tracked_queries: Vec<TrackedQuery>,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    ) -> Result<(), AdvancedAutoqueryError> {
        let mut state = self.state.lock().await;
        *state = AdvancedAutoqueryFlow::ExecuteBeforeRetrievalHook {
            tracked_queries,
            collection_id,
            read_api_key,
        };
        Ok(())
    }

    async fn transition_to_hooks_executed(
        &self,
        tracked_queries: Vec<TrackedQuery>,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    ) -> Result<(), AdvancedAutoqueryError> {
        let processed_queries = self
            .transition_with_retry("execute_before_retrieval_hook", || {
                self.execute_before_retrieval_hook(
                    tracked_queries.clone(),
                    collection_id,
                    read_api_key,
                )
            })
            .await?;

        self.send_event(AdvancedAutoqueryEvent::StateChanged {
            state: "hooks_executed".to_string(),
            message: "Hooks executed".to_string(),
            data: Some(serde_json::json!({
                "processed_queries_count": processed_queries.len(),
                "processed_queries": processed_queries
            })),
        })
        .await;

        let mut state = self.state.lock().await;
        *state = AdvancedAutoqueryFlow::HooksExecuted {
            processed_queries,
            collection_id,
            read_api_key,
        };
        Ok(())
    }

    async fn transition_to_execute_searches(
        &self,
        processed_queries: Vec<ProcessedTrackedQuery>,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    ) -> Result<(), AdvancedAutoqueryError> {
        let mut state = self.state.lock().await;
        *state = AdvancedAutoqueryFlow::ExecuteSearches {
            processed_queries,
            collection_id,
            read_api_key,
        };
        Ok(())
    }

    async fn transition_to_search_results(
        &self,
        processed_queries: Vec<ProcessedTrackedQuery>,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    ) -> Result<(), AdvancedAutoqueryError> {
        let results = self
            .transition_with_retry("execute_searches", || {
                self.execute_concurrent_searches(
                    processed_queries.clone(),
                    collection_id,
                    read_api_key,
                )
            })
            .await?;

        self.send_event(AdvancedAutoqueryEvent::StateChanged {
            state: "search_results".to_string(),
            message: "Search results obtained".to_string(),
            data: Some(serde_json::json!({
                "processed_queries_count": processed_queries.len(),
                "search_results": results
            })),
        })
        .await;

        let mut state = self.state.lock().await;
        *state = AdvancedAutoqueryFlow::SearchResults { results };
        Ok(())
    }

    // ==== Core Business Logic ====

    async fn analyze_input(
        &self,
        conversation_json: String,
    ) -> Result<Vec<String>, AdvancedAutoqueryError> {
        let current_datetime = chrono::Utc::now();
        let variables = vec![
            ("conversation".to_string(), conversation_json),
            (
                "timestamp".to_string(),
                current_datetime.format("%Y-%m-%d %H:%M:%S").to_string(),
            ),
            (
                "timestamp_unix".to_string(),
                current_datetime.timestamp().to_string(),
            ),
        ];

        let result = self
            .llm_service
            .run_known_prompt(
                KnownPrompts::AdvancedAutoqueryQueryAnalyzer,
                vec![],
                variables,
                None,
                self.llm_config.clone(),
            )
            .await
            .map_err(|e| AdvancedAutoqueryError::LLMServiceError(e.to_string()))?;

        tracing::trace!(">> result {result}");

        let cleaned = repair_json(&result, &Default::default())
            .map_err(|e| AdvancedAutoqueryError::JsonParsingError(e.to_string()))?;

        serde_json::from_str::<Vec<String>>(&cleaned)
            .map_err(|e| AdvancedAutoqueryError::JsonParsingError(e.to_string()))
    }

    async fn select_properties(
        &self,
        queries: Vec<String>,
    ) -> Result<Vec<HashMap<String, CollectionSelectedProperties>>, AdvancedAutoqueryError> {
        let selectable_props = serde_json::to_string(&self.format_collection_stats())
            .map_err(|e| AdvancedAutoqueryError::CollectionStatsError(e.to_string()))?;

        let futures: Vec<_> = queries
            .iter()
            .map(|query| {
                let variables = vec![
                    ("query".to_string(), query.clone()),
                    ("properties_list".to_string(), selectable_props.clone()),
                ];
                self.llm_service.run_known_prompt(
                    KnownPrompts::AdvancedAutoQueryPropertiesSelector,
                    vec![],
                    variables,
                    None,
                    self.llm_config.clone(),
                )
            })
            .collect();

        let results = join_all(futures).await;
        let mut parsed_results = Vec::new();

        for (index, result) in results.into_iter().enumerate() {
            match result {
                Ok(response) => {
                    let cleaned = repair_json(&response, &Default::default())
                        .map_err(|e| AdvancedAutoqueryError::JsonParsingError(e.to_string()))?;
                    let parsed = parse_properties_response(&cleaned).unwrap_or_else(|e| {
                        error!("Failed to parse LLM response at index {index}: {e}");
                        HashMap::new()
                    });
                    parsed_results.push(parsed);
                }
                Err(e) => {
                    error!("LLM call failed at index {index}: {e}");
                    parsed_results.push(HashMap::new());
                }
            }
        }

        Ok(parsed_results)
    }

    fn combine_queries_and_properties(
        &self,
        queries: Vec<String>,
        properties: Vec<HashMap<String, CollectionSelectedProperties>>,
    ) -> Vec<QueryAndProperties> {
        queries
            .into_iter()
            .zip(properties)
            .map(|(query, properties)| QueryAndProperties {
                query,
                properties,
                filter_properties: HashMap::new(),
            })
            .collect()
    }

    async fn generate_tracked_search_queries(
        &self,
        mut query_plan: Vec<QueryAndProperties>,
    ) -> Result<Vec<TrackedQuery>, AdvancedAutoqueryError> {
        // Extract filter properties for all queries
        let filter_properties = self.extract_filter_properties(&query_plan)?;

        // Update query plan with filter properties
        for query_and_props in &mut query_plan {
            query_and_props.filter_properties = filter_properties.clone();
        }

        // Convert to LLM variables and run queries in parallel
        let variables_list = self.create_llm_variables(&query_plan);
        let futures: Vec<_> = variables_list
            .into_iter()
            .map(|variables| {
                self.llm_service.run_known_prompt(
                    KnownPrompts::AdvancedAutoQueryQueryComposer,
                    vec![],
                    variables,
                    None,
                    self.llm_config.clone(),
                )
            })
            .collect();

        let results = join_all(futures).await;

        // Process results with tracking
        let tracked_queries: Result<Vec<TrackedQuery>, AdvancedAutoqueryError> = results
            .into_iter()
            .enumerate()
            .map(|(index, result)| {
                result
                    .map_err(|e| AdvancedAutoqueryError::LLMServiceError(e.to_string()))
                    .and_then(|response| {
                        let cleaned = repair_json(&response, &Default::default())
                            .map_err(|e| AdvancedAutoqueryError::JsonParsingError(e.to_string()))?;
                        let search_params = serde_json::from_str::<SearchParams>(&cleaned)
                            .map_err(|e| AdvancedAutoqueryError::JsonParsingError(e.to_string()))?;

                        Ok(TrackedQuery {
                            index,
                            original_query: query_plan[index].query.clone(),
                            generated_query_text: cleaned,
                            search_params,
                        })
                    })
            })
            .collect();

        tracked_queries
    }

    async fn execute_before_retrieval_hook(
        &self,
        tracked_queries: Vec<TrackedQuery>,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    ) -> Result<Vec<ProcessedTrackedQuery>, AdvancedAutoqueryError> {
        let (tx, mut rx) = mpsc::channel(self.config.max_concurrent_operations);

        // Spawn tasks for concurrent hook execution
        let mut handles = Vec::new();
        for (i, query) in tracked_queries.into_iter().enumerate() {
            let tx = tx.clone();
            let read_side = self.read_side.clone();

            let handle = tokio::spawn(async move {
                let result = Self::execute_single_hook_with_retry(
                    query,
                    collection_id,
                    read_api_key,
                    read_side,
                )
                .await;
                let _ = tx.send((i, result)).await;
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in handles {
            let _ = handle.await;
        }
        drop(tx);

        // Collect results in order
        let mut processed_queries = Vec::new();
        while let Some((index, result)) = rx.recv().await {
            match result {
                Ok(processed_query) => {
                    if processed_queries.len() <= index {
                        processed_queries.resize(index + 1, processed_query);
                    } else {
                        processed_queries[index] = processed_query;
                    }
                }
                Err(e) => {
                    error!("Hook execution at index {} failed: {:?}", index, e);
                    return Err(e);
                }
            }
        }

        Ok(processed_queries)
    }

    async fn execute_concurrent_searches(
        &self,
        processed_queries: Vec<ProcessedTrackedQuery>,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    ) -> Result<Vec<QueryMappedSearchResult>, AdvancedAutoqueryError> {
        let (tx, mut rx) = mpsc::channel(self.config.max_concurrent_operations);

        // Spawn tasks for concurrent execution
        let mut handles = Vec::new();
        for (i, query) in processed_queries.into_iter().enumerate() {
            let tx = tx.clone();
            let config = self.config.clone();
            let read_side = self.read_side.clone();

            let handle = tokio::spawn(async move {
                let result = Self::execute_single_search_with_retry(
                    query,
                    collection_id,
                    read_api_key,
                    read_side,
                    config,
                )
                .await;
                let _ = tx.send((i, result)).await;
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in handles {
            let _ = handle.await;
        }
        drop(tx);

        // Collect results in order
        let mut results = Vec::new();
        while let Some((index, result)) = rx.recv().await {
            match result {
                Ok(search_result) => {
                    if results.len() <= index {
                        results.resize(index + 1, search_result);
                    } else {
                        results[index] = search_result;
                    }
                }
                Err(e) => {
                    error!("Search at index {} failed: {:?}", index, e);
                    return Err(e);
                }
            }
        }

        Ok(results)
    }

    async fn execute_single_search_with_retry(
        query: ProcessedTrackedQuery,
        collection_id: CollectionId,
        read_api_key: ApiKey,
        read_side: Arc<ReadSide>,
        config: AdvancedAutoqueryConfig,
    ) -> Result<QueryMappedSearchResult, AdvancedAutoqueryError> {
        let mut backoff = ExponentialBackoffBuilder::new()
            .with_initial_interval(config.initial_backoff)
            .with_max_interval(config.max_backoff)
            .with_max_elapsed_time(Some(config.timeout))
            .build();

        let mut retry_count = 0;
        loop {
            match Self::execute_single_search(
                query.clone(),
                collection_id,
                read_api_key,
                read_side.clone(),
            )
            .await
            {
                Ok(search_result) => {
                    if retry_count > 0 {
                        info!(
                            "Search for query '{}' succeeded after {} retries",
                            query.original_query, retry_count
                        );
                    }
                    return Ok(search_result);
                }
                Err(e) => {
                    retry_count += 1;
                    if retry_count > config.max_retries {
                        error!(
                            "Search for query '{}' failed after {} retries: {:?}",
                            query.original_query, retry_count, e
                        );
                        return Err(e);
                    }

                    warn!(
                        "Search for query '{}' failed (attempt {}/{}), retrying: {:?}",
                        query.original_query, retry_count, config.max_retries, e
                    );

                    if let Some(duration) = Backoff::next_backoff(&mut backoff) {
                        sleep(duration).await;
                    }
                }
            }
        }
    }

    async fn execute_single_search(
        processed_query: ProcessedTrackedQuery,
        collection_id: CollectionId,
        read_api_key: ApiKey,
        read_side: Arc<ReadSide>,
    ) -> Result<QueryMappedSearchResult, AdvancedAutoqueryError> {
        let search_params = processed_query.processed_search_params.clone();

        let search_result = read_side
            .search(
                read_api_key,
                collection_id,
                SearchRequest {
                    search_params,
                    search_analytics_event_origin: Some(SearchAnalyticEventOrigin::RAG),
                    analytics_metadata: None,
                    interaction_id: None,
                },
            )
            .await
            .map_err(|e| AdvancedAutoqueryError::ExecuteSearchesError(e.to_string()))?;

        Ok(QueryMappedSearchResult {
            original_query: processed_query.original_query,
            generated_query: processed_query.generated_query_text,
            search_params: processed_query.processed_search_params,
            results: vec![search_result],
            query_index: processed_query.index,
        })
    }

    async fn execute_single_hook_with_retry(
        query: TrackedQuery,
        collection_id: CollectionId,
        read_api_key: ApiKey,
        read_side: Arc<ReadSide>,
    ) -> Result<ProcessedTrackedQuery, AdvancedAutoqueryError> {
        let mut backoff = ExponentialBackoffBuilder::new()
            .with_initial_interval(Duration::from_millis(100))
            .with_max_interval(Duration::from_secs(30))
            .with_max_elapsed_time(Some(Duration::from_secs(60)))
            .build();

        let mut retry_count = 0;
        loop {
            match Self::execute_single_hook(
                query.clone(),
                collection_id,
                read_api_key,
                read_side.clone(),
            )
            .await
            {
                Ok(processed_query) => {
                    if retry_count > 0 {
                        info!(
                            "Hook execution for query '{}' succeeded after {} retries",
                            query.original_query, retry_count
                        );
                    }
                    return Ok(processed_query);
                }
                Err(e) => {
                    retry_count += 1;
                    if retry_count > 3 {
                        error!(
                            "Hook execution for query '{}' failed after {} retries: {:?}",
                            query.original_query, retry_count, e
                        );
                        return Err(e);
                    }

                    warn!(
                        "Hook execution for query '{}' failed (attempt {}/{}), retrying: {:?}",
                        query.original_query, retry_count, 3, e
                    );

                    if let Some(duration) = Backoff::next_backoff(&mut backoff) {
                        sleep(duration).await;
                    }
                }
            }
        }
    }

    async fn execute_single_hook(
        tracked_query: TrackedQuery,
        collection_id: CollectionId,
        read_api_key: ApiKey,
        read_side: Arc<ReadSide>,
    ) -> Result<ProcessedTrackedQuery, AdvancedAutoqueryError> {
        let search_params = tracked_query.search_params.clone();
        let hook_storage = read_side
            .get_hook_storage(read_api_key, collection_id)
            .await
            .map_err(|e| AdvancedAutoqueryError::ExecuteBeforeRetrievalHookError(e.to_string()))?;

        let lock = hook_storage.read().await;
        let processed_search_params = run_before_retrieval(
            &lock,
            search_params.clone(),
            None, // log_sender
            ExecOption {
                allowed_hosts: Some(vec![]),
                timeout: Duration::from_millis(500),
            },
        )
        .await
        .map_err(|e| AdvancedAutoqueryError::ExecuteBeforeRetrievalHookError(e.to_string()))?;
        drop(lock);

        Ok(ProcessedTrackedQuery {
            index: tracked_query.index,
            original_query: tracked_query.original_query,
            generated_query_text: tracked_query.generated_query_text,
            original_search_params: tracked_query.search_params,
            processed_search_params,
        })
    }

    // ==== Helper Methods ====

    fn extract_filter_properties(
        &self,
        query_plan: &[QueryAndProperties],
    ) -> Result<HashMap<String, Vec<String>>, AdvancedAutoqueryError> {
        let properties_by_collection = self.get_properties_to_retrieve(query_plan);
        let mut all_filter_properties = HashMap::new();

        for (collection_name, properties) in &properties_by_collection {
            let index_id = IndexId::try_new(collection_name)
                .map_err(|e| AdvancedAutoqueryError::CollectionStatsError(e.to_string()))?;

            let index_stats = self
                .collection_stats
                .indexes_stats
                .iter()
                .find(|index| index.id == index_id)
                .ok_or_else(|| {
                    AdvancedAutoqueryError::CollectionStatsError(format!(
                        "Collection {collection_name} not found in stats"
                    ))
                })?;

            // Extract keys for each property
            for property in properties {
                let keys = self.extract_property_keys(index_stats, property)?;
                if !keys.is_empty() {
                    all_filter_properties.insert(property.clone(), keys);
                }
            }
        }

        Ok(all_filter_properties)
    }

    fn extract_property_keys(
        &self,
        index_stats: &crate::collection_manager::sides::read::IndexStats,
        property: &str,
    ) -> Result<Vec<String>, AdvancedAutoqueryError> {
        let mut keys = Vec::new();

        for field_stat in &index_stats.fields_stats {
            if field_stat.field_path == property {
                let stat_json = serde_json::to_string(&field_stat.stats)
                    .map_err(|e| AdvancedAutoqueryError::JsonParsingError(e.to_string()))?;

                if let Ok(stat_type) = serde_json::from_str::<FieldStatType>(&stat_json) {
                    if let Some(field_keys) = stat_type.extract_keys() {
                        keys.extend_from_slice(field_keys);
                    }
                }
            }
        }

        keys.sort();
        keys.dedup();
        Ok(keys)
    }

    fn get_properties_to_retrieve(
        &self,
        query_plan: &[QueryAndProperties],
    ) -> HashMap<String, Vec<String>> {
        let mut properties_map: HashMap<String, Vec<String>> = HashMap::new();

        for query_and_props in query_plan {
            for (collection_name, collection_props) in &query_and_props.properties {
                let props_to_get: Vec<String> = collection_props
                    .selected_properties
                    .iter()
                    .filter(|prop| prop.get)
                    .map(|prop| prop.property.clone())
                    .collect();

                properties_map
                    .entry(collection_name.clone())
                    .or_default()
                    .extend(props_to_get);
            }
        }

        // Remove duplicates and sort
        for properties in properties_map.values_mut() {
            properties.sort();
            properties.dedup();
        }

        properties_map
    }

    fn create_llm_variables(
        &self,
        queries_and_properties: &[QueryAndProperties],
    ) -> Vec<Vec<(String, String)>> {
        queries_and_properties
            .iter()
            .map(|qp| {
                let mut variables = vec![("query".to_string(), qp.query.clone())];

                // Flatten all properties from all collections
                let all_properties: Vec<_> = qp
                    .properties
                    .values()
                    .flat_map(|cp| cp.selected_properties.clone())
                    .collect();

                if let Ok(props_json) = serde_json::to_string(&all_properties) {
                    variables.push(("properties_list".to_string(), props_json));
                }

                if let Ok(filter_json) = serde_json::to_string(&qp.filter_properties) {
                    variables.push(("filter_properties".to_string(), filter_json));
                }

                variables
            })
            .collect()
    }

    fn format_collection_stats(&self) -> PropertiesSelectorInput {
        use regex::Regex;
        let prefix_regex = Regex::new(r"^(committed|uncommitted)_").expect("Valid regex pattern");

        let mut indexes_stats = Vec::new();
        let mut seen_fields = HashMap::new();

        for index in &self.collection_stats.indexes_stats {
            // Skip temporary indexes
            if index.is_temp {
                continue;
            }

            let fields_stats = self.extract_valid_fields(index, &prefix_regex, &mut seen_fields);

            if !fields_stats.is_empty() {
                indexes_stats.push(IndexStats {
                    id: index.id.as_str().to_string(),
                    fields_stats,
                });
            }
        }

        PropertiesSelectorInput { indexes_stats }
    }

    fn extract_valid_fields(
        &self,
        index: &crate::collection_manager::sides::read::IndexStats,
        prefix_regex: &Regex,
        seen_fields: &mut HashMap<String, bool>,
    ) -> Vec<FieldStats> {
        let result: Vec<FieldStats> = index
            .fields_stats
            .iter()
            .filter_map(|stat| {
                let field_path = prefix_regex.replace(&stat.field_path, "").to_string();

                // Skip auto-generated and already seen fields
                if field_path == "___orama_auto_embedding" {
                    return None;
                }

                if seen_fields.contains_key(&field_path) {
                    return None;
                }

                // Try to deserialize and validate the field
                let stat_json = match serde_json::to_string(&stat.stats) {
                    Ok(json) => json,
                    Err(_e) => {
                        return None;
                    }
                };

                let field_stat_type = match serde_json::from_str::<FieldStatType>(&stat_json) {
                    Ok(stat_type) => stat_type,
                    Err(_e) => {
                        return None;
                    }
                };

                // Skip unfilterable strings and fields with no documents
                if field_stat_type.is_unfilterable_string() {
                    return None;
                }

                if field_stat_type.document_count() == 0 {
                    return None;
                }

                seen_fields.insert(field_path.clone(), true);

                Some(FieldStats {
                    field_path,
                    field_type: field_stat_type.field_type_name().to_string(),
                })
            })
            .collect();

        result
    }

    /// Get current state for monitoring/debugging
    pub async fn current_state(&self) -> AdvancedAutoqueryFlow {
        let state = self.state.lock().await;
        state.clone()
    }

    /// Get retry statistics
    pub async fn retry_stats(&self) -> HashMap<String, usize> {
        let counts = self.retry_count.lock().await;
        counts.clone()
    }
}

// ==== Helper Types and Functions ====

#[derive(Serialize)]
struct PropertiesSelectorInput {
    indexes_stats: Vec<IndexStats>,
}

#[derive(Serialize)]
struct IndexStats {
    pub id: String,
    fields_stats: Vec<FieldStats>,
}

#[derive(Serialize)]
struct FieldStats {
    field_path: String,
    field_type: String,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
#[allow(dead_code)]
enum FieldStatType {
    #[serde(rename = "uncommitted_bool")]
    UncommittedBoolean {
        false_count: usize,
        true_count: usize,
    },
    #[serde(rename = "committed_bool")]
    CommittedBoolean {
        false_count: usize,
        true_count: usize,
    },
    #[serde(rename = "uncommitted_number")]
    UncommittedNumber { _min: f64, _max: f64, count: usize },
    #[serde(rename = "committed_number")]
    CommittedNumber { _min: f64, _max: f64 },
    #[serde(rename = "uncommitted_string_filter")]
    UncommittedStringFilter {
        key_count: usize,
        document_count: usize,
        keys: Option<Vec<String>>,
    },
    #[serde(rename = "committed_string_filter")]
    CommittedStringFilter {
        key_count: usize,
        document_count: usize,
        keys: Option<Vec<String>>,
    },
    #[serde(rename = "uncommitted_string")]
    UncommittedString {
        key_count: usize,
        global_info: GlobalInfo,
    },
    #[serde(rename = "committed_string")]
    CommittedString {
        key_count: usize,
        global_info: GlobalInfo,
    },
    #[serde(rename = "uncommitted_vector")]
    UncommittedVector {
        document_count: usize,
        vector_count: usize,
    },
    #[serde(rename = "committed_vector")]
    CommittedVector {
        dimensions: usize,
        vector_count: usize,
    },
}

impl FieldStatType {
    fn document_count(&self) -> usize {
        use FieldStatType::*;
        match self {
            UncommittedBoolean {
                false_count,
                true_count,
            }
            | CommittedBoolean {
                false_count,
                true_count,
            } => false_count + true_count,
            UncommittedNumber { count, .. } => *count,
            CommittedNumber { .. } => 0,
            UncommittedStringFilter { document_count, .. }
            | CommittedStringFilter { document_count, .. } => *document_count,
            UncommittedString { global_info, .. } | CommittedString { global_info, .. } => {
                global_info.total_documents
            }
            UncommittedVector { document_count, .. } => *document_count,
            CommittedVector { .. } => 0,
        }
    }

    fn field_type_name(&self) -> &'static str {
        use FieldStatType::*;
        match self {
            UncommittedBoolean { .. } | CommittedBoolean { .. } => "boolean",
            UncommittedNumber { .. } | CommittedNumber { .. } => "number",
            UncommittedStringFilter { .. } | CommittedStringFilter { .. } => "string",
            UncommittedString { .. } | CommittedString { .. } => "string",
            UncommittedVector { .. } | CommittedVector { .. } => "vector",
        }
    }

    fn is_unfilterable_string(&self) -> bool {
        matches!(
            self,
            FieldStatType::UncommittedString { .. } | FieldStatType::CommittedString { .. }
        )
    }

    fn extract_keys(&self) -> Option<&[String]> {
        match self {
            FieldStatType::CommittedStringFilter { keys: Some(k), .. }
            | FieldStatType::UncommittedStringFilter { keys: Some(k), .. } => Some(k),
            _ => None,
        }
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
struct GlobalInfo {
    total_documents: usize,
    total_document_length: usize,
}

#[derive(Deserialize, Debug)]
struct LLMPropertiesResponse {
    #[serde(flatten)]
    collections: HashMap<String, CollectionPropertiesWrapper>,
}

#[derive(Deserialize, Debug)]
struct CollectionPropertiesWrapper {
    selected_properties: Vec<SelectedProperty>,
}

fn parse_properties_response(
    response: &str,
) -> Result<HashMap<String, CollectionSelectedProperties>> {
    let llm_response: LLMPropertiesResponse =
        serde_json::from_str(response).context("Failed to parse LLM properties response")?;

    Ok(llm_response
        .collections
        .into_iter()
        .map(|(name, props)| {
            (
                name,
                CollectionSelectedProperties {
                    selected_properties: props.selected_properties,
                },
            )
        })
        .collect())
}
