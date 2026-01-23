use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use backoff::{backoff::Backoff, ExponentialBackoffBuilder};
use futures::future::Future;
use orama_js_pool::{ExecOption, OutputChannel};
use serde::Serialize;
use tokio::sync::{mpsc, Mutex};
use tokio::time::sleep;
use tokio_stream::{wrappers::UnboundedReceiverStream, StreamExt};
use tracing::{error, info, warn};

use crate::ai::llms::{KnownPrompts, LLMService};
use crate::ai::ragat::{ContextComponent, GeneralRagAtError, RAGAtParser};
use crate::ai::run_hooks::{run_before_answer, run_before_retrieval};
use crate::ai::state_machines::advanced_autoquery::{
    AdvancedAutoqueryConfig, AdvancedAutoqueryStateMachine,
};
use crate::collection_manager::sides::read::{
    AnalyticsHolder, SearchAnalyticEventOrigin, SearchRequest,
};
use crate::collection_manager::sides::{read::ReadSide, system_prompts::SystemPrompt};
use crate::types::{
    CollectionId, IndexId, Interaction, InteractionLLMConfig, Limit, Properties, ReadApiKey,
    SearchMode, SearchOffset, SearchParams, SearchResultHit, Similarity, VectorMode,
};
use futures::TryFutureExt;

// ==== SSE Event Types ====

#[derive(Debug, Clone, Serialize)]
pub enum AnswerEvent {
    #[serde(rename = "state_changed")]
    StateChanged {
        state: String,
        message: String,
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
    #[serde(rename = "acknowledged")]
    Acknowledged,
    #[serde(rename = "selected_llm")]
    SelectedLLM { provider: String, model: String },
    #[serde(rename = "optimizing_query")]
    OptimizingQuery {
        original_query: String,
        optimized_query: String,
    },
    #[serde(rename = "search_results")]
    SearchResults { results: Vec<SearchResultHit> },
    #[serde(rename = "answer_token")]
    AnswerToken { token: String },
    #[serde(rename = "related_queries")]
    RelatedQueries { queries: String },
    #[serde(rename = "result_action")]
    ResultAction { action: String, result: String },
}

// ==== Data Models ====

#[derive(Debug, Clone)]
pub struct ComponentResult {
    hits: Vec<SearchResultHit>,
}

#[derive(Debug, Clone)]
pub struct GeneratedAnswer {
    pub answer: String,
    pub search_results: Vec<SearchResultHit>,
    pub related_queries: Option<String>,
}

// ==== Error Types ====

#[derive(Debug, thiserror::Error, Clone)]
pub enum AnswerError {
    #[error("Failed to initialize: {0}")]
    InitializeError(String),
    #[error("Failed to handle GPU overload: {0}")]
    GPUOverloadError(String),
    #[error("Failed to get LLM config: {0}")]
    LLMConfigError(String),
    #[error("Failed to handle system prompt: {0}")]
    SystemPromptError(String),
    #[error("Failed to optimize query: {0}")]
    QueryOptimizationError(String),
    #[error("Failed to execute RAG-AT specification: {0}")]
    RagAtError(String),
    #[error("Failed to execute search: {0}")]
    SearchError(String),
    #[error("Failed to execute before retrieval hook: {0}")]
    BeforeRetrievalHookError(String),
    #[error("Failed to execute before answer hook: {0}")]
    BeforeAnswerHookError(String),
    #[error("Failed to generate answer: {0}")]
    AnswerGenerationError(String),
    #[error("Failed to generate related queries: {0}")]
    RelatedQueriesError(String),
    #[error("LLM service error: {0}")]
    LLMServiceError(String),
    #[error("Read error: {0}")]
    ReadError(String),
    #[error("Hook read error: {0}")]
    HookError(String),
    #[error("JS run error: {0}")]
    JSError(String),
    #[error("JSON parsing error: {0}")]
    JsonParsingError(String),
    #[error("Advanced autoquery error: {0}")]
    AdvancedAutoqueryError(String),
}

// ==== State Machine States ====

#[derive(Debug, Clone)]
pub enum AnswerFlow {
    Initialize {
        interaction: Interaction,
        collection_id: CollectionId,
        read_api_key: ReadApiKey,
    },
    HandleGPUOverload {
        interaction: Interaction,
        collection_id: CollectionId,
        read_api_key: ReadApiKey,
    },
    GetLLMConfig {
        interaction: Interaction,
        llm_config: InteractionLLMConfig,
        collection_id: CollectionId,
        read_api_key: ReadApiKey,
    },
    DetermineQueryStrategy {
        interaction: Interaction,
        llm_config: InteractionLLMConfig,
        collection_id: CollectionId,
        read_api_key: ReadApiKey,
    },
    ExecuteBeforeRetrievalHook {
        interaction: Interaction,
        llm_config: InteractionLLMConfig,
        collection_id: CollectionId,
        read_api_key: ReadApiKey,
    },
    OptimizeQuery {
        interaction: Interaction,
        llm_config: InteractionLLMConfig,
        system_prompt: Option<SystemPrompt>,
        collection_id: CollectionId,
        read_api_key: ReadApiKey,
        optimized_query: String,
    },
    ExecuteSearch {
        interaction: Interaction,
        llm_config: InteractionLLMConfig,
        optimized_query: String,
        collection_id: CollectionId,
        read_api_key: ReadApiKey,
        log_sender: Option<Arc<tokio::sync::broadcast::Sender<(OutputChannel, String)>>>,
    },
    HandleSystemPrompt {
        interaction: Interaction,
        llm_config: InteractionLLMConfig,
        search_results: Vec<SearchResultHit>,
        collection_id: CollectionId,
        read_api_key: ReadApiKey,
    },
    ExecuteAfterRetrievalHook {
        interaction: Interaction,
        llm_config: InteractionLLMConfig,
        search_results: Vec<SearchResultHit>,
        collection_id: CollectionId,
        read_api_key: ReadApiKey,
    },
    ExecuteBeforeAnswerHook {
        interaction: Interaction,
        llm_config: InteractionLLMConfig,
        system_prompt: Option<SystemPrompt>,
        search_results: Vec<SearchResultHit>,
        collection_id: CollectionId,
        read_api_key: ReadApiKey,
    },
    GenerateAnswer {
        interaction: Interaction,
        llm_config: InteractionLLMConfig,
        system_prompt: Option<SystemPrompt>,
        search_results: Vec<SearchResultHit>,
        variables: Vec<(String, String)>,
        collection_id: CollectionId,
        read_api_key: ReadApiKey,
    },
    Completed {
        answer: String,
        search_results: Vec<SearchResultHit>,
    },
    Error(AnswerError),
}

// ==== Configuration ====

#[derive(Debug, Clone)]
pub struct AnswerConfig {
    pub max_retries: usize,
    pub initial_backoff: Duration,
    pub max_backoff: Duration,
    pub max_concurrent_operations: usize,
    pub timeout: Duration,
    pub llm_timeout: Duration,
    pub hook_timeout: Duration,
}

impl Default for AnswerConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(30),
            max_concurrent_operations: 5,
            timeout: Duration::from_secs(60),
            llm_timeout: Duration::from_secs(30),
            hook_timeout: Duration::from_millis(500),
        }
    }
}

// ==== Main State Machine ====

pub struct AnswerStateMachine {
    config: AnswerConfig,
    state: Arc<Mutex<AnswerFlow>>,
    retry_count: Arc<Mutex<HashMap<String, usize>>>,
    llm_service: Arc<LLMService>,
    read_side: Arc<ReadSide>,
    collection_id: CollectionId,
    read_api_key: ReadApiKey,
    event_sender: Option<mpsc::UnboundedSender<AnswerEvent>>,
    analytics_holder: Option<Arc<Mutex<AnalyticsHolder>>>,
}

impl AnswerStateMachine {
    pub fn new(
        config: AnswerConfig,
        llm_service: Arc<LLMService>,
        read_side: Arc<ReadSide>,
        collection_id: CollectionId,
        read_api_key: ReadApiKey,
        analytics_holder: Option<Arc<Mutex<AnalyticsHolder>>>,
    ) -> Self {
        Self {
            config,
            state: Arc::new(Mutex::new(AnswerFlow::Error(AnswerError::InitializeError(
                "State machine not initialized".to_string(),
            )))),
            retry_count: Arc::new(Mutex::new(HashMap::new())),
            llm_service,
            read_side,
            collection_id,
            read_api_key,
            event_sender: None,
            analytics_holder,
        }
    }

    pub fn with_event_sender(mut self, event_sender: mpsc::UnboundedSender<AnswerEvent>) -> Self {
        self.event_sender = Some(event_sender);
        self
    }

    /// Send an event if the event sender is available
    async fn send_event(&self, event: AnswerEvent) {
        if let Some(sender) = &self.event_sender {
            let _ = sender.send(event);
        }
    }

    /// Convert state to JSON for progress events
    fn state_to_json(&self, state: &AnswerFlow) -> serde_json::Value {
        match state {
            AnswerFlow::Initialize { interaction, .. } => {
                serde_json::json!({
                    "type": "Initialize",
                    "interaction_id": interaction.interaction_id
                })
            }
            AnswerFlow::HandleGPUOverload { interaction, .. } => {
                serde_json::json!({
                    "type": "HandleGPUOverload",
                    "interaction_id": interaction.interaction_id
                })
            }
            AnswerFlow::GetLLMConfig {
                interaction,
                llm_config,
                ..
            } => {
                serde_json::json!({
                    "type": "GetLLMConfig",
                    "interaction_id": interaction.interaction_id,
                    "llm_provider": llm_config.provider.to_string(),
                    "llm_model": llm_config.model
                })
            }
            AnswerFlow::DetermineQueryStrategy { interaction, .. } => {
                serde_json::json!({
                    "type": "DetermineQueryStrategy",
                    "interaction_id": interaction.interaction_id,
                    "query": interaction.query
                })
            }
            AnswerFlow::ExecuteBeforeRetrievalHook { interaction, .. } => {
                serde_json::json!({
                    "type": "ExecuteBeforeRetrievalHook",
                    "interaction_id": interaction.interaction_id
                })
            }
            AnswerFlow::OptimizeQuery {
                interaction,
                optimized_query,
                ..
            } => {
                serde_json::json!({
                    "type": "OptimizeQuery",
                    "interaction_id": interaction.interaction_id,
                    "original_query": interaction.query,
                    "optimized_query": optimized_query
                })
            }
            AnswerFlow::ExecuteSearch {
                interaction,
                optimized_query,
                ..
            } => {
                serde_json::json!({
                    "type": "ExecuteSearch",
                    "interaction_id": interaction.interaction_id,
                    "optimized_query": optimized_query
                })
            }
            AnswerFlow::ExecuteAfterRetrievalHook {
                interaction,
                search_results,
                ..
            } => {
                serde_json::json!({
                    "type": "ExecuteAfterRetrievalHook",
                    "interaction_id": interaction.interaction_id,
                    "search_results_count": search_results.len()
                })
            }
            AnswerFlow::HandleSystemPrompt {
                interaction,
                search_results,
                ..
            } => {
                serde_json::json!({
                    "type": "HandleSystemPrompt",
                    "interaction_id": interaction.interaction_id,
                    "search_results_count": search_results.len(),
                    "system_prompt_id": interaction.system_prompt_id
                })
            }
            AnswerFlow::ExecuteBeforeAnswerHook {
                interaction,
                search_results,
                ..
            } => {
                serde_json::json!({
                    "type": "ExecuteBeforeAnswerHook",
                    "interaction_id": interaction.interaction_id,
                    "search_results_count": search_results.len()
                })
            }
            AnswerFlow::GenerateAnswer {
                interaction,
                search_results,
                variables,
                ..
            } => {
                serde_json::json!({
                    "type": "GenerateAnswer",
                    "interaction_id": interaction.interaction_id,
                    "search_results_count": search_results.len(),
                    "variables_count": variables.len()
                })
            }

            AnswerFlow::Completed { search_results, .. } => {
                serde_json::json!({
                    "type": "Completed",
                    "search_results_count": search_results.len()
                })
            }
            AnswerFlow::Error(error) => {
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
        interaction: Interaction,
        collection_id: CollectionId,
        read_api_key: ReadApiKey,
        _log_sender: Option<Arc<tokio::sync::broadcast::Sender<(OutputChannel, String)>>>,
    ) -> Result<GeneratedAnswer, AnswerError> {
        info!(
            "Starting answer generation for collection: {}",
            collection_id
        );

        // Send initial event
        self.send_event(AnswerEvent::StateChanged {
            state: "initializing".to_string(),
            message: "Starting answer generation".to_string(),
            data: None,
        })
        .await;

        // Initialize state
        {
            let mut state = self.state.lock().await;
            *state = AnswerFlow::Initialize {
                interaction,
                collection_id,
                read_api_key,
            };
        }

        let total_steps = 10; // Total number of states in the flow
        let mut current_step = 0;

        loop {
            current_step += 1;
            let current_state = {
                let state = self.state.lock().await;
                state.clone()
            };

            // Send progress event with JSON-formatted current_step
            let current_step_json = self.state_to_json(&current_state);

            if let Some(analytics_holder) = self.analytics_holder.as_ref() {
                let mut lock = analytics_holder.lock().await;
                lock.add_rag_step(current_step_json.clone());
            }

            self.send_event(AnswerEvent::Progress {
                current_step: current_step_json,
                total_steps,
                message: format!("Processing step {current_step}/{total_steps}"),
            })
            .await;

            match current_state {
                AnswerFlow::Initialize {
                    interaction,
                    collection_id,
                    read_api_key,
                } => {
                    self.send_event(AnswerEvent::StateChanged {
                        state: "handle_gpu_overload".to_string(),
                        message: "Checking GPU overload".to_string(),
                        data: None,
                    })
                    .await;
                    self.transition_to_handle_gpu_overload(
                        interaction,
                        collection_id,
                        read_api_key,
                    )
                    .await?;
                }
                AnswerFlow::HandleGPUOverload {
                    interaction,
                    collection_id,
                    read_api_key,
                } => {
                    let gpu_status = self
                        .transition_to_get_llm_config(interaction, collection_id, read_api_key)
                        .await?;

                    self.send_event(AnswerEvent::StateChanged {
                        state: "get_llm_config".to_string(),
                        message: "Getting LLM configuration".to_string(),
                        data: Some(serde_json::json!({
                            "provider": gpu_status.provider,
                            "model": gpu_status.model
                        })),
                    })
                    .await;
                }
                AnswerFlow::GetLLMConfig {
                    interaction,
                    llm_config,
                    ..
                    // collection_id,
                    // read_api_key,
                } => {
                    let query_strategy = self
                        .transition_to_determine_query_strategy(interaction, llm_config)
                        .await?;

                    self.send_event(AnswerEvent::StateChanged {
                        state: "determine_query_strategy".to_string(),
                        message: "Determining query strategy".to_string(),
                        data: Some(serde_json::json!({
                            "strategy": query_strategy
                        })),
                    })
                    .await;
                }
                AnswerFlow::DetermineQueryStrategy {
                    ..
                    // interaction,
                    // llm_config,
                    // collection_id,
                    // read_api_key,
                } => {
                    // This state is handled by the transition method
                    // The transition method will route to either SimpleRAG or AdvancedAutoquery
                    unreachable!(
                        "DetermineQueryStrategy state should be handled by transition method"
                    );
                }
                AnswerFlow::ExecuteBeforeRetrievalHook {
                    interaction,
                    llm_config,
                    collection_id,
                    read_api_key,
                } => {
                    let original_query = interaction.query.clone();
                    let optimized_query = self
                        .transition_to_optimize_query(
                            interaction,
                            llm_config,
                            collection_id,
                            read_api_key,
                        )
                        .await?;

                    self.send_event(AnswerEvent::StateChanged {
                        state: "optimize_query".to_string(),
                        message: "Optimizing query".to_string(),
                        data: Some(serde_json::json!({
                            "original_query": original_query,
                            "optimized_query": optimized_query
                        })),
                    })
                    .await;
                }
                AnswerFlow::OptimizeQuery {
                    interaction,
                    llm_config,
                    collection_id,
                    read_api_key,
                    optimized_query,
                    // system_prompt,
                    ..
                } => {
                    // Execute before retrieval hook
                    self.send_event(AnswerEvent::StateChanged {
                        state: "execute_before_retrieval_hook".to_string(),
                        message: "Executing before retrieval hook".to_string(),
                        data: None,
                    })
                    .await;

                    self.transition_to_execute_before_retrieval_hook(
                        interaction,
                        llm_config,
                        optimized_query,
                        collection_id,
                        read_api_key,
                    )
                    .await?;
                }
                AnswerFlow::ExecuteSearch {
                    interaction,
                    llm_config,
                    optimized_query,
                    collection_id,
                    read_api_key,
                    // log_sender,
                    ..
                } => {
                    let search_results = self
                        .execute_search(
                            interaction.clone(),
                            llm_config.clone(),
                            collection_id,
                            read_api_key.clone(),
                            optimized_query,
                        )
                        .await?;

                    self.send_event(AnswerEvent::SearchResults {
                        results: search_results.clone(),
                    })
                    .await;

                    self.send_event(AnswerEvent::StateChanged {
                        state: "execute_search".to_string(),
                        message: "Executing search".to_string(),
                        data: Some(serde_json::json!({
                            "search_results_count": search_results.len(),
                            "results": search_results
                        })),
                    })
                    .await;

                    // Execute after retrieval hook
                    self.send_event(AnswerEvent::StateChanged {
                        state: "execute_after_retrieval_hook".to_string(),
                        message: "Executing after retrieval hook".to_string(),
                        data: None,
                    })
                    .await;

                    self.transition_to_execute_after_retrieval_hook(
                        interaction,
                        llm_config,
                        search_results,
                        collection_id,
                        read_api_key,
                    )
                    .await?;
                }
                AnswerFlow::HandleSystemPrompt {
                    interaction,
                    llm_config,
                    search_results,
                    collection_id,
                    read_api_key,
                } => {
                    let system_prompt = self
                        .handle_system_prompt(interaction.system_prompt_id.clone())
                        .await?;

                    self.send_event(AnswerEvent::StateChanged {
                        state: "handle_system_prompt".to_string(),
                        message: "Processing system prompt".to_string(),
                        data: Some(serde_json::json!({
                            "has_system_prompt": system_prompt.is_some(),
                            "system_prompt_id": interaction.system_prompt_id
                        })),
                    })
                    .await;

                    // Now transition to execute before answer hook with the system prompt
                    let (variables, processed_system_prompt) = self
                        .transition_to_execute_before_answer_hook(
                            interaction,
                            llm_config,
                            system_prompt,
                            search_results,
                            collection_id,
                            read_api_key,
                            None, // log_sender
                        )
                        .await?;

                    self.send_event(AnswerEvent::StateChanged {
                        state: "execute_before_answer_hook".to_string(),
                        message: "Executing before answer hook".to_string(),
                        data: Some(serde_json::json!({
                            "variables_count": variables.len(),
                            "system_prompt_updated": processed_system_prompt.is_some()
                        })),
                    })
                    .await;
                }
                AnswerFlow::ExecuteAfterRetrievalHook {
                    interaction,
                    llm_config,
                    search_results,
                    collection_id,
                    read_api_key,
                } => {
                    // Transition to handle system prompt
                    self.transition_to_handle_system_prompt(
                        interaction,
                        llm_config,
                        search_results,
                        collection_id,
                        read_api_key,
                    )
                    .await?;
                }
                AnswerFlow::ExecuteBeforeAnswerHook {
                    interaction,
                    llm_config,
                    system_prompt,
                    search_results,
                    collection_id,
                    read_api_key,
                } => {
                    let answer = self
                        .transition_to_generate_answer(
                            interaction,
                            llm_config,
                            system_prompt,
                            search_results.clone(),
                            collection_id,
                            read_api_key,
                        )
                        .await?;

                    let bpe = tiktoken_rs::get_bpe_from_model("gpt-4o");

                    let token_count = match bpe {
                        Ok(bpe) => bpe.encode_with_special_tokens(&answer).len(),
                        Err(_) => 0,
                    };

                    self.send_event(AnswerEvent::StateChanged {
                        state: "generate_answer".to_string(),
                        message: "Generating answer".to_string(),
                        data: Some(serde_json::json!({
                            "answer": answer,
                            "answer_token_count": token_count
                        })),
                    })
                    .await;

                    // Complete the flow
                    self.send_event(AnswerEvent::StateChanged {
                        state: "completed".to_string(),
                        message: "Answer generation completed".to_string(),
                        data: None,
                    })
                    .await;

                    return Ok(GeneratedAnswer {
                        answer,
                        search_results,
                        related_queries: None,
                    });
                }
                AnswerFlow::GenerateAnswer {
                    interaction: _,
                    llm_config: _,
                    system_prompt: _,
                    search_results: _,
                    variables: _,
                    collection_id: _,
                    read_api_key: _,
                } => {
                    // This should not be reached as we return directly from ExecuteBeforeAnswerHook
                    unreachable!("GenerateAnswer should not be reached in normal flow");
                }
                AnswerFlow::Completed {
                    answer,
                    search_results,
                } => {
                    info!("Answer generation completed successfully");
                    return Ok(GeneratedAnswer {
                        answer,
                        search_results,
                        related_queries: None,
                    });
                }
                AnswerFlow::Error(error) => {
                    error!("Answer generation failed: {:?}", error);
                    self.send_event(AnswerEvent::Error {
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
        interaction: Interaction,
        collection_id: CollectionId,
        read_api_key: ReadApiKey,
        log_sender: Option<Arc<tokio::sync::broadcast::Sender<(OutputChannel, String)>>>,
    ) -> Result<UnboundedReceiverStream<AnswerEvent>, AnswerError> {
        let (event_sender, event_receiver) = mpsc::unbounded_channel();

        self.event_sender = Some(event_sender);

        // Spawn the state machine in a separate task
        let state_machine = self;
        tokio::spawn(async move {
            let result = state_machine
                .run(interaction, collection_id, read_api_key, log_sender)
                .await;
            match result {
                Ok(_) => {
                    // Success - the final event will be sent by the state machine
                }
                Err(e) => {
                    // Error - send terminal error event
                    if let Some(sender) = &state_machine.event_sender {
                        let _ = sender.send(AnswerEvent::Error {
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

    // ==== Transition Methods ====

    async fn transition_to_handle_gpu_overload(
        &self,
        mut interaction: Interaction,
        collection_id: CollectionId,
        read_api_key: ReadApiKey,
    ) -> Result<(), AnswerError> {
        self.handle_gpu_overload(&mut interaction).await;

        let mut state = self.state.lock().await;
        *state = AnswerFlow::HandleGPUOverload {
            interaction,
            collection_id,
            read_api_key,
        };
        Ok(())
    }

    async fn transition_to_get_llm_config(
        &self,
        interaction: Interaction,
        collection_id: CollectionId,
        read_api_key: ReadApiKey,
    ) -> Result<InteractionLLMConfig, AnswerError> {
        let llm_config = self.get_llm_config(&interaction);

        // Send LLM config event
        self.send_event(AnswerEvent::SelectedLLM {
            provider: llm_config.provider.to_string(),
            model: llm_config.model.clone(),
        })
        .await;

        self.send_event(AnswerEvent::Acknowledged).await;

        let mut state = self.state.lock().await;
        *state = AnswerFlow::GetLLMConfig {
            interaction,
            llm_config: llm_config.clone(),
            collection_id,
            read_api_key,
        };
        Ok(llm_config)
    }
    async fn transition_to_determine_query_strategy(
        &self,
        interaction: Interaction,
        llm_config: InteractionLLMConfig,
    ) -> Result<String, AnswerError> {
        let query_strategy = self
            .transition_with_retry("determine_query_strategy", || {
                self.determine_query_strategy(interaction.query.clone(), llm_config.clone())
            })
            .await?;

        if query_strategy == "advanced_autoquery" {
            // Execute advanced autoquery flow
            self.send_event(AnswerEvent::StateChanged {
                state: "advanced_autoquery".to_string(),
                message: "Executing advanced autoquery".to_string(),
                data: None,
            })
            .await;

            // Get collection stats for advanced autoquery
            let collection_stats = self
                .read_side
                .collection_stats(
                    &self.read_api_key,
                    self.collection_id,
                    crate::types::CollectionStatsRequest { with_keys: false },
                )
                .await
                .map_err(|e| AnswerError::ReadError(e.to_string()))?;

            // Create and run the advanced autoquery state machine with event streaming
            let advanced_state_machine = AdvancedAutoqueryStateMachine::new(
                AdvancedAutoqueryConfig::default(),
                self.llm_service.clone(),
                Some(llm_config.clone()),
                collection_stats,
                self.read_side.clone(),
                self.collection_id,
                self.read_api_key.clone(),
            );

            // Run the advanced autoquery with streaming and capture events
            let mut messages = interaction.messages.clone();
            if messages.is_empty() {
                messages.push(crate::types::InteractionMessage {
                    role: crate::types::Role::User,
                    content: interaction.query.clone(),
                });
            }
            let advanced_event_stream = advanced_state_machine
                .run_stream(messages, self.collection_id, self.read_api_key.clone())
                .await
                .map_err(|e| AnswerError::AdvancedAutoqueryError(e.to_string()))?;

            // Forward advanced autoquery events to the client
            let mut advanced_event_stream = advanced_event_stream;
            while let Some(advanced_event) = advanced_event_stream.next().await {
                // Convert advanced autoquery events to answer events
                match advanced_event {
                    crate::ai::state_machines::advanced_autoquery::AdvancedAutoqueryEvent::StateChanged { state, message, data } => {
                        self.send_event(AnswerEvent::StateChanged {
                            state: format!("advanced_autoquery_{state}"),
                            message: format!("Advanced Autoquery: {message}"),
                            data,
                        })
                        .await;
                    }
                    crate::ai::state_machines::advanced_autoquery::AdvancedAutoqueryEvent::Progress { current_step, total_steps, message } => {
                        self.send_event(AnswerEvent::Progress {
                            current_step: serde_json::json!({"type": "advanced_autoquery", "step": current_step}),
                            total_steps,
                            message: format!("Advanced Autoquery: {message}"),
                        })
                        .await;
                    }
                    crate::ai::state_machines::advanced_autoquery::AdvancedAutoqueryEvent::SearchResults { results } => {
                        // Convert to our search results format
                        let search_hits = results.into_iter()
                            .flat_map(|query_result| query_result.results.into_iter().flat_map(|search_result| search_result.hits))
                            .collect::<Vec<_>>();

                        self.send_event(AnswerEvent::SearchResults {
                            results: search_hits,
                        })
                        .await;
                    }
                    crate::ai::state_machines::advanced_autoquery::AdvancedAutoqueryEvent::Error { error, state, is_terminal } => {
                        self.send_event(AnswerEvent::Error {
                            error: format!("Advanced Autoquery Error: {error}"),
                            state: format!("advanced_autoquery_{state}"),
                            is_terminal,
                        })
                        .await;
                    }
                }
            }

            // Continue with before retrieval hook
            let mut state = self.state.lock().await;
            *state = AnswerFlow::ExecuteBeforeRetrievalHook {
                interaction,
                llm_config,
                collection_id: self.collection_id,
                read_api_key: self.read_api_key.clone(),
            };
        } else {
            // Simple RAG flow - go directly to system prompt handling
            self.send_event(AnswerEvent::StateChanged {
                state: "simple_rag".to_string(),
                message: "Executing simple RAG".to_string(),
                data: None,
            })
            .await;

            let mut state = self.state.lock().await;
            *state = AnswerFlow::OptimizeQuery {
                interaction: interaction.clone(),
                llm_config: llm_config.clone(),
                system_prompt: None,
                collection_id: self.collection_id,
                read_api_key: self.read_api_key.clone(),
                optimized_query: interaction.query.clone(),
            };
        }
        Ok(query_strategy)
    }

    async fn transition_to_optimize_query(
        &self,
        interaction: Interaction,
        llm_config: InteractionLLMConfig,
        collection_id: CollectionId,
        read_api_key: ReadApiKey,
    ) -> Result<String, AnswerError> {
        let optimized_query = self
            .transition_with_retry("optimize_query", || {
                self.optimize_query(interaction.query.clone(), llm_config.clone())
            })
            .await?;

        // Send optimizing query event
        self.send_event(AnswerEvent::OptimizingQuery {
            original_query: interaction.query.clone(),
            optimized_query: optimized_query.clone(),
        })
        .await;

        let mut state = self.state.lock().await;
        *state = AnswerFlow::OptimizeQuery {
            interaction: interaction.clone(),
            llm_config: llm_config.clone(),
            system_prompt: None,
            collection_id,
            read_api_key,
            optimized_query: optimized_query.clone(),
        };

        Ok(optimized_query)
    }

    async fn transition_to_execute_before_retrieval_hook(
        &self,
        interaction: Interaction,
        llm_config: InteractionLLMConfig,
        optimized_query: String,
        collection_id: CollectionId,
        read_api_key: ReadApiKey,
    ) -> Result<(), AnswerError> {
        // Execute before retrieval hook - this could process the query before search
        // For now, we'll just pass through the optimized query

        let mut state = self.state.lock().await;
        *state = AnswerFlow::ExecuteSearch {
            interaction,
            llm_config,
            optimized_query,
            collection_id,
            read_api_key,
            log_sender: None,
        };

        Ok(())
    }

    async fn transition_to_execute_after_retrieval_hook(
        &self,
        interaction: Interaction,
        llm_config: InteractionLLMConfig,
        search_results: Vec<SearchResultHit>,
        collection_id: CollectionId,
        read_api_key: ReadApiKey,
    ) -> Result<(), AnswerError> {
        // Execute after retrieval hook logic would go here
        // For now, we'll just pass through the search results

        let mut state = self.state.lock().await;
        *state = AnswerFlow::ExecuteAfterRetrievalHook {
            interaction,
            llm_config,
            search_results,
            collection_id,
            read_api_key,
        };

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    async fn transition_to_execute_before_answer_hook(
        &self,
        interaction: Interaction,
        llm_config: InteractionLLMConfig,
        system_prompt: Option<SystemPrompt>,
        search_results: Vec<SearchResultHit>,
        collection_id: CollectionId,
        read_api_key: ReadApiKey,
        log_sender: Option<Arc<tokio::sync::broadcast::Sender<(OutputChannel, String)>>>,
    ) -> Result<(Vec<(String, String)>, Option<SystemPrompt>), AnswerError> {
        let (variables, processed_system_prompt) = self
            .transition_with_retry("execute_before_answer_hook", || {
                let read_api_key = read_api_key.clone();
                self.execute_before_answer_hook(
                    interaction.clone(),
                    system_prompt.clone(),
                    collection_id,
                    read_api_key,
                    log_sender.clone(),
                )
            })
            .await?;

        let mut state = self.state.lock().await;
        *state = AnswerFlow::ExecuteBeforeAnswerHook {
            interaction,
            llm_config,
            system_prompt: processed_system_prompt.clone(),
            search_results,
            collection_id,
            read_api_key,
        };
        Ok((variables, processed_system_prompt))
    }

    async fn transition_to_handle_system_prompt(
        &self,
        interaction: Interaction,
        llm_config: InteractionLLMConfig,
        search_results: Vec<SearchResultHit>,
        collection_id: CollectionId,
        read_api_key: ReadApiKey,
    ) -> Result<(), AnswerError> {
        let mut state = self.state.lock().await;
        *state = AnswerFlow::HandleSystemPrompt {
            interaction,
            llm_config,
            search_results,
            collection_id,
            read_api_key,
        };
        Ok(())
    }

    async fn transition_to_generate_answer(
        &self,
        interaction: Interaction,
        llm_config: InteractionLLMConfig,
        system_prompt: Option<SystemPrompt>,
        search_results: Vec<SearchResultHit>,
        collection_id: CollectionId,
        read_api_key: ReadApiKey,
    ) -> Result<String, AnswerError> {
        let answer = self
            .transition_with_retry("generate_answer", || {
                let read_api_key = read_api_key.clone();
                self.generate_answer(
                    interaction.clone(),
                    llm_config.clone(),
                    system_prompt.clone(),
                    search_results.clone(),
                    collection_id,
                    read_api_key,
                )
            })
            .await?;

        let mut state = self.state.lock().await;
        *state = AnswerFlow::GenerateAnswer {
            interaction,
            llm_config,
            system_prompt,
            search_results,
            variables: vec![], // Will be populated
            collection_id,
            read_api_key,
        };
        Ok(answer)
    }

    /// Transition with retry logic
    async fn transition_with_retry<F, Fut, T>(
        &self,
        operation_name: &str,
        operation: F,
    ) -> Result<T, AnswerError>
    where
        F: Fn() -> Fut + Send + Sync,
        Fut: Future<Output = Result<T, AnswerError>> + Send,
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

    // ==== Core Business Logic ====

    async fn handle_gpu_overload(&self, interaction: &mut Interaction) {
        if self.read_side.is_gpu_overloaded() {
            match self.read_side.select_random_remote_llm_service() {
                Some((provider, model)) => {
                    info!("GPU is overloaded. Switching to \"{}\" as a remote LLM provider for this request.", provider);
                    interaction.llm_config = Some(InteractionLLMConfig { model, provider });
                }
                None => {
                    warn!("GPU is overloaded and no remote LLM is available. Using local LLM, but it's gonna be slow.");
                }
            }
        }
    }

    fn get_llm_config(&self, interaction: &Interaction) -> InteractionLLMConfig {
        interaction.llm_config.clone().unwrap_or_else(|| {
            let (provider, model) = self.read_side.get_default_llm_config();
            InteractionLLMConfig { model, provider }
        })
    }

    async fn handle_system_prompt(
        &self,
        system_prompt_id: Option<String>,
    ) -> Result<Option<SystemPrompt>, AnswerError> {
        match system_prompt_id {
            Some(id) => {
                let full_prompt = self
                    .read_side
                    .get_system_prompt(&self.read_api_key, self.collection_id, id)
                    .await
                    .map_err(|e| AnswerError::SystemPromptError(e.to_string()))?;
                Ok(full_prompt)
            }
            None => {
                let has_system_prompts = self
                    .read_side
                    .has_system_prompts(&self.read_api_key, self.collection_id)
                    .await
                    .map_err(|e| AnswerError::SystemPromptError(e.to_string()))?;

                if has_system_prompts {
                    let chosen_system_prompt = self
                        .read_side
                        .perform_system_prompt_selection(&self.read_api_key, self.collection_id)
                        .await
                        .map_err(|e| AnswerError::SystemPromptError(e.to_string()))?;
                    Ok(chosen_system_prompt)
                } else {
                    Ok(None)
                }
            }
        }
    }

    async fn optimize_query(
        &self,
        query: String,
        llm_config: InteractionLLMConfig,
    ) -> Result<String, AnswerError> {
        let variables = vec![("input".to_string(), query.clone())];
        let optimized_query = self
            .llm_service
            .run_known_prompt(
                KnownPrompts::OptimizeQuery,
                vec![],
                variables,
                None,
                Some(llm_config),
            )
            .await
            .map_err(|e| AnswerError::LLMServiceError(e.to_string()))?;

        Ok(optimized_query)
    }

    async fn determine_query_strategy(
        &self,
        query: String,
        llm_config: InteractionLLMConfig,
    ) -> Result<String, AnswerError> {
        let variables = vec![("query".to_string(), query)];

        let result = self
            .llm_service
            .run_known_prompt(
                KnownPrompts::DetermineQueryStrategy,
                vec![],
                variables,
                None,
                Some(llm_config),
            )
            .await
            .map_err(|e| AnswerError::LLMServiceError(e.to_string()))?;

        // Handle empty or invalid responses
        if result.trim().is_empty() {
            return Ok("simple_rag".to_string());
        }

        // Try to parse the JSON response to determine strategy
        match serde_json::from_str::<Vec<String>>(&result) {
            Ok(strategy_code) => {
                if strategy_code.is_empty() {
                    return Ok("simple_rag".to_string());
                }

                let code = &strategy_code[0];
                match code.as_str() {
                    "000" => Ok("simple_rag".to_string()),
                    "001" | "011" | "100" => Ok("advanced_autoquery".to_string()),
                    _ => Ok("simple_rag".to_string()), // Default to simple RAG
                }
            }
            Err(_) => {
                // If JSON parsing fails, try to extract the code from the response
                let cleaned_result = result.trim();
                if cleaned_result.contains("000") {
                    Ok("simple_rag".to_string())
                } else if cleaned_result.contains("001")
                    || cleaned_result.contains("011")
                    || cleaned_result.contains("100")
                {
                    Ok("advanced_autoquery".to_string())
                } else {
                    // Default to simple RAG if we can't determine
                    Ok("simple_rag".to_string())
                }
            }
        }
    }

    async fn execute_search(
        &self,
        interaction: Interaction,
        _llm_config: InteractionLLMConfig,
        collection_id: CollectionId,
        read_api_key: ReadApiKey,
        optimized_query: String,
    ) -> Result<Vec<SearchResultHit>, AnswerError> {
        let search_results = if let Some(ref notation) = interaction.ragat_notation {
            let parsed = RAGAtParser::parse(notation);

            let components = self
                .execute_rag_at_specification(&parsed.components, interaction.clone())
                .await
                .map_err(|e| AnswerError::RagAtError(format!("{e:?}")))?;

            self.merge_component_results(components)
                .await
                .map_err(|e| AnswerError::RagAtError(format!("{e:?}")))?
        } else {
            let max_documents = Limit(interaction.max_documents.unwrap_or(5));
            let min_similarity = Similarity(interaction.min_similarity.unwrap_or(0.5));

            let search_mode = match interaction
                .search_mode
                .as_ref()
                .map_or("vector", |s| s.as_str())
            {
                "vector" => SearchMode::Vector(VectorMode {
                    term: optimized_query,
                    similarity: min_similarity,
                }),
                mode => SearchMode::from_str(mode, optimized_query),
            };

            let params = SearchParams {
                mode: search_mode,
                limit: max_documents,
                offset: SearchOffset(0),
                where_filter: Default::default(),
                boost: HashMap::new(),
                facets: HashMap::new(),
                properties: Properties::Star,
                indexes: None, // Search all indexes
                sort_by: None,
                user_id: None,
                group_by: None,
            };

            let hook_storage = self
                .read_side
                .get_hook_storage(&read_api_key, collection_id)
                .await
                .map_err(|e| AnswerError::HookError(e.to_string()))?;
            let lock = hook_storage.read("run_before_retrieval").await;
            let hooks_config = self.read_side.get_hooks_config();
            let params = run_before_retrieval(
                &lock,
                params.clone(),
                None, // log_sender
                ExecOption {
                    allowed_hosts: Some(hooks_config.allowed_hosts.clone()),
                    timeout: Duration::from_millis(hooks_config.execution_timeout_ms),
                },
                hooks_config,
            )
            .await
            .map_err(|e| AnswerError::BeforeRetrievalHookError(e.to_string()))?;
            drop(lock);

            let result = self
                .read_side
                .search(
                    &read_api_key,
                    collection_id,
                    SearchRequest {
                        search_params: params,
                        search_analytics_event_origin: Some(SearchAnalyticEventOrigin::RAG),
                        analytics_metadata: None,
                        interaction_id: None,
                    },
                )
                .await
                .map_err(|e| AnswerError::SearchError(e.to_string()))?;
            result.hits
        };

        Ok(search_results)
    }

    async fn execute_before_answer_hook(
        &self,
        interaction: Interaction,
        system_prompt: Option<SystemPrompt>,
        collection_id: CollectionId,
        read_api_key: ReadApiKey,
        log_sender: Option<Arc<tokio::sync::broadcast::Sender<(OutputChannel, String)>>>,
    ) -> Result<(Vec<(String, String)>, Option<SystemPrompt>), AnswerError> {
        let search_result_str = serde_json::to_string(&vec![] as &Vec<SearchResultHit>)
            .map_err(|e| AnswerError::JsonParsingError(e.to_string()))?;

        let variables = vec![
            ("question".to_string(), interaction.query.clone()),
            ("context".to_string(), search_result_str.clone()),
        ];

        let hook_storage = self
            .read_side
            .get_hook_storage(&read_api_key, collection_id)
            .await
            .map_err(|e| AnswerError::HookError(e.to_string()))?;
        let lock = hook_storage.read("run_before_answer").await;
        let hooks_config = self.read_side.get_hooks_config();
        let (variables, system_prompt) = run_before_answer(
            &lock,
            (variables, system_prompt),
            log_sender,
            ExecOption {
                allowed_hosts: Some(hooks_config.allowed_hosts.clone()),
                timeout: Duration::from_millis(hooks_config.execution_timeout_ms),
            },
            hooks_config,
        )
        .await
        .map_err(|e| AnswerError::BeforeAnswerHookError(e.to_string()))?;
        drop(lock);

        Ok((variables, system_prompt))
    }

    async fn generate_answer(
        &self,
        interaction: Interaction,
        llm_config: InteractionLLMConfig,
        system_prompt: Option<SystemPrompt>,
        search_results: Vec<SearchResultHit>,
        _collection_id: CollectionId,
        _read_api_key: ReadApiKey,
    ) -> Result<String, AnswerError> {
        let search_result_str = serde_json::to_string(&search_results)
            .map_err(|e| AnswerError::JsonParsingError(e.to_string()))?;

        let variables = vec![
            ("question".to_string(), interaction.query.clone()),
            ("context".to_string(), search_result_str.clone()),
        ];

        if let Some(system_prompt) = &system_prompt {
            if let Some(ana) = self.analytics_holder.as_ref() {
                let mut lock = ana.lock().await;
                lock.set_system_prompt_id(system_prompt.id.clone());
            }
        }

        let answer_stream = self
            .llm_service
            .run_known_prompt_stream(
                KnownPrompts::Answer,
                interaction.messages,
                variables,
                system_prompt,
                Some(llm_config.clone()),
            )
            .await
            .map_err(|e| AnswerError::LLMServiceError(e.to_string()))?;

        let mut answer_stream = answer_stream;
        let mut answer = String::new();

        let start_time_to_first_token = Instant::now();
        let mut start_first_token = None;
        while let Some(resp) = answer_stream.next().await {
            if start_first_token.is_none() {
                start_first_token = Some(Instant::now());

                if let Some(analytics_holder) = self.analytics_holder.as_ref() {
                    let mut lock = analytics_holder.lock().await;
                    lock.set_time_to_first_token(start_time_to_first_token.elapsed());
                }
            }

            match resp {
                Ok(chunk) => {
                    answer.push_str(&chunk);
                    // Send answer token event
                    self.send_event(AnswerEvent::AnswerToken { token: chunk })
                        .await;
                }
                Err(e) => {
                    return Err(AnswerError::AnswerGenerationError(e.to_string()));
                }
            }
        }

        if let Some(analytics_holder) = self.analytics_holder.as_ref() {
            let mut lock = analytics_holder.lock().await;
            let delta = if let Some(start_first_token) = start_first_token {
                start_first_token.elapsed()
            } else {
                Default::default()
            };
            lock.set_assistant_response(answer.clone(), delta);
        }

        Ok(answer)
    }

    async fn execute_rag_at_specification(
        &self,
        components: &[ContextComponent],
        interaction: Interaction,
    ) -> Result<Vec<ComponentResult>, GeneralRagAtError> {
        let mut results = Vec::new();

        for component in components {
            let component_result = self
                .execute_single_component(component, interaction.clone())
                .await?;

            results.push(component_result);
        }

        Ok(results)
    }

    async fn execute_single_component(
        &self,
        component: &ContextComponent,
        interaction: Interaction,
    ) -> Result<ComponentResult, GeneralRagAtError> {
        let index_ids: Result<Vec<IndexId>, GeneralRagAtError> = component
            .source_ids
            .iter()
            .map(|source_id| {
                IndexId::try_new(source_id)
                    .map_err(|_| GeneralRagAtError::InvalidIndexId(source_id.clone()))
            })
            .collect();

        let index_ids = index_ids?;

        let analytics_metadata = if let Some(analytics_holder) = self.analytics_holder.as_ref() {
            let lock = analytics_holder.lock().await;
            Some(lock.get_analytics_metadata().clone())
        } else {
            None
        };

        let search_results = self
            .read_side
            .search(
                &self.read_api_key,
                self.collection_id,
                SearchRequest {
                    search_params: SearchParams {
                        mode: SearchMode::Vector(VectorMode {
                            term: interaction.query.clone(),
                            similarity: Similarity(component.threshold),
                        }),
                        limit: Limit(component.max_documents),
                        offset: SearchOffset(0),
                        where_filter: Default::default(),
                        boost: HashMap::new(),
                        facets: HashMap::new(),
                        properties: Properties::Star,
                        indexes: Some(index_ids),
                        sort_by: None,
                        user_id: None, // @todo: handle user_id if needed
                        group_by: None,
                    },
                    search_analytics_event_origin: Some(SearchAnalyticEventOrigin::RAG),
                    analytics_metadata,
                    interaction_id: Some(interaction.interaction_id.clone()),
                },
            )
            .map_err(|_| GeneralRagAtError::ReadError)
            .await?;

        let all_hits = search_results.hits.clone();

        Ok(ComponentResult { hits: all_hits })
    }

    async fn merge_component_results(
        &self,
        component_results: Vec<ComponentResult>,
    ) -> Result<Vec<SearchResultHit>, GeneralRagAtError> {
        let mut final_hits = Vec::new();

        for result in component_results {
            final_hits.extend(result.hits);
        }

        Ok(final_hits)
    }

    /// Get current state for monitoring/debugging
    pub async fn current_state(&self) -> AnswerFlow {
        let state = self.state.lock().await;
        state.clone()
    }

    /// Get retry statistics
    pub async fn retry_stats(&self) -> HashMap<String, usize> {
        let counts = self.retry_count.lock().await;
        counts.clone()
    }
}
