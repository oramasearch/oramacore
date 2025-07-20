use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use backoff::{backoff::Backoff, ExponentialBackoffBuilder};
use futures::future::Future;
use orama_js_pool::ExecOption;
use serde::Serialize;
use tokio::sync::{mpsc, Mutex};
use tokio::time::sleep;
use tokio_stream::{wrappers::UnboundedReceiverStream, StreamExt};
use tracing::{error, info, warn};

use crate::ai::llms::{KnownPrompts, LLMService};
use crate::ai::party_planner::PartyPlanner;
use crate::ai::ragat::{ContextComponent, GeneralRagAtError, RAGAtParser};
use crate::ai::run_hooks::{run_before_answer, run_before_retrieval};
use crate::collection_manager::sides::{read::ReadSide, system_prompts::SystemPrompt};
use crate::types::{
    ApiKey, CollectionId, IndexId, Interaction, InteractionLLMConfig, Limit, Properties,
    SearchMode, SearchOffset, SearchParams, SearchResultHit, Similarity, VectorMode,
};

// ==== SSE Event Types ====

#[derive(Debug, Clone, Serialize)]
pub enum AnswerEvent {
    #[serde(rename = "state_changed")]
    StateChanged {
        state: String,
        message: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        data: Option<serde_json::Value>,
    },
    #[serde(rename = "error")]
    Error { error: String, state: String },
    #[serde(rename = "progress")]
    Progress {
        current_step: String,
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
    #[error("Failed to execute party planner: {0}")]
    PartyPlannerError(String),
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
}

// ==== State Machine States ====

#[derive(Debug, Clone)]
pub enum AnswerFlow {
    Initialize {
        interaction: Interaction,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    },
    HandleGPUOverload {
        interaction: Interaction,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    },
    GetLLMConfig {
        interaction: Interaction,
        llm_config: InteractionLLMConfig,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    },
    HandleSystemPrompt {
        interaction: Interaction,
        llm_config: InteractionLLMConfig,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    },
    OptimizeQuery {
        interaction: Interaction,
        llm_config: InteractionLLMConfig,
        system_prompt: Option<SystemPrompt>,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    },
    ExecuteSearch {
        interaction: Interaction,
        llm_config: InteractionLLMConfig,
        system_prompt: Option<SystemPrompt>,
        optimized_query: String,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    },
    ExecuteBeforeAnswerHook {
        interaction: Interaction,
        llm_config: InteractionLLMConfig,
        system_prompt: Option<SystemPrompt>,
        search_results: Vec<SearchResultHit>,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    },
    GenerateAnswer {
        interaction: Interaction,
        llm_config: InteractionLLMConfig,
        system_prompt: Option<SystemPrompt>,
        search_results: Vec<SearchResultHit>,
        variables: Vec<(String, String)>,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    },
    GenerateRelatedQueries {
        interaction: Interaction,
        llm_config: InteractionLLMConfig,
        search_results: Vec<SearchResultHit>,
        answer: String,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    },
    PartyPlanner {
        interaction: Interaction,
        llm_config: InteractionLLMConfig,
        system_prompt: Option<SystemPrompt>,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    },
    Completed {
        answer: String,
        search_results: Vec<SearchResultHit>,
        related_queries: Option<String>,
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
    read_api_key: ApiKey,
    event_sender: Option<mpsc::UnboundedSender<AnswerEvent>>,
}

impl AnswerStateMachine {
    pub fn new(
        config: AnswerConfig,
        llm_service: Arc<LLMService>,
        read_side: Arc<ReadSide>,
        collection_id: CollectionId,
        read_api_key: ApiKey,
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

    /// Run the state machine with the given input
    pub async fn run(
        &self,
        interaction: Interaction,
        collection_id: CollectionId,
        read_api_key: ApiKey,
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

        let total_steps = 9; // Total number of states in the flow
        let mut current_step = 0;

        loop {
            current_step += 1;
            let current_state = {
                let state = self.state.lock().await;
                state.clone()
            };

            // Send progress event
            self.send_event(AnswerEvent::Progress {
                current_step: format!("{:?}", current_state),
                total_steps,
                message: format!("Processing step {}/{}", current_step, total_steps),
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
                    self.send_event(AnswerEvent::StateChanged {
                        state: "get_llm_config".to_string(),
                        message: "Getting LLM configuration".to_string(),
                        data: None,
                    })
                    .await;
                    self.transition_to_get_llm_config(interaction, collection_id, read_api_key)
                        .await?;
                }
                AnswerFlow::GetLLMConfig {
                    interaction,
                    llm_config,
                    collection_id,
                    read_api_key,
                } => {
                    self.send_event(AnswerEvent::StateChanged {
                        state: "handle_system_prompt".to_string(),
                        message: "Handling system prompt".to_string(),
                        data: None,
                    })
                    .await;
                    self.transition_to_handle_system_prompt(interaction, llm_config)
                        .await?;
                }
                AnswerFlow::HandleSystemPrompt {
                    interaction,
                    llm_config,
                    collection_id,
                    read_api_key,
                } => {
                    self.send_event(AnswerEvent::StateChanged {
                        state: "optimize_query".to_string(),
                        message: "Optimizing query".to_string(),
                        data: None,
                    })
                    .await;
                    self.transition_to_optimize_query(
                        interaction,
                        llm_config,
                        collection_id,
                        read_api_key,
                    )
                    .await?;
                }
                AnswerFlow::OptimizeQuery {
                    interaction,
                    llm_config,
                    system_prompt,
                    collection_id,
                    read_api_key,
                } => {
                    self.send_event(AnswerEvent::StateChanged {
                        state: "execute_search".to_string(),
                        message: "Executing search".to_string(),
                        data: None,
                    })
                    .await;
                    self.transition_to_execute_search(
                        interaction,
                        llm_config,
                        system_prompt,
                        collection_id,
                        read_api_key,
                    )
                    .await?;
                }
                AnswerFlow::ExecuteSearch {
                    interaction,
                    llm_config,
                    system_prompt,
                    optimized_query,
                    collection_id,
                    read_api_key,
                } => {
                    self.send_event(AnswerEvent::StateChanged {
                        state: "execute_before_answer_hook".to_string(),
                        message: "Executing before answer hook".to_string(),
                        data: None,
                    })
                    .await;
                    self.transition_to_execute_before_answer_hook(
                        interaction,
                        llm_config,
                        system_prompt,
                        optimized_query,
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
                    self.send_event(AnswerEvent::StateChanged {
                        state: "generate_answer".to_string(),
                        message: "Generating answer".to_string(),
                        data: None,
                    })
                    .await;
                    self.transition_to_generate_answer(
                        interaction,
                        llm_config,
                        system_prompt,
                        search_results,
                        collection_id,
                        read_api_key,
                    )
                    .await?;
                }
                AnswerFlow::GenerateAnswer {
                    interaction,
                    llm_config,
                    system_prompt: _,
                    search_results,
                    variables,
                    collection_id,
                    read_api_key,
                } => {
                    self.send_event(AnswerEvent::StateChanged {
                        state: "generate_related_queries".to_string(),
                        message: "Generating related queries".to_string(),
                        data: None,
                    })
                    .await;
                    self.transition_to_generate_related_queries(
                        interaction,
                        llm_config,
                        search_results,
                        collection_id,
                        read_api_key,
                    )
                    .await?;
                }
                AnswerFlow::GenerateRelatedQueries {
                    interaction: _,
                    llm_config: _,
                    search_results,
                    answer,
                    collection_id: _,
                    read_api_key: _,
                } => {
                    self.send_event(AnswerEvent::StateChanged {
                        state: "completed".to_string(),
                        message: "Answer generation completed".to_string(),
                        data: None,
                    })
                    .await;
                    // Return the generated answer
                    return Ok(GeneratedAnswer {
                        answer,
                        search_results,
                        related_queries: None, // Will be populated if related queries were generated
                    });
                }
                AnswerFlow::PartyPlanner {
                    interaction,
                    llm_config,
                    system_prompt,
                    collection_id,
                    read_api_key,
                } => {
                    self.send_event(AnswerEvent::StateChanged {
                        state: "party_planner".to_string(),
                        message: "Executing party planner".to_string(),
                        data: None,
                    })
                    .await;
                    self.transition_to_party_planner(
                        interaction,
                        llm_config,
                        system_prompt,
                        collection_id,
                        read_api_key,
                    )
                    .await?;
                }
                AnswerFlow::Completed {
                    answer,
                    search_results,
                    related_queries,
                } => {
                    info!("Answer generation completed successfully");
                    return Ok(GeneratedAnswer {
                        answer,
                        search_results,
                        related_queries,
                    });
                }
                AnswerFlow::Error(error) => {
                    error!("Answer generation failed: {:?}", error);
                    self.send_event(AnswerEvent::Error {
                        error: error.to_string(),
                        state: format!("{:?}", error),
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
        read_api_key: ApiKey,
    ) -> Result<UnboundedReceiverStream<AnswerEvent>, AnswerError> {
        let (event_sender, event_receiver) = mpsc::unbounded_channel();

        self.event_sender = Some(event_sender);

        // Spawn the state machine in a separate task
        let state_machine = self;
        tokio::spawn(async move {
            let result = state_machine
                .run(interaction, collection_id, read_api_key)
                .await;
            match result {
                Ok(_) => {
                    // Success - the final event will be sent by the state machine
                }
                Err(e) => {
                    // Error - send error event
                    if let Some(sender) = &state_machine.event_sender {
                        let _ = sender.send(AnswerEvent::Error {
                            error: e.to_string(),
                            state: format!("{:?}", e),
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
        read_api_key: ApiKey,
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
        read_api_key: ApiKey,
    ) -> Result<(), AnswerError> {
        let llm_config = self.get_llm_config(&interaction);

        // Send LLM config event
        self.send_event(AnswerEvent::SelectedLLM {
            provider: llm_config.provider.to_string(),
            model: llm_config.model.clone(),
        })
        .await;

        // Send acknowledged event
        self.send_event(AnswerEvent::Acknowledged).await;

        let mut state = self.state.lock().await;
        *state = AnswerFlow::GetLLMConfig {
            interaction,
            llm_config,
            collection_id,
            read_api_key,
        };
        Ok(())
    }

    async fn transition_to_handle_system_prompt(
        &self,
        interaction: Interaction,
        llm_config: InteractionLLMConfig,
    ) -> Result<(), AnswerError> {
        let system_prompt = self
            .transition_with_retry("handle_system_prompt", || {
                self.handle_system_prompt(interaction.system_prompt_id.clone())
            })
            .await?;

        let mut state = self.state.lock().await;
        *state = AnswerFlow::HandleSystemPrompt {
            interaction,
            llm_config,
            collection_id: self.collection_id.clone(),
            read_api_key: self.read_api_key.clone(),
        };
        Ok(())
    }

    async fn transition_to_optimize_query(
        &self,
        interaction: Interaction,
        llm_config: InteractionLLMConfig,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    ) -> Result<(), AnswerError> {
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
            interaction,
            llm_config,
            system_prompt: None, // Will be populated in the next transition
            collection_id,
            read_api_key,
        };
        Ok(())
    }

    async fn transition_to_execute_search(
        &self,
        interaction: Interaction,
        llm_config: InteractionLLMConfig,
        system_prompt: Option<SystemPrompt>,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    ) -> Result<(), AnswerError> {
        let search_results = self
            .transition_with_retry("execute_search", || {
                self.execute_search(
                    interaction.clone(),
                    llm_config.clone(),
                    collection_id,
                    read_api_key,
                )
            })
            .await?;

        // Send search results event
        self.send_event(AnswerEvent::SearchResults {
            results: search_results.clone(),
        })
        .await;

        let mut state = self.state.lock().await;
        *state = AnswerFlow::ExecuteSearch {
            interaction,
            llm_config,
            system_prompt,
            optimized_query: String::new(), // Will be populated
            collection_id,
            read_api_key,
        };
        Ok(())
    }

    async fn transition_to_execute_before_answer_hook(
        &self,
        interaction: Interaction,
        llm_config: InteractionLLMConfig,
        system_prompt: Option<SystemPrompt>,
        optimized_query: String,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    ) -> Result<(), AnswerError> {
        let (variables, processed_system_prompt) = self
            .transition_with_retry("execute_before_answer_hook", || {
                self.execute_before_answer_hook(
                    interaction.clone(),
                    system_prompt.clone(),
                    collection_id,
                    read_api_key,
                )
            })
            .await?;

        let mut state = self.state.lock().await;
        *state = AnswerFlow::ExecuteBeforeAnswerHook {
            interaction,
            llm_config,
            system_prompt: processed_system_prompt,
            search_results: vec![], // Will be populated
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
        read_api_key: ApiKey,
    ) -> Result<(), AnswerError> {
        let answer = self
            .transition_with_retry("generate_answer", || {
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
        Ok(())
    }

    async fn transition_to_generate_related_queries(
        &self,
        interaction: Interaction,
        llm_config: InteractionLLMConfig,
        search_results: Vec<SearchResultHit>,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    ) -> Result<(), AnswerError> {
        let related_queries = self
            .transition_with_retry("generate_related_queries", || {
                self.generate_related_queries(
                    interaction.clone(),
                    llm_config.clone(),
                    search_results.clone(),
                    collection_id,
                    read_api_key,
                )
            })
            .await?;

        let mut state = self.state.lock().await;
        *state = AnswerFlow::GenerateRelatedQueries {
            interaction,
            llm_config,
            search_results,
            answer: String::new(), // Will be populated
            collection_id,
            read_api_key,
        };
        Ok(())
    }

    async fn transition_to_party_planner(
        &self,
        interaction: Interaction,
        llm_config: InteractionLLMConfig,
        system_prompt: Option<SystemPrompt>,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    ) -> Result<(), AnswerError> {
        let party_planner_result = self
            .transition_with_retry("party_planner", || {
                self.execute_party_planner(
                    interaction.clone(),
                    llm_config.clone(),
                    system_prompt.clone(),
                    collection_id,
                    read_api_key,
                )
            })
            .await?;

        let mut state = self.state.lock().await;
        *state = AnswerFlow::Completed {
            answer: party_planner_result,
            search_results: vec![],
            related_queries: None,
        };
        Ok(())
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
                    .get_system_prompt(self.read_api_key.clone(), self.collection_id.clone(), id)
                    .await
                    .map_err(|e| AnswerError::SystemPromptError(e.to_string()))?;
                Ok(full_prompt)
            }
            None => {
                let has_system_prompts = self
                    .read_side
                    .has_system_prompts(self.read_api_key.clone(), self.collection_id.clone())
                    .await
                    .map_err(|e| AnswerError::SystemPromptError(e.to_string()))?;

                if has_system_prompts {
                    let chosen_system_prompt = self
                        .read_side
                        .perform_system_prompt_selection(
                            self.read_api_key.clone(),
                            self.collection_id.clone(),
                        )
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
            .run_known_prompt(KnownPrompts::OptimizeQuery, variables, Some(llm_config))
            .await
            .map_err(|e| AnswerError::LLMServiceError(e.to_string()))?;

        Ok(optimized_query)
    }

    async fn execute_search(
        &self,
        interaction: Interaction,
        llm_config: InteractionLLMConfig,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    ) -> Result<Vec<SearchResultHit>, AnswerError> {
        let search_results = if let Some(ref notation) = interaction.ragat_notation {
            let parsed = RAGAtParser::parse(notation);

            let components = self
                .execute_rag_at_specification(&parsed.components, interaction.clone())
                .await
                .map_err(|e| AnswerError::RagAtError(format!("{:?}", e)))?;

            let results = self
                .merge_component_results(components)
                .await
                .map_err(|e| AnswerError::RagAtError(format!("{:?}", e)))?;
            results
        } else {
            let max_documents = Limit(interaction.max_documents.unwrap_or(5));
            let min_similarity = Similarity(interaction.min_similarity.unwrap_or(0.5));

            let search_mode = match interaction
                .search_mode
                .as_ref()
                .map_or("vector", |s| s.as_str())
            {
                "vector" => SearchMode::Vector(VectorMode {
                    term: interaction.query.clone(),
                    similarity: min_similarity,
                }),
                mode => SearchMode::from_str(mode, interaction.query.clone()),
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
            };

            let hook_storage = self
                .read_side
                .get_hook_storage(read_api_key, collection_id)
                .await
                .map_err(|e| AnswerError::HookError(e.to_string()))?;
            let lock = hook_storage.read().await;
            let params = run_before_retrieval(
                &lock,
                params.clone(),
                None, // log_sender
                ExecOption {
                    allowed_hosts: Some(vec![]),
                    timeout: self.config.hook_timeout,
                },
            )
            .await
            .map_err(|e| AnswerError::BeforeRetrievalHookError(e.to_string()))?;
            drop(lock);

            let result = self
                .read_side
                .search(read_api_key, collection_id, params)
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
        read_api_key: ApiKey,
    ) -> Result<(Vec<(String, String)>, Option<SystemPrompt>), AnswerError> {
        let search_result_str = serde_json::to_string(&vec![] as &Vec<SearchResultHit>)
            .map_err(|e| AnswerError::JsonParsingError(e.to_string()))?;

        let mut variables = vec![
            ("question".to_string(), interaction.query.clone()),
            ("context".to_string(), search_result_str.clone()),
        ];

        let hook_storage = self
            .read_side
            .get_hook_storage(read_api_key, collection_id)
            .await
            .map_err(|e| AnswerError::HookError(e.to_string()))?;
        let lock = hook_storage.read().await;
        let (variables, system_prompt) = run_before_answer(
            &lock,
            (variables, system_prompt),
            None, // log_sender
            ExecOption {
                allowed_hosts: Some(vec![]),
                timeout: self.config.hook_timeout,
            },
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
        collection_id: CollectionId,
        read_api_key: ApiKey,
    ) -> Result<String, AnswerError> {
        let search_result_str = serde_json::to_string(&search_results)
            .map_err(|e| AnswerError::JsonParsingError(e.to_string()))?;

        let mut variables = vec![
            ("question".to_string(), interaction.query.clone()),
            ("context".to_string(), search_result_str.clone()),
        ];

        let answer_stream = self
            .llm_service
            .run_known_prompt_stream(
                KnownPrompts::Answer,
                variables,
                system_prompt,
                Some(llm_config.clone()),
            )
            .await
            .map_err(|e| AnswerError::LLMServiceError(e.to_string()))?;

        let mut answer_stream = answer_stream;
        let mut answer = String::new();

        while let Some(resp) = answer_stream.next().await {
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

        Ok(answer)
    }

    async fn generate_related_queries(
        &self,
        interaction: Interaction,
        llm_config: InteractionLLMConfig,
        search_results: Vec<SearchResultHit>,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    ) -> Result<Option<String>, AnswerError> {
        let mut related_queries_params = self
            .llm_service
            .get_related_questions_params(interaction.related);

        if related_queries_params.is_empty() {
            return Ok(None);
        }

        let search_result_str = serde_json::to_string(&search_results)
            .map_err(|e| AnswerError::JsonParsingError(e.to_string()))?;

        related_queries_params.push(("context".to_string(), search_result_str));
        related_queries_params.push(("query".to_string(), interaction.query.clone()));

        let related_questions_stream = self
            .llm_service
            .run_known_prompt_stream(
                KnownPrompts::GenerateRelatedQueries,
                related_queries_params,
                None,
                Some(llm_config),
            )
            .await
            .map_err(|e| AnswerError::LLMServiceError(e.to_string()))?;

        let mut related_questions_stream = related_questions_stream;
        let mut related_queries = String::new();

        while let Some(resp) = related_questions_stream.next().await {
            match resp {
                Ok(chunk) => {
                    related_queries.push_str(&chunk);
                    // Send related queries event
                    self.send_event(AnswerEvent::RelatedQueries { queries: chunk })
                        .await;
                }
                Err(e) => {
                    return Err(AnswerError::RelatedQueriesError(e.to_string()));
                }
            }
        }

        Ok(Some(related_queries))
    }

    async fn execute_party_planner(
        &self,
        interaction: Interaction,
        llm_config: InteractionLLMConfig,
        system_prompt: Option<SystemPrompt>,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    ) -> Result<String, AnswerError> {
        let party_planner = PartyPlanner::new(self.read_side.clone(), Some(llm_config.clone()));

        let mut party_planner_stream = party_planner.run(
            self.read_side.clone(),
            collection_id,
            read_api_key,
            interaction.query.clone(),
            interaction.messages.clone(),
            system_prompt,
        );

        let mut result = String::new();

        while let Some(message) = party_planner_stream.next().await {
            // Send result action event
            self.send_event(AnswerEvent::ResultAction {
                action: message.action.clone(),
                result: message.result.clone(),
            })
            .await;
            result.push_str(&format!(
                "Action: {}\nResult: {}\n",
                message.action, message.result
            ));
        }

        Ok(result)
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

        let search_results = self
            .read_side
            .search(
                self.read_api_key,
                self.collection_id,
                SearchParams {
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
                },
            )
            .await
            .map_err(|_| GeneralRagAtError::ReadError)?;

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
