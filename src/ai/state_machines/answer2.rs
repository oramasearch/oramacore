use std::{
    collections::HashMap,
    fmt::{Debug, Display},
    future::Future,
    sync::Arc,
    time::{Duration, Instant},
};

use backoff::{backoff::Backoff, ExponentialBackoffBuilder};
use orama_js_pool::ExecOption;
use serde::Serialize;
use tokio::{sync::mpsc, time::sleep};
use tokio_stream::StreamExt;
use tracing::{error, info, warn};

use crate::{
    ai::{
        llms::{KnownPrompts, LLMService},
        ragat::{ContextComponent, RAGAtParser},
        run_hooks::{run_before_answer, run_before_retrieval},
        state_machines::{
            advanced_autoquery::{AdvancedAutoqueryConfig, AdvancedAutoqueryStateMachine},
            answer::{AnswerError, ComponentResult},
        },
    },
    collection_manager::sides::{
        read::{
            AnalyticsHolder, AnalyticsMetadataFromRequest, ReadSide, SearchAnalyticEventOrigin,
            SearchRequest,
        },
        system_prompts::SystemPrompt,
    },
    types::{
        ApiKey, CollectionId, IndexId, Interaction, InteractionLLMConfig, Limit, Properties,
        SearchMode, SearchOffset, SearchParams, SearchResultHit, Similarity, VectorMode,
    },
};

#[derive(Debug, Serialize)]
#[cfg_attr(test, derive(PartialEq))]
pub enum PublicAnswerEvent {
    #[serde(rename = "state_changed")]
    StateChanged {
        state: String,
        message: String,
        data: Option<serde_json::Value>,
    },
    #[serde(rename = "selected_llm")]
    SelectedLLM { provider: String, model: String },
    #[serde(rename = "acknowledged")]
    Acknowledged,
    #[serde(rename = "progress")]
    Progress {
        current_step: serde_json::Value,
        total_steps: usize,
        message: String,
    },
    #[serde(rename = "search_results")]
    SearchResults { results: Vec<SearchResultHit> },
    #[serde(rename = "error")]
    Error {
        error: String,
        state: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_terminal: Option<bool>,
    },
    #[serde(rename = "optimizing_query")]
    OptimizingQuery {
        original_query: String,
        optimized_query: String,
    },
    #[serde(rename = "answer_token")]
    AnswerToken { token: String },
}

#[derive(Debug)]
enum Command {
    CalculateGPUOverload,
    CalculateLLMConfig,
    DetermineQueryStrategy,
    RunSimpleRAG,
    RunAdvancedAutoquery,
    OptimizeQuery,
    ExecuteSearch,
    ExecuteAnswer,
    Complete,
}
impl Display for Command {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Command::CalculateGPUOverload => write!(f, "calculate_gpu_overload"),
            Command::CalculateLLMConfig => write!(f, "calculate_llm_config"),
            Command::DetermineQueryStrategy => write!(f, "determine_query_strategy"),
            Command::RunSimpleRAG => write!(f, "run_simple_rag"),
            Command::RunAdvancedAutoquery => write!(f, "run_advanced_autoquery"),
            Command::OptimizeQuery => write!(f, "optimize_query"),
            Command::ExecuteSearch => write!(f, "execute_search"),
            Command::ExecuteAnswer => write!(f, "execute_answer"),
            Command::Complete => write!(f, "complete"),
        }
    }
}

async fn next<'context>(
    current_state: AnswerStateMachineState<'context>,
    command: Command,
) -> Result<AnswerStateMachineState<'context>, AnswerError> {
    match (current_state, command) {
        (
            AnswerStateMachineState::Initialized(InitializedState(context)),
            Command::CalculateGPUOverload,
        ) => {
            context.send_event(PublicAnswerEvent::StateChanged {
                state: "handle_gpu_overload".to_string(),
                message: "Checking GPU overload".to_string(),
                data: None,
            });

            if context.run.read_side.is_gpu_overloaded() {
                match context.run.read_side.select_random_remote_llm_service() {
                    Some((provider, model)) => {
                        info!("GPU is overloaded. Switching to \"{}\" as a remote LLM provider for this request.", provider);
                        context.run.interaction.llm_config =
                            Some(InteractionLLMConfig { model, provider });
                    }
                    None => {
                        warn!("GPU is overloaded and no remote LLM is available. Using local LLM, but it's gonna be slow.");
                    }
                }
            }

            Ok(AnswerStateMachineState::GPUOverloadHandled(
                GPUOverloadHandledState(context),
            ))
        }
        (
            AnswerStateMachineState::GPUOverloadHandled(GPUOverloadHandledState(context)),
            Command::CalculateLLMConfig,
        ) => {
            let llm_config = context
                .run
                .interaction
                .llm_config
                .clone()
                .unwrap_or_else(|| {
                    let (provider, model) = context.run.read_side.get_default_llm_config();
                    InteractionLLMConfig { model, provider }
                });

            context.send_event(PublicAnswerEvent::StateChanged {
                state: "get_llm_config".to_string(),
                message: "Getting LLM configuration".to_string(),
                data: Some(serde_json::json!({
                    "provider": llm_config.provider,
                    "model": llm_config.model
                })),
            });

            context.send_event(PublicAnswerEvent::SelectedLLM {
                provider: llm_config.provider.to_string(),
                model: llm_config.model.clone(),
            });
            context.send_event(PublicAnswerEvent::Acknowledged);

            Ok(AnswerStateMachineState::LLMConfigGot(LLMConfigGotState {
                context,
                llm_config,
            }))
        }
        (
            AnswerStateMachineState::LLMConfigGot(LLMConfigGotState {
                context,
                llm_config,
            }),
            Command::DetermineQueryStrategy,
        ) => {
            let variables = vec![("query".to_string(), context.run.interaction.query.clone())];

            let query_strategy: QueryStrategy = retry(
                &mut context.config.retry_config,
                "determine_query_strategy",
                &mut (),
                |_| {
                    calculate_query_strategy(
                        context.run.read_side.get_llm_service(),
                        &variables,
                        llm_config.clone(),
                    )
                },
            )
            .await?;

            context.send_event(PublicAnswerEvent::StateChanged {
                state: "determine_query_strategy".to_string(),
                message: "Determining query strategy".to_string(),
                data: Some(serde_json::json!({
                    "strategy": query_strategy.to_value()
                })),
            });

            Ok(AnswerStateMachineState::QueryStrategyDeterminated(
                QueryStrategyDeterminatedState {
                    context,
                    query_strategy,
                },
            ))
        }
        (
            AnswerStateMachineState::QueryStrategyDeterminated(QueryStrategyDeterminatedState {
                context,
                ..
            }),
            Command::RunSimpleRAG,
        ) => {
            context.send_event(PublicAnswerEvent::StateChanged {
                state: "simple_rag".to_string(),
                message: "Executing simple RAG".to_string(),
                data: None,
            });

            Ok(AnswerStateMachineState::SimpleRAGRun(SimpleRAGRunState(
                context,
            )))
        }
        (
            AnswerStateMachineState::QueryStrategyDeterminated(QueryStrategyDeterminatedState {
                context,
                ..
            }),
            Command::RunAdvancedAutoquery,
        ) => {
            // Execute advanced autoquery flow
            context.send_event(PublicAnswerEvent::StateChanged {
                state: "advanced_autoquery".to_string(),
                message: "Executing advanced autoquery".to_string(),
                data: None,
            });

            // I don't understand how this "run autoquery"
            // impacts the context
            // TODO: Ask to michele because it seems it doens't anything
            run_advanced_autoquery(&context).await?;

            Ok(AnswerStateMachineState::AdvancedAutoqueryRun(
                AdvancedAutoqueryRunState(context),
            ))
        }
        (
            AnswerStateMachineState::SimpleRAGRun(SimpleRAGRunState(context)),
            Command::OptimizeQuery,
        ) => {
            let optimized_query = context.run.interaction.query.clone();
            Ok(AnswerStateMachineState::QueryOptimized(
                QueryOptimizedState {
                    context,
                    optimized_query,
                },
            ))
        }
        (
            AnswerStateMachineState::AdvancedAutoqueryRun(AdvancedAutoqueryRunState(context)),
            Command::OptimizeQuery,
        ) => {
            let original_query = context.run.interaction.query.clone();

            let optimized_query = retry(
                &mut context.config.retry_config,
                "optimize_query",
                &mut (),
                |_| {
                    optimize_query(
                        context.run.read_side.get_llm_service(),
                        original_query.clone(),
                        context.run.interaction.llm_config.clone(),
                    )
                },
            )
            .await?;

            // Send optimizing query event
            context.send_event(PublicAnswerEvent::OptimizingQuery {
                original_query: context.run.interaction.query.clone(),
                optimized_query: optimized_query.clone(),
            });

            context.send_event(PublicAnswerEvent::StateChanged {
                state: "optimize_query".to_string(),
                message: "Optimizing query".to_string(),
                data: Some(serde_json::json!({
                    "original_query": original_query,
                    "optimized_query": optimized_query
                })),
            });

            Ok(AnswerStateMachineState::QueryOptimized(
                QueryOptimizedState {
                    context,
                    optimized_query: optimized_query.clone(),
                },
            ))
        }
        (
            AnswerStateMachineState::QueryOptimized(QueryOptimizedState {
                context,
                optimized_query,
            }),
            Command::ExecuteSearch,
        ) => {
            // TODO: execute Before Retrieval Hook
            context.send_event(PublicAnswerEvent::StateChanged {
                state: "execute_before_retrieval_hook".to_string(),
                message: "Executing before retrieval hook".to_string(),
                data: None,
            });

            let search_results = execute_search(
                &context.run,
                context.config.hook_timeout.clone(),
                optimized_query.clone(),
            )
            .await?;

            context.send_event(PublicAnswerEvent::SearchResults {
                results: search_results.clone(),
            });

            context.send_event(PublicAnswerEvent::StateChanged {
                state: "execute_search".to_string(),
                message: "Executing search".to_string(),
                data: Some(serde_json::json!({
                    "search_results_count": search_results.len(),
                    "results": search_results
                })),
            });

            // TODO: execute After Retrieval Hook
            context.send_event(PublicAnswerEvent::StateChanged {
                state: "execute_after_retrieval_hook".to_string(),
                message: "Executing after retrieval hook".to_string(),
                data: None,
            });

            Ok(AnswerStateMachineState::SearchExecuted(
                SearchExecutedAnswerState {
                    context,
                    search_results,
                    optimized_query,
                },
            ))
        }
        (
            AnswerStateMachineState::SearchExecuted(SearchExecutedAnswerState {
                context,
                search_results,
                ..
            }),
            Command::ExecuteAnswer,
        ) => {
            let system_prompt = get_system_prompt(
                &context.run,
                context.run.interaction.system_prompt_id.clone(),
            )
            .await?;

            if let Some(system_prompt) = &system_prompt {
                if let Some(analytics_holder) = context.analytics_holder.as_mut() {
                    analytics_holder.set_system_prompt_id(system_prompt.id.clone());
                }
            }

            context.send_event(PublicAnswerEvent::StateChanged {
                state: "handle_system_prompt".to_string(),
                message: "Processing system prompt".to_string(),
                data: Some(serde_json::json!({
                    "has_system_prompt": system_prompt.is_some(),
                    "system_prompt_id": context.run.interaction.system_prompt_id
                })),
            });

            let hook_timeout = context.config.hook_timeout.clone();
            let (variables, processed_system_prompt) = retry(
                &mut context.config.retry_config,
                "execute_before_answer_hook",
                &mut (),
                |_| {
                    execute_before_answer_hook(
                        &context.run,
                        hook_timeout.clone(),
                        system_prompt.as_ref(),
                    )
                },
            )
            .await?;

            context.send_event(PublicAnswerEvent::StateChanged {
                state: "execute_before_answer_hook".to_string(),
                message: "Executing before answer hook".to_string(),
                data: Some(serde_json::json!({
                    "variables_count": variables.len(),
                    "system_prompt_updated": processed_system_prompt.is_some()
                })),
            });

            let search_result_str = serde_json::to_string(&search_results)
                .map_err(|e| AnswerError::JsonParsingError(e.to_string()))?;
            let answer = retry(
                &mut context.config.retry_config,
                "execute_before_answer_hook",
                &mut context.analytics_holder,
                |analytics_holder| {
                    generate_answer(
                        &context.run,
                        analytics_holder,
                        &search_result_str,
                        processed_system_prompt.as_ref(),
                    )
                },
            )
            .await?;

            // if let Some(ana) = self.analytics_holder.as_ref() {
            //     let mut lock = ana.lock().await;
            //     let delta = if let Some(start_first_token) = start_first_token {
            //         start_first_token.elapsed()
            //     } else {
            //         Default::default()
            //     };
            //     lock.set_assistant_response(answer.clone(), delta);
            // }

            let bpe = tiktoken_rs::get_bpe_from_model("gpt-4o");
            let token_count = match bpe {
                Ok(bpe) => bpe.encode_with_special_tokens(&answer).len(),
                Err(_) => 0,
            };

            context.send_event(PublicAnswerEvent::StateChanged {
                state: "generate_answer".to_string(),
                message: "Generating answer".to_string(),
                data: Some(serde_json::json!({
                    "answer": answer,
                    "answer_token_count": token_count
                })),
            });

            Ok(AnswerStateMachineState::AnswerGenerated(
                AnswerGeneratedState {
                    context,
                    search_results,
                    variables,
                },
            ))
        }
        (
            AnswerStateMachineState::AnswerGenerated(AnswerGeneratedState {
                context,
                search_results,
                ..
            }),
            Command::Complete,
        ) => {
            context.send_event(PublicAnswerEvent::StateChanged {
                state: "completed".to_string(),
                message: "Answer generation completed".to_string(),
                data: None,
            });
            Ok(AnswerStateMachineState::Completed(CompletedState {
                context,
                search_results,
            }))
        }
        (current_state, command) => Err(AnswerError::InvalidCommandOnState(
            state_to_json(&current_state).to_string(),
            command.to_string(),
        )),
    }
}

fn state_to_json(state: &AnswerStateMachineState) -> serde_json::Value {
    match state {
        AnswerStateMachineState::Initialized(InitializedState(context)) => {
            serde_json::json!({
                "type": "Initialize",
                "interaction_id": context.run.interaction.interaction_id
            })
        }
        AnswerStateMachineState::GPUOverloadHandled(GPUOverloadHandledState(context)) => {
            serde_json::json!({
                "type": "HandleGPUOverload",
                "interaction_id": context.run.interaction.interaction_id
            })
        }
        AnswerStateMachineState::LLMConfigGot(LLMConfigGotState {
            context,
            llm_config,
        }) => {
            serde_json::json!({
                "type": "GetLLMConfig",
                "interaction_id": context.run.interaction.interaction_id,
                "llm_provider": llm_config.provider.to_string(),
                "llm_model": llm_config.model
            })
        }
        AnswerStateMachineState::QueryStrategyDeterminated(QueryStrategyDeterminatedState {
            context,
            query_strategy,
        }) => {
            serde_json::json!({
                "type": "DetermineQueryStrategy",
                "interaction_id": context.run.interaction.interaction_id,
                "query": context.run.interaction.query,
                "strategy": query_strategy.to_value()
            })
        }
        AnswerStateMachineState::SimpleRAGRun(SimpleRAGRunState(context)) => {
            serde_json::json!({
                "type": "RunSimpleRAG",
                "interaction_id": context.run.interaction.interaction_id,
                "query": context.run.interaction.query
            })
        }
        AnswerStateMachineState::AdvancedAutoqueryRun(AdvancedAutoqueryRunState(context)) => {
            serde_json::json!({
                "type": "RunAdvancedAutoquery",
                "interaction_id": context.run.interaction.interaction_id,
                "query": context.run.interaction.query
            })
        }
        AnswerStateMachineState::QueryOptimized(QueryOptimizedState {
            context,
            optimized_query,
        }) => {
            serde_json::json!({
                "type": "OptimizeQuery",
                "interaction_id": context.run.interaction.interaction_id,
                "original_query": context.run.interaction.query,
                "optimized_query": optimized_query
            })
        }
        AnswerStateMachineState::SearchExecuted(SearchExecutedAnswerState {
            context,
            optimized_query,
            search_results,
        }) => {
            serde_json::json!({
                "type": "ExecuteSearch",
                "interaction_id": context.run.interaction.interaction_id,
                "optimized_query": optimized_query,
                "search_results_count": search_results.len(),
            })
        }
        AnswerStateMachineState::AnswerGenerated(AnswerGeneratedState {
            context,
            search_results,
            variables,
        }) => {
            serde_json::json!({
                "type": "GenerateAnswer",
                "interaction_id": context.run.interaction.interaction_id,
                "search_results_count": search_results.len(),
                "variables_count": variables.len()
            })
        }
        AnswerStateMachineState::Completed(CompletedState { search_results, .. }) => {
            serde_json::json!({
                "type": "Completed",
                "search_results_count": search_results.len()
            })
        }
        AnswerStateMachineState::Error(ErrorState { error, .. }) => {
            serde_json::json!({
                "type": "Error",
                "error": error.to_string()
            })
        }
    }
}

#[derive(Debug)]
struct InitializedState<'context>(&'context mut AnswerStateMachineContext);
#[derive(Debug)]
struct GPUOverloadHandledState<'context>(&'context mut AnswerStateMachineContext);
#[derive(Debug)]
struct LLMConfigGotState<'context> {
    context: &'context mut AnswerStateMachineContext,
    llm_config: InteractionLLMConfig,
}
#[derive(Debug)]
struct QueryStrategyDeterminatedState<'context> {
    context: &'context mut AnswerStateMachineContext,
    query_strategy: QueryStrategy,
}
#[derive(Debug)]
struct AdvancedAutoqueryRunState<'context>(&'context mut AnswerStateMachineContext);
#[derive(Debug)]
struct SimpleRAGRunState<'context>(&'context mut AnswerStateMachineContext);
#[derive(Debug)]
struct QueryOptimizedState<'context> {
    context: &'context mut AnswerStateMachineContext,
    optimized_query: String,
}
#[derive(Debug)]
struct SearchExecutedAnswerState<'context> {
    context: &'context mut AnswerStateMachineContext,
    search_results: Vec<SearchResultHit>,
    optimized_query: String,
}
#[derive(Debug)]
struct CompletedState<'context> {
    context: &'context mut AnswerStateMachineContext,
    search_results: Vec<SearchResultHit>,
}
#[derive(Debug)]
struct AnswerGeneratedState<'context> {
    context: &'context mut AnswerStateMachineContext,
    search_results: Vec<SearchResultHit>,
    variables: Vec<(String, String)>,
}
#[derive(Debug)]
struct ErrorState {
    error: AnswerError,
}

#[derive(Debug)]
enum AnswerStateMachineState<'context> {
    Initialized(InitializedState<'context>),
    GPUOverloadHandled(GPUOverloadHandledState<'context>),
    LLMConfigGot(LLMConfigGotState<'context>),
    QueryStrategyDeterminated(QueryStrategyDeterminatedState<'context>),
    AdvancedAutoqueryRun(AdvancedAutoqueryRunState<'context>),
    SimpleRAGRun(SimpleRAGRunState<'context>),
    QueryOptimized(QueryOptimizedState<'context>),
    SearchExecuted(SearchExecutedAnswerState<'context>),
    AnswerGenerated(AnswerGeneratedState<'context>),
    Completed(CompletedState<'context>),
    Error(ErrorState),
}

impl AnswerStateMachineState<'_> {
    fn get_next_event(&self) -> Option<Command> {
        tracing::trace!("Calculating next event from state {:?}", self);
        let next_event = match self {
            AnswerStateMachineState::Initialized(_) => Some(Command::CalculateGPUOverload),
            AnswerStateMachineState::GPUOverloadHandled(_) => Some(Command::CalculateLLMConfig),
            AnswerStateMachineState::LLMConfigGot(_) => Some(Command::DetermineQueryStrategy),
            AnswerStateMachineState::QueryStrategyDeterminated(s) => match &s.query_strategy {
                QueryStrategy::SimpleRAG => Some(Command::RunSimpleRAG),
                QueryStrategy::AdvancedAutoquery => Some(Command::RunAdvancedAutoquery),
            },
            AnswerStateMachineState::AdvancedAutoqueryRun(_) => Some(Command::OptimizeQuery),
            AnswerStateMachineState::SimpleRAGRun(_) => Some(Command::OptimizeQuery),
            AnswerStateMachineState::QueryOptimized(_) => Some(Command::ExecuteSearch),
            AnswerStateMachineState::SearchExecuted(_) => Some(Command::ExecuteAnswer),
            AnswerStateMachineState::AnswerGenerated(_) => Some(Command::Complete),
            AnswerStateMachineState::Completed(_) => None,
            AnswerStateMachineState::Error(_) => None,
        };

        let next_event = next_event?;
        tracing::debug!("Calculated next event: {:?}", next_event);

        Some(next_event)
    }
}

#[derive(Debug)]
pub struct AnswerStateMachineConfig {
    pub max_concurrent_operations: usize,

    pub llm_timeout: Duration,
    pub hook_timeout: Duration,

    pub retry_config: RetryConfig,
}

impl Default for AnswerStateMachineConfig {
    fn default() -> Self {
        Self {
            max_concurrent_operations: 5,
            llm_timeout: Duration::from_secs(30),
            hook_timeout: Duration::from_millis(500),
            retry_config: RetryConfig::default(),
        }
    }
}

#[derive(Debug)]
pub struct RetryConfig {
    pub max_retries: usize,
    pub initial_backoff: Duration,
    pub max_backoff: Duration,
    pub timeout: Duration,
    pub retry_count: HashMap<String, usize>,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(30),
            timeout: Duration::from_secs(60),
            retry_count: Default::default(),
        }
    }
}

pub struct AnswerStateMachineRunContext {
    interaction: Interaction,
    read_side: Arc<ReadSide>,
    collection_id: CollectionId,
    read_api_key: ApiKey,
    event_sender: Option<mpsc::UnboundedSender<PublicAnswerEvent>>,
}
impl AnswerStateMachineRunContext {
    fn send_event(&self, ev: PublicAnswerEvent) {
        if let Some(event_sender) = self.event_sender.as_ref() {
            if let Err(e) = event_sender.send(ev) {
                error!(error = ?e, "Cannot send public answer event");
            }
        }
    }
}
pub struct AnswerStateMachineContext {
    run: AnswerStateMachineRunContext,
    analytics_holder: Option<AnalyticsHolder>,
    config: AnswerStateMachineConfig,
}
impl Debug for AnswerStateMachineContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnswerStateMachineContext")
            .field("run", &self.run.collection_id)
            .field("config", &self.config)
            .finish()
    }
}

impl AnswerStateMachineContext {
    fn send_event(&self, ev: PublicAnswerEvent) {
        self.run.send_event(ev);
    }
}

pub struct AnswerStateMachine {}

impl AnswerStateMachine {
    pub fn new() -> Self {
        Self {}
    }

    pub async fn execute(
        self,
        read_side: Arc<ReadSide>,
        collection_id: CollectionId,
        read_api_key: ApiKey,
        interaction: Interaction,
        config: Option<AnswerStateMachineConfig>,
        event_sender: Option<mpsc::UnboundedSender<PublicAnswerEvent>>,
        analytics_metadata: Option<AnalyticsMetadataFromRequest>,
    ) -> Result<(), AnswerError> {
        let analytics_holder = if let Some(analytics_metadata) = analytics_metadata {
            Some(AnalyticsHolder::new(
                read_side.clone(),
                collection_id,
                &interaction,
                analytics_metadata,
            ))
        } else {
            None
        };

        let mut context = AnswerStateMachineContext {
            run: AnswerStateMachineRunContext {
                read_side,
                collection_id: collection_id,
                read_api_key,
                interaction,
                event_sender,
            },
            analytics_holder,
            config: config.unwrap_or_default(),
        };
        context.send_event(PublicAnswerEvent::StateChanged {
            state: "initializing".to_string(),
            message: "Starting answer generation".to_string(),
            data: None,
        });
        let mut current_state =
            AnswerStateMachineState::Initialized(InitializedState(&mut context));

        let mut rag_steps = vec![state_to_json(&current_state)];
        loop {
            let event = current_state.get_next_event();

            let Some(event) = event else {
                info!("Last state {:?}", current_state);
                break;
            };

            info!("current_state {:?}, event {:?}", current_state, event);
            match next(current_state, event).await {
                Ok(next_state) => {
                    current_state = next_state;
                }
                Err(error) => {
                    current_state = AnswerStateMachineState::Error(ErrorState { error });
                    break;
                }
            }
            rag_steps.push(state_to_json(&current_state));
        }

        match current_state {
            AnswerStateMachineState::Completed(CompletedState { context, .. }) => {
                if let Some(analytics_holder) = context.analytics_holder.as_mut() {
                    analytics_holder.set_rag_steps(rag_steps);
                }
                context.send_event(PublicAnswerEvent::StateChanged {
                    state: "finished".to_string(),
                    message: "Answer generation finished".to_string(),
                    data: None,
                });
            }
            AnswerStateMachineState::Error(ErrorState { error }) => {
                let error_str = error.to_string();
                context.send_event(PublicAnswerEvent::Error {
                    error: error_str.clone(),
                    state: "error".to_string(),
                    is_terminal: Some(true),
                });
                if let Some(analytics_holder) = context.analytics_holder.as_mut() {
                    analytics_holder.set_rag_steps(rag_steps);
                    analytics_holder.set_error(error_str);
                }
                error!(error = ?error, "Answer state machine failed");
                return Err(error);
            }
            _ => {}
        };

        Ok(())
    }
}

#[derive(Debug)]
enum QueryStrategy {
    SimpleRAG,
    AdvancedAutoquery,
}
impl QueryStrategy {
    fn to_value(&self) -> &'static str {
        match self {
            Self::SimpleRAG => "simple_rag",
            Self::AdvancedAutoquery => "advanced_autoquery",
        }
    }
}

/// Retry logic
async fn retry<'param, 'output, 'config, F, P, Fut, T>(
    config: &'config mut RetryConfig,
    operation_name: &str,
    param: &'param mut P,
    mut operation: F,
) -> Result<T, AnswerError>
where
    F: FnMut(&mut P) -> Fut + Send + Sync,
    Fut: Future<Output = Result<T, AnswerError>> + Send + 'output,
    'param: 'output,
{
    let mut backoff = ExponentialBackoffBuilder::new()
        .with_initial_interval(config.initial_backoff)
        .with_max_interval(config.max_backoff)
        .with_max_elapsed_time(Some(config.timeout))
        .build();

    let mut retry_count = 0;
    loop {
        match operation(param).await {
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
                if retry_count > config.max_retries {
                    error!(
                        "Operation {} failed after {} retries: {:?}",
                        operation_name, retry_count, e
                    );
                    return Err(e);
                }

                warn!(
                    "Operation {} failed (attempt {}/{}), retrying: {:?}",
                    operation_name, retry_count, config.max_retries, e
                );

                // Update retry count
                {
                    config
                        .retry_count
                        .insert(operation_name.to_string(), retry_count);
                }

                // Wait before retry
                if let Some(duration) = Backoff::next_backoff(&mut backoff) {
                    sleep(duration).await;
                }
            }
        }
    }
}

async fn calculate_query_strategy(
    llm_service: Arc<LLMService>,
    variables: &[(String, String)],
    llm_config: InteractionLLMConfig,
) -> Result<QueryStrategy, AnswerError> {
    let result = llm_service
        .run_known_prompt(
            KnownPrompts::DetermineQueryStrategy,
            vec![],
            variables.to_vec(),
            None,
            Some(llm_config),
        )
        .await
        .map_err(|e| AnswerError::LLMServiceError(e.to_string()))?;
    // Try to parse the JSON response to determine strategy
    // We want to identify only 2 strategy:
    // - simple rag: search directly and perform the rag
    // - advanced_autoquery: run the autoquery state machine
    let query_strategy = if result.trim().is_empty() {
        QueryStrategy::SimpleRAG
    } else {
        match serde_json::from_str::<Vec<String>>(&result) {
            Ok(strategy_code) => {
                if strategy_code.is_empty() {
                    QueryStrategy::SimpleRAG
                } else {
                    let code = &strategy_code[0];
                    match code.as_str() {
                        "000" => QueryStrategy::SimpleRAG,
                        "001" | "011" | "100" => QueryStrategy::AdvancedAutoquery,
                        _ => QueryStrategy::SimpleRAG, // Default
                    }
                }
            }
            Err(_) => {
                // If JSON parsing fails, try to extract the code from the response
                // This commonly happen with LLM because it could return the response
                // with ` or " or ' or other wraps.
                let cleaned_result = result.trim();
                if cleaned_result.contains("000") {
                    QueryStrategy::SimpleRAG
                } else if cleaned_result.contains("001")
                    || cleaned_result.contains("011")
                    || cleaned_result.contains("100")
                {
                    QueryStrategy::AdvancedAutoquery
                } else {
                    // Default
                    QueryStrategy::SimpleRAG
                }
            }
        }
    };

    Ok(query_strategy)
}

async fn optimize_query(
    llm_service: Arc<LLMService>,
    query: String,
    llm_config: Option<InteractionLLMConfig>,
) -> Result<String, AnswerError> {
    let variables = vec![("input".to_string(), query.clone())];
    let optimized_query = llm_service
        .run_known_prompt(
            KnownPrompts::OptimizeQuery,
            vec![],
            variables,
            None,
            llm_config,
        )
        .await
        .map_err(|e| AnswerError::LLMServiceError(e.to_string()))?;

    Ok(optimized_query)
}

async fn run_advanced_autoquery(context: &AnswerStateMachineContext) -> Result<(), AnswerError> {
    use crate::ai::state_machines::advanced_autoquery::AdvancedAutoqueryEvent;

    // Get collection stats for advanced autoquery
    let collection_stats = context
        .run
        .read_side
        .collection_stats(
            context.run.read_api_key,
            context.run.collection_id,
            crate::types::CollectionStatsRequest { with_keys: false },
        )
        .await
        .map_err(|e| AnswerError::ReadError(e.to_string()))?;

    // Create and run the advanced autoquery state machine with event streaming
    let advanced_state_machine = AdvancedAutoqueryStateMachine::new(
        AdvancedAutoqueryConfig::default(),
        context.run.read_side.get_llm_service(),
        context.run.interaction.llm_config.clone(),
        collection_stats,
        context.run.read_side.clone(),
        context.run.collection_id,
        context.run.read_api_key,
    );

    // Run the advanced autoquery with streaming and capture events
    let mut messages = context.run.interaction.messages.clone();
    if messages.is_empty() {
        messages.push(crate::types::InteractionMessage {
            role: crate::types::Role::User,
            content: context.run.interaction.query.clone(),
        });
    }
    let advanced_event_stream = advanced_state_machine
        .run_stream(
            messages,
            context.run.collection_id,
            context.run.read_api_key,
        )
        .await
        .map_err(|e| AnswerError::AdvancedAutoqueryError(e.to_string()))?;

    // Forward advanced autoquery events to the client
    let mut advanced_event_stream = advanced_event_stream;
    while let Some(advanced_event) = advanced_event_stream.next().await {
        // Convert advanced autoquery events to answer events
        match advanced_event {
            AdvancedAutoqueryEvent::StateChanged {
                state,
                message,
                data,
            } => {
                context.send_event(PublicAnswerEvent::StateChanged {
                    state: format!("advanced_autoquery_{state}"),
                    message: format!("Advanced Autoquery: {message}"),
                    data,
                });
            }
            AdvancedAutoqueryEvent::Progress {
                current_step,
                total_steps,
                message,
            } => {
                context.send_event(PublicAnswerEvent::Progress {
                    current_step: serde_json::json!({"type": "advanced_autoquery", "step": current_step}),
                    total_steps,
                    message: format!("Advanced Autoquery: {message}"),
                });
            }
            AdvancedAutoqueryEvent::SearchResults { results } => {
                // Convert to our search results format
                let search_hits = results
                    .into_iter()
                    .flat_map(|query_result| {
                        query_result
                            .results
                            .into_iter()
                            .flat_map(|search_result| search_result.hits)
                    })
                    .collect::<Vec<_>>();

                context.send_event(PublicAnswerEvent::SearchResults {
                    results: search_hits,
                });
            }
            AdvancedAutoqueryEvent::Error {
                error,
                state,
                is_terminal,
            } => {
                context.send_event(PublicAnswerEvent::Error {
                    error: format!("Advanced Autoquery Error: {error}"),
                    state: format!("advanced_autoquery_{state}"),
                    is_terminal,
                });
            }
        }
    }

    Ok(())
}

async fn execute_search(
    context: &AnswerStateMachineRunContext,
    hook_timeout: Duration,
    optimized_query: String,
) -> Result<Vec<SearchResultHit>, AnswerError> {
    let search_results = if let Some(ref notation) = context.interaction.ragat_notation {
        let parsed = RAGAtParser::parse(notation);

        let components = execute_rag_at_specification(
            context,
            &parsed.components,
            context.interaction.query.clone(),
            context.interaction.interaction_id.clone(),
        )
        .await
        .map_err(|e| AnswerError::RagAtError(format!("{e:?}")))?;

        components.into_iter().flat_map(|c| c.hits).collect()
    } else {
        let max_documents = Limit(context.interaction.max_documents.unwrap_or(5));
        let min_similarity = Similarity(context.interaction.min_similarity.unwrap_or(0.5));

        let search_mode = match context
            .interaction
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
            indexes: None,
            sort_by: None,
            user_id: None,
            group_by: None,
        };

        let hook_storage = context
            .read_side
            .get_hook_storage(context.read_api_key, context.collection_id)
            .await
            .map_err(|e| AnswerError::HookError(e.to_string()))?;
        let lock = hook_storage.read().await;
        let params = run_before_retrieval(
            &lock,
            params.clone(),
            None, // log_sender
            ExecOption {
                allowed_hosts: Some(vec![]),
                timeout: hook_timeout,
            },
        )
        .await
        .map_err(|e| AnswerError::BeforeRetrievalHookError(e.to_string()))?;
        drop(lock);

        let result = context
            .read_side
            .search(
                context.read_api_key,
                context.collection_id,
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

use crate::ai::ragat::*;

async fn execute_rag_at_specification(
    context: &AnswerStateMachineRunContext,
    components: &[ContextComponent],
    query: String,
    interaction_id: String,
) -> Result<Vec<ComponentResult>, GeneralRagAtError> {
    let mut results = Vec::with_capacity(components.len());

    for component in components {
        let component_result =
            execute_single_component(context, component, query.clone(), interaction_id.clone())
                .await?;

        results.push(component_result);
    }

    Ok(results)
}

async fn execute_single_component(
    context: &AnswerStateMachineRunContext,
    component: &ContextComponent,
    query: String,
    interaction_id: String,
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

    let search_results = context
        .read_side
        .search(
            context.read_api_key,
            context.collection_id,
            SearchRequest {
                search_params: SearchParams {
                    mode: SearchMode::Vector(VectorMode {
                        term: query,
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
                analytics_metadata: None,
                interaction_id: Some(interaction_id),
            },
        )
        .await
        .map_err(|_| GeneralRagAtError::ReadError)?;

    let all_hits = search_results.hits.clone();

    Ok(ComponentResult { hits: all_hits })
}

async fn get_system_prompt(
    context: &AnswerStateMachineRunContext,
    system_prompt_id: Option<String>,
) -> Result<Option<SystemPrompt>, AnswerError> {
    match system_prompt_id {
        Some(id) => {
            let full_prompt = context
                .read_side
                .get_system_prompt(context.read_api_key, context.collection_id, id)
                .await
                .map_err(|e| AnswerError::SystemPromptError(e.to_string()))?;
            Ok(full_prompt)
        }
        None => {
            let has_system_prompts = context
                .read_side
                .has_system_prompts(context.read_api_key, context.collection_id)
                .await
                .map_err(|e| AnswerError::SystemPromptError(e.to_string()))?;

            if has_system_prompts {
                let chosen_system_prompt = context
                    .read_side
                    .perform_system_prompt_selection(context.read_api_key, context.collection_id)
                    .await
                    .map_err(|e| AnswerError::SystemPromptError(e.to_string()))?;
                Ok(chosen_system_prompt)
            } else {
                Ok(None)
            }
        }
    }
}

async fn execute_before_answer_hook(
    context: &AnswerStateMachineRunContext,
    hook_timeout: Duration,
    system_prompt: Option<&SystemPrompt>,
) -> Result<(Vec<(String, String)>, Option<SystemPrompt>), AnswerError> {
    let search_result_str = serde_json::to_string(&vec![] as &Vec<SearchResultHit>)
        .map_err(|e| AnswerError::JsonParsingError(e.to_string()))?;

    let variables = vec![
        ("question".to_string(), context.interaction.query.clone()),
        ("context".to_string(), search_result_str.clone()),
    ];

    let hook_storage = context
        .read_side
        .get_hook_storage(context.read_api_key, context.collection_id)
        .await
        .map_err(|e| AnswerError::HookError(e.to_string()))?;
    let lock = hook_storage.read().await;
    let (variables, system_prompt) = run_before_answer(
        &lock,
        (variables, system_prompt.cloned()),
        None,
        ExecOption {
            allowed_hosts: Some(vec![]),
            timeout: hook_timeout,
        },
    )
    .await
    .map_err(|e| AnswerError::BeforeAnswerHookError(e.to_string()))?;
    drop(lock);

    Ok((variables, system_prompt))
}

async fn generate_answer<'context, 'other>(
    context: &'context AnswerStateMachineRunContext,
    analytics_holder: &'context mut Option<AnalyticsHolder>,
    search_result_str: &str,
    processed_system_prompt: Option<&'other SystemPrompt>,
) -> Result<String, AnswerError>
where
    'context: 'other,
{
    let variables = vec![
        ("question".to_string(), context.interaction.query.clone()),
        ("context".to_string(), search_result_str.to_string()),
    ];
    let mut answer_stream = context
        .read_side
        .get_llm_service()
        .run_known_prompt_stream(
            KnownPrompts::Answer,
            context.interaction.messages.clone(),
            variables,
            processed_system_prompt.cloned(),
            context.interaction.llm_config.clone(),
        )
        .await
        .map_err(|e| AnswerError::LLMServiceError(e.to_string()))?;

    let mut answer = String::new();

    let start_time_to_first_token = Instant::now();
    let mut start_first_token = None;
    while let Some(resp) = answer_stream.next().await {
        if start_first_token.is_none() {
            start_first_token = Some(Instant::now());

            if let Some(analytics_holder) = analytics_holder.as_mut() {
                analytics_holder.set_time_to_first_token(start_time_to_first_token.elapsed());
            }
        }

        match resp {
            Ok(chunk) => {
                answer.push_str(&chunk);
                // Send answer token event
                context.send_event(PublicAnswerEvent::AnswerToken { token: chunk });
            }
            Err(e) => {
                return Err(AnswerError::AnswerGenerationError(e.to_string()));
            }
        }
    }

    Ok(answer)
}

#[cfg(test)]
mod tests_answer2 {
    use itertools::Itertools;
    use tokio::sync::RwLock;

    use crate::{
        tests::utils::{create_ai_server_mock, create_oramacore_config, init_log, TestContext},
        types::{Document, DocumentList},
    };

    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    async fn test_answer_state_machine_simple_rag() {
        init_log();

        let completition_mock = Arc::new(RwLock::new(vec![
            vec!["000".to_string()], // SIMPLE RAG
            vec![
                "I'm Tommaso, a ".to_string(),
                " software developer".to_string(),
                " bla bla bla".to_string(),
            ],
        ]));
        let completition_req = Arc::new(RwLock::new(vec![]));

        let output = create_ai_server_mock(completition_mock, completition_req.clone())
            .await
            .unwrap();
        let mut config = create_oramacore_config();
        config.ai_server.llm.port = Some(output.port());

        let test_context = TestContext::new_with_config(config).await;

        let collection_client = test_context.create_collection().await.unwrap();
        let index_client = collection_client.create_index().await.unwrap();

        let docs = r#" [
            {"id":"1","name":"I'm Tommaso, a software developer"}
        ]"#;
        let docs = serde_json::from_str::<Vec<Document>>(docs).unwrap();
        index_client
            .insert_documents(DocumentList(docs))
            .await
            .unwrap();

        sleep(Duration::from_millis(1_000)).await;

        let state_machine = AnswerStateMachine::new();
        let (answer_sender, mut answer_receiver) = mpsc::unbounded_channel();

        state_machine
            .execute(
                test_context.reader.clone(),
                collection_client.collection_id,
                collection_client.read_api_key,
                Interaction {
                    conversation_id: "the-conversation-id".to_string(),
                    interaction_id: "the-interaction-id".to_string(),
                    llm_config: None,
                    max_documents: None,
                    messages: vec![],
                    min_similarity: None,
                    query: "Who is Tommaso?".to_string(),
                    system_prompt_id: None,
                    ragat_notation: None,
                    related: None,
                    search_mode: None,
                    visitor_id: "the-visitor-id".to_string(),
                },
                None,
                Some(answer_sender),
                None,
            )
            .await
            .unwrap();

        let mut buffer = vec![];
        answer_receiver.recv_many(&mut buffer, 100).await;

        let expected_state_changed = vec![
            "initializing",
            "handle_gpu_overload",
            "get_llm_config",
            "determine_query_strategy",
            "simple_rag",
            "execute_before_retrieval_hook",
            "execute_search",
            "execute_after_retrieval_hook",
            "handle_system_prompt",
            "execute_before_answer_hook",
            "generate_answer",
            "completed",
        ];
        let state_changed: Vec<_> = buffer
            .iter()
            .filter_map(|s| {
                if let PublicAnswerEvent::StateChanged { state, .. } = s {
                    Some(state)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(state_changed, expected_state_changed);

        let answer = buffer
            .iter()
            .filter_map(|s| {
                if let PublicAnswerEvent::AnswerToken { token } = s {
                    Some(token)
                } else {
                    None
                }
            })
            .join("");
        assert_eq!(answer, "I'm Tommaso, a  software developer bla bla bla");

        drop(test_context);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_answer_state_machine_advance_autoquery() {
        init_log();

        let completition_mock = Arc::new(RwLock::new(vec![
            vec!["001".to_string()], // ADVANCE AUTOQUERY
            vec![r#"["Tommaso", "software", "developer"]"#.to_string()],
            // For each term, list of interesting properties
            vec![r#"{}"#.to_string()],
            vec![r#"{}"#.to_string()],
            vec![r#"{}"#.to_string()],
            // For each term, generate the search params
            vec![r#"{"term": "Tommaso"}"#.to_string()],
            vec![r#"{"term": "software"}"#.to_string()],
            vec![r#"{"term": "developer"}"#.to_string()],
            // optimize query
            vec![r#"Tommaso software developer"#.to_string()],
            // Real answer
            vec![r#"Yes, Tommaso is a software developer"#.to_string()],
        ]));
        let completition_req = Arc::new(RwLock::new(vec![]));

        let output = create_ai_server_mock(completition_mock, completition_req.clone())
            .await
            .unwrap();
        let mut config = create_oramacore_config();
        config.ai_server.llm.port = Some(output.port());

        let test_context = TestContext::new_with_config(config).await;

        let collection_client = test_context.create_collection().await.unwrap();
        let index_client = collection_client.create_index().await.unwrap();

        let docs = r#" [
            {"id":"1","content":"I'm Tommaso, a software developer"}
        ]"#;
        let docs = serde_json::from_str::<Vec<Document>>(docs).unwrap();
        index_client
            .insert_documents(DocumentList(docs))
            .await
            .unwrap();

        sleep(Duration::from_millis(1_000)).await;

        let state_machine = AnswerStateMachine::new();
        let (answer_sender, mut answer_receiver) = mpsc::unbounded_channel();
        state_machine
            .execute(
                test_context.reader.clone(),
                collection_client.collection_id,
                collection_client.read_api_key,
                Interaction {
                    conversation_id: "the-conversation-id".to_string(),
                    interaction_id: "the-interaction-id".to_string(),
                    llm_config: None,
                    max_documents: None,
                    messages: vec![],
                    min_similarity: None,
                    query: "Is Tommaso be a software developer?".to_string(),
                    system_prompt_id: None,
                    ragat_notation: None,
                    related: None,
                    search_mode: None,
                    visitor_id: "the-visitor-id".to_string(),
                },
                None,
                Some(answer_sender),
                None,
            )
            .await
            .unwrap();

        let mut buffer = vec![];
        answer_receiver.recv_many(&mut buffer, 100).await;

        let expected_state_changed = vec![
            "initializing",
            "handle_gpu_overload",
            "get_llm_config",
            "determine_query_strategy",
            "advanced_autoquery",
            "advanced_autoquery_initializing",
            "advanced_autoquery_analyzing_input",
            "advanced_autoquery_query_optimized",
            "advanced_autoquery_query_optimized",
            "advanced_autoquery_select_properties",
            "advanced_autoquery_properties_selected",
            "advanced_autoquery_combine_queries",
            "advanced_autoquery_combine_queries",
            "advanced_autoquery_queries_combined",
            "advanced_autoquery_queries_combined",
            "advanced_autoquery_generate_tracked_queries",
            "advanced_autoquery_tracked_queries_generated",
            "advanced_autoquery_tracked_queries_generated",
            "advanced_autoquery_execute_before_retrieval_hook",
            "advanced_autoquery_hooks_executed",
            "advanced_autoquery_hooks_executed",
            "advanced_autoquery_execute_searches",
            "advanced_autoquery_search_results",
            "advanced_autoquery_search_results",
            "advanced_autoquery_completed",
            "optimize_query",
            "execute_before_retrieval_hook",
            "execute_search",
            "execute_after_retrieval_hook",
            "handle_system_prompt",
            "execute_before_answer_hook",
            "generate_answer",
            "completed",
        ];
        let state_changed: Vec<_> = buffer
            .iter()
            .filter_map(|s| {
                if let PublicAnswerEvent::StateChanged { state, .. } = s {
                    Some(state)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(state_changed, expected_state_changed);

        let answer = buffer
            .iter()
            .filter_map(|s| {
                if let PublicAnswerEvent::AnswerToken { token } = s {
                    Some(token)
                } else {
                    None
                }
            })
            .join("");
        assert_eq!(answer, "Yes, Tommaso is a software developer");

        drop(test_context);
    }
}
