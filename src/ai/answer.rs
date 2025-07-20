use futures::TryFutureExt;
use hook_storage::HookReaderError;
use orama_js_pool::{ExecOption, JSRunnerError, OutputChannel};
use std::{collections::HashMap, sync::Arc, time::Duration};
use thiserror::Error;
use tokio::sync::mpsc::error::SendError;
use tokio_stream::StreamExt;
use tracing::{info, warn};

use crate::{
    ai::{
        llms,
        party_planner::PartyPlanner,
        ragat::{ContextComponent, GeneralRagAtError, RAGAtParser},
        run_hooks::{run_before_answer, run_before_retrieval},
    },
    collection_manager::sides::{
        read::{ReadError, ReadSide},
        system_prompts::SystemPrompt,
    },
    types::{
        ApiKey, CollectionId, IndexId, Interaction, InteractionLLMConfig, Limit, Properties,
        SearchMode, SearchOffset, SearchParams, SearchResultHit, Similarity, VectorMode,
    },
};

#[derive(Debug, Error)]
pub enum AnswerError {
    #[error("Generic error: {0}")]
    Generic(#[from] anyhow::Error),
    #[error("read error: {0}")]
    ReadError(#[from] ReadError),
    #[error("channel is closed: {0}")]
    ChannelClosed(#[from] SendError<AnswerEvent>),
    #[error("Hook read error: {0:?}")]
    HookError(#[from] HookReaderError),
    #[error("JS run error: {0:?}")]
    JSError(#[from] JSRunnerError),
}

pub struct Answer {
    read_side: Arc<ReadSide>,
    collection_id: CollectionId,
    read_api_key: ApiKey,
}

#[derive(Debug)]
struct ComponentResult {
    hits: Vec<SearchResultHit>,
}

impl Answer {
    pub async fn try_new(
        read_side: Arc<ReadSide>,
        collection_id: CollectionId,
        read_api_key: ApiKey,
    ) -> Result<Self, AnswerError> {
        read_side
            .check_read_api_key(collection_id, read_api_key)
            .await?;

        Ok(Self {
            read_side,
            collection_id,
            read_api_key,
        })
    }

    pub async fn planned_answer(
        self,
        mut interaction: Interaction,
        sender: tokio::sync::mpsc::UnboundedSender<AnswerEvent>,
    ) -> Result<(), AnswerError> {
        self.handle_gpu_overload(&mut interaction).await;

        let llm_config = self.get_llm_config(&interaction);

        sender.send(AnswerEvent::Acknowledged)?;
        sender.send(AnswerEvent::SelectedLLM(llm_config.clone()))?;

        let system_prompt = self
            .handle_system_prompt(interaction.system_prompt_id)
            .await?;

        let party_planner = PartyPlanner::new(self.read_side.clone(), Some(llm_config.clone()));

        let mut party_planner_stream = party_planner.run(
            self.read_side.clone(),
            self.collection_id,
            self.read_api_key,
            interaction.query.clone(),
            interaction.messages.clone(),
            system_prompt,
        );

        while let Some(message) = party_planner_stream.next().await {
            sender.send(AnswerEvent::ResultAction {
                action: message.action,
                result: message.result,
            })?;
        }

        let llm_service = self.read_side.get_llm_service();
        let mut related_queries_params =
            llm_service.get_related_questions_params(interaction.related);
        if related_queries_params.is_empty() {
            warn!("related_queries_params is empty. stop streaming");
            return Ok(());
        }
        related_queries_params.push(("context".to_string(), "{}".to_string())); // @todo: check if we can retrieve additional context
        related_queries_params.push(("query".to_string(), interaction.query.clone()));

        self.handle_related_queries(&llm_service, llm_config, related_queries_params, &sender)
            .await?;

        Ok(())
    }

    pub async fn answer(
        self,
        mut interaction: Interaction,
        sender: tokio::sync::mpsc::UnboundedSender<AnswerEvent>,
        log_sender: Option<Arc<tokio::sync::broadcast::Sender<(OutputChannel, String)>>>,
    ) -> Result<(), AnswerError> {
        info!("Answering interaction...");
        self.handle_gpu_overload(&mut interaction).await;

        let llm_config = self.get_llm_config(&interaction);

        sender.send(AnswerEvent::SelectedLLM(llm_config.clone()))?;
        sender.send(AnswerEvent::Acknowledged)?;

        let system_prompt = self
            .handle_system_prompt(interaction.system_prompt_id.clone())
            .await?;
        info!("With system prompt: {:?}", system_prompt.is_some());

        let llm_service = self.read_side.get_llm_service();
        let optimized_query_variables = vec![("input".to_string(), interaction.query.clone())];
        let optimized_query = llm_service
            .run_known_prompt(
                llms::KnownPrompts::OptimizeQuery,
                optimized_query_variables,
                Some(llm_config.clone()),
            )
            .await
            .unwrap_or_else(|_| interaction.query.clone()); // fallback to the original query if the optimization fails
        info!("Optimized query: {}", optimized_query);

        sender.send(AnswerEvent::OptimizeingQuery(optimized_query.clone()))?;

        let search_results = if let Some(ref notation) = interaction.ragat_notation {
            let parsed = RAGAtParser::parse(notation);

            let components = self
                .execute_rag_at_specification(&parsed.components, interaction.clone())
                .map_err(|_| AnswerError::Generic(anyhow::anyhow!("Error")))
                .await?;

            let results = self
                .merge_component_results(components)
                .map_err(|_| AnswerError::Generic(anyhow::anyhow!("Error")))
                .await?;
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
                .get_hook_storage(self.read_api_key, self.collection_id)
                .await?;
            let lock = hook_storage.read().await;
            let params = run_before_retrieval(
                &lock,
                params.clone(),
                log_sender.clone(),
                ExecOption {
                    allowed_hosts: Some(vec![]),
                    timeout: Duration::from_millis(500),
                },
            )
            .await?;
            drop(lock);

            let result = self
                .read_side
                .search(self.read_api_key, self.collection_id, params)
                .await?;
            result.hits
        };

        let search_result_str = serde_json::to_string(&search_results).unwrap();
        sender.send(AnswerEvent::SearchResults(search_results.clone()))?;

        let mut variables = vec![
            ("question".to_string(), interaction.query.clone()),
            ("context".to_string(), search_result_str.clone()),
        ];

        let hook_storage = self
            .read_side
            .get_hook_storage(self.read_api_key, self.collection_id)
            .await?;
        let lock = hook_storage.read().await;
        let (variables, system_prompt) = run_before_answer(
            &lock,
            (variables, system_prompt),
            log_sender,
            ExecOption {
                allowed_hosts: Some(vec![]),
                timeout: Duration::from_millis(500),
            },
        )
        .await?;
        drop(lock);

        info!("Variables for LLM: {:?}", variables);
        let answer_stream = llm_service
            .run_known_prompt_stream(
                llms::KnownPrompts::Answer,
                variables,
                system_prompt,
                Some(llm_config.clone()),
            )
            .await;
        let mut answer_stream = match answer_stream {
            Ok(s) => s,
            Err(e) => {
                sender.send(AnswerEvent::FailedToRunPrompt(e))?;
                return Ok(());
            }
        };

        while let Some(resp) = answer_stream.next().await {
            match resp {
                Ok(chunk) => {
                    sender.send(AnswerEvent::AnswerResponse(chunk))?;
                }
                Err(e) => {
                    sender.send(AnswerEvent::FailedToFetchAnswer(e))?;
                }
            }
        }

        let mut related_queries_params =
            llm_service.get_related_questions_params(interaction.related);

        if related_queries_params.is_empty() {
            sender.send(AnswerEvent::AnswerResponse("".to_string()))?;
            return Ok(());
        }

        related_queries_params.push(("context".to_string(), search_result_str));
        related_queries_params.push(("query".to_string(), interaction.query.clone()));

        self.handle_related_queries(&llm_service, llm_config, related_queries_params, &sender)
            .await?;

        sender.send(AnswerEvent::AnswerResponse("".to_string()))?;

        Ok(())
    }

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
                    .get_system_prompt(self.read_api_key, self.collection_id, id)
                    .await?;
                Ok(full_prompt)
            }
            None => {
                let has_system_prompts = self
                    .read_side
                    .has_system_prompts(self.read_api_key, self.collection_id)
                    .await?;

                if has_system_prompts {
                    let chosen_system_prompt = self
                        .read_side
                        .perform_system_prompt_selection(self.read_api_key, self.collection_id)
                        .await?;
                    Ok(chosen_system_prompt)
                } else {
                    Ok(None)
                }
            }
        }
    }

    async fn handle_related_queries(
        &self,
        llm_service: &llms::LLMService,
        llm_config: InteractionLLMConfig,
        related_queries_params: Vec<(String, String)>,
        sender: &tokio::sync::mpsc::UnboundedSender<AnswerEvent>,
    ) -> Result<(), AnswerError> {
        let related_questions_stream = llm_service
            .run_known_prompt_stream(
                llms::KnownPrompts::GenerateRelatedQueries,
                related_queries_params,
                None,
                Some(llm_config),
            )
            .await;

        let mut related_questions_stream = match related_questions_stream {
            Ok(s) => s,
            Err(e) => {
                sender.send(AnswerEvent::FailedToRunRelatedQuestion(e))?;
                return Ok(());
            }
        };

        while let Some(resp) = related_questions_stream.next().await {
            match resp {
                Ok(chunk) => {
                    sender.send(AnswerEvent::RelatedQueries(chunk))?;
                }
                Err(e) => {
                    sender.send(AnswerEvent::FailedToFetchRelatedQuestion(e))?;
                    break;
                }
            }
        }

        Ok(())
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
}

#[derive(Debug)]
pub enum AnswerEvent {
    Acknowledged,
    SelectedLLM(InteractionLLMConfig),
    ResultAction { action: String, result: String },
    FailedToRunRelatedQuestion(anyhow::Error),
    RelatedQueries(String),
    FailedToFetchRelatedQuestion(anyhow::Error),
    OptimizeingQuery(String),
    SearchResults(Vec<SearchResultHit>),
    FailedToRunPrompt(anyhow::Error),
    AnswerResponse(String),
    FailedToFetchAnswer(anyhow::Error),
}
