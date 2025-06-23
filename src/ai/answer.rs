use anyhow::Context;
use futures::Stream;
use std::{collections::HashMap, sync::Arc};
use thiserror::Error;
use tokio::sync::mpsc::error::SendError;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};
use tracing::{info, warn};

use crate::{
    ai::{llms, party_planner::PartyPlanner},
    collection_manager::sides::{
        read::{ReadError, ReadSide},
        segments::{Segment, SegmentError},
        system_prompts::SystemPrompt,
        triggers::Trigger,
    },
    types::{
        ApiKey, CollectionId, Interaction, InteractionLLMConfig, InteractionMessage, Limit,
        Properties, Role, SearchMode, SearchOffset, SearchParams, SearchResultHit, Similarity,
        VectorMode,
    },
};

#[derive(Debug, Error)]
pub enum AnswerError {
    #[error("Generic error: {0}")]
    Generic(#[from] anyhow::Error),
    #[error("read error: {0}")]
    ReadError(#[from] ReadError),
    #[error("segment error: {0}")]
    SegmentError(#[from] SegmentError),
    #[error("channel is closed: {0}")]
    ChannelClosed(#[from] SendError<AnswerEvent>),
}

pub struct Answer {
    read_side: Arc<ReadSide>,
    collection_id: CollectionId,
    read_api_key: ApiKey,
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

        // Always make sure that the conversation is not empty, or else the AI will not be able to
        // determine the segment and trigger.
        let segments_and_triggers_conversation = if interaction.messages.is_empty() {
            vec![InteractionMessage {
                role: Role::User,
                content: interaction.query.clone(),
            }]
        } else {
            interaction.messages.clone()
        };

        let mut segments_and_triggers_stream = select_triggers_and_segments(
            self.read_side.clone(),
            self.read_api_key,
            self.collection_id,
            Some(segments_and_triggers_conversation),
            Some(llm_config.clone()),
        )
        .await;

        let mut trigger: Option<Trigger> = None;
        let mut segment: Option<Segment> = None;

        while let Some(result) = segments_and_triggers_stream.next().await {
            match result {
                AudienceManagementResult::Segment(s) => {
                    segment = s.clone();
                    sender.send(AnswerEvent::GetSegment(s))?;
                }
                AudienceManagementResult::Trigger(t) => {
                    trigger = t.clone();
                    sender.send(AnswerEvent::GetTrigger(t))?;
                }
            }
        }

        let party_planner = PartyPlanner::new(self.read_side.clone(), Some(llm_config.clone()));

        let mut party_planner_stream = party_planner.run(
            self.read_side.clone(),
            self.collection_id,
            self.read_api_key,
            interaction.query.clone(),
            interaction.messages.clone(),
            segment.clone(),
            trigger.clone(),
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
    ) -> Result<(), AnswerError> {
        self.handle_gpu_overload(&mut interaction).await;

        let llm_config = self.get_llm_config(&interaction);

        sender.send(AnswerEvent::SelectedLLM(llm_config.clone()))?;
        sender.send(AnswerEvent::Acknowledged)?;

        let system_prompt = self
            .handle_system_prompt(interaction.system_prompt_id.clone())
            .await?;

        // Always make sure that the conversation is not empty, or else the AI will not be able to
        // determine the segment and trigger.
        let segments_and_triggers_conversation = if interaction.messages.is_empty() {
            vec![InteractionMessage {
                role: Role::User,
                content: interaction.query.clone(),
            }]
        } else {
            interaction.messages.clone()
        };

        let mut segments_and_triggers_stream = select_triggers_and_segments(
            self.read_side.clone(),
            self.read_api_key,
            self.collection_id,
            Some(segments_and_triggers_conversation),
            Some(llm_config.clone()),
        )
        .await;

        let mut trigger: Option<Trigger> = None;
        let mut segment: Option<Segment> = None;

        while let Some(result) = segments_and_triggers_stream.next().await {
            match result {
                AudienceManagementResult::Segment(s) => {
                    segment = s;
                }
                AudienceManagementResult::Trigger(t) => {
                    trigger = t;
                }
            }
        }

        sender.send(AnswerEvent::GetSegment(segment.clone()))?;

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

        sender.send(AnswerEvent::OptimizeingQuery(optimized_query.clone()))?;

        // Set the limit based on the interaction's max_documents with 5 by default
        let max_documents = Limit(interaction.max_documents.unwrap_or(5));
        let min_similarity = Similarity(interaction.min_similarity.unwrap_or(0.5));

        let search_results = self
            .read_side
            .search(
                self.read_api_key,
                self.collection_id,
                SearchParams {
                    mode: SearchMode::Vector(VectorMode {
                        term: interaction.query.clone(), // Optimized query IS NOT working well enough, defaults for current term for now
                        similarity: min_similarity,
                    }),
                    limit: max_documents,
                    offset: SearchOffset(0),
                    where_filter: Default::default(),
                    boost: HashMap::new(),
                    facets: HashMap::new(),
                    properties: Properties::Star,
                    indexes: None, // Search all indexes
                },
            )
            .await?;

        let search_result_str = serde_json::to_string(&search_results.hits).unwrap();
        sender.send(AnswerEvent::SearchResults(search_results.hits))?;

        let mut variables = vec![
            ("question".to_string(), interaction.query.clone()),
            ("context".to_string(), search_result_str.clone()),
        ];
        if let Some(full_segment) = segment {
            variables.push(("segment".to_string(), full_segment.to_string()));
        }
        if let Some(full_trigger) = trigger {
            variables.push(("trigger".to_string(), full_trigger.response));
        }

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
}

enum AudienceManagementResult {
    Segment(Option<crate::collection_manager::sides::segments::Segment>),
    Trigger(Option<crate::collection_manager::sides::triggers::Trigger>),
}

async fn select_triggers_and_segments(
    read_side: Arc<ReadSide>,
    read_api_key: ApiKey,
    collection_id: CollectionId,
    conversation: Option<Vec<InteractionMessage>>,
    mut llm_config: Option<InteractionLLMConfig>,
) -> impl Stream<Item = AudienceManagementResult> {
    let segment_interface = read_side
        .get_segments_manager(read_api_key, collection_id)
        .await
        .expect("Failed to get segments for the collection");
    let all_segments = segment_interface
        .list()
        .await
        .expect("Failed to get segments for the collection");

    if read_side.is_gpu_overloaded() {
        match read_side.select_random_remote_llm_service() {
            Some((provider, model)) => {
                info!("GPU is overloaded. Switching to \"{}\" as a remote LLM provider for this request.", provider);
                llm_config = Some(InteractionLLMConfig { model, provider });
            }
            None => {
                warn!("GPU is overloaded and no remote LLM is available. Using local LLM, but it's gonna be slow.");
            }
        }
    }

    let (tx, rx) = tokio::sync::mpsc::channel(100);

    // Move to another task to notify the FE without waiting all the LLM responses
    tokio::spawn(async move {
        if all_segments.is_empty() {
            tx.send(AudienceManagementResult::Segment(None))
                .await
                .unwrap();
            return;
        };

        let chosen_segment = segment_interface
            .perform_segment_selection(conversation.clone(), llm_config.clone())
            .await
            .expect("Failed to choose a segment.");

        let trigger_interface = read_side
            .get_triggers_manager(read_api_key, collection_id)
            .await
            .expect("Failed to get triggers manager");

        match chosen_segment {
            None => {
                tx.send(AudienceManagementResult::Segment(None))
                    .await
                    .unwrap();

                tx.send(AudienceManagementResult::Trigger(None))
                    .await
                    .unwrap();
            }
            Some(segment) => {
                let full_segment = segment_interface
                    .get(segment.clone().id.clone())
                    .await
                    .expect("Failed to get full segment");

                tx.send(AudienceManagementResult::Segment(full_segment.clone()))
                    .await
                    .unwrap();

                let all_segments_triggers = trigger_interface
                    .get_all_triggers_by_segment(full_segment.unwrap().id.clone())
                    .await
                    .expect("Failed to get triggers for the segment");

                if all_segments_triggers.is_empty() {
                    tx.send(AudienceManagementResult::Trigger(None))
                        .await
                        .unwrap();
                    return;
                }

                let chosen_trigger = trigger_interface
                    .perform_trigger_selection(
                        conversation,
                        all_segments_triggers,
                        llm_config.clone(),
                    )
                    .await
                    .context(
                        "Failed to choose a trigger for the given segment. Will fall back to None.",
                    )
                    .unwrap_or(None);

                match chosen_trigger {
                    None => {
                        tx.send(AudienceManagementResult::Trigger(None))
                            .await
                            .unwrap();
                    }
                    Some(chosen_trigger) => {
                        let full_trigger = trigger_interface
                            .get_trigger(chosen_trigger.id.clone())
                            .await
                            .expect("Failed to get full trigger");

                        tx.send(AudienceManagementResult::Trigger(full_trigger))
                            .await
                            .unwrap();
                    }
                }
            }
        }
    });

    ReceiverStream::new(rx)
}

pub enum AnswerEvent {
    Acknowledged,
    SelectedLLM(InteractionLLMConfig),
    GetSegment(Option<Segment>),
    GetTrigger(Option<Trigger>),
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
