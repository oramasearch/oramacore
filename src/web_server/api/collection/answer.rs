use crate::ai::llms;
use crate::ai::party_planner::PartyPlanner;
use crate::collection_manager::sides::segments::Segment;
use crate::collection_manager::sides::system_prompts::SystemPrompt;
use crate::collection_manager::sides::triggers::Trigger;
use crate::collection_manager::sides::ReadSide;
use crate::types::{
    ApiKey, AutoMode, Interaction, InteractionLLMConfig, InteractionMessage, Limit, Properties,
    Role, SearchMode, SearchParams,
};
use crate::types::{CollectionId, SearchOffset};
use crate::web_server::api::util::print_error;
use anyhow::Context;
use axum::extract::Query;
use axum::response::sse::Event;
use axum::response::Sse;
use axum::routing::post;
use axum::{
    extract::{Path, State},
    Json, Router,
};
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tracing::{info, warn};

#[derive(Serialize, Deserialize, Debug)]
struct MessageChunk {
    text: String,
    is_final: bool,
}

#[derive(Serialize, Debug)]
#[serde(tag = "type")]
enum SseMessage {
    #[serde(rename = "acknowledgement")]
    Acknowledge { message: String },
    #[serde(rename = "error")]
    Error { message: String },
    #[serde(rename = "response")]
    Response { message: String },
}

pub fn apis(read_side: Arc<ReadSide>) -> Router {
    Router::new()
        .route("/v1/collections/{id}/answer", post(answer_v1))
        .route(
            "/v1/collections/{id}/planned_answer",
            post(planned_answer_v1),
        )
        .with_state(read_side)
}

#[derive(Deserialize)]
struct AnswerQueryParams {
    #[serde(rename = "api-key")]
    api_key: ApiKey,
}

async fn planned_answer_v1(
    Path(id): Path<String>,
    read_side: State<Arc<ReadSide>>,
    Query(query_params): Query<AnswerQueryParams>,
    Json(mut interaction): Json<Interaction>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let collection_id = CollectionId::try_new(id).expect("Invalid collection ID");
    let read_side = read_side.clone();

    let query = interaction.query.clone();
    let conversation = interaction.messages.clone();
    let api_key = query_params.api_key;

    if read_side.is_gpu_overloaded() {
        match read_side.select_random_remote_llm_service() {
            Some((provider, model)) => {
                info!("GPU is overloaded. Switching to \"{}\" as a remote LLM provider for this request.", provider);
                interaction.llm_config = Some(InteractionLLMConfig { model, provider });
            }
            None => {
                warn!("GPU is overloaded and no remote LLM is available. Using local LLM, but it's gonna be slow.");
            }
        }
    }

    read_side
        .clone()
        .check_read_api_key(collection_id, api_key)
        .await
        .expect("Invalid API key");

    let (tx, rx) = mpsc::channel(100);
    let rx_stream = ReceiverStream::new(rx);

    tokio::spawn(async move {
        let llm_service = read_side.clone().get_llm_service();

        let _ = tx
            .send(Ok(Event::default().data(
                serde_json::to_string(&SseMessage::Acknowledge {
                    message: "Acknowledged".to_string(),
                })
                .unwrap(),
            )))
            .await;

        let llm_config = interaction.llm_config.clone().unwrap_or_else(|| {
            let (provider, model) = read_side.get_default_llm_config();

            InteractionLLMConfig { model, provider }
        });

        let _ = tx
            .send(Ok(Event::default().data(
                serialize_response(
                    "SELECTED_LLM",
                    &serde_json::to_string(&llm_config).unwrap(),
                    true,
                )
                .unwrap(),
            )))
            .await;

        let mut trigger: Option<Trigger> = None;
        let mut segment: Option<Segment> = None;
        let mut system_prompt: Option<SystemPrompt> = None;

        // Check if we have to select a random system prompt or a specific one based on the `system_prompt_id` param.
        match interaction.system_prompt_id {
            Some(id) => {
                let full_prompt = read_side
                    .get_system_prompt(api_key, collection_id, id)
                    .await
                    .context("Failed to get full system prompt")
                    .unwrap();

                system_prompt = full_prompt;
            }
            None => {
                let has_system_prompts = read_side
                    .has_system_prompts(api_key, collection_id)
                    .await
                    .context("Failed to check if the collection has system prompts")
                    .unwrap();

                if has_system_prompts {
                    let chosen_system_prompt = read_side
                        .perform_system_prompt_selection(api_key, collection_id)
                        .await
                        .context("Failed to choose a system prompt")
                        .unwrap();

                    system_prompt = chosen_system_prompt;
                }
            }
        }

        // Always make sure that the conversation is not empty, or else the AI will not be able to
        // determine the segment and trigger.
        let segments_and_triggers_conversation = match conversation.len() {
            0 => Some(vec![InteractionMessage {
                role: Role::User,
                content: query.clone(),
            }]),
            _ => Some(conversation.clone()),
        };

        let mut segments_and_triggers_stream = select_triggers_and_segments(
            read_side.clone(),
            api_key,
            collection_id,
            segments_and_triggers_conversation,
            interaction.llm_config.clone(),
        )
        .await;

        while let Some(result) = segments_and_triggers_stream.next().await {
            match result {
                AudienceManagementResult::Segment(s) => {
                    segment = s.clone();

                    let _ = tx
                        .send(Ok(Event::default().data(
                            serialize_response(
                                "GET_SEGMENT",
                                &serde_json::to_string(&s).unwrap(),
                                true,
                            )
                            .unwrap(),
                        )))
                        .await;
                }
                AudienceManagementResult::Trigger(t) => {
                    trigger = t.clone();

                    let _ = tx
                        .send(Ok(Event::default().data(
                            serialize_response(
                                "GET_TRIGGER",
                                &serde_json::to_string(&t).unwrap(),
                                true,
                            )
                            .unwrap(),
                        )))
                        .await;
                }
            }
        }

        let party_planner = PartyPlanner::new(read_side.clone(), interaction.llm_config.clone());

        let mut party_planner_stream = party_planner.run(
            read_side.clone(),
            collection_id,
            api_key,
            interaction.query.clone(),
            interaction.messages.clone(),
            segment.clone(),
            trigger.clone(),
            system_prompt,
        );

        while let Some(message) = party_planner_stream.next().await {
            let _ = tx
                .send(Ok(Event::default().data(
                    serde_json::to_string(&SseMessage::Response {
                        message: json!({
                            "action": message.action,
                            "result": message.result,
                        })
                        .to_string(),
                    })
                    .unwrap(),
                )))
                .await;
        }

        let mut related_queries_params =
            llm_service.get_related_questions_params(interaction.related);

        if !related_queries_params.is_empty() {
            related_queries_params.push(("context".to_string(), "{}".to_string())); // @todo: check if we can retrieve additional context
            related_queries_params.push(("query".to_string(), query.clone()));

            let mut related_questions_stream = llm_service
                .run_known_prompt_stream(
                    llms::KnownPrompts::GenerateRelatedQueries,
                    related_queries_params,
                    None,
                    interaction.llm_config,
                )
                .await;

            while let Some(resp) = related_questions_stream.next().await {
                match resp {
                    Ok(chunk) => {
                        tx.send(Ok(Event::default().data(
                            serialize_response("RELATED_QUERIES", &chunk, false).unwrap(),
                        )))
                        .await
                        .unwrap();
                    }
                    Err(e) => {
                        print_error(&e, "Error during streaming");
                        let _ = tx
                            .send(Ok(Event::default().data(
                                serde_json::to_string(&SseMessage::Error {
                                    message: format!("Error during streaming: {}", e),
                                })
                                .unwrap(),
                            )))
                            .await;
                        break;
                    }
                }
            }
        }
    });

    Sse::new(rx_stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("{ \"type\": \"keepalive\", \"message\": \"ok\" }"),
    )
}

// @todo: this function needs some cleaning. It works but it's not well structured.
async fn answer_v1(
    Path(id): Path<String>,
    read_side: State<Arc<ReadSide>>,
    Query(query): Query<AnswerQueryParams>,
    Json(mut interaction): Json<Interaction>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let collection_id = CollectionId::try_new(id).expect("Invalid collection ID");
    let read_side = read_side.clone();
    let read_api_key = query.api_key;

    if read_side.is_gpu_overloaded() {
        match read_side.select_random_remote_llm_service() {
            Some((provider, model)) => {
                info!("GPU is overloaded. Switching to \"{}\" as a remote LLM provider for this request.", provider);
                interaction.llm_config = Some(InteractionLLMConfig { model, provider });
            }
            None => {
                warn!("GPU is overloaded and no remote LLM is available. Using local LLM, but it's gonna be slow.");
            }
        }
    }

    read_side
        .clone()
        .check_read_api_key(collection_id, read_api_key)
        .await
        .expect("Invalid API key");

    // let context_evaluator = ContextEvaluator::try_new(read_side.get_ai_service())
    //     .context("Unable to instantiate the context evaluator")
    //     .unwrap();

    let query = interaction.query;
    let conversation = interaction.messages;

    let (tx, rx) = mpsc::channel(100);
    let rx_stream = ReceiverStream::new(rx);

    tokio::spawn(async move {
        let llm_service = read_side.clone().get_llm_service();

        let llm_config = interaction.llm_config.clone().unwrap_or_else(|| {
            let (provider, model) = read_side.get_default_llm_config();

            InteractionLLMConfig { model, provider }
        });

        let _ = tx
            .send(Ok(Event::default().data(
                serialize_response(
                    "SELECTED_LLM",
                    &serde_json::to_string(&llm_config).unwrap(),
                    true,
                )
                .unwrap(),
            )))
            .await;

        let _ = tx
            .send(Ok(Event::default().data(
                serde_json::to_string(&SseMessage::Acknowledge {
                    message: "Acknowledged".to_string(),
                })
                .unwrap(),
            )))
            .await;

        let mut trigger: Option<Trigger> = None;
        let mut segment: Option<Segment> = None;
        let mut system_prompt: Option<SystemPrompt> = None;

        // Check if we have to select a random system prompt or a specific one based on the `system_prompt_id` param.
        match interaction.system_prompt_id {
            Some(id) => {
                let full_prompt = read_side
                    .get_system_prompt(read_api_key, collection_id, id)
                    .await
                    .context("Failed to get full system prompt")
                    .unwrap();

                system_prompt = full_prompt;
            }
            None => {
                let has_system_prompts = read_side
                    .has_system_prompts(read_api_key, collection_id)
                    .await
                    .context("Failed to check if the collection has system prompts")
                    .unwrap();

                if has_system_prompts {
                    let chosen_system_prompt = read_side
                        .perform_system_prompt_selection(read_api_key, collection_id)
                        .await
                        .context("Failed to choose a system prompt")
                        .unwrap();

                    system_prompt = chosen_system_prompt;
                }
            }
        }

        // Always make sure that the conversation is not empty, or else the AI will not be able to
        // determine the segment and trigger.
        let segments_and_triggers_conversation = match conversation.len() {
            0 => Some(vec![InteractionMessage {
                role: Role::User,
                content: query.clone(),
            }]),
            _ => Some(conversation.clone()),
        };

        let mut segments_and_triggers_stream = select_triggers_and_segments(
            read_side.clone(),
            read_api_key,
            collection_id,
            segments_and_triggers_conversation,
            interaction.llm_config.clone(),
        )
        .await;

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

        let _ = tx
            .send(Ok(Event::default().data(
                serialize_response(
                    "GET_SEGMENT",
                    &serde_json::to_string(&segment).unwrap(),
                    true,
                )
                .unwrap(),
            )))
            .await;

        let optimized_query_variables = vec![("input".to_string(), query.clone())];

        let optimized_query = llm_service
            .run_known_prompt(
                llms::KnownPrompts::OptimizeQuery,
                optimized_query_variables,
                interaction.llm_config.clone(),
            )
            .await
            .unwrap_or(query.clone()); // fallback to the original query if the optimization fails

        let _ = tx
            .send(Ok(Event::default().data(
                serialize_response("OPTIMIZING_QUERY", &optimized_query, true).unwrap(),
            )))
            .await;

        // @todo: derive limit, boost, and where filters depending on the schema and the input query
        let search_results = read_side
            .search(
                read_api_key,
                collection_id,
                SearchParams {
                    mode: SearchMode::Auto(AutoMode {
                        term: optimized_query,
                    }),
                    limit: Limit(5),
                    offset: SearchOffset(0),
                    where_filter: HashMap::new(),
                    boost: HashMap::new(),
                    facets: HashMap::new(),
                    properties: Properties::Star,
                },
            )
            .await
            .unwrap();

        let _ = tx
            .send(Ok(Event::default().data(
                serialize_response(
                    "SEARCH_RESULTS",
                    &serde_json::to_string(&search_results.hits).unwrap(),
                    true,
                )
                .unwrap(),
            )))
            .await;

        let search_result_str = serde_json::to_string(&search_results.hits).unwrap();

        // Run the context evaluator to determine if the context is good enough to proceed with the answer.
        // We assume that vector search should pull highly relevant results, but sometimes it can happen that on very specific
        // queries the context is not good enough to provide a good answer.
        // match context_evaluator
        //     .evaluate(query.clone(), search_results)
        //     .await
        // {
        //     Ok(evaluation_result) => {
        //         // In case the context does not support the query, we'll ask for clarifications.
        //         // For now, the context must be at least 70% relevant to the query.
        //         if evaluation_result < 0.7 {
        //             let variables = vec![
        //             ("input".to_string(), query.clone()),
        //             ("description".to_string(), format!("Ask clarifications about {} in relationship to the following retrieved context: \n\n{}", query.clone(), search_result_str.clone())),
        //         ];

        //             let mut answer_stream = vllm_service
        //                 .run_known_prompt_stream(vllm::KnownPrompts::Followup, variables, None)
        //                 .await;

        //             while let Some(resp) = answer_stream.next().await {
        //                 match resp {
        //                     Ok(chunk) => {
        //                         tx.send(Ok(Event::default().data(
        //                             serialize_response("ANSWER_RESPONSE", &chunk, false).unwrap(),
        //                         )))
        //                         .await
        //                         .unwrap();
        //                     }
        //                     Err(e) => {
        //                         let _ = tx
        //                             .send(Ok(Event::default().data(
        //                                 serde_json::to_string(&SseMessage::Error {
        //                                     message: format!("Error during streaming: {}", e),
        //                                 })
        //                                 .unwrap(),
        //                             )))
        //                             .await;
        //                         break;
        //                     }
        //                 }
        //             }

        //             return;
        //         }
        //     }

        //     // If for any reason the context evaluation fails, we'll exit early to avoid hallucinations.
        //     Err(err) => {
        //         let _ = tx
        //             .send(Ok(Event::default().data(
        //                 serde_json::to_string(&SseMessage::Error {
        //                     message: format!("Error during context evaluation: {}", err),
        //                 })
        //                 .unwrap(),
        //             )))
        //             .await;

        //         return;
        //     }
        // }

        let mut variables = vec![
            ("question".to_string(), query.clone()),
            ("context".to_string(), search_result_str.clone()),
        ];

        if let Some(full_segment) = segment {
            variables.push(("segment".to_string(), full_segment.to_string()));
        }

        if let Some(full_trigger) = trigger {
            variables.push(("trigger".to_string(), full_trigger.response));
        }

        let mut answer_stream = llm_service
            .run_known_prompt_stream(
                llms::KnownPrompts::Answer,
                variables,
                system_prompt,
                interaction.llm_config.clone(),
            )
            .await;

        while let Some(resp) = answer_stream.next().await {
            match resp {
                Ok(chunk) => {
                    tx.send(Ok(Event::default().data(
                        serialize_response("ANSWER_RESPONSE", &chunk, false).unwrap(),
                    )))
                    .await
                    .unwrap();
                }
                Err(e) => {
                    print_error(&e, "Error during streaming");
                    let _ = tx
                        .send(Ok(Event::default().data(
                            serde_json::to_string(&SseMessage::Error {
                                message: format!("Error during streaming: {}", e),
                            })
                            .unwrap(),
                        )))
                        .await;
                    break;
                }
            }
        }

        let mut related_queries_params =
            llm_service.get_related_questions_params(interaction.related);

        if !related_queries_params.is_empty() {
            related_queries_params.push(("context".to_string(), search_result_str.clone()));
            related_queries_params.push(("query".to_string(), query.clone()));

            let mut related_questions_stream = llm_service
                .run_known_prompt_stream(
                    llms::KnownPrompts::GenerateRelatedQueries,
                    related_queries_params,
                    None,
                    interaction.llm_config,
                )
                .await;

            while let Some(resp) = related_questions_stream.next().await {
                match resp {
                    Ok(chunk) => {
                        tx.send(Ok(Event::default().data(
                            serialize_response("RELATED_QUERIES", &chunk, false).unwrap(),
                        )))
                        .await
                        .unwrap();
                    }
                    Err(e) => {
                        print_error(&e, "Error during streaming");
                        let _ = tx
                            .send(Ok(Event::default().data(
                                serde_json::to_string(&SseMessage::Error {
                                    message: format!("Error during streaming: {}", e),
                                })
                                .unwrap(),
                            )))
                            .await;
                        break;
                    }
                }
            }
        }

        let _ = tx
            .send(Ok(Event::default().data(
                serialize_response("ANSWER_RESPONSE", "", true).unwrap(),
            )))
            .await;
    });

    Sse::new(rx_stream).keep_alive(
        axum::response::sse::KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("{ \"type\": \"keepalive\", \"message\": \"ok\" }"),
    )
}

enum AudienceManagementResult {
    Segment(Option<crate::collection_manager::sides::segments::Segment>),
    Trigger(Option<crate::collection_manager::sides::triggers::Trigger>),
}

async fn select_triggers_and_segments(
    read_side: State<Arc<ReadSide>>,
    read_api_key: ApiKey,
    collection_id: CollectionId,
    conversation: Option<Vec<InteractionMessage>>,
    mut llm_config: Option<InteractionLLMConfig>,
) -> impl Stream<Item = AudienceManagementResult> {
    let all_segments = read_side
        .get_all_segments_by_collection(read_api_key, collection_id)
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

    let (tx, rx) = mpsc::channel(100);

    tokio::spawn(async move {
        if all_segments.is_empty() {
            tx.send(AudienceManagementResult::Segment(None))
                .await
                .unwrap();
            return;
        };

        let chosen_segment = read_side
            .perform_segment_selection(
                read_api_key,
                collection_id,
                conversation.clone(),
                llm_config.clone(),
            )
            .await
            .expect("Failed to choose a segment.");

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
                let full_segment = read_side
                    .get_segment(read_api_key, collection_id, segment.clone().id.clone())
                    .await
                    .expect("Failed to get full segment");

                tx.send(AudienceManagementResult::Segment(full_segment.clone()))
                    .await
                    .unwrap();

                let all_segments_triggers = read_side
                    .get_all_triggers_by_segment(
                        read_api_key,
                        collection_id,
                        full_segment.unwrap().id.clone(),
                    )
                    .await
                    .expect("Failed to get triggers for the segment");

                if all_segments_triggers.is_empty() {
                    tx.send(AudienceManagementResult::Trigger(None))
                        .await
                        .unwrap();
                    return;
                }

                let chosen_trigger = read_side
                    .perform_trigger_selection(
                        read_api_key,
                        collection_id,
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
                        let full_trigger = read_side
                            .get_trigger(read_api_key, collection_id, chosen_trigger.id.clone())
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

fn serialize_response(action: &str, result: &str, done: bool) -> serde_json::Result<String> {
    serde_json::to_string(&SseMessage::Response {
        message: json!({
            "action": action,
            "result": result,
            "done": done,
        })
        .to_string(),
    })
}
