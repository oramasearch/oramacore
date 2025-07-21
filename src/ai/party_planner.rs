use anyhow::{Context, Result};
use async_openai::config::OpenAIConfig;
use futures::{Stream, StreamExt};
use llm_json::repair_json;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::HashMap, sync::Arc};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::{
    collection_manager::sides::{
        read::{AnalyticSearchEventInvocationType, ReadSide},
        system_prompts::SystemPrompt,
    },
    types::{
        ApiKey, AutoMode, CollectionId, InteractionLLMConfig, InteractionMessage, Limit,
        Properties, Role, SearchMode, SearchOffset, SearchParams, SearchResult,
    },
};

use super::{
    llms::{KnownPrompts, LLMService},
    RemoteLLMProvider,
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Action {
    step: String,
    description: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Step {
    pub name: String,
    pub description: String,
    pub returns_json: bool,
    pub should_stream: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum ActionPlanResponse {
    Wrapped { actions: Vec<Action> },
    Unwrapped(Vec<Action>),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PartyPlannerMessage {
    pub action: String,
    pub result: String,
    pub done: bool,
}

pub struct PartyPlanner {
    // chosen_model: String,
    // llm_client: async_openai::Client<OpenAIConfig>,
    llm_config: Option<InteractionLLMConfig>,
}

impl PartyPlanner {
    pub fn new(_read_side: Arc<ReadSide>, llm_config: Option<InteractionLLMConfig>) -> Self {
        // let llm_service = read_side.get_llm_service();
        // Let the user choose a remote LLM model / client if they want to.

        /*
        let chosen_model = get_chosen_model(llm_config.clone(), llm_service.model.clone());
        let llm_client = get_chosen_llm_client(
            llm_config.clone(),
            llm_service.remote_clients.clone(),
            llm_service.local_vllm_client.clone(),
        );
        */

        Self {
            // chosen_model,
            // llm_client,
            llm_config,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        read_side: Arc<ReadSide>,
        collection_id: CollectionId,
        api_key: ApiKey,
        input: String,
        mut history: Vec<InteractionMessage>,
        custom_system_prompt: Option<SystemPrompt>,
    ) -> impl Stream<Item = PartyPlannerMessage> {
        let llm_service = read_side.get_llm_service();
        let llm_config = self.llm_config.clone();

        // Add a system prompt to the history if the first entry is not a system prompt.
        let mut system_prompt = InteractionMessage {
            role: Role::System,
            content: include_str!("../prompts/v1/party_planner/system_short.md").to_string(),
        };

        // If there's a custom system prompt, append it to the default system prompt.
        if let Some(custom_system_prompt) = custom_system_prompt {
            system_prompt.content += &format!(
                "\n\n### Important Additional Information - FOLLOW STRICTLY\n\n{}",
                custom_system_prompt.prompt
            );
        }

        if history.is_empty()
            || history
                .first()
                .map(|msg| msg.role != Role::System)
                .unwrap_or(true)
        {
            history.insert(0, system_prompt);
        }

        // Create the full user input. If possible, add trigger and segment information.
        let full_input = format!("### User Input\n{}", input.clone());

        history.push(InteractionMessage {
            role: Role::User,
            content: full_input.clone(),
        });

        // Now it's time to create a channel for the AI service to send messages to the caller.
        let (tx, rx) = mpsc::channel(100);

        tokio::spawn(async move {
            // Get the action plan from the AI service.
            // If there is an error, send an error message to the caller and exit early.
            let action_plan =
                match Self::get_action_plan(llm_service.clone(), full_input, llm_config.clone())
                    .await
                {
                    Ok(plan) => plan,
                    Err(e) => {
                        tx.send(PartyPlannerMessage {
                            action: "PARTY_PLANNER_ERROR".to_string(),
                            result: format!("{e:?}"),
                            done: true,
                        })
                        .await
                        .unwrap();
                        return;
                    }
                };

            // Set the action plan as followup prompt in the history
            history.push(InteractionMessage {
                role: Role::Assistant,
                content: Self::format_action_plan_assistant(action_plan.clone()),
            });

            // Make sure that the last step is either "GIVE_REPLY" or "ASK_FOLLOWUP".
            // If that's not the case, append them to the action plan.
            // for now, we'll always append "GIVE_REPLY" as the last step.
            if action_plan
                .last()
                .map(|action| action.step != "GIVE_REPLY")
                .unwrap_or(true)
            {
                todo!("Append 'GIVE_REPLY' to the action plan."); // @todo: implement this.
            }

            // Send the full action plan to the caller.
            tx.send(PartyPlannerMessage {
                action: "ACTION_PLAN".to_string(),
                result: serde_json::to_string(&action_plan)
                    .expect("Could not serialize action plan to a valid JSON string."),
                done: true,
            })
            .await
            .unwrap();

            // Loop over each action in the action plan and send it to the caller.
            for action in action_plan {
                // Push each action as a user message to the history.
                history.push(InteractionMessage {
                    role: Role::User,
                    content: action.description.clone(),
                });

                let step = Self::create_step(action.clone());

                // From now on, different steps could be handled at different places.
                // In this case, we may need to perform search inside the search index as part of the RAG pipeline.
                if step.name == "PERFORM_ORAMA_SEARCH" {
                    let result = Self::handle_orama_search(
                        read_side.clone(),
                        input.clone(),
                        collection_id,
                        api_key,
                    )
                    .await
                    .context("Unable to perform Orama Search as part of the RAG pipeline.")
                    .unwrap();

                    // Send the Orama search results as part of the "sources" for the RAG pipeline.
                    tx.send(PartyPlannerMessage {
                        action: "PERFORM_ORAMA_SEARCH".to_string(),
                        result: serde_json::to_string(&result).expect(
                            "Could not serialize Orama search results to a valid JSON string.",
                        ),
                        done: true,
                    })
                    .await
                    .unwrap();

                // For now, Orama-specific steps do not stream tokens. But there are other steps that may not stream them,
                // so we need to handle them here.
                } else if !step.should_stream {
                    let result = llm_service
                        .run_party_planner_prompt(
                            step.clone(),
                            &input,
                            &history,
                            llm_config.clone(),
                        )
                        .await
                        .context(format!(
                            "Unable to run party planner prompt for step: {}",
                            step.name
                        ))
                        .unwrap();

                    let value = match step.returns_json {
                        true => repair_json(&result, &Default::default())
                            .context(format!(
                                "Unable to repair JSON for step: {}.\nOriginal value:\n{}",
                                step.name, result
                            ))
                            .unwrap(),
                        false => result,
                    };

                    let _ = tx
                        .send(PartyPlannerMessage {
                            action: step.name.clone(),
                            result: value.clone(),
                            done: true,
                        })
                        .await;

                    history.push(InteractionMessage {
                        role: Role::Assistant,
                        content: value,
                    });
                }
                // Just like we did with non-streaming steps, we need to handle streaming steps here.
                else if step.clone().should_stream {
                    let mut acc = String::new();
                    let mut stream = llm_service
                        .run_party_planner_prompt_stream(
                            step.clone(),
                            &input,
                            &history,
                            llm_config.clone(),
                        )
                        .await
                        .unwrap();

                    while let Some(msg) = stream.next().await {
                        match msg {
                            Ok(m) => {
                                acc += &m;
                                tx.send(PartyPlannerMessage {
                                    action: step.name.clone(),
                                    result: m,
                                    done: false,
                                })
                                .await
                                .unwrap();
                            }
                            Err(e) => {
                                tx.send(PartyPlannerMessage {
                                    action: step.name.clone(),
                                    result: format!("{e:?}"),
                                    done: true,
                                })
                                .await
                                .unwrap();
                                break;
                            }
                        }
                    }

                    // Send the last message specifying `done: true`.
                    // This will be used on the front-end to determine if the step is done.
                    tx.send(PartyPlannerMessage {
                        action: step.name.clone(),
                        result: acc.clone(),
                        done: true,
                    })
                    .await
                    .unwrap();

                    // Push the accumulated messages to the history. This will be used
                    // as an additional context for subsequent steps.
                    history.push(InteractionMessage {
                        role: Role::Assistant,
                        content: acc,
                    });
                }
            }
        });

        ReceiverStream::new(rx)
    }

    async fn get_action_plan(
        llm_service: Arc<LLMService>,
        input: String,
        llm_config: Option<InteractionLLMConfig>,
    ) -> Result<Vec<Action>> {
        let action_plan = llm_service
            .run_known_prompt(
                KnownPrompts::PartyPlanner,
                vec![("input".to_string(), input)],
                llm_config,
            )
            .await?;

        let repaired = repair_json(&action_plan, &Default::default())?;
        let action_plan_deser: ActionPlanResponse = serde_json::from_str(&repaired)?;

        let plan = match action_plan_deser {
            ActionPlanResponse::Wrapped { actions } => actions,
            ActionPlanResponse::Unwrapped(actions) => actions,
        };

        Ok(plan)
    }

    fn create_step(action: Action) -> Step {
        let returns_json = !matches!(
            action.step.as_str(),
            "ASK_FOLLOWUP" | "IMPROVE_INPUT" | "GIVE_REPLY"
        );
        let should_stream = !matches!(
            action.step.as_str(),
            "OPTIMIZE_QUERY" | "GENERATE_QUERIES" | "PERFORM_ORAMA_SEARCH"
        );

        Step {
            name: action.step.clone(),
            description: action.description.clone(),
            returns_json,
            should_stream,
        }
    }

    fn format_action_plan_assistant(plan: Vec<Action>) -> String {
        let plan_as_value = serde_json::to_value(plan).unwrap();
        let as_md = json_to_md(&plan_as_value, 0);

        format!(
            "Alright! Here's the action plan I've come up with based on your request:\n\n{as_md}\n\nAsk me to proceed with the next step, one step at a time when you're ready."
        )
        .to_string()
    }

    async fn handle_orama_search(
        read_side: Arc<ReadSide>,
        input: String,
        collection_id: CollectionId,
        api_key: ApiKey,
    ) -> Result<SearchResult> {
        let results = read_side
            .search(
                api_key,
                collection_id,
                SearchParams {
                    mode: SearchMode::Auto(AutoMode { term: input }),
                    limit: Limit(5),
                    offset: SearchOffset(0),
                    boost: HashMap::new(),
                    facets: HashMap::new(),
                    properties: Properties::Star,
                    where_filter: Default::default(),
                    indexes: None, // Search all indexes.
                    sort_by: None,
                    user_id: None, // @todo: handle user_id if needed
                },
                AnalyticSearchEventInvocationType::PartyPlanner,
            )
            .await?;

        Ok(results)
    }
}

#[allow(dead_code)]
fn get_chosen_model(llm_config: Option<InteractionLLMConfig>, default: String) -> String {
    if let Some(config) = llm_config {
        return config.model;
    }

    default
}

#[allow(dead_code)]
fn get_chosen_llm_client(
    config: Option<InteractionLLMConfig>,
    remote_clients: Option<HashMap<RemoteLLMProvider, async_openai::Client<OpenAIConfig>>>,
    default: async_openai::Client<OpenAIConfig>,
) -> async_openai::Client<OpenAIConfig> {
    if let Some(config) = config {
        if let Some(remote_clients) = remote_clients {
            if let Some(client) = remote_clients.get(&config.provider) {
                return client.clone();
            }
        }
    }

    default
}

fn json_to_md(data: &Value, level: usize) -> String {
    let indent = "  ".repeat(level);
    let mut md = String::new();

    match data {
        Value::String(s) => {
            if let Ok(parsed) = serde_json::from_str::<Value>(s) {
                return json_to_md(&parsed, level);
            } else {
                return format!("{indent}`{s}`\n");
            }
        }
        Value::Array(arr) => {
            if !arr.is_empty() && arr[0].is_array() {
                return json_to_md(&arr[0], level);
            }

            for item in arr {
                if item.is_object() || item.is_array() {
                    md.push_str(&json_to_md(item, level));
                } else {
                    md.push_str(&format!("{indent}- `{item}`\n"));
                }
            }
        }
        Value::Object(map) => {
            for (key, value) in map {
                md.push_str(&format!("{indent}- **{key}**: "));
                if value.is_object() || value.is_array() {
                    md.push('\n');
                    md.push_str(&json_to_md(value, level + 1));
                } else {
                    md.push_str(&format!("`{value}`\n"));
                }
            }
        }
        _ => {
            md.push_str(&format!("{indent}`{data}`\n"));
        }
    }

    md
}
