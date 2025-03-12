use anyhow::{Context, Result};
use axum::extract::State;
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::HashMap, sync::Arc};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::{
    collection_manager::{
        dto::{
            ApiKey, AutoMode, InteractionMessage, Limit, Role, SearchMode, SearchParams,
            SearchResult,
        },
        sides::{segments::Segment, triggers::Trigger, ReadSide},
    },
    types::CollectionId,
};

use super::vllm::{run_known_prompt, run_party_planner_prompt, KnownPrompts};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Action {
    step: String,
    description: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Step {
    name: String,
    description: String,
    returns_json: bool,
    should_stream: bool,
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

pub struct PartyPlanner {}

impl PartyPlanner {
    pub fn run(
        read_side: State<Arc<ReadSide>>,
        collection_id: String,
        api_key: ApiKey,
        input: String,
        mut history: Vec<InteractionMessage>,
        segment: Option<Segment>,
        trigger: Option<Trigger>,
    ) -> impl Stream<Item = PartyPlannerMessage> {
        // Add a system prompt to the history if the first entry is not a system prompt.
        let system_prompt = InteractionMessage {
            role: Role::System,
            content: include_str!("../prompts/v1/party_planner/system_short.txt").to_string(),
        };

        if history.is_empty()
            || history
                .first()
                .map(|msg| msg.role != Role::System)
                .unwrap_or(true)
        {
            history.insert(0, system_prompt);
        }

        // Create the full user input. If possible, add trigger and segment information.
        let mut full_input = format!("### User Input\n{}", input.clone());

        if let Some(segment) = segment {
            full_input.push_str(&format!("\n\n### Segment\n{}", segment));
        }

        if let Some(trigger) = trigger {
            full_input.push_str(&format!("\n\n### Trigger\n{}", trigger));
        }

        // Now it's time to create a channel for the AI service to send messages to the caller.
        let (tx, rx) = mpsc::channel(100);

        tokio::spawn(async move {
            // Get the action plan from the AI service.
            // If there is an error, send an error message to the caller and exit early.
            let action_plan = match Self::get_action_plan(full_input).await {
                Ok(plan) => plan,
                Err(e) => {
                    tx.send(PartyPlannerMessage {
                        action: "PARTY_PLANNER_ERROR".to_string(),
                        result: format!("{:?}", e),
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
                        CollectionId(collection_id.clone()),
                        api_key.clone(),
                    )
                    .await
                    .context("Unable to perform Orama Search as part of the RAG pipeline.")
                    .unwrap();

                    // Send the Orama search results as part of the "sources" for the RAG pipeline.
                    let _ = tx
                        .send(PartyPlannerMessage {
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
                    dbg!(step);
                    // let result = Self::execute_non_streaming_step(
                    //     action.clone(),
                    //     input.clone(),
                    //     step.clone(),
                    //     history.clone(),
                    // ).await;
                }
            }
        });

        ReceiverStream::new(rx)
    }

    async fn get_action_plan(input: String) -> Result<Vec<Action>> {
        let action_plan = run_known_prompt(
            KnownPrompts::PartyPlanner,
            vec![("input".to_string(), input)],
        )
        .await?;

        let repaired = repair_json::repair(action_plan)?;
        let action_plan_deser: ActionPlanResponse = serde_json::from_str(&repaired)?;

        let plan = match action_plan_deser {
            ActionPlanResponse::Wrapped { actions } => actions,
            ActionPlanResponse::Unwrapped(actions) => actions,
        };

        Ok(plan)
    }

    fn create_step(action: Action) -> Step {
        let returns_json = match action.step.as_str() {
            "ASK_FOLLOWUP" => false,
            "IMPROVE_INPUT" => false,
            "GIVE_REPLY" => false,
            _ => true,
        };

        let should_stream = match action.step.as_str() {
            "OPTIMIZE_QUERY" => false,
            "GENERATE_QUERIES" => false,
            "PERFORM_ORAMA_SEARCH" => false,
            _ => true,
        };

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

        return format!(
            r"Alright! Here's the action plan I've come up with based on your request:\n\n{}\n\nAsk me to proceed with the next step, one step at a time when you're ready.",
            as_md
        )
        .to_string();
    }

    async fn handle_orama_search(
        read_side: State<Arc<ReadSide>>,
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
                    boost: HashMap::new(),
                    facets: HashMap::new(),
                    properties: crate::collection_manager::dto::Properties::Star,
                    where_filter: HashMap::new(),
                },
            )
            .await?;

        Ok(results)
    }

    async fn execute_non_streaming_step(
        action: Action,
        input: String,
        step: Step,
        history: Vec<InteractionMessage>,
    ) -> String {
        // let step_result = run_party_planner_prompt(step.name, variables).await;
        todo!();
    }
}

fn json_to_md(data: &Value, level: usize) -> String {
    let indent = "  ".repeat(level);
    let mut md = String::new();

    match data {
        Value::String(s) => {
            if let Ok(parsed) = serde_json::from_str::<Value>(s) {
                return json_to_md(&parsed, level);
            } else {
                return format!("{}`{}`\n", indent, s);
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
                    md.push_str(&format!("{}- `{}`\n", indent, item));
                }
            }
        }
        Value::Object(map) => {
            for (key, value) in map {
                md.push_str(&format!("{}- **{}**: ", indent, key));
                if value.is_object() || value.is_array() {
                    md.push_str("\n");
                    md.push_str(&json_to_md(value, level + 1));
                } else {
                    md.push_str(&format!("`{}`\n", value));
                }
            }
        }
        _ => {
            md.push_str(&format!("{}`{}`\n", indent, data));
        }
    }

    md
}
