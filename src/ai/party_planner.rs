use anyhow::Result;
use futures::Stream;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::collection_manager::{
    dto::{ApiKey, InteractionMessage, Role},
    sides::{segments::Segment, triggers::Trigger},
};

use super::vllm::{run_known_prompt, KnownPrompts};

#[derive(Serialize, Deserialize, Debug)]
pub struct Action {
    step: String,
    description: String,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(untagged)]
pub enum ActionPlanResponse {
    Wrapped { actions: Vec<Action> },
    Unwrapped(Vec<Action>),
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PartyPlannerMessage {
    pub action: String,
    pub result: String,
    pub done: bool,
}

pub struct PartyPlanner {}

impl PartyPlanner {
    pub fn run(
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
        let mut full_input = format!("### User Input\n{}", input);

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
            let action_plan = match Self::get_action_plan(full_input) {
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

            // Send the full action plan to the caller.
            tx.send(PartyPlannerMessage {
                action: "ACTION_PLAN".to_string(),
                result: serde_json::to_string(&action_plan)
                    .expect("Could not serialize action plan to a valid JSON string."),
                done: true,
            })
            .await
            .unwrap();

            println!("Action plan:");
            println!("{:?}", action_plan);

            // Loop over each action in the action plan and send it to the caller.
            for action in action_plan {
                dbg!(&action);
                tx.send(PartyPlannerMessage {
                    action: action.step,
                    result: action.description,
                    done: true,
                })
                .await
                .unwrap();
            }
        });

        ReceiverStream::new(rx)
    }

    fn get_action_plan(input: String) -> Result<Vec<Action>> {
        let action_plan = run_known_prompt(
            KnownPrompts::PartyPlanner,
            vec![("input".to_string(), input)],
        )?;

        let repaired = repair_json::repair(action_plan)?;
        let action_plan_deser: ActionPlanResponse = serde_json::from_str(&repaired)?;

        let plan = match action_plan_deser {
            ActionPlanResponse::Wrapped { actions } => actions,
            ActionPlanResponse::Unwrapped(actions) => actions,
        };

        Ok(plan)
    }
}
