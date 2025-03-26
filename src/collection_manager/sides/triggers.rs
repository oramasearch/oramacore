use super::generic_kv::{format_key, KV};
use crate::{
    ai::vllm::{self, VLLMService},
    collection_manager::dto::InteractionMessage,
    types::CollectionId,
};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{fmt, sync::Arc};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TriggerIdContent {
    pub collection_id: CollectionId,
    pub trigger_id: String,
    pub segment_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Trigger {
    pub id: String,
    pub name: String,
    pub description: String,
    pub response: String,
    pub segment_id: Option<String>,
}

impl fmt::Display for Trigger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let displayed = format!(
            "**name**: {}\n**description**: {}\n**response**: {}",
            self.name, self.description, self.response
        );

        write!(f, "{}", displayed)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SelectedTrigger {
    pub id: String,
    pub name: String,
    pub response: String,
    pub probability: f32,
}

pub struct TriggerInterface {
    kv: Arc<KV>,
    vllm_service: Arc<VLLMService>,
}

impl TriggerInterface {
    pub fn new(kv: Arc<KV>, vllm_service: Arc<VLLMService>) -> Self {
        Self { kv, vllm_service }
    }

    pub async fn insert(&self, trigger: Trigger) -> Result<String> {
        self.kv.insert(trigger.id.clone(), trigger.clone()).await?;
        Ok(trigger.id)
    }

    pub async fn get(
        &self,
        collection_id: CollectionId,
        trigger_id: String,
    ) -> Result<Option<Trigger>> {
        // This function has O(n) complexity. But it's fine since a single collection will have at most a dozen of triggers.
        // If that ever changes, we can add a secondary index.
        let triggers = self.list_by_collection(collection_id).await?;

        if triggers.is_empty() {
            return Ok(None);
        }

        let trigger_prefix = format!("{}:trigger:t_{}", collection_id.0, trigger_id);

        match triggers
            .into_iter()
            .find(|trigger| trigger.id.starts_with(&trigger_prefix))
        {
            Some(trigger) => Ok(Some(trigger)),
            None => Ok(None),
        }
    }

    pub async fn delete(
        &self,
        collection_id: CollectionId,
        trigger_id: String,
    ) -> Result<Option<Trigger>> {
        match self.get(collection_id.clone(), trigger_id.clone()).await? {
            Some(trigger) => {
                let trigger_id = trigger.id.clone();

                match self.kv.remove_and_get(&trigger_id).await? {
                    None => Ok(None),
                    Some(Err(e)) => Err(e),
                    Some(Ok(trigger)) => Ok(Some(trigger)),
                }
            }
            None => {
                return Err(anyhow::anyhow!(
                    "No trigger {} found for collection {}",
                    trigger_id,
                    collection_id.0
                ));
            }
        }
    }

    pub async fn has_triggers(&self, collection_id: CollectionId) -> Result<bool> {
        let triggers = self.list_by_collection(collection_id).await?;

        Ok(!triggers.is_empty())
    }

    pub async fn list_by_collection(&self, collection_id: CollectionId) -> Result<Vec<Trigger>> {
        let prefix = format!("{}:trigger:", collection_id.0.clone());

        let triggers: Vec<Trigger> = self.kv.prefix_scan(&prefix).await.context(format!(
            "Cannot scan triggers for collection {}",
            collection_id.0
        ))?;

        Ok(triggers)
    }

    pub async fn list_by_segment(
        &self,
        collection_id: CollectionId,
        segment_id: String,
    ) -> Result<Vec<Trigger>> {
        let all_triggers = self.list_by_collection(collection_id).await?;

        let triggers = all_triggers
            .into_iter()
            .filter(|trigger| trigger.segment_id == Some(segment_id.clone()))
            .collect();

        Ok(triggers)
    }

    pub async fn perform_trigger_selection(
        &self,
        _collection_id: CollectionId,
        conversation: Option<Vec<InteractionMessage>>,
        triggers: Vec<Trigger>,
    ) -> Result<Option<SelectedTrigger>> {
        let response = self
            .vllm_service
            .run_known_prompt(
                vllm::KnownPrompts::Trigger,
                vec![
                    ("triggers".to_string(), serde_json::to_string(&triggers)?),
                    (
                        "conversation".to_string(),
                        serde_json::to_string(&conversation)?,
                    ),
                ],
            )
            .await?;

        let repaired = repair_json::repair(response)?;

        // @todo: improve this.
        if repaired == "{}" {
            return Ok(None);
        }

        let deserialized = serde_json::from_str::<SelectedTrigger>(&repaired)?;

        Ok(Some(deserialized))
    }
}

pub fn get_trigger_key(
    collection_id: CollectionId,
    trigger_id: String,
    segment_id: Option<String>,
) -> String {
    match segment_id {
        Some(segment_id) => format_key(
            collection_id,
            &format!("trigger:t_{}:s_{}", trigger_id, segment_id),
        ),
        None => format_key(collection_id, &format!("trigger:t_{}", trigger_id)),
    }
}

pub fn parse_trigger_id(trigger_id: String) -> Option<TriggerIdContent> {
    let parts = trigger_id.split(':').collect::<Vec<&str>>();
    let collection_id = CollectionId::from(parts[0].to_string());

    let segment_id = parts
        .iter()
        .find(|part| part.starts_with("s_"))
        .map(|s| s.to_string());

    if let Some(trigger_id) = parts.iter().find(|part| part.starts_with("t_")) {
        return Some(TriggerIdContent {
            collection_id,
            trigger_id: trigger_id.replace("t_", ""),
            segment_id: segment_id.map(|s| s.replace("s_", "")),
        });
    };

    None
}
