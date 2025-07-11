use super::generic_kv::{format_key, KV};
use crate::{
    ai::llms::{self, LLMService},
    collection_manager::sides::{
        read::{CollectionReadLock as ReadCollectionReadLock, ReadError},
        write::{CollectionReadLock as WriteCollectionReadLock, WriteError},
    },
    types::{CollectionId, InteractionLLMConfig, InteractionMessage},
};
use anyhow::{Context, Result};
use llm_json::{repair_json, JsonRepairError};
use serde::{Deserialize, Serialize};
use std::{fmt, sync::Arc};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TriggerError {
    #[error("Generic error: {0}")]
    Generic(#[from] anyhow::Error),
    #[error("Trigger {0} not found in collection {1}")]
    NotFound(CollectionId, String),
    #[error("write error: {0}")]
    WriteError(#[from] WriteError),
    #[error("read error: {0}")]
    ReadError(#[from] ReadError),
    #[error("Invalid trigger id: {0}")]
    InvalidTriggerId(String),
    #[error("Cannot repair error: {0}")]
    RepairError(#[from] JsonRepairError),
    #[error("Deserialization error: {0}")]
    DeserializationError(#[from] serde_json::Error),
}

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

        write!(f, "{displayed}")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SelectedTrigger {
    pub id: String,
    pub name: String,
    pub response: String,
    pub probability: f32,
}

#[derive(Clone)]
pub struct TriggerInterface {
    kv: Arc<KV>,
    llm_service: Arc<LLMService>,
}

impl TriggerInterface {
    pub fn new(kv: Arc<KV>, llm_service: Arc<LLMService>) -> Self {
        Self { kv, llm_service }
    }

    pub async fn insert(&self, trigger: Trigger) -> Result<String, TriggerError> {
        self.kv
            .insert(trigger.id.clone(), trigger.clone())
            .await
            .map_err(TriggerError::Generic)?;
        Ok(trigger.id)
    }

    pub async fn get(
        &self,
        collection_id: CollectionId,
        trigger_id: String,
    ) -> Result<Option<Trigger>, TriggerError> {
        // This function has O(n) complexity. But it's fine since a single collection will have at most a dozen of triggers.
        // If that ever changes, we can add a secondary index.
        let triggers = self.list_by_collection(collection_id).await?;

        if triggers.is_empty() {
            return Ok(None);
        }

        let trigger_prefix = format!("{}:trigger:t_{}", collection_id.as_str(), trigger_id);

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
    ) -> Result<Option<Trigger>, TriggerError> {
        match self.get(collection_id, trigger_id.clone()).await? {
            Some(trigger) => {
                let trigger_id = trigger.id.clone();

                match self
                    .kv
                    .remove_and_get(&trigger_id)
                    .await
                    .map_err(TriggerError::Generic)?
                {
                    None => Ok(None),
                    Some(Err(e)) => Err(e.into()),
                    Some(Ok(trigger)) => Ok(Some(trigger)),
                }
            }
            None => Err(TriggerError::NotFound(collection_id, trigger_id)),
        }
    }

    pub async fn has_triggers(&self, collection_id: CollectionId) -> Result<bool, TriggerError> {
        let triggers = self.list_by_collection(collection_id).await?;

        Ok(!triggers.is_empty())
    }

    pub async fn list_by_collection(
        &self,
        collection_id: CollectionId,
    ) -> Result<Vec<Trigger>, TriggerError> {
        let prefix = format!("{}:trigger:", collection_id.as_str());

        let triggers: Vec<Trigger> = self.kv.prefix_scan(&prefix).await.context(format!(
            "Cannot scan triggers for collection {}",
            collection_id.as_str()
        ))?;

        Ok(triggers)
    }

    pub async fn list_by_segment(
        &self,
        collection_id: CollectionId,
        segment_id: String,
    ) -> Result<Vec<Trigger>, TriggerError> {
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
        llm_config: Option<InteractionLLMConfig>,
    ) -> Result<Option<SelectedTrigger>, TriggerError> {
        let response = self
            .llm_service
            .run_known_prompt(
                llms::KnownPrompts::Trigger,
                vec![
                    ("triggers".to_string(), serde_json::to_string(&triggers)?),
                    (
                        "conversation".to_string(),
                        serde_json::to_string(&conversation)?,
                    ),
                ],
                llm_config,
            )
            .await?;

        let repaired = repair_json(&response, &Default::default())?;

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
            &format!("trigger:t_{trigger_id}:s_{segment_id}"),
        ),
        None => format_key(collection_id, &format!("trigger:t_{trigger_id}")),
    }
}

pub fn parse_trigger_id(trigger_id: String) -> Option<TriggerIdContent> {
    let parts = trigger_id.split(':').collect::<Vec<&str>>();
    let collection_id =
        CollectionId::try_new(parts[0]).expect("Invalid collection ID in trigger ID");

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

pub struct WriteCollectionTriggerInterface<'a> {
    triggers: TriggerInterface,
    collection: WriteCollectionReadLock<'a>,
}

impl<'a> WriteCollectionTriggerInterface<'a> {
    pub fn new(triggers: TriggerInterface, collection: WriteCollectionReadLock<'a>) -> Self {
        Self {
            triggers,
            collection,
        }
    }

    pub async fn insert_trigger(
        &self,
        trigger: Trigger,
        trigger_id: Option<String>,
    ) -> Result<Trigger, TriggerError> {
        let final_trigger_id = match trigger_id {
            Some(mut id) => {
                let required_prefix = format!("{}:trigger:", self.collection.id.as_str());

                if !id.starts_with(&required_prefix) {
                    id = get_trigger_key(self.collection.id, id, trigger.segment_id.clone());
                }

                id
            }
            None => {
                let cuid = cuid2::create_id();
                get_trigger_key(self.collection.id, cuid, trigger.segment_id.clone())
            }
        };

        let trigger = Trigger {
            id: final_trigger_id,
            name: trigger.name,
            description: trigger.description,
            response: trigger.response,
            segment_id: trigger.segment_id,
        };

        self.triggers
            .insert(trigger.clone())
            .await
            .context("Cannot insert trigger")?;

        Ok(trigger)
    }

    pub async fn get_trigger(&self, trigger_id: String) -> Result<Trigger, TriggerError> {
        let trigger = self
            .triggers
            .get(self.collection.id, trigger_id.clone())
            .await?;

        match trigger {
            Some(trigger) => Ok(trigger),
            None => Err(TriggerError::NotFound(self.collection.id, trigger_id)),
        }
    }

    pub async fn delete_trigger(
        &self,
        trigger_id: String,
    ) -> Result<Option<Trigger>, TriggerError> {
        self.triggers.delete(self.collection.id, trigger_id).await
    }

    pub async fn update_trigger(&self, trigger: Trigger) -> Result<Option<Trigger>, TriggerError> {
        let trigger_key = get_trigger_key(
            self.collection.id,
            trigger.id.clone(),
            trigger.segment_id.clone(),
        );

        let new_trigger = Trigger {
            id: trigger_key.clone(),
            ..trigger
        };

        self.insert_trigger(new_trigger, Some(trigger.id))
            .await
            .context("Cannot insert updated trigger")?;

        match parse_trigger_id(trigger_key.clone()) {
            Some(key_content) => {
                let updated_trigger = self
                    .triggers
                    .get(self.collection.id, key_content.trigger_id.clone())
                    .await
                    .context("Cannot get updated trigger")?;

                match updated_trigger {
                    Some(trigger) => Ok(Some(Trigger {
                        id: key_content.trigger_id,
                        ..trigger
                    })),
                    None => Err(TriggerError::Generic(anyhow::anyhow!(
                        "Cannot get updated trigger"
                    ))),
                }
            }
            None => Err(TriggerError::InvalidTriggerId(trigger_key)),
        }
    }
}

pub struct ReadCollectionTriggerInterface<'a> {
    triggers: TriggerInterface,
    collection: ReadCollectionReadLock<'a>,
}

impl<'a> ReadCollectionTriggerInterface<'a> {
    pub fn new(triggers: TriggerInterface, collection: ReadCollectionReadLock<'a>) -> Self {
        Self {
            triggers,
            collection,
        }
    }

    pub async fn perform_trigger_selection(
        &self,
        conversation: Option<Vec<InteractionMessage>>,
        triggers: Vec<Trigger>,
        llm_config: Option<InteractionLLMConfig>,
    ) -> Result<Option<SelectedTrigger>, TriggerError> {
        self.triggers
            .perform_trigger_selection(self.collection.id, conversation, triggers, llm_config)
            .await
    }

    pub async fn get_all_triggers_by_segment(
        &self,
        segment_id: String,
    ) -> Result<Vec<Trigger>, TriggerError> {
        self.triggers
            .list_by_segment(self.collection.id, segment_id)
            .await
    }

    pub async fn get_trigger(&self, trigger_id: String) -> Result<Option<Trigger>, TriggerError> {
        self.triggers.get(self.collection.id, trigger_id).await
    }

    pub async fn get_all_triggers_by_collection(&self) -> Result<Vec<Trigger>, TriggerError> {
        self.triggers.list_by_collection(self.collection.id).await
    }
}
