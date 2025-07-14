use crate::{
    ai::llms::{KnownPrompts, LLMService},
    collection_manager::sides::{
        generic_kv::{format_key, KV},
        write::WriteError,
    },
    types::{CollectionId, InteractionLLMConfig, InteractionMessage},
};
use anyhow::{Context, Result};
use core::fmt;
use llm_json::{repair_json, JsonRepairError};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Segment {
    pub id: String,
    pub name: String,
    pub description: String,
    pub goal: Option<String>,
}

impl fmt::Display for Segment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut displayed = format!(
            "**name**: {}\n**description**: {}",
            self.name, self.description
        );

        if let Some(goal) = &self.goal {
            displayed.push_str(&format!("\n**goal**: {goal}"));
        }

        write!(f, "{displayed}")
    }
}

#[derive(Error, Debug)]
pub enum SegmentError {
    #[error("Generic error: {0}")]
    Generic(#[from] anyhow::Error),
    #[error("write error: {0}")]
    WriteError(#[from] WriteError),
    #[error("Cannot repair error: {0}")]
    RepairError(#[from] JsonRepairError),
    #[error("Deserialization error: {0}")]
    DeserializationError(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SelectedSegment {
    pub id: String,
    pub name: String,
    pub probability: f32,
}

#[derive(Clone)]
pub struct SegmentInterface {
    kv: Arc<KV>,
    llm_service: Arc<LLMService>,
}

impl SegmentInterface {
    pub fn new(kv: Arc<KV>, llm_service: Arc<LLMService>) -> Self {
        Self { kv, llm_service }
    }

    pub async fn insert(
        &self,
        collection_id: CollectionId,
        segment: Segment,
    ) -> Result<(), SegmentError> {
        let key = format_key(collection_id, &format!("segment:{}", segment.id));
        self.kv.insert(key, segment).await?;
        Ok(())
    }

    pub async fn get(
        &self,
        collection_id: CollectionId,
        segment_id: String,
    ) -> Result<Option<Segment>, SegmentError> {
        let segment_key = format!("segment:{segment_id}");
        let key = format_key(collection_id, &segment_key);

        match self.kv.get(&key).await {
            None => Ok(None),
            Some(Err(e)) => Err(e.into()),
            Some(Ok(segment)) => Ok(Some(segment)),
        }
    }

    pub async fn delete(
        &self,
        collection_id: CollectionId,
        segment_id: String,
    ) -> Result<Option<Segment>, SegmentError> {
        let segment_key = format!("segment:{segment_id}");
        let key = format_key(collection_id, &segment_key);

        match self.kv.remove_and_get(&key).await? {
            None => Ok(None),
            Some(Err(e)) => Err(e.into()),
            Some(Ok(segment)) => Ok(Some(segment)),
        }
    }

    pub async fn has_segments(&self, collection_id: CollectionId) -> Result<bool, SegmentError> {
        let segments = self.list_by_collection(collection_id).await?;

        Ok(!segments.is_empty())
    }

    pub async fn list_by_collection(
        &self,
        collection_id: CollectionId,
    ) -> Result<Vec<Segment>, SegmentError> {
        let prefix = format!("{}:segment:", collection_id.as_str());

        let segments = self.kv.prefix_scan(&prefix).await.context(format!(
            "Cannot scan segments for collection {}",
            collection_id.as_str()
        ))?;

        Ok(segments)
    }

    pub async fn perform_segment_selection(
        &self,
        collection_id: CollectionId,
        conversation: Option<Vec<InteractionMessage>>,
        llm_config: Option<InteractionLLMConfig>,
    ) -> Result<Option<SelectedSegment>, SegmentError> {
        let segments = self.list_by_collection(collection_id).await?;

        let response = self
            .llm_service
            .run_known_prompt(
                KnownPrompts::Segmenter,
                vec![
                    (
                        "personas".to_string(),
                        serde_json::to_string(&segments).unwrap(),
                    ),
                    (
                        "conversation".to_string(),
                        serde_json::to_string(&conversation).unwrap(),
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

        let deserialized = serde_json::from_str::<SelectedSegment>(&repaired)?;

        Ok(Some(deserialized))
    }
}

pub fn parse_segment_id_from_trigger(id: String) -> Option<String> {
    id.split(":")
        .find(|s| s.starts_with("s_"))
        .map(|s| s.to_string())
}

pub struct CollectionSegmentInterface {
    segments: SegmentInterface,
    collection_id: CollectionId,
}

impl CollectionSegmentInterface {
    pub fn new(segments: SegmentInterface, collection_id: CollectionId) -> Self {
        Self {
            segments,
            collection_id,
        }
    }

    pub async fn insert(&self, segment: Segment) -> Result<(), SegmentError> {
        self.segments.insert(self.collection_id, segment).await
    }

    pub async fn get(&self, segment_id: String) -> Result<Option<Segment>, SegmentError> {
        self.segments.get(self.collection_id, segment_id).await
    }

    pub async fn delete(&self, segment_id: String) -> Result<Option<Segment>, SegmentError> {
        self.segments.delete(self.collection_id, segment_id).await
    }

    pub async fn update(&self, segment: Segment) -> Result<(), SegmentError> {
        self.delete(segment.id.clone())
            .await
            .context("Cannot delete segment")?;
        self.insert(segment)
            .await
            .context("Cannot insert segment")?;

        Ok(())
    }

    pub async fn has_segments(&self) -> Result<bool, SegmentError> {
        self.segments.has_segments(self.collection_id).await
    }

    pub async fn list(&self) -> Result<Vec<Segment>, SegmentError> {
        self.segments.list_by_collection(self.collection_id).await
    }

    pub async fn perform_segment_selection(
        &self,
        conversation: Option<Vec<InteractionMessage>>,
        llm_config: Option<InteractionLLMConfig>,
    ) -> Result<Option<SelectedSegment>, SegmentError> {
        self.segments
            .perform_segment_selection(self.collection_id, conversation, llm_config)
            .await
    }
}
