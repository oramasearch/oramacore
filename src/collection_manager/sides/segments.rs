use crate::{
    ai::llms::{KnownPrompts, LLMService},
    collection_manager::sides::generic_kv::{format_key, KV},
    types::{CollectionId, InteractionLLMConfig, InteractionMessage},
};
use anyhow::{Context, Result};
use core::fmt;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

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
            displayed.push_str(&format!("\n**goal**: {}", goal));
        }

        write!(f, "{}", displayed)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SelectedSegment {
    pub id: String,
    pub name: String,
    pub probability: f32,
}

pub struct SegmentInterface {
    kv: Arc<KV>,
    llm_service: Arc<LLMService>,
}

impl SegmentInterface {
    pub fn new(kv: Arc<KV>, llm_service: Arc<LLMService>) -> Self {
        Self { kv, llm_service }
    }

    pub async fn insert(&self, collection_id: CollectionId, segment: Segment) -> Result<()> {
        let key = format_key(collection_id, &format!("segment:{}", segment.id));
        self.kv.insert(key, segment).await?;
        Ok(())
    }

    pub async fn get(
        &self,
        collection_id: CollectionId,
        segment_id: String,
    ) -> Result<Option<Segment>> {
        let segment_key = format!("segment:{}", segment_id);
        let key = format_key(collection_id, &segment_key);

        match self.kv.get(&key).await {
            None => Ok(None),
            Some(Err(e)) => Err(e),
            Some(Ok(segment)) => Ok(Some(segment)),
        }
    }

    pub async fn delete(
        &self,
        collection_id: CollectionId,
        segment_id: String,
    ) -> Result<Option<Segment>> {
        let segment_key = format!("segment:{}", segment_id);
        let key = format_key(collection_id, &segment_key);

        match self.kv.remove_and_get(&key).await? {
            None => Ok(None),
            Some(Err(e)) => Err(e),
            Some(Ok(segment)) => Ok(Some(segment)),
        }
    }

    pub async fn has_segments(&self, collection_id: CollectionId) -> Result<bool> {
        let segments = self.list_by_collection(collection_id).await?;

        Ok(!segments.is_empty())
    }

    pub async fn list_by_collection(&self, collection_id: CollectionId) -> Result<Vec<Segment>> {
        let prefix = format!("{}:segment:", collection_id.0.clone());

        let segments = self.kv.prefix_scan(&prefix).await.context(format!(
            "Cannot scan segments for collection {}",
            collection_id.0
        ))?;

        Ok(segments)
    }

    pub async fn perform_segment_selection(
        &self,
        collection_id: CollectionId,
        conversation: Option<Vec<InteractionMessage>>,
        llm_config: Option<InteractionLLMConfig>,
    ) -> Result<Option<SelectedSegment>> {
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

        let repaired = repair_json::repair(response)?;

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

// #[cfg(test)]
// mod tests {
//     use std::path::PathBuf;

//     use http::uri::Scheme;

//     use crate::{ai::AIServiceConfig, collection_manager::sides::generic_kv::KVConfig};

//     use super::*;

//     #[tokio::test]
//     async fn insert_and_retrieve_segments() {
//         let kv_config = KVConfig {
//             data_dir: PathBuf::from("/tmp/segments"),
//             sender: None,
//         };

//         let ai_service_conf = AIServiceConfig {
//             scheme: Scheme::HTTP,
//             max_connections: 1,
//             port: 8080,
//             api_key: None,
//             host: std::net::IpAddr::V4(std::net::Ipv4Addr::new(127, 0, 0, 1)),
//         };

//         let kv = Arc::new(KV::try_load(kv_config).unwrap());
//         let segment_interface =
//             SegmentInterface::new(kv.clone(), Arc::new(AIService::new(ai_service_conf)));

//         let collection_id = CollectionId("test_collection".to_string());

//         let segment = Segment {
//             id: "test_segment".to_string(),
//             name: "Test Segment".to_string(),
//             description: "This is a test segment".to_string(),
//             goal: Some("Test goal".to_string()),
//         };

//         segment_interface
//             .insert(collection_id.clone(), segment.clone())
//             .await
//             .unwrap();

//         let retrieved_segment = segment_interface
//             .get(collection_id.clone(), segment.id.clone())
//             .await
//             .unwrap()
//             .unwrap();

//         assert_eq!(segment, retrieved_segment);
//     }

//     #[tokio::test]
//     async fn insert_and_delete_segments() {
//         let kv_config = KVConfig {
//             data_dir: PathBuf::from("/tmp/segments"),
//             sender: None,
//         };

//         let ai_service_conf = AIServiceConfig {
//             scheme: Scheme::HTTP,
//             max_connections: 1,
//             port: 8080,
//             api_key: None,
//             host: std::net::IpAddr::V4(std::net::Ipv4Addr::new(127, 0, 0, 1)),
//         };

//         let kv = Arc::new(KV::try_load(kv_config).unwrap());
//         let segment_interface =
//             SegmentInterface::new(kv.clone(), Arc::new(AIService::new(ai_service_conf)));

//         let collection_id = CollectionId("test_collection".to_string());

//         let segment = Segment {
//             id: "test_segment".to_string(),
//             name: "Test Segment".to_string(),
//             description: "This is a test segment".to_string(),
//             goal: Some("Test goal".to_string()),
//         };

//         segment_interface
//             .insert(collection_id.clone(), segment.clone())
//             .await
//             .unwrap();

//         let deleted_segment = segment_interface
//             .delete(collection_id.clone(), segment.id.clone())
//             .await
//             .unwrap()
//             .unwrap();

//         let after_delete_result = segment_interface
//             .get(collection_id.clone(), segment.id.clone())
//             .await
//             .unwrap();

//         assert_eq!(segment, deleted_segment);
//         assert!(after_delete_result.is_none());
//     }

//     #[tokio::test]
//     async fn insert_and_list() {
//         let kv_config = KVConfig {
//             data_dir: PathBuf::from("/tmp/segments"),
//             sender: None,
//         };

//         let ai_service_conf = AIServiceConfig {
//             scheme: Scheme::HTTP,
//             max_connections: 1,
//             port: 8080,
//             api_key: None,
//             host: std::net::IpAddr::V4(std::net::Ipv4Addr::new(127, 0, 0, 1)),
//         };

//         let kv = Arc::new(KV::try_load(kv_config).unwrap());
//         let segment_interface =
//             SegmentInterface::new(kv.clone(), Arc::new(AIService::new(ai_service_conf)));

//         let collection_id = CollectionId("test_collection".to_string());

//         let segment1 = Segment {
//             id: "test_segment1".to_string(),
//             name: "Test Segment 1".to_string(),
//             description: "This is a test segment 1".to_string(),
//             goal: Some("Test goal 1".to_string()),
//         };

//         let segment2 = Segment {
//             id: "test_segment2".to_string(),
//             name: "Test Segment 2".to_string(),
//             description: "This is a test segment 2".to_string(),
//             goal: Some("Test goal 2".to_string()),
//         };

//         segment_interface
//             .insert(collection_id.clone(), segment1.clone())
//             .await
//             .context("Cannot insert segment 1")
//             .unwrap();
//         segment_interface
//             .insert(collection_id.clone(), segment2.clone())
//             .await
//             .context("Cannot insert segment 2")
//             .unwrap();

//         let segments = segment_interface
//             .list_by_collection(collection_id.clone())
//             .await
//             .unwrap();

//         assert_eq!(segments.len(), 2);
//         assert!(segments.contains(&segment1));
//         assert!(segments.contains(&segment2));
//     }
// }
