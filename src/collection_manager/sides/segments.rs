use crate::{
    ai::AIService,
    collection_manager::sides::generic_kv::{format_key, KV},
    types::CollectionId,
};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Segment {
    pub id: String,
    pub name: String,
    pub description: String,
    pub goal: Option<String>,
}

pub struct SegmentInterface {
    kv: Arc<KV>,
    ai_service: Arc<AIService>,
}

impl SegmentInterface {
    pub fn new(kv: Arc<KV>, ai_service: Arc<AIService>) -> Self {
        Self { kv, ai_service }
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

    pub async fn list_by_collection(&self, collection_id: CollectionId) -> Result<Vec<Segment>> {
        let prefix = format_key(collection_id.clone(), "segment:");

        let segments = self
            .kv
            .prefix_scan(&prefix)
            .await
            .context("Cannot scan segments")?;

        if segments.is_empty() {
            Err(anyhow::anyhow!(format!(
                "No segments found for collection {}",
                collection_id.0
            )))
        } else {
            Ok(segments)
        }
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn insert_and_retrieve_segments() {
//         let kv = Arc::new(RwLock::new(KV::new()));
//         let segment_interface = SegmentInterface::new(kv.clone());

//         let collection_id = CollectionId("test_collection".to_string());

//         let segment = Segment {
//             id: "test_segment".to_string(),
//             name: "Test Segment".to_string(),
//             description: "This is a test segment".to_string(),
//             goal: Some("Test goal".to_string()),
//         };

//         segment_interface
//             .insert(collection_id.clone(), segment.clone())
//             .unwrap();

//         let retrieved_segment = segment_interface
//             .get(collection_id.clone(), segment.id.clone())
//             .unwrap()
//             .unwrap();

//         assert_eq!(segment, retrieved_segment);
//     }

//     #[test]
//     fn insert_and_delete_segments() {
//         let kv = Arc::new(RwLock::new(KV::new()));
//         let segment_interface = SegmentInterface::new(kv.clone());

//         let collection_id = CollectionId("test_collection".to_string());

//         let segment = Segment {
//             id: "test_segment".to_string(),
//             name: "Test Segment".to_string(),
//             description: "This is a test segment".to_string(),
//             goal: Some("Test goal".to_string()),
//         };

//         segment_interface
//             .insert(collection_id.clone(), segment.clone())
//             .unwrap();

//         let deleted_segment = segment_interface
//             .delete(collection_id.clone(), segment.id.clone())
//             .unwrap()
//             .unwrap();

//         let after_delete_result = segment_interface
//             .get(collection_id.clone(), segment.id.clone())
//             .unwrap();

//         assert_eq!(segment, deleted_segment);
//         assert!(after_delete_result.is_none());
//     }

//     #[test]
//     fn insert_and_list() {
//         let kv = Arc::new(RwLock::new(KV::new()));
//         let segment_interface = SegmentInterface::new(kv.clone());

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
//             .unwrap();
//         segment_interface
//             .insert(collection_id.clone(), segment2.clone())
//             .unwrap();

//         let segments = segment_interface
//             .list_by_collection(collection_id.clone())
//             .unwrap();

//         assert_eq!(segments.len(), 2);
//         assert!(segments.contains(&segment1));
//         assert!(segments.contains(&segment2));
//     }
// }
