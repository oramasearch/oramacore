use crate::{
    collection_manager::sides::generic_kv::{format_key, KV},
    types::CollectionId,
};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Segment {
    id: String,
    name: String,
    description: String,
    goal: Option<String>,
}

pub struct SegmentInterface {
    kv: Arc<RwLock<KV<Segment>>>,
}

impl SegmentInterface {
    pub fn new(kv: Arc<RwLock<KV<Segment>>>) -> Self {
        Self { kv }
    }

    pub fn insert(&self, collection_id: CollectionId, segment: Segment) -> Result<()> {
        let key = format_key(collection_id, &format!("segment:{}", segment.id));
        let mut kv = self.kv.write().unwrap();
        kv.insert(key, segment);
        Ok(())
    }

    pub fn get(&self, collection_id: CollectionId, segment_id: String) -> Result<Option<Segment>> {
        let segment_key = format!("segment:{}", segment_id);
        let key = format_key(collection_id, &segment_key);
        let kv = self.kv.read().unwrap();
        Ok(kv.get(&key).cloned())
    }

    pub fn delete(
        &self,
        collection_id: CollectionId,
        segment_id: String,
    ) -> Result<Option<Segment>> {
        let segment_key = format!("segment:{}", segment_id);
        let key = format_key(collection_id, &segment_key);
        let mut kv = self.kv.write().unwrap();
        Ok(kv.remove(&key))
    }

    pub fn list_by_collection(&self, collection_id: CollectionId) -> Result<Vec<Segment>> {
        let kv = self.kv.read().unwrap();
        let prefix = format_key(collection_id.clone(), "segment:");

        let segments: Vec<Segment> = kv.prefix_scan(&prefix).map(|(_, value)| value).collect();

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_retrieve_segments() {
        let kv = Arc::new(RwLock::new(KV::new()));
        let segment_interface = SegmentInterface::new(kv.clone());

        let collection_id = CollectionId("test_collection".to_string());

        let segment = Segment {
            id: "test_segment".to_string(),
            name: "Test Segment".to_string(),
            description: "This is a test segment".to_string(),
            goal: Some("Test goal".to_string()),
        };

        segment_interface
            .insert(collection_id.clone(), segment.clone())
            .unwrap();

        let retrieved_segment = segment_interface
            .get(collection_id.clone(), segment.id.clone())
            .unwrap()
            .unwrap();

        assert_eq!(segment, retrieved_segment);
    }

    #[test]
    fn insert_and_delete_segments() {
        let kv = Arc::new(RwLock::new(KV::new()));
        let segment_interface = SegmentInterface::new(kv.clone());

        let collection_id = CollectionId("test_collection".to_string());

        let segment = Segment {
            id: "test_segment".to_string(),
            name: "Test Segment".to_string(),
            description: "This is a test segment".to_string(),
            goal: Some("Test goal".to_string()),
        };

        segment_interface
            .insert(collection_id.clone(), segment.clone())
            .unwrap();

        let deleted_segment = segment_interface
            .delete(collection_id.clone(), segment.id.clone())
            .unwrap()
            .unwrap();

        let after_delete_result = segment_interface
            .get(collection_id.clone(), segment.id.clone())
            .unwrap();

        assert_eq!(segment, deleted_segment);
        assert!(after_delete_result.is_none());
    }

    #[test]
    fn insert_and_list() {
        let kv = Arc::new(RwLock::new(KV::new()));
        let segment_interface = SegmentInterface::new(kv.clone());

        let collection_id = CollectionId("test_collection".to_string());

        let segment1 = Segment {
            id: "test_segment1".to_string(),
            name: "Test Segment 1".to_string(),
            description: "This is a test segment 1".to_string(),
            goal: Some("Test goal 1".to_string()),
        };

        let segment2 = Segment {
            id: "test_segment2".to_string(),
            name: "Test Segment 2".to_string(),
            description: "This is a test segment 2".to_string(),
            goal: Some("Test goal 2".to_string()),
        };

        segment_interface
            .insert(collection_id.clone(), segment1.clone())
            .unwrap();
        segment_interface
            .insert(collection_id.clone(), segment2.clone())
            .unwrap();

        let segments = segment_interface
            .list_by_collection(collection_id.clone())
            .unwrap();

        assert_eq!(segments.len(), 2);
        assert!(segments.contains(&segment1));
        assert!(segments.contains(&segment2));
    }
}
