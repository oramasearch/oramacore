use super::generic_kv::{format_key, KV};
use crate::{ai::AIService, collection_manager::dto::InteractionMessage, types::CollectionId};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

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

pub struct TriggerInterface {
    kv: Arc<KV>,
    ai_service: Arc<AIService>,
}

impl TriggerInterface {
    pub fn new(kv: Arc<KV>, ai_service: Arc<AIService>) -> Self {
        Self { kv, ai_service }
    }

    pub async fn insert(&self, collection_id: CollectionId, trigger: Trigger) -> Result<String> {
        let key = self.get_trigger_key(
            collection_id,
            trigger.id.clone(),
            trigger.segment_id.clone(),
        );

        self.kv.insert(key.clone(), trigger).await?;
        Ok(key)
    }

    pub async fn get(
        &self,
        collection_id: CollectionId,
        trigger_id: String,
    ) -> Result<Option<Trigger>> {
        let trigger_id_parts = parse_trigger_id(trigger_id.clone());

        let key = self.get_trigger_key(
            collection_id,
            trigger_id_parts.trigger_id,
            trigger_id_parts.segment_id,
        );

        match self.kv.get(&key).await {
            None => Ok(None),
            Some(Err(e)) => Err(e),
            Some(Ok(trigger)) => Ok(Some(Trigger { id: key, ..trigger })),
        }
    }

    pub async fn delete(
        &self,
        collection_id: CollectionId,
        trigger_id: String,
    ) -> Result<Option<Trigger>> {
        let trigger_id_parts = parse_trigger_id(trigger_id.clone());

        let key = self.get_trigger_key(
            collection_id,
            trigger_id_parts.trigger_id,
            trigger_id_parts.segment_id,
        );

        match self.kv.remove_and_get(&key).await? {
            None => Ok(None),
            Some(Err(e)) => Err(e),
            Some(Ok(trigger)) => Ok(Some(trigger)),
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

        let triggers_with_key = triggers
            .into_iter()
            .map(|trigger| Trigger {
                id: self.get_trigger_key(
                    collection_id.clone(),
                    trigger.id.clone(),
                    trigger.segment_id.clone(),
                ),
                ..trigger
            })
            .collect();

        Ok(triggers_with_key)
    }

    pub async fn perform_trigger_selection(
        &self,
        collection_id: CollectionId,
        conversation: Option<Vec<InteractionMessage>>,
    ) -> Result<crate::ai::TriggerResponse> {
        let triggers = self.list_by_collection(collection_id).await?;
        self.ai_service.get_trigger(triggers, conversation).await
    }

    fn get_trigger_key(
        &self,
        collection_id: CollectionId,
        trigger_id: String,
        segment_id: Option<String>,
    ) -> String {
        match segment_id {
            Some(segment_id) => format_key(
                collection_id,
                &format!("trigger:s_{}:t_{}", segment_id, trigger_id),
            ),
            None => format_key(collection_id, &format!("trigger:t_{}", trigger_id)),
        }
    }
}

pub fn parse_trigger_id(trigger_id: String) -> TriggerIdContent {
    let parts = trigger_id.split(':').collect::<Vec<&str>>();
    let collection_id = CollectionId(parts[0].to_string());

    let segment_id = parts
        .iter()
        .find(|part| part.starts_with("s_"))
        .map(|s| s.to_string());

    let trigger_id = parts.iter().find(|part| part.starts_with("t_")).unwrap();

    TriggerIdContent {
        collection_id,
        trigger_id: trigger_id.replace("t_", ""),
        segment_id: segment_id.map(|s| s.replace("s_", "")),
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use http::uri::Scheme;

    use crate::{ai::AIServiceConfig, collection_manager::sides::generic_kv::KVConfig};

    use super::*;

    #[tokio::test]
    async fn insert_and_retrieve_triggers() {
        let kv_config = KVConfig {
            data_dir: PathBuf::from("/tmp/triggers"),
            sender: None,
        };

        let ai_service_conf = AIServiceConfig {
            scheme: Scheme::HTTP,
            max_connections: 1,
            port: 8080,
            api_key: None,
            host: std::net::IpAddr::V4(std::net::Ipv4Addr::new(127, 0, 0, 1)),
        };

        let kv = Arc::new(KV::try_load(kv_config).unwrap());
        let trigger_interface =
            TriggerInterface::new(kv.clone(), Arc::new(AIService::new(ai_service_conf)));

        let collection_id = CollectionId("test_collection".to_string());

        let trigger = Trigger {
            id: "test_trigger".to_string(),
            name: "Test Trigger".to_string(),
            description: "This is a test trigger".to_string(),
            response: "Test response".to_string(),
            segment_id: None,
        };

        trigger_interface
            .insert(collection_id.clone(), trigger.clone())
            .await
            .unwrap();

        let retrieved_trigger = trigger_interface
            .get(collection_id.clone(), trigger.id.clone())
            .await
            .unwrap()
            .unwrap();

        assert_eq!(trigger, retrieved_trigger);
    }

    #[tokio::test]
    async fn insert_and_delete_triggers() {
        let kv_config = KVConfig {
            data_dir: PathBuf::from("/tmp/triggers"),
            sender: None,
        };

        let ai_service_conf = AIServiceConfig {
            scheme: Scheme::HTTP,
            max_connections: 1,
            port: 8080,
            api_key: None,
            host: std::net::IpAddr::V4(std::net::Ipv4Addr::new(127, 0, 0, 1)),
        };

        let kv = Arc::new(KV::try_load(kv_config).unwrap());
        let trigger_interface =
            TriggerInterface::new(kv.clone(), Arc::new(AIService::new(ai_service_conf)));

        let collection_id = CollectionId("test_collection".to_string());

        let trigger = Trigger {
            id: "test_trigger".to_string(),
            name: "Test Trigger".to_string(),
            description: "This is a test trigger".to_string(),
            response: "Test response".to_string(),
            segment_id: None,
        };

        trigger_interface
            .insert(collection_id.clone(), trigger.clone())
            .await
            .unwrap();

        let deleted_trigger = trigger_interface
            .delete(collection_id.clone(), trigger.id.clone())
            .await
            .unwrap()
            .unwrap();

        let after_delete_result = trigger_interface
            .get(collection_id.clone(), trigger.id.clone())
            .await
            .unwrap();

        assert_eq!(trigger, deleted_trigger);
        assert!(after_delete_result.is_none());
    }

    #[tokio::test]
    async fn insert_and_list() {
        let kv_config = KVConfig {
            data_dir: PathBuf::from("/tmp/triggers"),
            sender: None,
        };

        let ai_service_conf = AIServiceConfig {
            scheme: Scheme::HTTP,
            max_connections: 1,
            port: 8080,
            api_key: None,
            host: std::net::IpAddr::V4(std::net::Ipv4Addr::new(127, 0, 0, 1)),
        };

        let kv = Arc::new(KV::try_load(kv_config).unwrap());
        let trigger_interface =
            TriggerInterface::new(kv.clone(), Arc::new(AIService::new(ai_service_conf)));

        let collection_id = CollectionId("test_collection".to_string());

        let trigger1 = Trigger {
            id: "test_trigger1".to_string(),
            name: "Test Trigger 1".to_string(),
            description: "This is a test trigger 1".to_string(),
            response: "Test response 1".to_string(),
            segment_id: None,
        };

        let trigger2 = Trigger {
            id: "test_trigger2".to_string(),
            name: "Test Trigger 2".to_string(),
            description: "This is a test trigger 2".to_string(),
            response: "Test response 2".to_string(),
            segment_id: None,
        };

        trigger_interface
            .insert(collection_id.clone(), trigger1.clone())
            .await
            .context("Cannot insert trigger 1")
            .unwrap();
        trigger_interface
            .insert(collection_id.clone(), trigger2.clone())
            .await
            .context("Cannot insert trigger 2")
            .unwrap();

        let triggers = trigger_interface
            .list_by_collection(collection_id.clone())
            .await
            .unwrap();

        assert_eq!(triggers.len(), 2);
        assert!(triggers.contains(&trigger1));
        assert!(triggers.contains(&trigger2));
    }
}
