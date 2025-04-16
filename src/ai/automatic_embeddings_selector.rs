use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::{collection_manager::sides::generic_kv::KV, types::CollectionId};

use super::llms::LLMService;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChosenProperties {
    pub properties: Vec<String>,
    #[serde(rename = "includeKeys")]
    pub include_keys: Vec<String>,
    pub rename: HashMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChosenPropertiesError {
    pub error: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ChosenPropertiesResult {
    Properties(ChosenProperties),
    Error(ChosenPropertiesError),
}

pub struct AutomaticEmbeddingsSelector {
    pub llm_service: Arc<LLMService>,
    pub kv_service: Arc<KV>,
}

impl AutomaticEmbeddingsSelector {
    pub fn new(llm_service: Arc<LLMService>, kv_service: Arc<KV>) -> Self {
        Self {
            llm_service,
            kv_service,
        }
    }

    async fn get_properties(&self, collection_id: CollectionId, documents: Vec<Value>) {
        let key = self.get_key(collection_id.clone(), documents.clone());

        if self.key_exists(&key).await {
            return; // @todo: return keys here
        }
    }

    async fn choose_properties(
        &self,
        documents: Vec<Value>,
        llm_config: Option<crate::types::InteractionLLMConfig>,
    ) -> Result<ChosenProperties> {
        let documents_as_json = serde_json::to_string(&documents)?;
        let variables = vec![("documents".to_string(), documents_as_json)];

        let result = self
            .llm_service
            .run_known_prompt(
                super::llms::KnownPrompts::AutomaticEmbeddingsSelector,
                variables,
                llm_config,
            )
            .await?;

        match serde_json::from_str(&result)? {
            ChosenPropertiesResult::Properties(properties) => Ok(properties),
            ChosenPropertiesResult::Error(error) => Err(anyhow::anyhow!(error.error)),
        }
    }

    async fn key_exists(&self, key: &str) -> bool {
        self.kv_service.get::<String>(key).await.is_some()
    }

    fn get_key(&self, collection_id: CollectionId, documents: Vec<Value>) -> String {
        use std::collections::HashSet;
        let mut all_keys = HashSet::new();

        for doc in documents.iter() {
            if let Value::Object(map) = doc {
                self.extract_keys_with_dot_notation(map, "", &mut all_keys);
            }
        }

        let mut keys: Vec<String> = all_keys.into_iter().collect();

        keys.sort();

        if keys.is_empty() {
            return String::new();
        }

        format!("{}:{}", collection_id, keys.join(":"))
    }

    fn extract_keys_with_dot_notation(
        &self,
        obj: &serde_json::Map<String, Value>,
        prefix: &str,
        keys: &mut HashSet<String>,
    ) {
        for (key, value) in obj.iter() {
            let current_key = if prefix.is_empty() {
                key.clone()
            } else {
                format!("{}.{}", prefix, key)
            };

            keys.insert(current_key.clone());

            if let Value::Object(nested_obj) = value {
                self.extract_keys_with_dot_notation(nested_obj, &current_key, keys);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{collection_manager::sides::generic_kv::KVConfig, tests::utils::generate_new_path};

    use super::*;
    use serde_json::json;

    fn create_selector() -> AutomaticEmbeddingsSelector {
        let llm_service = Arc::new(
            LLMService::try_new(
                crate::ai::AIServiceLLMConfig {
                    host: "localhost".to_string(),
                    port: 8000,
                    model: "Qwen/Qwen2.5-3b-Instruct".to_string(),
                },
                None,
            )
            .unwrap(),
        );

        let kv_service = Arc::new(
            KV::try_load(KVConfig {
                data_dir: generate_new_path(),
                sender: None,
            })
            .unwrap(),
        );

        AutomaticEmbeddingsSelector {
            llm_service,
            kv_service,
        }
    }

    #[test]
    fn test_different_order_nested_objects_produce_same_key() {
        let selector = create_selector();
        let collection_id = CollectionId::try_new("test_collection".to_string()).unwrap();

        // Create first document with a specific order of nested keys
        let doc1 = json!({
            "user": {
                "name": "John",
                "address": {
                    "city": "New York",
                    "zip": "10001"
                }
            },
            "active": true
        });

        // Create second document with the same structure but different order of keys
        let doc2 = json!({
            "active": true,
            "user": {
                "address": {
                    "zip": "10001",
                    "city": "New York"
                },
                "name": "John"
            }
        });

        // Get keys for each document individually
        let key1 = selector.get_key(collection_id.clone(), vec![doc1.clone()]);
        let key2 = selector.get_key(collection_id.clone(), vec![doc2.clone()]);

        // Test that the keys are identical
        assert_eq!(key1, key2);

        // Verify the expected key format
        let expected_keys = format!(
            "{}:{}:{}:{}:{}:{}:{}",
            collection_id,
            "active",
            "user",
            "user.address",
            "user.address.city",
            "user.address.zip",
            "user.name"
        );
        assert_eq!(key1, expected_keys);
    }

    #[test]
    fn test_deeply_nested_objects_with_different_order() {
        let selector = create_selector();
        let collection_id = CollectionId::try_new("nested_collection".to_string()).unwrap();

        // First document with deep nesting
        let doc1 = json!({
            "metadata": {
                "created_at": "2023-01-01",
                "config": {
                    "settings": {
                        "enabled": true,
                        "options": {
                            "color": "blue",
                            "size": "medium"
                        }
                    },
                    "version": "1.0"
                }
            },
            "data": {
                "items": [1, 2, 3]
            }
        });

        // Second document with same structure but different order
        let doc2 = json!({
            "data": {
                "items": [1, 2, 3]
            },
            "metadata": {
                "config": {
                    "version": "1.0",
                    "settings": {
                        "options": {
                            "size": "medium",
                            "color": "blue"
                        },
                        "enabled": true
                    }
                },
                "created_at": "2023-01-01"
            }
        });

        // Get keys for each document
        let key1 = selector.get_key(collection_id.clone(), vec![doc1]);
        let key2 = selector.get_key(collection_id.clone(), vec![doc2]);

        // Test that the keys are identical
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_mixed_documents_produce_combined_key() {
        let selector = create_selector();
        let collection_id = CollectionId::try_new("mixed_collection".to_string()).unwrap();

        // Create documents with different structures
        let doc1 = json!({
            "id": 1,
            "name": "Document 1"
        });

        let doc2 = json!({
            "id": 2,
            "attributes": {
                "tags": ["important", "urgent"]
            }
        });

        // Get keys for individual documents
        let key1 = selector.get_key(collection_id.clone(), vec![doc1.clone()]);
        let key2 = selector.get_key(collection_id.clone(), vec![doc2.clone()]);

        // Get key for combined documents
        let combined_key = selector.get_key(collection_id.clone(), vec![doc1, doc2]);

        // Verify that combined key contains all keys from both documents
        assert_ne!(key1, combined_key);
        assert_ne!(key2, combined_key);

        // Check if combined key has all expected keys
        let expected_keys = ["attributes", "attributes.tags", "id", "name"];

        for key in expected_keys {
            assert!(
                combined_key.contains(key),
                "Combined key should contain '{}'",
                key
            );
        }
    }
}
