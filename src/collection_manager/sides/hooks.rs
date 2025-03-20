use anyhow::{anyhow, Context, Result};
use axum_openapi3::utoipa::ToSchema;
use axum_openapi3::utoipa::{self};
use chrono::Utc;
use duration_string::DurationString;
use orama_js_pool::{JSExecutorPoolConfig, OramaJSPool, OramaJSPoolConfig};
use oxc_allocator::Allocator;
use oxc_parser::Parser;
use oxc_span::SourceType;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::str::FromStr;
use std::sync::Arc;
use tracing::warn;

use crate::metrics::js::JS_CALCULATION_TIME;
use crate::metrics::JSOperationLabels;
use crate::types::{CollectionId, FlattenDocument};

use super::generic_kv::KV;

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct HookValue {
    pub code: String,
    pub created_at: i64,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct HookPair(HookName, HookValue);

impl HookValue {
    pub fn to_string(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Hash, Eq, ToSchema)]
pub enum HookName {
    #[serde(rename = "selectEmbeddingProperties")]
    SelectEmbeddingsProperties,
}

impl FromStr for HookName {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "selectEmbeddingProperties" => Ok(HookName::SelectEmbeddingsProperties),
            _ => Err(anyhow::anyhow!("Invalid hook name")),
        }
    }
}

impl Display for HookName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HookName::SelectEmbeddingsProperties => write!(f, "selectEmbeddingProperties"),
        }
    }
}

#[inline]
fn prefix(collection_id: &CollectionId) -> String {
    format!("{}:hook:", collection_id.0)
}
#[inline]
fn hook_key(collection_id: &CollectionId, name: &HookName) -> String {
    format!("{}{}", prefix(collection_id), name)
}

#[derive(Clone, Deserialize, Serialize)]
pub struct SelectEmbeddingsPropertiesHooksRuntimeConfig {
    pub check_interval: DurationString,
    pub max_idle_time: DurationString,
    pub instances_count_per_code: usize,
    pub queue_capacity: usize,
    pub max_execution_time: DurationString,
    pub max_startup_time: DurationString,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct HooksRuntimeConfig {
    pub select_embeddings_properties: SelectEmbeddingsPropertiesHooksRuntimeConfig,
}

pub struct HooksRuntime {
    kv: Arc<KV>,
    embedding_js_runtime: OramaJSPool<Value, SelectEmbeddingPropertiesReturnType>,
}

impl HooksRuntime {
    pub async fn new(kv: Arc<KV>, config: HooksRuntimeConfig) -> Self {
        Self {
            kv,
            embedding_js_runtime: OramaJSPool::new(OramaJSPoolConfig {
                check_interval: *config.select_embeddings_properties.check_interval,
                max_idle_time: *config.select_embeddings_properties.max_idle_time,
                pool_config: JSExecutorPoolConfig {
                    instances_count_per_code: config
                        .select_embeddings_properties
                        .instances_count_per_code,
                    queue_capacity: config.select_embeddings_properties.queue_capacity,
                    executor_config: orama_js_pool::JSExecutorConfig {
                        allowed_hosts: vec![],
                        function_name: "selectEmbeddingsProperties".to_string(),
                        is_async: false,
                        max_execution_time: *config.select_embeddings_properties.max_execution_time,
                        max_startup_time: *config.select_embeddings_properties.max_startup_time,
                    },
                },
            }),
        }
    }

    pub async fn has_hook(&self, collection_id: CollectionId, name: HookName) -> bool {
        self.get_hook(collection_id, name).await.is_some()
    }

    pub async fn get_hook(&self, collection_id: CollectionId, name: HookName) -> Option<HookValue> {
        let key = hook_key(&collection_id, &name);

        let k = self.kv.get::<HookPair>(&key).await;

        match k {
            Some(Ok(v)) => Some(v.1),
            None => None,
            Some(Err(e)) => {
                warn!("Error getting hook: {}. Ignored", e);
                None
            }
        }
    }

    pub async fn list_hooks(
        &self,
        collection_id: CollectionId,
    ) -> Result<HashMap<HookName, HookValue>> {
        let pairs = self
            .kv
            .prefix_scan::<HookPair>(&prefix(&collection_id))
            .await
            .context("Error listing hooks")?;
        let ret = pairs
            .into_iter()
            .map(|HookPair(name, value)| (name, value))
            .collect();
        Ok(ret)
    }

    pub async fn delete_hook(
        &self,
        collection_id: CollectionId,
        name: HookName,
    ) -> Option<(String, HookValue)> {
        let key = hook_key(&collection_id, &name);
        let a = self.kv.remove_and_get::<HookPair>(&key).await;

        match a {
            Ok(Some(Ok(v))) => Some((v.0.to_string(), v.1)),
            Ok(None) => None,
            Ok(Some(Err(e))) => {
                warn!("Error deleting hook: {}. Ignored", e);
                None
            }
            Err(e) => {
                warn!("Error deleting hook: {}. Ignored", e);
                None
            }
        }
    }

    pub async fn insert_hook(
        &self,
        collection_id: CollectionId,
        name: HookName,
        code: String,
    ) -> Result<()> {
        if !self.is_valid_js(&code) {
            return Err(anyhow::anyhow!("Invalid JavaScript code"));
        }

        let hook = HookValue {
            code,
            created_at: Utc::now().timestamp(),
        };
        let pair = HookPair(name.clone(), hook);

        let key = hook_key(&collection_id, &name);
        self.kv.insert(key, pair).await
    }

    pub async fn calculate_text_for_embedding(
        &self,
        collection_id: CollectionId,
        doc: FlattenDocument,
    ) -> Option<Result<SelectEmbeddingPropertiesReturnType>> {
        let key = hook_key(&collection_id, &HookName::SelectEmbeddingsProperties);
        let code = match self.kv.get::<HookPair>(&key).await {
            None => return None,
            Some(Err(e)) => return Some(Err(e)),
            Some(Ok(pair)) => pair.1.code,
        };

        let m = JS_CALCULATION_TIME.create(JSOperationLabels {
            operation: "selectEmbeddingsProperties",
            collection: collection_id.to_string().into(),
        });
        let output = self
            .embedding_js_runtime
            .execute(&code, serde_json::Value::Object(doc.into_inner()))
            .await
            .map_err(|e| anyhow!("Error in evaluate the embedding script: {:?}", e));
        drop(m);

        Some(output)
    }

    // @todo: make this more robust and possibly check arguments and returning types via AST
    fn is_valid_js(&self, code: &str) -> bool {
        let allocator = Allocator::default();
        let source_type = SourceType::default();
        let parser = Parser::new(&allocator, code, source_type);

        let result = parser.parse();

        result.errors.is_empty()
    }
}

impl Debug for HooksRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HooksRuntime")
            .field("hooks", &"...")
            .field("javascript_runtime", &"...")
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use crate::{collection_manager::sides::generic_kv::KVConfig, tests::utils::generate_new_path};

    use super::*;

    fn config() -> HooksRuntimeConfig {
        HooksRuntimeConfig {
            select_embeddings_properties: SelectEmbeddingsPropertiesHooksRuntimeConfig {
                check_interval: DurationString::from_str("1s").unwrap(),
                max_idle_time: DurationString::from_str("1s").unwrap(),
                instances_count_per_code: 1,
                queue_capacity: 1,
                max_execution_time: DurationString::from_str("1s").unwrap(),
                max_startup_time: DurationString::from_str("1s").unwrap(),
            },
        }
    }

    #[tokio::test]
    async fn test_hook_crud() {
        let hook_runtime = HooksRuntime::new(
            Arc::new(
                KV::try_load(KVConfig {
                    data_dir: generate_new_path(),
                    sender: None,
                })
                .unwrap(),
            ),
            config(),
        )
        .await;

        let collection_id = CollectionId::from("1".to_string());
        hook_runtime
            .insert_hook(
                collection_id,
                HookName::SelectEmbeddingsProperties,
                "function the_hook() { return 1; }".to_string(),
            )
            .await
            .unwrap();

        let hook = hook_runtime
            .get_hook(collection_id, HookName::SelectEmbeddingsProperties)
            .await
            .unwrap();
        assert_eq!(hook.code, "function the_hook() { return 1; }");

        let hooks = hook_runtime.list_hooks(collection_id).await.unwrap();
        assert_eq!(hooks.len(), 1);

        let hook = hook_runtime
            .delete_hook(collection_id, HookName::SelectEmbeddingsProperties)
            .await
            .unwrap();
        assert_eq!(hook.0, "selectEmbeddingProperties");

        let hook = hook_runtime
            .get_hook(collection_id, HookName::SelectEmbeddingsProperties)
            .await;
        assert!(hook.is_none());
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SelectEmbeddingPropertiesReturnType {
    Properties(Vec<String>),
    Text(String),
}
