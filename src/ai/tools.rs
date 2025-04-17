use std::sync::Arc;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::{collection_manager::sides::generic_kv::KV, types::CollectionId};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Tool {
    pub id: String,
    pub name: String,
    pub description: String,
    pub parameters: String, // @todo: check if we want to have a serde_json::Value here
}

impl Tool {
    pub fn new(id: String, name: String, description: String, parameters: String) -> Self {
        Tool {
            id,
            name,
            description,
            parameters,
        }
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

pub struct ToolsManager {
    pub kv: Arc<KV>,
}

impl ToolsManager {
    pub fn new(kv: Arc<KV>) -> Self {
        ToolsManager { kv }
    }

    pub async fn insert(&self, collection_id: CollectionId, tool: Tool) -> Result<()> {
        let key = self.format_key(collection_id, &tool.id);

        self.kv.insert(key, tool).await?;

        Ok(())
    }

    pub async fn get(&self, collection_id: CollectionId, tool_id: String) -> Result<Option<Tool>> {
        let key = self.format_key(collection_id, &tool_id);

        match self.kv.get::<Tool>(&key).await {
            None => Ok(None),
            Some(Err(e)) => Err(e),
            Some(Ok(tool)) => Ok(Some(tool)),
        }
    }

    pub async fn delete(&self, collection_id: CollectionId, tool_id: String) -> Result<()> {
        let key = self.format_key(collection_id, &tool_id);

        self.kv.remove(&key).await?;

        Ok(())
    }

    pub async fn has_tools(&self, collection_id: CollectionId) -> Result<bool> {
        let tools = self.list_by_collection(collection_id).await?;

        Ok(!tools.is_empty())
    }

    pub async fn list_by_collection(&self, collection_id: CollectionId) -> Result<Vec<Tool>> {
        let prefix = format!("{}:tool:", collection_id.as_str());

        let segments = self.kv.prefix_scan(&prefix).await.context(format!(
            "Cannot scan tools for collection {}",
            collection_id.as_str()
        ))?;

        Ok(segments)
    }

    fn format_key(&self, collection_id: CollectionId, tool_id: &str) -> String {
        format!("{}:tool:{}", collection_id, tool_id)
    }
}
