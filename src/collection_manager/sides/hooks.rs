use anyhow::Result;
use axum_openapi3::utoipa::ToSchema;
use axum_openapi3::utoipa::{self};
use chrono::Utc;
use dashmap::DashMap;
use oxc_allocator::Allocator;
use oxc_parser::Parser;
use oxc_span::SourceType;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::{Debug, Display};

use crate::js::deno::{JavaScript, Operation};
use crate::types::CollectionId;

#[derive(Serialize, Clone, Debug)]
pub struct HookValue {
    pub code: String,
    pub created_at: i64,
}

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

impl Display for HookName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HookName::SelectEmbeddingsProperties => write!(f, "selectEmbeddingProperties"),
        }
    }
}

#[derive(Debug)]
struct HookStorage {
    hooks_map: DashMap<CollectionId, HashMap<HookName, HookValue>>,
}

impl HookStorage {
    fn new() -> Self {
        Self {
            hooks_map: DashMap::new(),
        }
    }

    fn has_hook(&self, collection_id: CollectionId, name: HookName) -> bool {
        self.hooks_map
            .get(&collection_id)
            .map(|hooks| hooks.contains_key(&name))
            .unwrap_or(false)
    }

    fn get_hook(&self, collection_id: CollectionId, name: HookName) -> Option<HookValue> {
        self.hooks_map
            .get(&collection_id)
            .and_then(|hooks| hooks.get(&name).cloned())
    }

    fn list_hooks(&self, collection_id: CollectionId) -> HashMap<HookName, HookValue> {
        self.hooks_map
            .get(&collection_id)
            .map(|hooks| hooks.clone())
            .unwrap_or_default()
    }

    fn delete_hook(
        &self,
        collection_id: CollectionId,
        name: HookName,
    ) -> Option<(String, HookValue)> {
        self.hooks_map
            .get_mut(&collection_id)
            .and_then(|mut hooks| hooks.remove(&name).map(|value| (name.to_string(), value)))
    }

    fn insert_hook(&self, collection_id: CollectionId, name: HookName, code: String) -> Result<()> {
        if !self.is_valid_js(&code) {
            return Err(anyhow::anyhow!("Invalid JavaScript code"));
        }

        let hook = HookValue {
            code,
            created_at: Utc::now().timestamp(),
        };

        self.hooks_map
            .entry(collection_id)
            .or_default()
            .insert(name, hook);

        Ok(())
    }

    // @todo: make this more robust and possibly check arguments and returning types via AST
    fn is_valid_js(&self, code: &str) -> bool {
        let allocator = Allocator::default();
        let source_type = SourceType::default();
        let parser = Parser::new(&allocator, code, source_type);

        let result = parser.parse();
        !result.errors.is_empty()
    }
}

pub struct HooksRuntime {
    hooks: HookStorage,
    javascript_runtime: JavaScript,
}

impl Debug for HooksRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HooksRuntime")
            .field("hooks", &self.hooks)
            .field("javascript_runtime", &"...")
            .finish()
    }
}

impl Default for HooksRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl HooksRuntime {
    pub fn new() -> Self {
        Self {
            hooks: HookStorage::new(),
            javascript_runtime: JavaScript::new(),
        }
    }

    pub fn has_hook(&self, collection_id: CollectionId, name: HookName) -> bool {
        self.hooks.has_hook(collection_id, name)
    }

    pub fn get_hook(&self, collection_id: CollectionId, name: HookName) -> Option<HookValue> {
        self.hooks.get_hook(collection_id, name)
    }

    pub fn list_hooks(&self, collection_id: CollectionId) -> HashMap<HookName, HookValue> {
        self.hooks.list_hooks(collection_id)
    }

    pub fn delete_hook(
        &self,
        collection_id: CollectionId,
        name: HookName,
    ) -> Option<(String, HookValue)> {
        self.hooks.delete_hook(collection_id, name)
    }

    pub fn insert_hook(
        &self,
        collection_id: CollectionId,
        name: HookName,
        code: String,
    ) -> Result<()> {
        self.hooks.insert_hook(collection_id, name, code)
    }

    pub async fn eval<T: Serialize, R: DeserializeOwned>(
        &self,
        collection_id: CollectionId,
        name: HookName,
        input: T,
    ) -> Option<Result<R>> {
        let hook = self.hooks.get_hook(collection_id, name)?;

        let operation = match name {
            HookName::SelectEmbeddingsProperties => Operation::SelectEmbeddingsProperties,
        };

        let result = self.javascript_runtime.eval(operation, hook.code, input);

        Some(result.await)
    }
}
