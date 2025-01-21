use anyhow::Result;
use axum_openapi3::utoipa::ToSchema;
use axum_openapi3::utoipa::{self, IntoParams};
use chrono::Utc;
use dashmap::DashMap;
use oxc_allocator::Allocator;
use oxc_parser::Parser;
use oxc_span::SourceType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Display;
use thiserror::Error;

use crate::types::CollectionId;

#[derive(Serialize, Clone)]
pub struct HookValue {
    pub code: String,
    pub created_at: i64,
}

impl HookValue {
    pub fn to_string(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

#[derive(Error, Debug)]
pub enum HookError {
    #[error("Invalid hook name: {0}")]
    InvalidHook(String),
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

pub struct HookStorage {
    hooks_map: DashMap<CollectionId, HashMap<HookName, HookValue>>,
}

impl HookStorage {
    pub fn new() -> Self {
        Self {
            hooks_map: DashMap::new(),
        }
    }

    pub fn has_hook(&self, collection_id: CollectionId, name: HookName) -> bool {
        self.hooks_map
            .get(&collection_id)
            .map(|hooks| hooks.contains_key(&name))
            .unwrap_or(false)
    }

    pub fn get_hook(&self, collection_id: CollectionId, name: HookName) -> Option<HookValue> {
        self.hooks_map
            .get(&collection_id)
            .and_then(|hooks| hooks.get(&name).cloned())
    }

    pub fn list_hooks(&self, collection_id: CollectionId) -> HashMap<HookName, HookValue> {
        self.hooks_map
            .get(&collection_id)
            .map(|hooks| hooks.clone())
            .unwrap_or_default()
    }

    pub fn delete_hook(
        &self,
        collection_id: CollectionId,
        name: HookName,
    ) -> Option<(String, HookValue)> {
        self.hooks_map
            .get_mut(&collection_id)
            .and_then(|mut hooks| hooks.remove(&name).map(|value| (name.to_string(), value)))
    }

    pub fn insert_hook(
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
