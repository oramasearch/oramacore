use anyhow::{Context, Result};
use axum_openapi3::utoipa::ToSchema;
use axum_openapi3::utoipa::{self};
use chrono::Utc;
use oxc_allocator::Allocator;
use oxc_parser::Parser;
use oxc_span::SourceType;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::path::PathBuf;
use std::str::FromStr;
use tokio::sync::RwLock;

use crate::file_utils::{create_if_not_exists, BufferedFile};
use crate::js::deno::{JavaScript, Operation};
use crate::types::CollectionId;

#[derive(Deserialize, Serialize, Clone, Debug)]
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

#[derive(Debug)]
struct HookStorage {
    hooks_map: RwLock<HashMap<CollectionId, HashMap<HookName, HookValue>>>,
}

impl HookStorage {
    fn try_load(data_dir: PathBuf) -> Result<Self> {
        let hooks_path = data_dir.join("hooks.bin");
        let hooks_map = BufferedFile::open(hooks_path)
            .and_then(|f| f.read_bincode_data())
            .unwrap_or_default();

        Ok(Self {
            hooks_map: RwLock::new(hooks_map),
        })
    }

    async fn has_hook(&self, collection_id: CollectionId, name: HookName) -> bool {
        let hooks_map = self.hooks_map.read().await;
        hooks_map
            .get(&collection_id)
            .map(|hooks| hooks.contains_key(&name))
            .unwrap_or(false)
    }

    async fn get_hook(&self, collection_id: CollectionId, name: HookName) -> Option<HookValue> {
        let hooks_map = self.hooks_map.read().await;
        hooks_map
            .get(&collection_id)
            .and_then(|hooks| hooks.get(&name).cloned())
    }

    async fn list_hooks(&self, collection_id: CollectionId) -> HashMap<HookName, HookValue> {
        let hooks_map = self.hooks_map.read().await;
        hooks_map
            .get(&collection_id)
            .map(|hooks| hooks.clone())
            .unwrap_or_default()
    }

    async fn delete_hook(
        &self,
        collection_id: CollectionId,
        name: HookName,
    ) -> Option<(String, HookValue)> {
        let mut hooks_map = self.hooks_map.write().await;
        hooks_map
            .get_mut(&collection_id)
            .and_then(|hooks| hooks.remove(&name).map(|value| (name.to_string(), value)))
    }

    async fn insert_hook(
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

        let mut hooks_map = self.hooks_map.write().await;
        hooks_map
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

        result.errors.is_empty()
    }

    async fn commit(&self, data_dir: PathBuf) -> Result<()> {
        let hooks_map = self.hooks_map.write().await;

        create_if_not_exists(&data_dir).context("Cannot create hook dir")?;

        let hooks_path = data_dir.join("hooks.bin");
        BufferedFile::create_or_overwrite(hooks_path)
            .context("Cannot create hooks file")?
            .write_bincode_data(&hooks_map.clone())
            .context("Cannot serialize hooks file")?;

        Ok(())
    }
}

#[derive(Clone, Deserialize, Serialize)]
pub struct HooksRuntimeConfig {
    pub channel_limit: usize,
    pub data_dir: PathBuf,
}

pub struct HooksRuntime {
    data_dir: PathBuf,
    hooks: HookStorage,
    javascript_runtime: JavaScript,
}

impl HooksRuntime {
    pub async fn try_load(config: HooksRuntimeConfig) -> Result<Self> {
        let storage =
            HookStorage::try_load(config.data_dir.clone()).context("Cannot load hooks")?;

        Ok(Self {
            data_dir: config.data_dir,
            hooks: storage,
            javascript_runtime: JavaScript::new(config.channel_limit).await,
        })
    }

    pub async fn has_hook(&self, collection_id: CollectionId, name: HookName) -> bool {
        self.hooks.has_hook(collection_id, name).await
    }

    pub async fn get_hook(&self, collection_id: CollectionId, name: HookName) -> Option<HookValue> {
        self.hooks.get_hook(collection_id, name).await
    }

    pub async fn list_hooks(&self, collection_id: CollectionId) -> HashMap<HookName, HookValue> {
        self.hooks.list_hooks(collection_id).await
    }

    pub async fn delete_hook(
        &self,
        collection_id: CollectionId,
        name: HookName,
    ) -> Option<(String, HookValue)> {
        self.hooks.delete_hook(collection_id, name).await
    }

    pub async fn insert_hook(
        &self,
        collection_id: CollectionId,
        name: HookName,
        code: String,
    ) -> Result<()> {
        self.hooks.insert_hook(collection_id, name, code).await
    }

    pub async fn eval<T: Serialize, R: DeserializeOwned>(
        &self,
        collection_id: CollectionId,
        name: HookName,
        input: T,
    ) -> Option<Result<R>> {
        let hook = self.hooks.get_hook(collection_id, name).await?;

        let operation = match name {
            HookName::SelectEmbeddingsProperties => Operation::SelectEmbeddingsProperties,
        };

        let result = self
            .javascript_runtime
            .eval(operation, hook.code, input)
            .await;

        Some(result)
    }

    pub async fn commit(&self) -> Result<()> {
        self.hooks.commit(self.data_dir.clone()).await
    }
}

impl Debug for HooksRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HooksRuntime")
            .field("hooks", &self.hooks)
            .field("javascript_runtime", &"...")
            .finish()
    }
}
