use std::str::FromStr;

use anyhow::Result;
use chrono::Utc;
use dashmap::DashMap;
use oxc_allocator::{Allocator, HashMap};
use oxc_parser::Parser;
use oxc_span::SourceType;
use serde::Serialize;
use thiserror::Error;

const HOOK_PREFIX: &str = "hook:";

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

#[derive(Serialize, Debug, Clone, PartialEq)]
pub enum Hook {
    SelectEmbeddingsProperties,
}

impl Hook {
    // @todo: this may not be necessary, but we may want to prefix "official" hooks with "hook:" to distingish them from
    // user defined hooks that are not part of the specs.
    pub fn to_key_name(&self) -> String {
        format!("{}{}", HOOK_PREFIX, self.to_string().to_lowercase())
    }
}

impl ToString for Hook {
    fn to_string(&self) -> String {
        match self {
            Self::SelectEmbeddingsProperties => "selectEmbeddingProperties".to_string(),
        }
    }
}

impl FromStr for Hook {
    type Err = HookError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let name = s.strip_prefix(HOOK_PREFIX).unwrap_or(s);

        match name {
            "selectEmbeddingProperties" => Ok(Hook::SelectEmbeddingsProperties),
            _ => Err(HookError::InvalidHook(s.to_string())),
        }
    }
}

impl TryFrom<String> for Hook {
    type Error = HookError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Hook::from_str(&value)
    }
}

pub struct WriteHooks {
    hooks_map: DashMap<String, HookValue>,
}

impl WriteHooks {
    pub fn new() -> Self {
        Self {
            hooks_map: DashMap::new(),
        }
    }

    pub fn has_hook(&self, name: Hook) -> bool {
        let key = name.to_key_name();
        self.hooks_map.contains_key(&key)
    }

    pub fn get_hook(&self, name: Hook) -> Option<HookValue> {
        let key = name.to_key_name();
        self.hooks_map.get(&key).map(|ref_| ref_.clone())
    }

    pub fn list_hooks(&self) -> Vec<(String, HookValue)> {
        self.hooks_map
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect()
    }

    pub fn delete_hook(&self, name: Hook) -> Option<(String, HookValue)> {
        self.hooks_map.remove(&name.to_key_name())
    }

    pub fn insert_hook(&self, name: Hook, code: String) -> Result<()> {
        match self.is_valid_js(&code) {
            true => {
                let now = Utc::now();
                self.hooks_map.insert(
                    name.to_key_name(),
                    HookValue {
                        code,
                        created_at: now.timestamp(),
                    },
                );
                Ok(())
            }
            false => anyhow::bail!("The provided JS code is not valid."),
        }
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
