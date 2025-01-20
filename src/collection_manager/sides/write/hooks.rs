use anyhow::Result;
use chrono::Utc;
use dashmap::DashMap;
use oxc_allocator::Allocator;
use oxc_parser::Parser;
use oxc_span::SourceType;
use serde::Serialize;

#[derive(Serialize, Clone)]
struct HookValue {
    pub code: String,
    pub created_at: i64,
}

#[derive(Serialize)]
pub enum Hook {
    SelectEmbeddingsProperties,
}

impl Hook {
    // @todo: this may not be necessary, but we may want to prefix "official" hooks with "hook:" to distingish them from
    // user defined hooks that are not part of the specs.
    pub fn to_key_name(&self) -> String {
        match self {
            Self::SelectEmbeddingsProperties => "hook:selectEmbeddingProperties".to_string(),
        }
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

    pub fn get_hook(&self, name: Hook) -> Option<HookValue> {
        let key = name.to_key_name();
        self.hooks_map.get(&key).map(|ref_| ref_.clone())
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
