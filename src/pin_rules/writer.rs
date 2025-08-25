use fs::*;
use std::{fmt::Debug, path::PathBuf};
use thiserror::Error;

use super::PinRule;

#[derive(Error, Debug)]
pub enum PinRulesWriterError {
    #[error("Cannot perform operation on FS: {0:?}")]
    FSError(#[from] std::io::Error),
    #[error("Unknown error: {0:?}")]
    Generic(#[from] anyhow::Error),
}

pub struct PinRulesWriter {
    base_dir: PathBuf,
}

impl PinRulesWriter {
    pub fn try_new(base_dir: PathBuf) -> Result<Self, PinRulesWriterError> {
        create_if_not_exists(&base_dir)?;

        Ok(Self {
            base_dir,
        })
    }

    pub async fn insert_pin_rule(&self, rule: PinRule<String>) -> Result<(), PinRulesWriterError> {
        let id = &rule.id;
        let path = self.base_dir.join(format!("{}.rules", id));
        BufferedFile::create_or_overwrite(path)?.write_json_data(&rule)?;

        Ok(())
    }

    pub async fn delete_pin_rule(&self, id: &str) -> Result<(), PinRulesWriterError> {
        let path = self.base_dir.join(format!("{}.rules", id));
        match std::fs::remove_file(&path) {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // File does not exist, treat as success
            }
            Err(e) => return Err(PinRulesWriterError::FSError(e)),
        };

        Ok(())
    }

    pub fn list_pin_rules(&self) -> Result<Vec<PinRule<String>>, PinRulesWriterError> {
        let r = std::fs::read_dir(&self.base_dir)?;
        let mut ret: Vec<PinRule<String>> = Vec::new();
        for entry in r {
            let entry = entry?;
            let path = entry.path();
            if path.extension() == Some(std::ffi::OsStr::new("rules")) {
                let content = match BufferedFile::open(path)
                    .and_then(|f| f.read_json_data())
                    {
                        Ok(c) => c,
                        Err(e) => {
                            return Err(PinRulesWriterError::Generic(e));
                        }
                    };
                ret.push(content);
            }
        }

        // Stable rules order
        ret.sort_by(|a, b| a.id.cmp(&b.id));

        Ok(ret)
    }
}

#[cfg(test)]
mod pin_rules_tests {
    use super::*;
    use fs::generate_new_path;
    use futures::FutureExt;

    #[tokio::test]
    async fn test_simple() {
        let path = generate_new_path();

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<PinRuleOperation<u64>>();

        let writer = PinRulesWriter::try_new(path)
        .unwrap();

        writer.insert_pin_rule(PinRule {
            id: "test-rule-1".to_string(),
            conditions: vec![],
            consequence: crate::Consequence { promote: vec![] },
        }).await.unwrap();

        writer.insert_pin_rule(PinRule {
            id: "test-rule-2".to_string(),
            conditions: vec![],
            consequence: crate::Consequence { promote: vec![] },
        }).await.unwrap();

        let rules = writer.list_pin_rules().unwrap();
        assert_eq!(rules.len(), 2);
        assert_eq!(rules[0].id, "test-rule-1");
        assert_eq!(rules[1].id, "test-rule-2");

        writer.delete_pin_rule("test-rule-1".to_string()).await.unwrap();

        let rules = writer.list_pin_rules().unwrap();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].id, "test-rule-2");
    }
}