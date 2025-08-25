use std::{fmt::Debug, path::PathBuf};

use anyhow::Context;
use fs::*;
use thiserror::Error;

use crate::types::DocumentId;

use super::{Condition, Consequence, PinRule, PinRuleOperation};

#[derive(Error, Debug)]
pub enum PinRulesReaderError {
    #[error("Io error {0:?}")]
    Io(std::io::Error),
    #[error("generic {0:?}")]
    Generic(#[from] anyhow::Error),
}

pub struct PinRulesReader {
    data_dir: PathBuf,
    rules: Vec<PinRule<DocumentId>>,
}

impl PinRulesReader {
    pub fn try_new(data_dir: PathBuf) -> Result<Self, PinRulesReaderError> {
        create_if_not_exists(&data_dir)?;

        let rules: Vec<_> = std::fs::read_dir(&data_dir)
            .context("Cannot read pin rules directory")?
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                if path.extension()?.to_str()? == "rules" {
                    let rule: PinRule<DocumentId> = BufferedFile::open(path)
                        .ok()?
                        .read_json_data()
                        .ok()?;
                    Some(rule)
                } else {
                    None
                }
            })
            .collect();

        Ok(Self {
            data_dir,
            rules,
        })
    }

    pub fn update(&mut self, op: PinRuleOperation) -> Result<(), PinRulesReaderError> {
        match op {
            PinRuleOperation::Insert(rule) => {
                self.rules.push(rule);
            }
            PinRuleOperation::Delete(rule_id) => {
                self.rules.retain(|r| &r.id != &rule_id);
            }
        }
        Ok(())
    }

    pub fn commit(&mut self) -> Result<(), PinRulesReaderError> {
        for rule in &self.rules {
            let file_path = self.data_dir.join(format!("{}.rules", rule.id));
            BufferedFile::create_or_overwrite(file_path)
                .context("Cannot create file")?
                .write_json_data(rule)
                .context("Cannot write rule to file")?;
        }

        Ok(())
    }

    /// List all pin rules. Returns the content as well if present.
    pub fn list(&self) -> Result<Vec<PinRule<DocumentId>>, PinRulesReaderError> {
        Ok(self.rules.clone())
    }

    pub fn apply<'s>(&'s self, term: &str) -> Result<Vec<&'s Consequence<DocumentId>>, PinRulesReaderError> {
        let mut results = Vec::new();
        for rule in &self.rules {
            for c in &rule.conditions {
                match c {
                    Condition::Is { pattern } if pattern == term  => {
                        results.push(&rule.consequence);
                        break
                    },
                    _ => continue,
                }
            }
        }
        Ok(results)
    }
}

#[cfg(test)]
mod pin_rules_tests {
    use super::*;

    #[test]
    fn test_pin_rules_reader_empty() {
        let base_dir = generate_new_path();
        let reader = PinRulesReader::try_new(base_dir).expect("Failed to create PinRulesReader");

        // Test listing rules
        let rules = reader.list().expect("Failed to list rules");
        assert!(rules.is_empty());

        // Test applying rules
        let consequences = reader.apply("test").expect("Failed to apply rules");
        assert!(consequences.is_empty());
    }

    #[test]
    fn test_apply_pin_rules() {
        let base_dir = generate_new_path();
        let reader = PinRulesReader::try_new(base_dir.clone()).expect("Failed to create PinRulesReader");

        reader.update(PinRuleOperation::Insert(PinRule {
            id: "test-rule-1".to_string(),
            conditions: vec![
                Condition::Is { pattern: "test".to_string() },
            ],
            consequence: crate::Consequence { promote: vec![
                PromoteItem {
                    doc_id: 1,
                    position: 1,
                }
            ] },
        })).expect("Failed to insert rule");

        let consequences = reader.apply("term").expect("Failed to apply rules");
        assert_eq!(consequences.len(), 0);

        let consequences = reader.apply("test").expect("Failed to apply rules");
        assert_eq!(consequences.len(), 1);
        assert_eq!(consequences[0].promote.len(), 1);
        assert_eq!(consequences[0].promote[0].doc_id, 1);
        assert_eq!(consequences[0].promote[0].position, 1);

        reader.commit().expect("Failed to commit rules");

        let reader = PinRulesReader::try_new(base_dir.clone()).expect("Failed to create PinRulesReader");

        let consequences = reader.apply("term").expect("Failed to apply rules");
        assert_eq!(consequences.len(), 0);

        let consequences = reader.apply("test").expect("Failed to apply rules");
        assert_eq!(consequences.len(), 1);
        assert_eq!(consequences[0].promote.len(), 1);
        assert_eq!(consequences[0].promote[0].doc_id, 1);
        assert_eq!(consequences[0].promote[0].position, 1);

        reader.update(PinRuleOperation::Delete("test-rule-1".to_string())).expect("Failed to delete rule");

        let consequences = reader.apply("test").expect("Failed to apply rules");
        assert_eq!(consequences.len(), 0);

        reader.commit().expect("Failed to commit rules");

        let reader = PinRulesReader::try_new(base_dir.clone()).expect("Failed to create PinRulesReader");

        let consequences = reader.apply("test").expect("Failed to apply rules");
        assert_eq!(consequences.len(), 0);
    }
}
