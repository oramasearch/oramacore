use super::PinRule;
use crate::pin_rules::file_util::{is_rule_file, remove_rule_file};
use anyhow::Context;
use debug_panic::debug_panic;
use fs::*;
use std::{collections::HashMap, fmt::Debug, path::PathBuf};
use thiserror::Error;
use tracing::error;

#[derive(Error, Debug)]
pub enum PinRulesWriterError {
    #[error("Cannot perform operation on FS: {0:?}")]
    FSError(#[from] std::io::Error),
    #[error("Unknown error: {0:?}")]
    Generic(#[from] anyhow::Error),
}

pub struct PinRulesWriter {
    rules: Vec<PinRule<String>>,
    rule_ids_to_delete: Vec<String>,
}

impl PinRulesWriter {
    pub fn empty() -> Result<Self, PinRulesWriterError> {
        Ok(Self {
            rules: Vec::new(),
            rule_ids_to_delete: Vec::new(),
        })
    }

    pub fn try_new(data_dir: PathBuf) -> Result<Self, PinRulesWriterError> {
        create_if_not_exists(&data_dir)?;

        let dir = std::fs::read_dir(data_dir).context("Cannot read dir")?;

        let mut rules = Vec::new();
        for entry in dir {
            let Ok(entry) = entry else {
                debug_panic!("This shouldn't happen");
                error!("Error occurred while trying to read pin rule entry. Rule skipped");
                continue;
            };

            if is_rule_file(&entry.path()) {
                let rule = BufferedFile::open(entry.path())
                    .context("cannot open rules file")?
                    .read_json_data()
                    .context("cannot read rules file")?;
                rules.push(rule);
            }
        }

        Ok(Self {
            rules,
            rule_ids_to_delete: Vec::new(),
        })
    }

    pub fn commit(&mut self, data_dir: PathBuf) -> Result<(), PinRulesWriterError> {
        create_if_not_exists(&data_dir)?;

        for rule in &self.rules {
            let p = data_dir.join(format!("{}.rule", rule.id));
            BufferedFile::create_or_overwrite(p)
                .context("Cannot open file")?
                .write_json_data(rule)
                .context("Cannot write to file")?;
        }

        for rule_id_to_remove in self.rule_ids_to_delete.drain(..) {
            let file_path = data_dir.join(format!("{rule_id_to_remove}.rule"));
            remove_rule_file(file_path);
        }

        Ok(())
    }

    pub async fn insert_pin_rule(
        &mut self,
        rule: PinRule<String>,
    ) -> Result<(), PinRulesWriterError> {
        self.rules.retain(|r| r.id != rule.id);
        self.rule_ids_to_delete.retain(|id| id != &rule.id);
        self.rules.push(rule);

        Ok(())
    }

    pub async fn delete_pin_rule(&mut self, id: &str) -> Result<(), PinRulesWriterError> {
        self.rules.retain(|r| r.id != id);
        self.rule_ids_to_delete.push(id.to_string());

        Ok(())
    }

    pub fn list_pin_rules(&self) -> &[PinRule<String>] {
        &self.rules
    }

    pub fn get_involved_doc_ids(&self) -> Result<HashMap<String, String>, PinRulesWriterError> {
        let rules = self.list_pin_rules();

        let mut ret = HashMap::new();
        for rule in rules {
            for p in &rule.consequence.promote {
                ret.insert(p.doc_id.clone(), rule.id.clone());
            }
        }

        Ok(ret)
    }

    pub fn get_matching_rules(&self, doc_id_str: &str) -> Vec<PinRule<String>> {
        let rules = self.list_pin_rules();

        rules
            .iter()
            .filter(|rule| {
                rule.consequence
                    .promote
                    .iter()
                    .any(|p| p.doc_id == doc_id_str)
            })
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod pin_rules_tests {
    use super::*;
    use crate::pin_rules::Consequence;
    use fs::generate_new_path;

    #[tokio::test]
    async fn test_simple() {
        let path = generate_new_path();

        let mut writer = PinRulesWriter::empty().unwrap();

        writer
            .insert_pin_rule(PinRule {
                id: "test-rule-1".to_string(),
                conditions: vec![],
                consequence: Consequence { promote: vec![] },
            })
            .await
            .unwrap();

        writer
            .insert_pin_rule(PinRule {
                id: "test-rule-2".to_string(),
                conditions: vec![],
                consequence: Consequence { promote: vec![] },
            })
            .await
            .unwrap();

        let rules = writer.list_pin_rules();
        assert_eq!(rules.len(), 2);
        assert_eq!(rules[0].id, "test-rule-1");
        assert_eq!(rules[1].id, "test-rule-2");

        writer.delete_pin_rule("test-rule-1").await.unwrap();

        let rules = writer.list_pin_rules();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].id, "test-rule-2");

        writer.commit(path.clone()).unwrap();

        let writer = PinRulesWriter::try_new(path).unwrap();

        let rules = writer.list_pin_rules();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].id, "test-rule-2");
    }
}
