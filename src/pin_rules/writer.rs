use fs::*;
use std::{collections::HashMap, fmt::Debug, path::PathBuf};
use anyhow::Context;
use debug_panic::debug_panic;
use thiserror::Error;
use tracing::error;
use super::PinRule;

#[derive(Error, Debug)]
pub enum PinRulesWriterError {
    #[error("Cannot perform operation on FS: {0:?}")]
    FSError(#[from] std::io::Error),
    #[error("Unknown error: {0:?}")]
    Generic(#[from] anyhow::Error),
}

pub struct PinRulesWriter {
    rules: Vec<PinRule<String>>,
    rules_to_delete: Vec<String>,
}

impl PinRulesWriter {
    pub fn empty() -> Result<Self, PinRulesWriterError> {
        Ok(Self {
            rules: Vec::new(),
            rules_to_delete: Vec::new(),
        })
    }

    pub fn try_new(
        data_dir: PathBuf
    ) -> Result<Self, PinRulesWriterError> {
        create_if_not_exists(&data_dir)?;

        let dir = std::fs::read_dir(data_dir)
            .context("Cannot read dir")?;

        let mut rules = Vec::new();
        for entry in dir {
            let Ok(entry) = entry else {
                debug_panic!("This shouldn't happen");
                continue;
            };
            let file_name = entry.file_name();

            if entry.path().extension().map(|os| os.as_encoded_bytes()) == Some("rules".as_bytes()) {
                let rule = BufferedFile::open(entry.path())
                    .context("cannot open rules file")?
                    .read_json_data()
                    .context("cannot read rules file")?;
                rules.push(rule);
            }
        }

        Ok(Self {
            rules,
            rules_to_delete: Vec::new(),
        })
    }

    pub fn commit(&mut self, data_dir: PathBuf) -> Result<(), PinRulesWriterError> {
        create_if_not_exists(&data_dir)?;

        let mut rules: Vec<PinRule<String>> = self.rules.drain(..).collect();
        rules.reverse();

        let mut r = HashMap::with_capacity(rules.len());
        for rule in rules {
            if r.contains_key(&rule.id) {
                continue;
            }
            r.insert(rule.id.clone(), rule);
        }

        // Add "rule-id-1" -> Delete "rule-id-1" -> Add "rule-id-1"
        // should work correctly. IE we shouldn't remove a rule if it is in `r`
        self.rules_to_delete.retain(|rule_id| {
            !r.contains_key(rule_id)
        });

        self.rules = r.into_values().collect();

        for rule in &self.rules {
            let p = data_dir.join(format!("{}.rule", rule.id));
            BufferedFile::create_or_overwrite(p)
                .context("Cannot open file")?
                .write_json_data(rule)
                .context("Cannot write to file")?;
        }

        for rule_id in self.rules_to_delete.drain(..) {
            let p = data_dir.join(format!("{}.rule", rule_id));
            if let Err(e) = std::fs::remove_file(&p).context("Cannot remove file") {
                error!(error = ?e, "Cannot remove rule file {:?} from disk", p);
            }
        }

        Ok(())
    }

    pub async fn insert_pin_rule(&mut self, rule: PinRule<String>) -> Result<(), PinRulesWriterError> {
        self.rules.retain(|r| r.id != rule.id);
        self.rules.push(rule);
        Ok(())
    }

    pub async fn delete_pin_rule(&mut self, id: &str) -> Result<(), PinRulesWriterError> {
        self.rules.retain(|r| r.id != id);
        self.rules_to_delete.push(id.to_string());

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

    pub fn get_matching_rules(
        &self,
        doc_id_str: &str,
    ) -> Result<Vec<PinRule<String>>, PinRulesWriterError> {
        let rules = self.list_pin_rules();

        Ok(rules.into_iter()
            .filter(|rule| rule.consequence.promote.iter().any(|p| p.doc_id == doc_id_str))
            .cloned()
            .collect())
    }
}

#[cfg(test)]
mod pin_rules_tests {
    use super::*;
    use crate::pin_rules::{PinRuleOperation, Consequence};
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

        let mut rules =  PinRulesWriter::try_new(path).unwrap();

        let rules = writer.list_pin_rules();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].id, "test-rule-2");
    }
}
