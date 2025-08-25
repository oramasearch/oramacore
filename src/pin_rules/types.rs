use std::future::Future;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(test, derive(PartialEq))]
pub struct PinRule<DocId> {
    pub id: String,
    pub conditions: Vec<Condition>,
    pub consequence: Consequence<DocId>,
}

impl<DocId> PinRule<DocId> {
    pub async fn convert_ids<F, Fut, NewId>(self, f: F) -> PinRule<NewId>
    where
        F: Fn(DocId) -> Fut + Send,
        Fut: Future<Output = NewId> + Send,
    {
        use futures::StreamExt;

        PinRule {
            id: self.id,
            conditions: self.conditions,
            consequence: Consequence {
                promote: futures::stream::iter(self.consequence.promote).then(|item| {
                    let doc_id_fut = f(item.doc_id);
                    async move {
                        
                        PromoteItem { doc_id: doc_id_fut.await, position: item.position }
                    }
                }).collect().await,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(test, derive(PartialEq))]
#[serde(tag = "anchoring")]
pub enum Condition {
    #[serde(rename = "is")]
    Is { pattern: String },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Anchoring {
    Is,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(test, derive(PartialEq))]
pub struct Consequence<DocId> {
    pub promote: Vec<PromoteItem<DocId>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(test, derive(PartialEq))]
pub struct PromoteItem<DocId> {
    pub doc_id: DocId,
    pub position: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pin_rule_deserialization() {
        let json = r#"{
            "id": "promote-red-jacket",
            "conditions": [
                {
                    "pattern": "red jacket",
                    "anchoring": "is"
                }
            ],
            "consequence": {
                "promote": [
                    {
                        "doc_id": "JACKET42",
                        "position": 1
                    },
                    {
                        "doc_id": "PANTS77",
                        "position": 2
                    }
                ]
            }
        }"#;

        let pin_rule: PinRule<String> =
            serde_json::from_str(json).expect("Failed to deserialize JSON");

        assert_eq!(pin_rule.id, "promote-red-jacket");
        assert_eq!(pin_rule.conditions.len(), 1);

        match &pin_rule.conditions[0] {
            Condition::Is { pattern } => {
                assert_eq!(pattern, "red jacket");
            }
        }

        assert_eq!(pin_rule.consequence.promote.len(), 2);
        assert_eq!(pin_rule.consequence.promote[0].doc_id, "JACKET42");
        assert_eq!(pin_rule.consequence.promote[0].position, 1);
        assert_eq!(pin_rule.consequence.promote[1].doc_id, "PANTS77");
        assert_eq!(pin_rule.consequence.promote[1].position, 2);
    }
}
