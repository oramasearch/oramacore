use std::future::Future;
use futures::StreamExt;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use crate::types::SearchParams;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(test, derive(PartialEq))]
pub struct PinRule<DocId> {
    pub id: String,
    pub conditions: Vec<Condition>,
    pub consequence: Consequence<DocId>,
}

impl<DocId> PinRule<DocId> {

    pub async fn convert_ids<F, NewId>(self, f: F) -> PinRule<NewId>
    where
        F: Fn(DocId) -> NewId,
    {
        PinRule {
            id: self.id,
            conditions: self.conditions,
            consequence: Consequence {
                promote: self.consequence.promote.into_iter()
                    .map(|item| {
                        let new_doc_id = f(item.doc_id);
                        PromoteItem {
                            doc_id: new_doc_id,
                            position: item.position,
                        }

                    })
                    .collect()
            },
        }
    }

    pub async fn async_convert_ids<F, Fut, NewId>(self, f: F) -> PinRule<NewId>
    where
        F: Fn(DocId) -> Fut + Send,
        Fut: Future<Output = NewId> + Send,
    {
        use futures::StreamExt;

        PinRule {
            id: self.id,
            conditions: self.conditions,
            consequence: Consequence {
                promote: futures::stream::iter(self.consequence.promote)
                    .then(|item| {
                        let doc_id_fut = f(item.doc_id);
                        async move {
                            PromoteItem {
                                doc_id: doc_id_fut.await,
                                position: item.position,
                            }
                        }
                    })
                    .collect()
                    .await,
            },
        }
    }
}

#[cfg(test)]
impl TryFrom<serde_json::Value> for PinRule<String> {
    type Error = serde_json::Error;

    fn try_from(value: serde_json::Value) -> anyhow::Result<Self, Self::Error> {
        serde_json::from_value(value)
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(test, derive(PartialEq))]
pub enum Condition {
    Is { pattern: String },
}

/// Required for the bincode deserialization
/// In fact, bincode doesn't support deserialize_any
/// So, we implemented it manually
#[derive(Serialize, Deserialize, Debug)]
struct SerdeCondition {
    anchoring: String,
    pattern: Option<String>,
}

impl Serialize for Condition {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer
    {
        let c = match self {
            Condition::Is { pattern} => SerdeCondition {
                anchoring: "is".to_string(),
                pattern: Some(pattern.clone()),
            }
        };
        c.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Condition {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>
    {
        println!("AA {}", std::any::type_name::<D>());


        let c = SerdeCondition::deserialize(deserializer)?;

        println!("c {:?}", c);

        match c.anchoring.as_str() {
            "is" => {
                if let Some(pattern) = c.pattern {
                    Ok(Condition::Is { pattern })
                } else {
                    Err(serde::de::Error::custom("Unexpected pattern"))
                }
            },
            _ => Err(serde::de::Error::custom("Unexpected anchoring")),
        }
    }
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
