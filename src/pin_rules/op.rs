use serde::{Deserialize, Serialize};

use crate::{pin_rules::PinRule, types::DocumentId};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(test, derive(PartialEq))]
pub enum PinRuleOperation {
    Insert(PinRule<DocumentId>),
    Delete(String),
}
