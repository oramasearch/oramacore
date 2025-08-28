pub mod histogram;
use metrics::{Label, SharedString};

use crate::types::FieldId;

pub struct Empty;
impl From<Empty> for Vec<Label> {
    fn from(_: Empty) -> Self {
        vec![]
    }
}

impl From<FieldId> for SharedString {
    fn from(val: FieldId) -> Self {
        SharedString::from(val.0.to_string())
    }
}
