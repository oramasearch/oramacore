use std::hash::Hash;
use crate::collection_manager::sides::read::index::uncommitted_field::{UncommittedBoolField, UncommittedNumberField, UncommittedStringFilterField};
use crate::types::{DocumentId, Number, NumberFilter};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroupValue {
    String(String),
    Number(Number),
    Bool(bool),
}
impl Hash for GroupValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            GroupValue::String(string) => string.hash(state),
            GroupValue::Number(Number::I32(number)) => number.hash(state),
            GroupValue::Number(Number::F32(number)) => {
                f32::to_be_bytes(*number).hash(state);
            },
            GroupValue::Bool(bool) => bool.hash(state),
        }
    }
}

impl Into<serde_json::Value> for GroupValue {
    fn into(self) -> serde_json::Value {
        match self {
            GroupValue::String(string) => serde_json::Value::String(string),
            GroupValue::Number(number) => number.into(),
            GroupValue::Bool(b) => serde_json::Value::Bool(b),
        }
    }
}

pub trait Groupable {
    fn get_values(&self) -> Box<dyn Iterator<Item=GroupValue> + '_>;

    fn get_doc_ids(&self, variant: &GroupValue) -> Box<dyn Iterator<Item=DocumentId> + '_>;
}

impl Groupable for UncommittedBoolField {

    fn get_values(&self) -> Box<dyn Iterator<Item=GroupValue> + '_> {
        Box::new(vec![true, false].into_iter().map(GroupValue::Bool))
    }

    fn get_doc_ids(&self, variant: &GroupValue) -> Box<dyn Iterator<Item=DocumentId> + '_> {
        let GroupValue::Bool(b) = variant else {
            panic!()
        };
        Box::new(self.filter(*b))
    }
}

impl Groupable for UncommittedNumberField {

    fn get_values(&self) -> Box<dyn Iterator<Item=GroupValue> + '_> {
        Box::new(self.iter()
            .map(|(n, _)| n).map(GroupValue::Number))
    }

    fn get_doc_ids(&self, variant: &GroupValue) -> Box<dyn Iterator<Item=DocumentId> + '_> {
        let GroupValue::Number(n) = variant else {
            panic!()
        };
        Box::new(self.filter(&NumberFilter::Equal(*n)))
    }
}

impl Groupable for UncommittedStringFilterField {

    fn get_values(&self) -> Box<dyn Iterator<Item=GroupValue> + '_> {
        Box::new(self.iter()
            .map(|(n, _)| n).map(GroupValue::String))
    }

    fn get_doc_ids(&self, variant: &GroupValue) -> Box<dyn Iterator<Item=DocumentId> + '_> {
        let GroupValue::String(s) = variant else {
            panic!()
        };
        Box::new(self.filter(s))
    }
}