use std::collections::HashMap;

use crate::types::FieldId;

use super::FieldType;

pub struct PathToIndexId {
    map: HashMap<Box<[String]>, (FieldId, FieldType)>,
}

impl PathToIndexId {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    pub fn insert(&mut self, path: Box<[String]>, field_id: FieldId, field_type: FieldType) {
        self.map.insert(path, (field_id, field_type));
    }

    pub fn get(&self, field_name: &str) -> Option<(FieldId, FieldType)> {
        let path = field_name
            .split('.')
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        match self.map.get(&path) {
            None => None,
            Some((field_id, field_type)) => Some((*field_id, *field_type)),
        }
    }
}
