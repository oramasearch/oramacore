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

    pub fn from(map: Vec<(Box<[String]>, (FieldId, FieldType))>) -> Self {
        Self {
            map: map.into_iter().collect(),
        }
    }

    pub fn serialize(&self) -> Vec<(Box<[String]>, (FieldId, FieldType))> {
        self.map.iter().map(|(k, v)| (k.clone(), *v)).collect()
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
        self.map
            .get(&path)
            .map(|(field_id, field_type)| (*field_id, *field_type))
    }
}
