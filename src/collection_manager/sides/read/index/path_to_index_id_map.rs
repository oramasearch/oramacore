use std::collections::HashMap;

use crate::{collection_manager::sides::field_name_to_path, types::FieldId};

use super::FieldType;

#[derive(Debug)]
pub struct PathToIndexId {
    filter_fields: HashMap<Box<[String]>, (FieldId, FieldType)>,
    score_fields: HashMap<Box<[String]>, (FieldId, FieldType)>,
}

impl PathToIndexId {
    pub fn empty() -> Self {
        Self {
            filter_fields: HashMap::new(),
            score_fields: HashMap::new(),
        }
    }

    pub fn new(
        filter_fields: HashMap<Box<[String]>, (FieldId, FieldType)>,
        score_fields: HashMap<Box<[String]>, (FieldId, FieldType)>,
    ) -> Self {
        Self {
            filter_fields,
            score_fields,
        }
    }

    pub fn insert_filter_field(
        &mut self,
        path: Box<[String]>,
        field_id: FieldId,
        field_type: FieldType,
    ) {
        self.filter_fields.insert(path, (field_id, field_type));
    }

    pub fn insert_score_field(
        &mut self,
        path: Box<[String]>,
        field_id: FieldId,
        field_type: FieldType,
    ) {
        self.score_fields.insert(path, (field_id, field_type));
    }

    pub fn get(&self, field_name: &str) -> Option<(FieldId, FieldType)> {
        let path = field_name_to_path(field_name);
        self.score_fields
            .get(&path)
            .map(|(field_id, field_type)| (*field_id, *field_type))
            .or_else(|| {
                self.filter_fields
                    .get(&path)
                    .map(|(field_id, field_type)| (*field_id, *field_type))
            })
    }

    pub fn get_filter_field(&self, field_name: &str) -> Option<(FieldId, FieldType)> {
        let path = field_name_to_path(field_name);
        self.filter_fields
            .get(&path)
            .map(|(field_id, field_type)| (*field_id, *field_type))
    }
}
