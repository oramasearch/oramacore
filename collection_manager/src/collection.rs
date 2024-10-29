use std::{collections::HashMap, sync::{atomic::AtomicUsize, Arc}};

use anyhow::{anyhow, Context};
use dashmap::DashMap;
use serde_json::Value;
use storage::Storage;
use string_index::{DocumentId, FieldId, StringIndex};
use string_utils::{Language, Parser};

use crate::{document::{Document, DocumentList, ScalarType, ValueType}, dto::CollectionDTO};

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct CollectionId(pub String);

pub struct Collection {
    document_id_generator: AtomicUsize,
    pub(crate) id: CollectionId,
    description: Option<String>,
    string_index: StringIndex,
    language: Language,
    field_id_generator: AtomicUsize,
    string_fields: DashMap<String, FieldId>
}

impl Collection {
    pub fn new(
        storage: Arc<Storage>,
        id: CollectionId,
        description: Option<String>,
        language: Language,
    ) -> Self {
        let parser = Parser::from_language(language);
        Collection {
            id,
            description,
            string_index: StringIndex::new(storage, parser),
            language,
            string_fields: Default::default(),
            document_id_generator: AtomicUsize::new(0),
            field_id_generator: AtomicUsize::new(0),
        }
    }

    pub fn as_dto(&self) -> CollectionDTO {
        CollectionDTO {
            id: self.id.0.clone(),
            description: self.description.clone(),
            language: self.language.into(),
        }
    }

    pub fn insert_batch(&self, document_list: DocumentList) -> Result<(), anyhow::Error> {
        let mut strings_per_field: HashMap<FieldId, Vec<(DocumentId, String)>> = HashMap::with_capacity(self.string_fields.len());

        for doc in document_list.0 {
            let mut flatten = doc.into_flatten();
            let schema = flatten.get_field_schema();

            let internal_document_id = self.generate_document_id();

            for (key, field_type) in schema.0 {
                if field_type == ValueType::Scalar(ScalarType::String) {
                    // Remove from the original document because we are consuming it (the above `for` loop)
                    // Because `schema` is calculated on the current document, this removal is safe
                    // and never fails.
                    let value = match flatten.0.remove(&key) {
                        Some(Value::String(value)) => value,
                        _ => Err(anyhow!("value is not string. This should never happen"))?,
                    };
                    let field_id = self.string_fields.entry(key.clone()).or_insert_with(|| {
                        self.create_field()
                    });

                    strings_per_field.entry(*field_id).or_default().push((internal_document_id, value));
                }
            }
        }

        println!("Inserting batch {:?}", strings_per_field);

        for (field_id, batch) in strings_per_field {
            self.string_index.insert_multiple(field_id, batch)?;
        }

        Ok(())
    }

    fn generate_document_id(&self) -> DocumentId {
        let id = self.document_id_generator.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        DocumentId(id)
    }

    fn create_field(&self) -> FieldId {
        let field_id = self.field_id_generator.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        FieldId(field_id)
    }
}
