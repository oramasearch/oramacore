use std::{
    collections::HashMap,
    sync::{atomic::AtomicU16, Arc},
};

use anyhow::anyhow;
use dashmap::DashMap;
use document_storage::DocumentStorage;
use nlp::Parser;
use nlp::locales::Locale;
use serde_json::Value;
use storage::Storage;
use string_index::StringIndex;
use types::{
    CollectionId, DocumentId, DocumentList, FieldId, ScalarType, SearchResult, SearchResultHit,
    ValueType,
};

use crate::dto::{CollectionDTO, SearchParams};

pub struct Collection {
    pub(crate) id: CollectionId,
    description: Option<String>,
    string_index: StringIndex,
    language: Locale,
    field_id_generator: AtomicU16,
    string_fields: DashMap<String, FieldId>,
    document_storage: Arc<DocumentStorage>,
}

impl Collection {
    pub fn new(
        storage: Arc<Storage>,
        id: CollectionId,
        description: Option<String>,
        language: Locale,
        document_storage: Arc<DocumentStorage>,
    ) -> Self {
        let parser = Parser::from_language(language);
        Collection {
            id,
            description,
            string_index: StringIndex::new(storage, parser),
            language,
            string_fields: Default::default(),
            field_id_generator: AtomicU16::new(0),
            document_storage,
        }
    }

    pub fn as_dto(&self) -> CollectionDTO {
        CollectionDTO {
            id: self.id.0.clone(),
            description: self.description.clone(),
            language: self.language.into(),
            document_count: self.string_index.get_total_documents(),
            string_fields: self
                .string_fields
                .iter()
                .map(|e| (e.key().clone(), *e.value()))
                .collect(),
        }
    }

    pub fn insert_batch(&self, document_list: DocumentList) -> Result<(), anyhow::Error> {
        let mut strings: HashMap<DocumentId, Vec<(FieldId, String)>> =
            HashMap::with_capacity(self.string_fields.len());
        let mut documents = Vec::with_capacity(document_list.len());
        for doc in document_list {
            let mut flatten = doc.into_flatten();
            let schema = flatten.get_field_schema();

            let internal_document_id = self.generate_document_id();

            for (key, field_type) in schema {
                if field_type == ValueType::Scalar(ScalarType::String) {
                    // flatten is a copy of the document, so we can remove the key
                    let value = match flatten.remove(&key) {
                        Some(Value::String(value)) => value,
                        _ => Err(anyhow!("value is not string. This should never happen"))?,
                    };
                    let field_id = self
                        .string_fields
                        .entry(key.clone())
                        .or_insert_with(|| self.create_field());

                    strings
                        .entry(internal_document_id)
                        .or_default()
                        .push((*field_id, value));
                }
            }

            documents.push((internal_document_id, doc));
        }

        // TODO: if the insert_multiple fails, should we rollback the `add_documents`?
        self.document_storage.add_documents(documents)?;
        self.string_index.insert_multiple(strings)?;

        Ok(())
    }

    pub fn search(&self, search_params: SearchParams) -> Result<SearchResult, anyhow::Error> {
        let (token_scores, count) = self.string_index.search(&search_params.term, 10, None)?;

        let docs = self
            .document_storage
            .get_all(token_scores.iter().map(|m| m.document_id).collect())?;

        println!("{:#?}", docs);

        let hits: Vec<_> = token_scores
            .into_iter()
            .zip(docs)
            .map(|(token_score, document)| {
                let id = document
                    .as_ref()
                    .and_then(|d| d.get("id"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_default();
                SearchResultHit {
                    id,
                    score: token_score.score,
                    document,
                }
            })
            .collect();

        Ok(SearchResult { count, hits })
    }

    fn generate_document_id(&self) -> DocumentId {
        self.document_storage.generate_document_id()
    }

    fn create_field(&self) -> FieldId {
        let field_id = self
            .field_id_generator
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        FieldId(field_id)
    }
}
