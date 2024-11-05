use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
    sync::{atomic::AtomicU16, Arc},
};

use anyhow::anyhow;
use dashmap::DashMap;
use document_storage::DocumentStorage;
use nlp::{locales::Locale, TextParser};
use ordered_float::NotNan;
use serde_json::Value;
use storage::Storage;
use string_index::StringIndex;
use types::{
    CollectionId, DocumentId, DocumentList, FieldId, ScalarType, SearchResult, SearchResultHit,
    StringParser, TokenScore, ValueType,
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
    parsers: HashMap<FieldId, Box<dyn StringParser>>,
    default_parser: Box<dyn StringParser>,
}

impl Collection {
    pub fn new(
        storage: Arc<Storage>,
        id: CollectionId,
        description: Option<String>,
        language: Locale,
        document_storage: Arc<DocumentStorage>,
        parsers: HashMap<String, Box<dyn StringParser>>,
    ) -> Self {
        let default_parser = TextParser::from_language(Locale::EN);

        let mut collection = Collection {
            id,
            description,
            language,
            string_fields: Default::default(),
            field_id_generator: AtomicU16::new(0),
            document_storage,
            string_index: StringIndex::new(storage.clone()),
            parsers: Default::default(),
            default_parser: Box::new(default_parser),
        };

        for (field_name, parser) in parsers {
            let field_id = collection.get_field_id(field_name.clone());
            collection.parsers.insert(field_id, parser);
        }

        collection
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
        let mut strings: HashMap<DocumentId, Vec<(FieldId, Vec<(String, Vec<String>)>)>> =
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
                    let field_id = self.get_field_id(key.clone());

                    let parser = self.parsers.get(&field_id).unwrap_or(&self.default_parser);

                    println!("tokenizing doc {doc:?}");
                    let tokens = parser.tokenize_str_and_stem(&value)?;

                    strings
                        .entry(internal_document_id)
                        .or_default()
                        .push((field_id, tokens));
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
        let tokens: Vec<_> = self
            .default_parser
            .tokenize_str_and_stem(&search_params.term)?
            .into_iter()
            .flat_map(|(token, stemmed)| {
                let mut terms = vec![token];
                terms.extend(stemmed);
                terms
            })
            .collect();
        let token_scores = self.string_index.search(tokens, None, None)?;
        let count = token_scores.len();

        let token_scores = top_n(token_scores, search_params.limit.0);

        let docs = self
            .document_storage
            .get_all(token_scores.iter().map(|m| m.document_id).collect())?;

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

    fn get_field_id(&self, field_name: String) -> FieldId {
        if let Some(field_id) = self.string_fields.get(&field_name) {
            return *field_id;
        }

        let field_id = self.string_fields.entry(field_name).or_insert_with(|| {
            let field_id = self
                .field_id_generator
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            FieldId(field_id)
        });

        *field_id
    }
}

fn top_n(map: HashMap<DocumentId, f32>, n: usize) -> Vec<TokenScore> {
    // A min-heap of size `n` to keep track of the top N elements
    let mut heap: BinaryHeap<Reverse<(NotNan<f32>, DocumentId)>> = BinaryHeap::with_capacity(n);

    for (key, value) in map {
        // Insert into the heap if it's not full, or replace the smallest element if the current one is larger
        if heap.len() < n {
            heap.push(Reverse((NotNan::new(value).unwrap(), key)));
        } else if let Some(Reverse((min_value, _))) = heap.peek() {
            if value > *min_value.as_ref() {
                heap.pop();
                heap.push(Reverse((NotNan::new(value).unwrap(), key)));
            }
        }
    }

    // Collect results into a sorted Vec (optional sorting based on descending values)
    let mut result: Vec<TokenScore> = heap
        .into_sorted_vec()
        .into_iter()
        .map(|Reverse((value, key))| TokenScore {
            document_id: key,
            score: value.into_inner(),
        })
        .collect();

    result
}
