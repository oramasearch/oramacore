use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
    sync::{atomic::AtomicU16, Arc},
};

use anyhow::anyhow;
use code_index::CodeIndex;
use dashmap::DashMap;
use document_storage::DocumentStorage;
use nlp::{locales::Locale, TextParser};
use ordered_float::NotNan;
use serde_json::Value;
use storage::Storage;
use string_index::{
    scorer::bm25::BM25Score, DocumentBatch, StringIndex
};
use types::{
    CollectionId, DocumentId, DocumentList, FieldId, ScalarType, SearchResult, SearchResultHit,
    StringParser, TokenScore, ValueType,
};

use crate::dto::{CollectionDTO, SearchParams, TypedField};

pub struct Collection {
    pub(crate) id: CollectionId,
    description: Option<String>,

    language: Locale,
    field_id_generator: AtomicU16,

    document_storage: Arc<DocumentStorage>,
    default_parser: Box<dyn StringParser>,

    // Strings
    string_index: StringIndex,
    string_fields: DashMap<String, FieldId>,
    // Code
    code_index: CodeIndex,
    code_fields: DashMap<String, FieldId>,
}

impl Collection {
    pub fn new(
        storage: Arc<Storage>,
        id: CollectionId,
        description: Option<String>,
        language: Locale,
        document_storage: Arc<DocumentStorage>,
        typed_fields: HashMap<String, TypedField>,
    ) -> Self {
        let default_parser = TextParser::from_language(Locale::EN);

        let collection = Collection {
            id,
            description,
            language,
            default_parser: Box::new(default_parser),
            field_id_generator: AtomicU16::new(0),
            document_storage,
            string_index: StringIndex::new(storage.clone()),
            string_fields: Default::default(),
            code_index: CodeIndex::new(),
            code_fields: Default::default(),
        };

        for (field_name, field_type) in typed_fields {
            let field_id = collection.get_field_id(field_name.clone());
            match field_type {
                // TODO: handle text language
                TypedField::Text(_) => {
                    collection.string_fields.insert(field_name, field_id);
                }
                // TODO: handle code language
                TypedField::Code(_) => {
                    collection.code_fields.insert(field_name, field_id);
                }
            }
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
            code_fields: self
                .code_fields
                .iter()
                .map(|e| (e.key().clone(), *e.value()))
                .collect(),
        }
    }

    pub fn insert_batch(&self, document_list: DocumentList) -> Result<(), anyhow::Error> {
        let mut strings: DocumentBatch =
            HashMap::with_capacity(document_list.len());
        let mut codes: HashMap<_, Vec<_>> = HashMap::with_capacity(document_list.len());
        let mut documents = Vec::with_capacity(document_list.len());
        for doc in document_list {
            let mut flatten = doc.into_flatten();
            let schema = flatten.get_field_schema();

            let internal_document_id = self.generate_document_id();

            for (key, field_type) in schema {

                if self.code_fields.contains_key(&key) {
                    let value = match flatten.remove(&key) {
                        Some(Value::String(value)) => value,
                        _ => Err(anyhow!("value is not string. This should never happen"))?,
                    };
                    let field_id = self.get_field_id(key.clone());

                    codes
                        .entry(internal_document_id)
                        .or_default()
                        .push((field_id, value));
                } else if field_type == ValueType::Scalar(ScalarType::String) {
                    // flatten is a copy of the document, so we can remove the key
                    let value = match flatten.remove(&key) {
                        Some(Value::String(value)) => value,
                        _ => Err(anyhow!("value is not string. This should never happen"))?,
                    };
                    let field_id = self.get_field_id(key.clone());

                    let tokens = self.default_parser.tokenize_str_and_stem(&value)?;

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
        self.code_index.insert_multiple(codes)?;

        Ok(())
    }

    pub fn search(&self, search_params: SearchParams) -> Result<SearchResult, anyhow::Error> {
        let boost: HashMap<_, _> = search_params
            .boost
            .into_iter()
            .map(|(field_name, boost)| {
                let field_id = self.get_field_id(field_name);
                (field_id, boost)
            })
            .collect();
        let properties: Vec<_> = match search_params.properties {
            Some(properties) => properties
                .into_iter()
                .map(|p| self.get_field_id(p))
                .collect(),
            None => self.string_fields.iter().map(|e| *e.value()).collect(),
        };


        let string_token_scores = {
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

            let fields_on_search_with_default_parser: Vec<_> = self
                .string_fields
                .iter()
                .filter(|field_id| properties.contains(field_id.value()))
                .map(|field_id| *field_id.value())
                .collect();
            println!(
                "searching for tokens {:?}",
                fields_on_search_with_default_parser,
            );

            self.string_index.search(
                tokens,
                Some(fields_on_search_with_default_parser),
                boost.clone(),
                BM25Score::default(),
            )
        }?;

        let code_token_scores = {
            let properties_on_code = self.code_fields.iter()
                .map(|e| *e.value())
                .filter(|field_name| properties.contains(field_name))
                .collect::<Vec<_>>();

            if !properties_on_code.is_empty() {
                self.code_index.search(
                    search_params.term.clone(),
                    Some(properties_on_code),
                    boost,
                )
            } else {
                Default::default()
            }
        };

        let token_scores = {
            let mut token_scores = string_token_scores;
            for (document_id, score) in code_token_scores {
                token_scores
                    .entry(document_id)
                    .and_modify(|s| *s += score)
                    .or_insert(score);
            }

            token_scores
        };

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
    let result: Vec<TokenScore> = heap
        .into_sorted_vec()
        .into_iter()
        .map(|Reverse((value, key))| TokenScore {
            document_id: key,
            score: value.into_inner(),
        })
        .collect();

    result
}
