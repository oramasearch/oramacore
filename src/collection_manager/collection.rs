use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap, HashSet},
    sync::{
        atomic::{AtomicU16, AtomicU64},
        Arc,
    },
};

use crate::{
    collection_manager::dto::{FacetResult, SearchResultHit},
    document_storage::{DocumentId, DocumentStorage},
    embeddings::{EmbeddingService, LoadedModel},
    indexes::{
        bool::BoolIndex,
        code::CodeIndex,
        number::{Number, NumberFilter, NumberIndex},
        string::{scorer::bm25::BM25Score, DocumentBatch, StringIndex},
        vector::VectorIndexConfig,
    },
    nlp::locales::Locale,
};
use crate::{indexes::vector::VectorIndex, nlp::TextParser};
use anyhow::{anyhow, Context, Result};
use dashmap::DashMap;
use num_traits::ToPrimitive;

use ordered_float::NotNan;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::info;

use crate::types::{Document, DocumentList, ScalarType, StringParser, ValueType};

use super::{
    dto::{
        CollectionDTO, EmbeddingTypedField, FacetDefinition, Filter, SearchParams, SearchResult,
        TypedField,
    },
    CollectionId,
};

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldId(pub u16);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenScore {
    pub document_id: DocumentId,
    pub score: f32,
}

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
    // Number
    number_index: NumberIndex,
    // Bool
    bool_index: BoolIndex,

    // Vector
    vector_fields: DashMap<String, (FieldId, EmbeddingTypedField)>,
    embedding_models: DashMap<FieldId, Arc<LoadedModel>>,
    vector_index: VectorIndex,
}

impl Collection {
    pub async fn try_new(
        id: CollectionId,
        description: Option<String>,
        language: Locale,
        document_storage: Arc<DocumentStorage>,
        typed_fields: HashMap<String, TypedField>,
        id_generator: Arc<AtomicU64>,
        embedding_service: Arc<EmbeddingService>,
    ) -> Result<Self> {
        let default_parser = TextParser::from_language(Locale::EN);

        let vector_index = VectorIndex::try_new(VectorIndexConfig {})?;

        let collection = Collection {
            id,
            description,
            language,
            default_parser: Box::new(default_parser),
            field_id_generator: AtomicU16::new(0),
            document_storage,
            string_index: StringIndex::new(id_generator),
            string_fields: Default::default(),
            code_index: CodeIndex::new(),
            code_fields: Default::default(),
            number_index: Default::default(),
            bool_index: Default::default(),

            vector_fields: Default::default(),
            vector_index,
            embedding_models: Default::default(),
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
                TypedField::Embedding(embedding) => {
                    let model = embedding_service
                        .get_model(embedding.model_name.clone())
                        .await
                        .with_context(|| {
                            format!(
                                "Unable to find a model named \"{}\"",
                                embedding.model_name.clone()
                            )
                        })?
                        .clone();

                    collection
                        .vector_index
                        .add_field(field_id, model)
                        .with_context(|| format!("Cannot add field \"{}\"", field_name))?;
                    let model = 
                        embedding_service.get_model(embedding.model_name.clone()).await?;
                    collection.embedding_models.insert(field_id, model);
                    collection
                        .vector_fields
                        .insert(field_name, (field_id, embedding));
                }
            }
        }

        Ok(collection)
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

    pub async fn insert_batch(&self, document_list: DocumentList) -> Result<(), anyhow::Error> {
        info!("Inserting batch of {} documents", document_list.len());

        let mut strings: DocumentBatch = HashMap::with_capacity(document_list.len());
        let mut codes: HashMap<_, Vec<_>> = HashMap::with_capacity(document_list.len());
        let mut numbers = Vec::new();
        let mut bools = Vec::new();
        let mut embeddings: Vec<(DocumentId, FieldId, Vec<String>)> = Vec::new();
        let mut documents = Vec::with_capacity(document_list.len());
        for doc in document_list {
            let mut flatten = doc.into_flatten();
            let schema = flatten.get_field_schema();

            let internal_document_id = self.generate_document_id();

            for e in self.vector_fields.iter() {
                let (field_id, embedding) = e.value();

                embeddings.push((
                    internal_document_id,
                    *field_id,
                    embedding
                        .document_fields
                        .iter()
                        .filter_map(|f| {
                            flatten
                                .get(f)
                                .and_then(|v| v.as_str())
                                .map(|v| v.to_string())
                        })
                        .collect(),
                ))
            }

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
                    // TODO: avoid this "id" hard coded
                    if key == "id" {
                        let value = match flatten.remove(&key) {
                            Some(Value::String(value)) => value,
                            _ => Err(anyhow!("value is not string. This should never happen"))?,
                        };

                        let field_id = self.get_field_id(key.clone());

                        strings
                            .entry(internal_document_id)
                            .or_default()
                            .push((field_id, vec![(value, vec![])]));
                    } else {
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
                } else if field_type == ValueType::Scalar(ScalarType::Number) {
                    let value = match flatten.remove(&key) {
                        Some(Value::Number(value)) => value,
                        _ => Err(anyhow!("value is not number. This should never happen"))?,
                    };

                    let v: Option<Number> = value
                        .as_i64()
                        .and_then(|v| v.to_i32())
                        .map(Number::from)
                        .or_else(|| value.as_f64().and_then(|v| v.to_f32()).map(Number::from));
                    let v = match v {
                        Some(v) => v,
                        // TODO: handle better the error
                        None => continue,
                    };

                    let field_id = self.get_field_id(key.clone());
                    numbers.push((internal_document_id, field_id, v));
                } else if field_type == ValueType::Scalar(ScalarType::Boolean) {
                    let value = match flatten.remove(&key) {
                        Some(Value::Bool(value)) => value,
                        _ => Err(anyhow!("value is not bool. This should never happen"))?,
                    };

                    let field_id = self.get_field_id(key.clone());
                    bools.push((internal_document_id, field_id, value));
                }
            }

            documents.push((internal_document_id, doc));
        }

        // TODO: if the insert_multiple fails, should we rollback the `add_documents`?
        self.document_storage.add_documents(documents).await?;

        self.string_index.insert_multiple(strings).await?;
        self.code_index.insert_multiple(codes)?;

        for (doc_id, field_id, value) in numbers {
            self.number_index.add(doc_id, field_id, value);
        }
        for (doc_id, field_id, value) in bools {
            self.bool_index.add(doc_id, field_id, value);
        }

        let mut v = Vec::new();
        for (doc_id, field_id, values) in embeddings {
            let model = self.embedding_models.get(&field_id)
                .ok_or_else(|| anyhow!("Model not found"))?
                .clone();
            let embedding = model.embed(values, Some(1))?;

            v.push((doc_id, field_id, embedding));
        }
        self.vector_index.insert_batch(v)?;

        Ok(())
    }

    pub async fn search(&self, search_params: SearchParams) -> Result<SearchResult, anyhow::Error> {
        info!("Searching with params: {:?}", search_params);

        let filtered_doc_ids = if search_params.where_filter.is_empty() {
            None
        } else {
            let mut filters: Vec<_> = search_params
                .where_filter
                .into_iter()
                .map(|(field_name, value)| {
                    let field_id = self.get_field_id(field_name);
                    (field_id, value)
                })
                .collect();
            let (field_id, filter) = filters
                .pop()
                .expect("where condition has to not be empty here.");

            let mut doc_ids = match filter {
                Filter::Number(filter_number) => self.number_index.filter(field_id, filter_number),
                Filter::Bool(filter_bool) => self.bool_index.filter(field_id, filter_bool),
            };
            for (field_id, filter) in filters {
                let doc_ids_ = match filter {
                    Filter::Number(filter_number) => {
                        self.number_index.filter(field_id, filter_number)
                    }
                    Filter::Bool(filter_bool) => self.bool_index.filter(field_id, filter_bool),
                };
                doc_ids = doc_ids.intersection(&doc_ids_).copied().collect();
            }

            info!("Matching doc from filters: {:?}", doc_ids);

            Some(doc_ids)
        };

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
                .filter(|field_id| field_id.key() != "id") // TODO: remove this
                .filter(|field_id| properties.contains(field_id.value()))
                .map(|field_id| *field_id.value())
                .collect();

            let mut output = self
                .string_index
                .search(
                    tokens,
                    Some(fields_on_search_with_default_parser),
                    boost.clone(),
                    BM25Score::default(),
                    filtered_doc_ids.as_ref(),
                )
                .await?;

            let id_field_id = self.get_field_id("id".to_string());
            if properties.contains(&id_field_id) {
                let id_output = self
                    .string_index
                    .search(
                        vec![search_params.term.clone()],
                        Some(vec![id_field_id]),
                        boost.clone(),
                        BM25Score::default(),
                        filtered_doc_ids.as_ref(),
                    )
                    .await?;

                for (key, v) in id_output {
                    let vv = output.entry(key).or_default();
                    *vv += v;
                }
            }

            Result::<HashMap<_, _>, anyhow::Error>::Ok(output)
        }?;

        let code_token_scores = {
            let properties_on_code = self
                .code_fields
                .iter()
                .map(|e| *e.value())
                .filter(|field_name| properties.contains(field_name))
                .collect::<Vec<_>>();

            if !properties_on_code.is_empty() {
                self.code_index.search(
                    search_params.term.clone(),
                    Some(properties_on_code),
                    boost,
                    filtered_doc_ids.as_ref(),
                )?
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

        info!("Document that match the term: {:?}", token_scores.len());

        let facets = if search_params.facets.is_empty() {
            None
        } else {
            info!("Computing facets on {:?}", search_params.facets.keys());

            let mut facets = HashMap::new();
            for (field_name, facet) in search_params.facets {
                let field_id = self.get_field_id(field_name.clone());
                match facet {
                    FacetDefinition::Number(facet) => {
                        let mut values = HashMap::new();

                        for range in facet.ranges {
                            let facet: HashSet<_> = self
                                .number_index
                                .filter(field_id, NumberFilter::Between(range.from, range.to))
                                .into_iter()
                                .filter(|doc_id| token_scores.contains_key(doc_id))
                                .collect();

                            values.insert(format!("{}-{}", range.from, range.to), facet.len());
                        }

                        facets.insert(
                            field_name,
                            FacetResult {
                                count: values.len(),
                                values,
                            },
                        );
                    }
                    FacetDefinition::Bool => {
                        let true_facet: HashSet<DocumentId> = self
                            .bool_index
                            .filter(field_id, true)
                            .into_iter()
                            .filter(|doc_id| token_scores.contains_key(doc_id))
                            .collect();
                        let false_facet: HashSet<DocumentId> = self
                            .bool_index
                            .filter(field_id, false)
                            .into_iter()
                            .filter(|doc_id| token_scores.contains_key(doc_id))
                            .collect();

                        facets.insert(
                            field_name,
                            FacetResult {
                                count: 2,
                                values: HashMap::from_iter([
                                    ("true".to_string(), true_facet.len()),
                                    ("false".to_string(), false_facet.len()),
                                ]),
                            },
                        );
                    }
                }
            }
            Some(facets)
        };

        let count = token_scores.len();

        let token_scores = top_n(token_scores, search_params.limit.0);

        let docs = self
            .document_storage
            .get_all(token_scores.iter().map(|m| m.document_id).collect())
            .await?;

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

        Ok(SearchResult {
            count,
            hits,
            facets,
        })
    }

    fn generate_document_id(&self) -> DocumentId {
        self.document_storage.generate_document_id()
    }

    fn get_field_id(&self, field_name: String) -> FieldId {
        if let Some(field_id) = self.string_fields.get(&field_name) {
            return *field_id;
        }

        let field_id = self
            .string_fields
            .entry(field_name.clone())
            .or_insert_with(|| {
                let field_id = self
                    .field_id_generator
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                FieldId(field_id)
            });

        info!(
            "Created new field id for \"{}\": {:?}",
            field_name, *field_id
        );

        *field_id
    }

    pub async fn get_doc_by_unique_field(
        &self,
        field_name: String,
        value: String,
    ) -> Result<Option<Document>> {
        let field_id = match self.string_fields.get(&field_name) {
            Some(field_id) => *field_id,
            None => return Ok(None),
        };

        let output = self
            .string_index
            .search(
                vec![value],
                Some(vec![field_id]),
                Default::default(),
                BM25Score::default(),
                None,
            )
            .await?;

        let doc_id = dbg!(match output.into_keys().next() {
            Some(doc_id) => doc_id,
            None => return Ok(None),
        });

        self.document_storage.get(doc_id).await
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
