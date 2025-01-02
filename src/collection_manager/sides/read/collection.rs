use std::{
    cmp::Reverse,
    collections::{HashMap, HashSet},
    sync::{atomic::AtomicU64, Arc},
};

use anyhow::{anyhow, Context, Result};
use dashmap::DashMap;
use ordered_float::NotNan;
use tokio::join;
use tracing::{debug, error, info, instrument};

use crate::{
    capped_heap::CappedHeap,
    collection_manager::{
        dto::{
            FacetDefinition, FacetResult, FieldId, Filter, SearchMode, SearchParams, SearchResult,
            SearchResultHit, TokenScore, TypedField,
        },
        sides::{document_storage::DocumentStorage, write::InsertStringTerms},
        CollectionId,
    },
    document_storage::DocumentId,
    embeddings::{EmbeddingService, LoadedModel},
    indexes::{
        bool::BoolIndex,
        number::{Number, NumberFilter, NumberIndex},
        string::{BM25Scorer, StringIndex, StringIndexConfig},
        vector::{VectorIndex, VectorIndexConfig},
    },
    metrics::{
        SearchFilterLabels, SearchLabels, SEARCH_FILTER_HISTOGRAM, SEARCH_FILTER_METRIC,
        SEARCH_METRIC,
    },
    nlp::TextParser,
    types::Document,
};

use super::IndexesConfig;

pub struct CollectionReader {
    id: CollectionId,
    embedding_service: Arc<EmbeddingService>,

    document_storage: Arc<dyn DocumentStorage>,

    fields: DashMap<String, (FieldId, TypedField)>,

    // indexes
    vector_index: VectorIndex,
    fields_per_model: DashMap<Arc<LoadedModel>, Vec<FieldId>>,

    string_index: StringIndex,
    number_index: NumberIndex,
    bool_index: BoolIndex,
    // TODO: textparser -> vec<field_id>
}

impl CollectionReader {
    pub fn try_new(
        id: CollectionId,
        embedding_service: Arc<EmbeddingService>,
        document_storage: Arc<dyn DocumentStorage>,
        posting_id_generator: Arc<AtomicU64>,
        indexes_config: IndexesConfig,
    ) -> Result<Self> {
        let collection_data_dir = indexes_config.data_dir.join(&id.0);

        let vector_index = VectorIndex::try_new(VectorIndexConfig {
            base_path: collection_data_dir.join("vectors"),
        })
        .context("Cannot create vector index during collection creation")?;

        let string_index = StringIndex::new(StringIndexConfig {
            posting_id_generator,
            base_path: collection_data_dir.join("strings"),
        });

        let number_index = NumberIndex::try_new(
            collection_data_dir.join("numbers"),
            indexes_config.max_size_per_chunk,
        )?;

        let bool_index = BoolIndex::new();

        Ok(Self {
            id,
            embedding_service,
            document_storage,

            vector_index,
            fields_per_model: Default::default(),

            string_index,

            number_index,

            bool_index,

            fields: Default::default(),
        })
    }

    pub async fn create_field(
        &self,
        field_id: FieldId,
        field_name: String,
        field: TypedField,
    ) -> Result<()> {
        self.fields
            .insert(field_name.clone(), (field_id, field.clone()));

        if let TypedField::Embedding(embedding) = field {
            let orama_model = self
                .embedding_service
                .get_model(embedding.model_name)
                .await?;

            self.vector_index
                .add_field(field_id, orama_model.dimensions())?;

            self.fields_per_model
                .entry(orama_model)
                .or_default()
                .push(field_id);
        };

        Ok(())
    }

    pub fn index_embedding(
        &self,
        doc_id: DocumentId,
        field_id: FieldId,
        value: Vec<f32>,
    ) -> Result<()> {
        // `insert_batch` is designed to process multiple values at once
        // We are inserting only one value, and this is not good for performance
        // We should add an API to accept a single value and avoid the rebuild step
        // Instead, we could move the "rebuild" logic to the `VectorIndex`
        // TODO: do it.
        self.vector_index
            .insert_batch(vec![(doc_id, field_id, vec![value])])
    }

    #[instrument(skip(self, terms), level="debug", fields(self.id = ?self.id))]
    pub fn index_string(
        &self,
        doc_id: DocumentId,
        field_id: FieldId,
        field_length: u16,
        terms: InsertStringTerms,
    ) -> Result<()> {
        self.string_index
            .insert(doc_id, field_id, field_length, terms)?;
        Ok(())
    }

    #[instrument(skip(self, value), level="debug", fields(self.id = ?self.id))]
    pub fn index_number(&self, doc_id: DocumentId, field_id: FieldId, value: Number) -> Result<()> {
        self.number_index.add(doc_id, field_id, value);
        Ok(())
    }

    #[instrument(skip(self, value), level="debug", fields(self.id = ?self.id))]
    pub fn index_boolean(&self, doc_id: DocumentId, field_id: FieldId, value: bool) -> Result<()> {
        self.bool_index.add(doc_id, field_id, value);
        Ok(())
    }

    #[instrument(skip(self), level="debug", fields(self.id = ?self.id))]
    pub async fn insert_document(&self, doc_id: DocumentId, doc: Document) -> Result<()> {
        // self.string_index.increment_total_document().await;
        self.document_storage.add_document(doc_id, doc).await
    }

    fn get_field_id(&self, field_name: String) -> Result<FieldId> {
        let field_id = self.fields.get(&field_name);

        match field_id {
            Some(field_id) => Ok(field_id.0),
            None => Err(anyhow!("Field not found")),
        }
    }

    fn get_field_id_with_type(&self, field_name: &str) -> Result<(FieldId, TypedField)> {
        self.fields
            .get(field_name)
            .map(|v| v.clone())
            .ok_or_else(|| anyhow!("Field not found"))
    }

    fn calculate_boost(&self, boost: HashMap<String, f32>) -> HashMap<FieldId, f32> {
        boost
            .into_iter()
            .filter_map(|(field_name, boost)| {
                let field_id = self.get_field_id(field_name).ok()?;
                Some((field_id, boost))
            })
            .collect()
    }

    fn calculate_filtered_doc_ids(
        &self,
        where_filter: HashMap<String, Filter>,
    ) -> Result<Option<HashSet<DocumentId>>> {
        if where_filter.is_empty() {
            return Ok(None);
        }

        let metric = SEARCH_FILTER_METRIC.create(SearchFilterLabels {
            collection: self.id.0.to_string(),
        });

        let filters: Result<Vec<_>> = where_filter
            .into_iter()
            .map(|(field_name, value)| {
                self.get_field_id_with_type(&field_name)
                    .with_context(|| format!("Unknown field \"{}\"", &field_name))
                    .map(|(field_id, field_type)| (field_name, field_id, field_type, value))
            })
            .collect();
        let mut filters = filters?;
        let last = filters.pop();

        let (field_name, field_id, field_type, filter) = match last {
            Some(v) => v,
            None => return Err(anyhow!("No filter provided")),
        };

        info!(
            "Filtering on field {:?}({:?}): {:?}",
            field_name, field_type, filter
        );

        let mut doc_ids = match (&field_type, filter) {
            (TypedField::Number, Filter::Number(filter_number)) => {
                self.number_index.filter(field_id, filter_number)?
            }
            (TypedField::Bool, Filter::Bool(filter_bool)) => {
                self.bool_index.filter(field_id, filter_bool)?
            }
            _ => {
                error!(
                    "Filter on field {:?}({:?}) not supported",
                    field_name, field_type
                );
                anyhow::bail!(
                    "Filter on field {:?}({:?}) not supported",
                    field_name,
                    field_type
                )
            }
        };
        for (field_name, field_id, field_type, filter) in filters {
            let doc_ids_ = match (&field_type, filter) {
                (TypedField::Number, Filter::Number(filter_number)) => {
                    self.number_index.filter(field_id, filter_number)?
                }
                (TypedField::Bool, Filter::Bool(filter_bool)) => {
                    self.bool_index.filter(field_id, filter_bool)?
                }
                _ => {
                    error!(
                        "Filter on field {:?}({:?}) not supported",
                        field_name, field_type
                    );
                    anyhow::bail!(
                        "Filter on field {:?}({:?}) not supported",
                        field_name,
                        field_type
                    )
                }
            };
            doc_ids = doc_ids.intersection(&doc_ids_).copied().collect();
        }

        drop(metric);

        SEARCH_FILTER_HISTOGRAM
            .create(SearchFilterLabels {
                collection: self.id.0.to_string(),
            })
            .record_usize(doc_ids.len());
        info!("Matching doc from filters: {:?}", doc_ids.len());

        Ok(Some(doc_ids))
    }

    fn calculate_properties(&self, properties: Option<Vec<String>>) -> Result<Vec<FieldId>> {
        let properties: Result<Vec<_>> = match properties {
            Some(properties) => properties
                .into_iter()
                .map(|p| self.get_field_id(p))
                .collect(),
            None => self.fields.iter().map(|e| Ok(e.value().0)).collect(),
        };

        properties
    }

    pub async fn get_total_documents(&self) -> Result<usize> {
        self.document_storage.get_total_documents().await
    }

    #[instrument(skip(self), level="debug", fields(self.id = ?self.id))]
    pub async fn search(&self, search_params: SearchParams) -> Result<SearchResult, anyhow::Error> {
        let metric = SEARCH_METRIC.create(SearchLabels {
            collection: self.id.0.to_string(),
        });
        let SearchParams {
            mode,
            properties,
            boost,
            facets,
            limit,
            where_filter,
        } = search_params;

        let filtered_doc_ids = self.calculate_filtered_doc_ids(where_filter)?;
        let boost = self.calculate_boost(boost);
        let properties = self.calculate_properties(properties)?;

        let token_scores = match mode {
            SearchMode::Default(search_params) | SearchMode::FullText(search_params) => {
                self.search_full_text(&search_params.term, properties, boost, filtered_doc_ids)
                    .await?
            }
            SearchMode::Vector(search_params) => {
                self.search_vector(&search_params.term, filtered_doc_ids)
                    .await?
            }
            SearchMode::Hybrid(search_params) => {
                let (vector, fulltext) = join!(
                    self.search_vector(&search_params.term, filtered_doc_ids.clone()),
                    self.search_full_text(&search_params.term, properties, boost, filtered_doc_ids)
                );
                let vector = vector?;
                let fulltext = fulltext?;

                // min-max normalization
                let max = vector.values().copied().fold(0.0, f32::max);
                let max = max.max(fulltext.values().copied().fold(0.0, f32::max));
                let min = vector.values().copied().fold(0.0, f32::min);
                let min = min.min(fulltext.values().copied().fold(0.0, f32::min));

                let vector: HashMap<_, _> = vector
                    .into_iter()
                    .map(|(k, v)| (k, (v - min) / (max - min)))
                    .collect();

                let mut fulltext: HashMap<_, _> = fulltext
                    .into_iter()
                    .map(|(k, v)| (k, (v - min) / (max - min)))
                    .collect();

                for (k, v) in vector {
                    let e = fulltext.entry(k).or_default();
                    *e += v;
                }
                fulltext
            }
        };

        info!("token_scores len: {:?}", token_scores.len());

        debug!("token_scores: {:?}", token_scores);

        let facets = self.calculate_facets(&token_scores, facets)?;

        let count = token_scores.len();

        let top_results = top_n(token_scores, limit.0);

        let docs = self
            .document_storage
            .get_documents_by_ids(top_results.iter().map(|m| m.document_id).collect())
            .await?;

        let hits: Vec<_> = top_results
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

        drop(metric);

        Ok(SearchResult {
            count,
            hits,
            facets,
        })
    }

    async fn search_full_text(
        &self,
        term: &str,
        properties: Vec<FieldId>,
        boost: HashMap<FieldId, f32>,
        filtered_doc_ids: Option<HashSet<DocumentId>>,
    ) -> Result<HashMap<DocumentId, f32>> {
        let text_parser = TextParser::from_language(crate::nlp::locales::Locale::EN);
        let tokens = text_parser.tokenize(term);

        let mut scorer: BM25Scorer<DocumentId> = BM25Scorer::new();
        self.string_index
            .search(
                tokens,
                // This option is not required.
                // It was introduced because for test purposes we
                // could avoid to pass every properties
                // Anyway the production code should always pass the properties
                // So we could avoid this option
                // TODO: remove this option
                Some(&properties),
                boost,
                &mut scorer,
                filtered_doc_ids.as_ref(),
            )
            .await?;

        Ok(scorer.get_scores())
    }

    async fn search_vector(
        &self,
        term: &str,
        filtered_doc_ids: Option<HashSet<DocumentId>>,
    ) -> Result<HashMap<DocumentId, f32>> {
        let mut ret: HashMap<DocumentId, f32> = HashMap::new();

        for e in &self.fields_per_model {
            let model = e.key();
            let fields = e.value();

            let e = model.embed(vec![term.to_string()], None)?;

            for k in e {
                let r = self.vector_index.search(fields, &k, 1)?;

                for (doc_id, score) in r {
                    if !filtered_doc_ids
                        .as_ref()
                        .map(|f| f.contains(&doc_id))
                        .unwrap_or(true)
                    {
                        continue;
                    }

                    let v = ret.entry(doc_id).or_default();
                    *v += score;
                }
            }
        }

        Ok(ret)
    }

    fn calculate_facets(
        &self,
        token_scores: &HashMap<DocumentId, f32>,
        facets: HashMap<String, FacetDefinition>,
    ) -> Result<Option<HashMap<String, FacetResult>>> {
        if facets.is_empty() {
            Ok(None)
        } else {
            info!("Computing facets on {:?}", facets.keys());

            let mut res_facets: HashMap<String, FacetResult> = HashMap::new();
            for (field_name, facet) in facets {
                let field_id = self.get_field_id(field_name.clone())?;

                // This calculation is inneficient
                // we have the doc_ids that matches:
                // - filters
                // - search
                // We should use them to calculate the facets
                // Instead here we are building an hashset and
                // iter again on it to filter the doc_ids.
                // We could create a dedicated method in the indexes that
                // accepts the matching doc_ids + facet definition and returns the count
                // TODO: do it
                match facet {
                    FacetDefinition::Number(facet) => {
                        let mut values = HashMap::new();

                        for range in facet.ranges {
                            let facet: HashSet<_> = self
                                .number_index
                                .filter(field_id, NumberFilter::Between((range.from, range.to)))?
                                .into_iter()
                                .filter(|doc_id| token_scores.contains_key(doc_id))
                                .collect();

                            values.insert(format!("{}-{}", range.from, range.to), facet.len());
                        }

                        res_facets.insert(
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
                            .filter(field_id, true)?
                            .into_iter()
                            .filter(|doc_id| token_scores.contains_key(doc_id))
                            .collect();
                        let false_facet: HashSet<DocumentId> = self
                            .bool_index
                            .filter(field_id, false)?
                            .into_iter()
                            .filter(|doc_id| token_scores.contains_key(doc_id))
                            .collect();

                        res_facets.insert(
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
            Ok(Some(res_facets))
        }
    }
}

fn top_n(map: HashMap<DocumentId, f32>, n: usize) -> Vec<TokenScore> {
    let mut capped_heap = CappedHeap::new(n);

    for (key, value) in map {
        let k = match NotNan::new(value) {
            Ok(k) => k,
            Err(_) => continue,
        };
        let v = key;
        capped_heap.insert(k, v);
    }

    let result: Vec<TokenScore> = capped_heap
        .into_top()
        .into_iter()
        .map(|Reverse((value, key))| TokenScore {
            document_id: key,
            score: value.into_inner(),
        })
        .collect();

    result
}
