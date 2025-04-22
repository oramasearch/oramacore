use committed_field::{
    CommittedBoolField, CommittedNumberField, CommittedStringField, CommittedStringFilterField,
    CommittedVectorField,
};
use futures::join;
use path_to_index_id_map::PathToIndexId;
use serde::Serialize;
use tokio::sync::RwLock;
use tracing::{debug, info, trace, warn};
use uncommitted_field::{
    bool::UncommittedBoolField, number::UncommittedNumberField, string::UncommittedStringField,
    string_filter::UncommittedStringFilterField, vector::UncommittedVectorField,
};

use crate::{
    ai::{llms, AIService},
    collection_manager::bm25::BM25Scorer,
    metrics::{
        search::{
            FILTER_COUNT_CALCULATION_COUNT, FILTER_PERC_CALCULATION_COUNT,
            MATCHING_COUNT_CALCULTATION_COUNT, MATCHING_PERC_CALCULATION_COUNT,
        },
        CollectionLabels,
    },
    nlp::{locales::Locale, TextParser},
    types::{
        DocumentId, Filter, FulltextMode, HybridMode, IndexId, Limit, Properties, SearchMode,
        SearchModeResult, SearchParams, Similarity, VectorMode,
    },
};

use super::collection::{IndexFieldStats, IndexFieldStatsType};
mod committed_field;
mod path_to_index_id_map;
mod uncommitted_field;

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use anyhow::{anyhow, bail, Context, Result};

use crate::{
    collection_manager::sides::{
        index::IndexedValue, IndexWriteOperation, IndexWriteOperationFieldType,
    },
    types::FieldId,
};

pub use committed_field::{
    CommittedBoolFieldStats, CommittedNumberFieldStats, CommittedStringFieldStats,
    CommittedStringFilterFieldStats, CommittedVectorFieldStats,
};
pub use uncommitted_field::{
    UncommittedBoolFieldStats, UncommittedNumberFieldStats, UncommittedStringFieldStats,
    UncommittedStringFilterFieldStats, UncommittedVectorFieldStats,
};

pub struct Index {
    id: IndexId,
    locale: Locale,

    llm_service: Arc<llms::LLMService>,
    ai_service: Arc<AIService>,

    document_count: u64,
    uncommitted_deleted_documents: HashSet<DocumentId>,

    committed_number_index: HashMap<FieldId, CommittedNumberField>,
    uncommitted_number_index: HashMap<FieldId, UncommittedNumberField>,

    committed_bool_index: HashMap<FieldId, CommittedBoolField>,
    uncommitted_bool_index: HashMap<FieldId, UncommittedBoolField>,

    committed_string_filter_index: HashMap<FieldId, CommittedStringFilterField>,
    uncommitted_string_filter_index: HashMap<FieldId, UncommittedStringFilterField>,

    committed_string_index: HashMap<FieldId, CommittedStringField>,
    uncommitted_string_index: HashMap<FieldId, UncommittedStringField>,

    committed_vector_index: HashMap<FieldId, CommittedVectorField>,
    uncommitted_vector_index: HashMap<FieldId, UncommittedVectorField>,

    path_to_index_id_map: PathToIndexId,
}

impl Index {
    pub fn new(
        id: IndexId,
        locale: Locale,
        llm_service: Arc<llms::LLMService>,
        ai_service: Arc<AIService>,
    ) -> Self {
        Self {
            id,
            locale,

            llm_service,
            ai_service,

            document_count: 0,
            uncommitted_deleted_documents: HashSet::new(),

            committed_number_index: HashMap::new(),
            uncommitted_number_index: HashMap::new(),

            committed_bool_index: HashMap::new(),
            uncommitted_bool_index: HashMap::new(),

            committed_string_filter_index: HashMap::new(),
            uncommitted_string_filter_index: HashMap::new(),

            committed_string_index: HashMap::new(),
            uncommitted_string_index: HashMap::new(),

            committed_vector_index: HashMap::new(),
            uncommitted_vector_index: HashMap::new(),

            path_to_index_id_map: PathToIndexId::new(),
        }
    }

    pub fn update(&mut self, op: IndexWriteOperation) -> Result<()> {
        match op {
            IndexWriteOperation::CreateField2 {
                field_id,
                field_path,
                is_array,
                field_type,
            } => {
                let field_type = match field_type {
                    IndexWriteOperationFieldType::Bool => {
                        self.uncommitted_bool_index
                            .insert(field_id, UncommittedBoolField::empty(field_path.clone()));
                        FieldType::Bool
                    }
                    IndexWriteOperationFieldType::Number => {
                        self.uncommitted_number_index
                            .insert(field_id, UncommittedNumberField::empty(field_path.clone()));
                        FieldType::Number
                    }
                    IndexWriteOperationFieldType::StringFilter => {
                        self.uncommitted_string_filter_index.insert(
                            field_id,
                            UncommittedStringFilterField::empty(field_path.clone()),
                        );
                        FieldType::StringFilter
                    }
                    IndexWriteOperationFieldType::String => {
                        self.uncommitted_string_index
                            .insert(field_id, UncommittedStringField::empty(field_path.clone()));
                        FieldType::String
                    }
                    IndexWriteOperationFieldType::Embedding(model) => {
                        self.uncommitted_vector_index.insert(
                            field_id,
                            UncommittedVectorField::empty(field_path.clone(), model.0),
                        );
                        FieldType::Vector
                    }
                };
                self.path_to_index_id_map
                    .insert(field_path, field_id, field_type);
            }
            IndexWriteOperation::Index {
                doc_id,
                indexed_values,
            } => {
                self.document_count += 1;
                for indexed_value in indexed_values {
                    match indexed_value {
                        IndexedValue::FilterBool(field_id, bool_value) => {
                            if let Some(field) = self.uncommitted_bool_index.get_mut(&field_id) {
                                field.insert(doc_id, bool_value);
                            }
                        }
                        IndexedValue::FilterNumber(field_id, number) => {
                            if let Some(field) = self.uncommitted_number_index.get_mut(&field_id) {
                                field.insert(doc_id, number.0);
                            }
                        }
                        IndexedValue::FilterString(field_id, string_value) => {
                            if let Some(field) =
                                self.uncommitted_string_filter_index.get_mut(&field_id)
                            {
                                field.insert(doc_id, string_value);
                            }
                        }
                        IndexedValue::ScoreString(field_id, len, values) => {
                            if let Some(field) = self.uncommitted_string_index.get_mut(&field_id) {
                                field.insert(doc_id, len, values);
                            }
                        }
                    }
                }
            }
            IndexWriteOperation::DeleteDocuments { doc_ids } => {
                let len = doc_ids.len() as u64;
                self.uncommitted_deleted_documents.extend(doc_ids);
                self.document_count = self.document_count.saturating_sub(len);

                println!("self.uncommitted_deleted_documents: {:?}", self.uncommitted_deleted_documents);
                println!("doc count: {}", self.document_count)
            }
        };

        Ok(())
    }

    pub async fn search(
        &self,
        search_params: &SearchParams,
        results: &mut HashMap<DocumentId, f32>,
    ) -> Result<()> {
        let uncommitted_deleted_documents = &self.uncommitted_deleted_documents;

        let filtered_doc_ids = self
            .calculate_filtered_doc_ids(&search_params.where_filter, uncommitted_deleted_documents)
            .await
            .with_context(|| format!("Cannot calculate filtered doc in index {:?}", self.id))?;
        if let Some(filtered_doc_ids) = &filtered_doc_ids {
            FILTER_PERC_CALCULATION_COUNT.track(
                CollectionLabels {
                    collection: self.id.to_string(),
                },
                filtered_doc_ids.len() as f64 / self.document_count as f64,
            );
            FILTER_COUNT_CALCULATION_COUNT.track_usize(
                CollectionLabels {
                    collection: self.id.to_string(),
                },
                filtered_doc_ids.len(),
            );
        }

        // Manage the "auto" search mode. OramaCore will automatically determine
        // wether to use full text search, vector search or hybrid search.
        let search_mode: SearchMode = match &search_params.mode {
            SearchMode::Auto(mode_result) => {
                let final_mode: String = self
                    .llm_service
                    .run_known_prompt(
                        llms::KnownPrompts::Autoquery,
                        vec![("query".to_string(), mode_result.term.clone())],
                        // @todo: determine if we want to allow the user to select which LLM to use here.
                        None,
                    )
                    .await?;

                let serilized_final_mode: SearchModeResult = serde_json::from_str(&final_mode)?;

                match serilized_final_mode.mode.as_str() {
                    "fulltext" => SearchMode::FullText(FulltextMode {
                        term: mode_result.term.clone(),
                    }),
                    "hybrid" => SearchMode::Hybrid(HybridMode {
                        term: mode_result.term.clone(),
                        similarity: Similarity(0.8),
                    }),
                    "vector" => SearchMode::Vector(VectorMode {
                        term: mode_result.term.clone(),
                        similarity: Similarity(0.8),
                    }),
                    _ => anyhow::bail!("Invalid search mode"),
                }
            }
            _ => search_params.mode.clone(),
        };

        let boost = self.calculate_boost(&search_params.boost);

        match search_mode {
            SearchMode::Default(search_mode) | SearchMode::FullText(search_mode) => {
                let properties = self.calculate_string_properties(&search_params.properties)?;
                results.extend(self.search_full_text(
                    &search_mode.term,
                    properties,
                    boost,
                    filtered_doc_ids.as_ref(),
                    uncommitted_deleted_documents,
                )
                .await?)
            }
            SearchMode::Vector(search_mode) => {
                let vector_properties = self.calculate_vector_properties()?;
                results.extend(self.search_vector(
                    &search_mode.term,
                    vector_properties,
                    search_mode.similarity,
                    filtered_doc_ids.as_ref(),
                    search_params.limit,
                    uncommitted_deleted_documents,
                )
                .await?)
            }
            SearchMode::Hybrid(search_mode) => {
                let vector_properties = self.calculate_vector_properties()?;
                let string_properties =
                    self.calculate_string_properties(&search_params.properties)?;

                let (vector, fulltext) = join!(
                    self.search_vector(
                        &search_mode.term,
                        vector_properties,
                        search_mode.similarity,
                        filtered_doc_ids.as_ref(),
                        search_params.limit,
                        uncommitted_deleted_documents
                    ),
                    self.search_full_text(
                        &search_mode.term,
                        string_properties,
                        boost,
                        filtered_doc_ids.as_ref(),
                        uncommitted_deleted_documents,
                    )
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
                results.extend(fulltext);
            }
            SearchMode::Auto(_) => unreachable!(),
        };

        MATCHING_COUNT_CALCULTATION_COUNT.track_usize(
            CollectionLabels {
                collection: self.id.to_string(),
            },
            results.len(),
        );
        MATCHING_PERC_CALCULATION_COUNT.track(
            CollectionLabels {
                collection: self.id.to_string(),
            },
            results.len() as f64 / self.document_count as f64,
        );

        debug!("token_scores: {:?}", results);

        Ok(())
    }

    fn calculate_vector_properties(&self) -> Result<Vec<FieldId>> {
        let properties: Vec<_> = self
            .uncommitted_vector_index
            .keys()
            .chain(self.committed_vector_index.keys())
            .copied()
            .collect();

        Ok(properties)
    }

    fn calculate_string_properties(&self, properties: &Properties) -> Result<Vec<FieldId>> {
        let properties: Vec<_> = match properties {
            Properties::Specified(properties) => {
                let mut field_ids = Vec::new();
                for field in properties {
                    let (field_id, field_type) = match self.path_to_index_id_map.get(field) {
                        None => {
                            bail!("Cannot filter by \"{}\": unknown field", &field);
                        }
                        Some((field_id, field_type)) => (field_id, field_type),
                    };
                    if !matches!(field_type, FieldType::String) {
                        bail!("Cannot filter by \"{}\": wrong type", &field);
                    }
                    field_ids.push(field_id);
                }
                field_ids
            }
            Properties::None | Properties::Star => self
                .uncommitted_string_index
                .keys()
                .chain(self.committed_string_index.keys())
                .copied()
                .collect(),
        };

        Ok(properties)
    }

    pub async fn stats(&self) -> Result<IndexStats> {
        let mut fields_stats = Vec::new();

        fields_stats.extend(self.uncommitted_bool_index.iter().map(|(k, v)| {
            let path = v.field_path().join(".");
            IndexFieldStats {
                field_id: *k,
                path,
                stats: IndexFieldStatsType::UncommittedBoolean(v.stats()),
            }
        }));
        fields_stats.extend(self.uncommitted_number_index.iter().map(|(k, v)| {
            let path = v.field_path().join(".");
            IndexFieldStats {
                field_id: *k,
                path,
                stats: IndexFieldStatsType::UncommittedNumber(v.stats()),
            }
        }));
        fields_stats.extend(self.uncommitted_string_filter_index.iter().map(|(k, v)| {
            let path = v.field_path().join(".");
            IndexFieldStats {
                field_id: *k,
                path,
                stats: IndexFieldStatsType::UncommittedStringFilter(v.stats()),
            }
        }));
        fields_stats.extend(self.uncommitted_string_index.iter().map(|(k, v)| {
            let path = v.field_path().join(".");
            IndexFieldStats {
                field_id: *k,
                path,
                stats: IndexFieldStatsType::UncommittedString(v.stats()),
            }
        }));
        fields_stats.extend(self.uncommitted_vector_index.iter().map(|(k, v)| {
            let path = v.field_path().join(".");
            IndexFieldStats {
                field_id: *k,
                path,
                stats: IndexFieldStatsType::UncommittedVector(v.stats()),
            }
        }));
        fields_stats.extend(self.committed_number_index.iter().filter_map(|(k, v)| {
            let path = v.field_path().join(".");
            Some(IndexFieldStats {
                field_id: *k,
                path,
                stats: IndexFieldStatsType::CommittedNumber(v.stats().ok()?),
            })
        }));
        fields_stats.extend(self.committed_bool_index.iter().filter_map(|(k, v)| {
            let path = v.field_path().join(".");
            Some(IndexFieldStats {
                field_id: *k,
                path,
                stats: IndexFieldStatsType::CommittedBoolean(v.stats().ok()?),
            })
        }));
        fields_stats.extend(
            self.committed_string_filter_index
                .iter()
                .filter_map(|(k, v)| {
                    let path = v.field_path().join(".");
                    Some(IndexFieldStats {
                        field_id: *k,
                        path,
                        stats: IndexFieldStatsType::CommittedStringFilter(v.stats()),
                    })
                }),
        );
        fields_stats.extend(self.committed_string_index.iter().filter_map(|(k, v)| {
            let path = v.field_path().join(".");
            Some(IndexFieldStats {
                field_id: *k,
                path,
                stats: IndexFieldStatsType::CommittedString(v.stats().ok()?),
            })
        }));
        fields_stats.extend(self.committed_vector_index.iter().filter_map(|(k, v)| {
            let path = v.field_path().join(".");
            Some(IndexFieldStats {
                field_id: *k,
                path,
                stats: IndexFieldStatsType::CommittedVector(v.stats().ok()?),
            })
        }));

        fields_stats.sort_by_key(|e| e.field_id.0);

        Ok(IndexStats {
            id: self.id,
            default_locale: self.locale,
            document_count: self.document_count,
            fields_stats,
        })
    }

    async fn calculate_filtered_doc_ids(
        &self,
        where_filter: &HashMap<String, Filter>,
        uncommitted_deleted_documents: &HashSet<DocumentId>,
    ) -> Result<Option<HashSet<DocumentId>>> {
        if where_filter.is_empty() {
            return Ok(None);
        }

        trace!("Calculating filtered doc ids",);

        let mut filtered: HashSet<DocumentId> = Default::default();
        for (k, filter) in where_filter {
            let (field_id, field_type) = match self.path_to_index_id_map.get(k) {
                None => {
                    bail!("Cannot filter by \"{}\": unknown field", &k);
                }
                Some((field_id, field_type)) => (field_id, field_type),
            };
            match (field_type, filter) {
                (FieldType::Bool, Filter::Bool(filter_bool)) => {
                    let uncommitted_field =
                        self.uncommitted_bool_index.get(&field_id).ok_or_else(|| {
                            anyhow::anyhow!("Cannot filter by \"{}\": unknown field", &k)
                        })?;

                    filtered.extend(uncommitted_field.filter(*filter_bool));

                    let committed_field = self.committed_bool_index.get(&field_id);
                    if let Some(committed_field) = committed_field {
                        let a = committed_field.filter(*filter_bool).with_context(|| {
                            format!("Cannot filter by \"{}\": unknown field", &k)
                        })?;
                        filtered.extend(a);
                    }
                }
                _ => panic!(
                    "Unsupported filter type: type {:?}. filter: {:?}",
                    field_type, filter
                ),
            }
        }

        let doc_ids: HashSet<_> = filtered
            .difference(uncommitted_deleted_documents)
            .copied()
            .collect();

        Ok(Some(doc_ids))
    }

    fn calculate_boost(&self, boost: &HashMap<String, f32>) -> HashMap<FieldId, f32> {
        boost
            .iter()
            .filter_map(|(field_name, boost)| {
                let a = self.path_to_index_id_map.get(field_name);
                let field_id = a?.0;
                Some((field_id, *boost))
            })
            .collect()
    }

    async fn search_full_text(
        &self,
        term: &str,
        properties: Vec<FieldId>,
        boost: HashMap<FieldId, f32>,
        filtered_doc_ids: Option<&HashSet<DocumentId>>,
        uncommitted_deleted_documents: &HashSet<DocumentId>,
    ) -> Result<HashMap<DocumentId, f32>> {
        let mut scorer: BM25Scorer<DocumentId> = BM25Scorer::new();

        let parser = TextParser::from_locale(self.locale.clone());
        let tokens = parser.tokenize_and_stem(term);
        let tokens: Vec<_> = tokens
            .into_iter()
            .flat_map(|e| std::iter::once(e.0).chain(e.1.into_iter()))
            .collect();

        for field_id in properties {
            let Some(field) = self.uncommitted_string_index.get(&field_id) else {
                bail!("Cannot search on field {:?}: unknown field", field_id);
            };
            let committed = self.committed_string_index.get(&field_id);

            let global_info = if let Some(committed) = committed {
                committed.global_info() + field.global_info()
            } else {
                field.global_info()
            };

            let boost = boost.get(&field_id).copied().unwrap_or(1.0);

            field
                .search(
                    &tokens,
                    boost,
                    &mut scorer,
                    filtered_doc_ids,
                    &global_info,
                    uncommitted_deleted_documents,
                )
                .context("Cannot perform search")?;
        }

        Ok(scorer.get_scores())
    }

    async fn search_vector(
        &self,
        term: &str,
        properties: Vec<FieldId>,
        similarity: Similarity,
        filtered_doc_ids: Option<&HashSet<DocumentId>>,
        limit: Limit,
        uncommitted_deleted_documents: &HashSet<DocumentId>,
    ) -> Result<HashMap<DocumentId, f32>> {
        let mut output: HashMap<DocumentId, f32> = HashMap::new();

        for field_id in properties {
            let Some(uncommitted) = self.uncommitted_vector_index.get(&field_id) else {
                bail!("Cannot search on field {:?}: unknown field", field_id);
            };
            let committed = self.committed_vector_index.get(&field_id);

            let model = uncommitted.get_model();

            // We don't cache the embedding.
            // We can do that because, for now, an index and an collection has only one embedding field.
            // Anyway, if the user seach in different index, we are re-calculating the embedding.
            // We should put a sort of cache here.
            // TODO: think about this.
            let targets = self
                .ai_service
                .embed_query(model, vec![&term.to_string()])
                .await?;

            for target in targets {
                uncommitted.search(
                    &target,
                    similarity.0,
                    filtered_doc_ids,
                    &mut output,
                    uncommitted_deleted_documents,
                )?;
                if let Some(committed) = committed {
                    committed.search(
                        &target,
                        similarity.0,
                        limit.0,
                        filtered_doc_ids,
                        &mut output,
                        uncommitted_deleted_documents,
                    )?;
                }
            }
        }

        Ok(output)
    }
}

#[derive(Serialize, Debug)]
pub struct IndexStats {
    pub id: IndexId,
    pub default_locale: Locale,
    pub document_count: u64,
    pub fields_stats: Vec<IndexFieldStats>,
}

#[derive(Debug, Clone, Copy)]
enum FieldType {
    Bool,
    Number,
    StringFilter,
    String,
    Vector,
}
