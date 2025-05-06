use chrono::{DateTime, Utc};
use committed_field::{
    BoolFieldInfo, CommittedBoolField, CommittedNumberField, CommittedStringField,
    CommittedStringFilterField, CommittedVectorField, NumberFieldInfo, StringFieldInfo,
    StringFilterFieldInfo, VectorFieldInfo,
};
use futures::join;
use merge::{
    merge_bool_field, merge_number_field, merge_string_field, merge_string_filter_field,
    merge_vector_field,
};
use path_to_index_id_map::PathToIndexId;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, error, info, trace};
use uncommitted_field::*;

use crate::{
    ai::{llms, AIService},
    collection_manager::{bm25::BM25Scorer, sides::Offset},
    file_utils::{create_if_not_exists, BufferedFile},
    metrics::{
        search::{
            FILTER_COUNT_CALCULATION_COUNT, FILTER_PERC_CALCULATION_COUNT,
            MATCHING_COUNT_CALCULTATION_COUNT, MATCHING_PERC_CALCULATION_COUNT,
        },
        CollectionLabels,
    },
    nlp::{locales::Locale, NLPService, TextParser},
    types::{
        DocumentId, FacetDefinition, FacetResult, Filter, FulltextMode, HybridMode, IndexId, Limit,
        NumberFilter, Properties, SearchMode, SearchModeResult, SearchParams, Similarity,
        Threshold, VectorMode,
    },
};

use super::collection::{IndexFieldStats, IndexFieldStatsType};
mod committed_field;
mod merge;
mod path_to_index_id_map;
mod uncommitted_field;

use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use anyhow::{bail, Context, Result};

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

#[derive(Debug, Clone, Copy)]
pub enum DeletionReason {
    UserWanted,
    CollectionReindexed { temp_index_id: IndexId },
    IndexResynced { temp_index_id: IndexId },
}

#[derive(Default)]
struct UncommittedFields {
    bool_fields: HashMap<FieldId, UncommittedBoolField>,
    number_fields: HashMap<FieldId, UncommittedNumberField>,
    string_filter_fields: HashMap<FieldId, UncommittedStringFilterField>,
    string_fields: HashMap<FieldId, UncommittedStringField>,
    vector_fields: HashMap<FieldId, UncommittedVectorField>,
}

#[derive(Default)]
struct CommittedFields {
    bool_fields: HashMap<FieldId, CommittedBoolField>,
    number_fields: HashMap<FieldId, CommittedNumberField>,
    string_filter_fields: HashMap<FieldId, CommittedStringFilterField>,
    string_fields: HashMap<FieldId, CommittedStringField>,
    vector_fields: HashMap<FieldId, CommittedVectorField>,
}

pub struct Index {
    id: IndexId,
    locale: Locale,
    text_parser: Arc<TextParser>,

    // This will contains all the temp index ids
    // This is needed because embedding calculation can arrive later,
    // after the index substitution.
    aliases: Vec<IndexId>,

    deleted: Option<DeletionReason>,
    pub promoted_to_runtime_index: AtomicBool,

    llm_service: Arc<llms::LLMService>,
    ai_service: Arc<AIService>,

    document_count: u64,
    uncommitted_deleted_documents: HashSet<DocumentId>,

    uncommitted_fields: RwLock<UncommittedFields>,
    committed_fields: RwLock<CommittedFields>,

    path_to_index_id_map: PathToIndexId,

    is_new: AtomicBool,

    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
}

impl Index {
    pub fn new(
        id: IndexId,
        text_parser: Arc<TextParser>,
        llm_service: Arc<llms::LLMService>,
        ai_service: Arc<AIService>,
    ) -> Self {
        Self {
            id,
            locale: text_parser.locale(),
            text_parser,
            aliases: vec![],
            deleted: None,
            promoted_to_runtime_index: AtomicBool::new(false),

            llm_service,
            ai_service,

            document_count: 0,
            uncommitted_deleted_documents: HashSet::new(),

            committed_fields: Default::default(),
            uncommitted_fields: Default::default(),

            path_to_index_id_map: PathToIndexId::new(),
            is_new: AtomicBool::new(true),

            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    pub fn try_load(
        index_id: IndexId,
        data_dir: PathBuf,
        nlp_service: Arc<NLPService>,
        llm_service: Arc<llms::LLMService>,
        ai_service: Arc<AIService>,
    ) -> Result<Self> {
        let dump: Dump = BufferedFile::open(data_dir.join("index.json"))
            .context("Cannot open index.json")?
            .read_json_data()
            .context("Cannot read index.json")?;
        let Dump::V1(dump) = dump;

        debug_assert_eq!(
            dump.id, index_id,
            "Index id mismatch: expected {:?}, got {:?}",
            index_id, dump.id
        );

        let mut uncommitted_fields = UncommittedFields::default();
        let mut committed_fields = CommittedFields::default();
        for (field_id, info) in dump.bool_field_ids {
            uncommitted_fields.bool_fields.insert(
                field_id,
                UncommittedBoolField::empty(info.field_path.clone()),
            );
            let field = CommittedBoolField::try_load(info).context("Cannot load bool field")?;
            committed_fields.bool_fields.insert(field_id, field);
        }
        for (field_id, info) in dump.number_field_ids {
            uncommitted_fields.number_fields.insert(
                field_id,
                UncommittedNumberField::empty(info.field_path.clone()),
            );
            let field = CommittedNumberField::try_load(info).context("Cannot load number field")?;
            committed_fields.number_fields.insert(field_id, field);
        }
        for (field_id, info) in dump.string_filter_field_ids {
            uncommitted_fields.string_filter_fields.insert(
                field_id,
                UncommittedStringFilterField::empty(info.field_path.clone()),
            );
            let field = CommittedStringFilterField::try_load(info)
                .context("Cannot load string filter field")?;
            committed_fields
                .string_filter_fields
                .insert(field_id, field);
        }
        for (field_id, info) in dump.string_field_ids {
            uncommitted_fields.string_fields.insert(
                field_id,
                UncommittedStringField::empty(info.field_path.clone()),
            );
            let field = CommittedStringField::try_load(info).context("Cannot load string field")?;
            committed_fields.string_fields.insert(field_id, field);
        }
        for (field_id, info) in dump.vector_field_ids {
            uncommitted_fields.vector_fields.insert(
                field_id,
                UncommittedVectorField::empty(info.field_path.clone(), info.model.0),
            );
            let field = CommittedVectorField::try_load(info).context("Cannot load vector field")?;
            committed_fields.vector_fields.insert(field_id, field);
        }

        Ok(Self {
            id: dump.id,
            locale: dump.locale,
            text_parser: nlp_service.get(dump.locale),
            deleted: None,
            promoted_to_runtime_index: AtomicBool::new(false),
            aliases: dump.aliases,

            llm_service,
            ai_service,

            document_count: dump.document_count,
            uncommitted_deleted_documents: HashSet::new(),

            committed_fields: RwLock::new(committed_fields),
            uncommitted_fields: RwLock::new(uncommitted_fields),

            path_to_index_id_map: PathToIndexId::from(dump.path_to_index_id_map),
            is_new: AtomicBool::new(false),

            created_at: dump.created_at,
            updated_at: dump.updated_at,
        })
    }

    pub async fn commit(&self, data_dir: PathBuf, offset: Offset) -> Result<()> {
        let data_dir_with_offset = data_dir.join(format!("offset-{}", offset.0));

        let uncommitted_fields = self.uncommitted_fields.read().await;

        let is_new = self.is_new.load(Ordering::Relaxed);

        // This index was a temp index and from the last commit, it was promoted to a runtime index
        // That means all committed fields are inside "temp_indexes" folder and
        // the metadata info points to the wrong folder.
        // If it is "true", we need to move the committed field to "indexes" folder.
        // To not income to a bug, if the uncommitted fields are empty, we need to copy the committed fields
        // to the new folder.
        let is_promoted = self.promoted_to_runtime_index.load(Ordering::Relaxed);

        // check if there's something to commit
        let something_to_commit = uncommitted_fields
            .bool_fields
            .iter()
            .any(|(_, field)| !field.is_empty())
            || uncommitted_fields
                .number_fields
                .iter()
                .any(|(_, field)| !field.is_empty())
            || uncommitted_fields
                .string_filter_fields
                .iter()
                .any(|(_, field)| !field.is_empty())
            || uncommitted_fields
                .string_fields
                .iter()
                .any(|(_, field)| !field.is_empty())
            || uncommitted_fields
                .vector_fields
                .iter()
                .any(|(_, field)| !field.is_empty())
            || !self.uncommitted_deleted_documents.is_empty();
        if !something_to_commit && !is_promoted && !is_new {
            // Nothing to commit
            debug!("Nothing to commit {:?}", self.id);
            return Ok(());
        }

        create_if_not_exists(data_dir_with_offset.clone())
            .context("Cannot create data directory")?;

        debug!("Committing index {:?}", self.id);

        let committed_fields = self.committed_fields.read().await;

        enum MergeResult<T> {
            Changed(T),
            Unchanged,
        }

        let mut merged_bools = HashMap::new();
        let bool_indexes: HashSet<_> = uncommitted_fields
            .bool_fields
            .keys()
            .chain(committed_fields.bool_fields.keys())
            .copied()
            .collect();
        for field_id in bool_indexes {
            let data_dir = data_dir_with_offset.join(field_id.0.to_string());
            let uncommitted = uncommitted_fields.bool_fields.get(&field_id);
            let committed = committed_fields.bool_fields.get(&field_id);
            if let Some(merged) = merge_bool_field(
                uncommitted,
                committed,
                data_dir,
                &self.uncommitted_deleted_documents,
                is_promoted,
            )
            .context("Cannot merge bool field")?
            {
                merged_bools.insert(field_id, MergeResult::Changed(merged));
            } else {
                merged_bools.insert(field_id, MergeResult::Unchanged);
            }
        }

        let mut merged_numbers = HashMap::new();
        let number_indexes: HashSet<_> = uncommitted_fields
            .number_fields
            .keys()
            .chain(committed_fields.number_fields.keys())
            .copied()
            .collect();
        for field_id in number_indexes {
            let data_dir = data_dir_with_offset.join(field_id.0.to_string());
            let uncommitted = uncommitted_fields.number_fields.get(&field_id);
            let committed = committed_fields.number_fields.get(&field_id);
            if let Some(merged) = merge_number_field(
                uncommitted,
                committed,
                data_dir,
                &self.uncommitted_deleted_documents,
                is_promoted,
            )
            .context("Cannot merge number field")?
            {
                merged_numbers.insert(field_id, MergeResult::Changed(merged));
            } else {
                merged_numbers.insert(field_id, MergeResult::Unchanged);
            }
        }

        let mut merged_string_filters = HashMap::new();
        let string_filter_indexes: HashSet<_> = uncommitted_fields
            .string_filter_fields
            .keys()
            .chain(committed_fields.string_filter_fields.keys())
            .copied()
            .collect();
        for field_id in string_filter_indexes {
            let data_dir = data_dir_with_offset.join(field_id.0.to_string());
            let uncommitted = uncommitted_fields.string_filter_fields.get(&field_id);
            let committed = committed_fields.string_filter_fields.get(&field_id);
            if let Some(merged) = merge_string_filter_field(
                uncommitted,
                committed,
                data_dir,
                &self.uncommitted_deleted_documents,
                is_promoted,
            )
            .context("Cannot merge string filter field")?
            {
                merged_string_filters.insert(field_id, MergeResult::Changed(merged));
            } else {
                merged_string_filters.insert(field_id, MergeResult::Unchanged);
            }
        }

        let mut merged_strings = HashMap::new();
        let string_indexes: HashSet<_> = uncommitted_fields
            .string_fields
            .keys()
            .chain(committed_fields.string_fields.keys())
            .copied()
            .collect();
        for field_id in string_indexes {
            let data_dir = data_dir_with_offset.join(field_id.0.to_string());
            let uncommitted = uncommitted_fields.string_fields.get(&field_id);
            let committed = committed_fields.string_fields.get(&field_id);
            if let Some(merged) = merge_string_field(
                uncommitted,
                committed,
                data_dir,
                &self.uncommitted_deleted_documents,
                is_promoted,
            )
            .context("Cannot merge string field")?
            {
                merged_strings.insert(field_id, MergeResult::Changed(merged));
            } else {
                merged_strings.insert(field_id, MergeResult::Unchanged);
            }
        }

        let mut merged_vectors = HashMap::new();
        let vector_indexes: HashSet<_> = uncommitted_fields
            .vector_fields
            .keys()
            .chain(committed_fields.vector_fields.keys())
            .copied()
            .collect();
        for field_id in vector_indexes {
            let data_dir = data_dir_with_offset.join(field_id.0.to_string());
            let uncommitted = uncommitted_fields.vector_fields.get(&field_id);
            let committed = committed_fields.vector_fields.get(&field_id);
            if let Some(merged) = merge_vector_field(
                uncommitted,
                committed,
                data_dir,
                &self.uncommitted_deleted_documents,
                is_promoted,
            )
            .context("Cannot merge vector field")?
            {
                merged_vectors.insert(field_id, MergeResult::Changed(merged));
            } else {
                merged_vectors.insert(field_id, MergeResult::Unchanged);
            }
        }

        // Once all changed in merged_* maps are collected,
        // we can proceed to replace the committed fields with the merged ones
        // and clear the uncommitted fields
        // NB: the following drop + write lock lines are not safe:
        // we calculate the "merged" fields in a read lock, which guarantees
        // that the fields are not changed while we are reading them.
        // But we are dropping the read lock and then acquiring a write lock
        // which is not safe because tokio can switch to another task changing
        // the fields while we are dropping the read lock.
        drop(uncommitted_fields);
        drop(committed_fields);
        // Here something bad can happen inside here
        // see: https://github.com/tokio-rs/tokio/issues/7282
        let mut uncommitted_fields = self.uncommitted_fields.write().await;
        let mut committed_fields = self.committed_fields.write().await;

        for (field_id, merged) in merged_bools {
            match merged {
                MergeResult::Changed(merged) => {
                    committed_fields.bool_fields.insert(field_id, merged);
                }
                MergeResult::Unchanged => {}
            }
            if let Some(uncommitted) = uncommitted_fields.bool_fields.get_mut(&field_id) {
                uncommitted.clear();
            }
        }
        for (field_id, merged) in merged_numbers {
            match merged {
                MergeResult::Changed(merged) => {
                    committed_fields.number_fields.insert(field_id, merged);
                }
                MergeResult::Unchanged => {}
            }
            if let Some(uncommitted) = uncommitted_fields.number_fields.get_mut(&field_id) {
                uncommitted.clear();
            }
        }
        for (field_id, merged) in merged_string_filters {
            match merged {
                MergeResult::Changed(merged) => {
                    committed_fields
                        .string_filter_fields
                        .insert(field_id, merged);
                }
                MergeResult::Unchanged => {}
            }
            if let Some(uncommitted) = uncommitted_fields.string_filter_fields.get_mut(&field_id) {
                uncommitted.clear();
            }
        }
        for (field_id, merged) in merged_strings {
            match merged {
                MergeResult::Changed(merged) => {
                    committed_fields.string_fields.insert(field_id, merged);
                }
                MergeResult::Unchanged => {}
            }
            if let Some(uncommitted) = uncommitted_fields.string_fields.get_mut(&field_id) {
                uncommitted.clear();
            }
        }
        for (field_id, merged) in merged_vectors {
            match merged {
                MergeResult::Changed(merged) => {
                    committed_fields.vector_fields.insert(field_id, merged);
                }
                MergeResult::Unchanged => {}
            }
            if let Some(uncommitted) = uncommitted_fields.vector_fields.get_mut(&field_id) {
                uncommitted.clear();
            }
        }

        let dump = Dump::V1(DumpV1 {
            id: self.id,
            document_count: self.document_count,
            locale: self.locale,
            aliases: self.aliases.clone(),
            bool_field_ids: committed_fields
                .bool_fields
                .iter()
                .map(|(k, v)| (*k, v.get_field_info()))
                .collect(),
            number_field_ids: committed_fields
                .number_fields
                .iter()
                .map(|(k, v)| (*k, v.get_field_info()))
                .collect(),
            string_filter_field_ids: committed_fields
                .string_filter_fields
                .iter()
                .map(|(k, v)| (*k, v.get_field_info()))
                .collect(),
            string_field_ids: committed_fields
                .string_fields
                .iter()
                .map(|(k, v)| (*k, v.get_field_info()))
                .collect(),
            vector_field_ids: committed_fields
                .vector_fields
                .iter()
                .map(|(k, v)| (*k, v.get_field_info()))
                .collect(),
            path_to_index_id_map: self.path_to_index_id_map.serialize(),
            created_at: self.created_at,
            updated_at: self.updated_at,
        });

        drop(uncommitted_fields);
        drop(committed_fields);

        BufferedFile::create_or_overwrite(data_dir.join("index.json"))
            .context("Cannot create index.json")?
            .write_json_data(&dump)
            .context("Cannot write index.json")?;

        // Force flags after commit
        self.promoted_to_runtime_index
            .store(false, Ordering::Relaxed);
        self.is_new.store(false, Ordering::Relaxed);

        debug!("Index committed: {:?}", self.id);

        Ok(())
    }

    #[inline]
    pub fn id(&self) -> IndexId {
        self.id
    }

    #[inline]
    pub fn aliases(&self) -> &[IndexId] {
        self.aliases.as_slice()
    }

    pub fn promote_to_runtime_index(&mut self, runtime_index_id: IndexId) {
        let previous_id = self.id;
        self.id = runtime_index_id;
        self.aliases.push(previous_id);
        self.promoted_to_runtime_index
            .store(true, Ordering::Relaxed);
        // We need to update the created_at and updated_at fields
        self.updated_at = Utc::now();
    }

    pub fn mark_as_deleted(&mut self, reason: DeletionReason) {
        self.deleted = Some(reason);
        self.updated_at = Utc::now();
    }

    pub fn get_deletion_reason(&self) -> Option<DeletionReason> {
        self.deleted
    }

    pub fn is_deleted(&self) -> bool {
        self.deleted.is_some()
    }

    pub fn update(&mut self, op: IndexWriteOperation) -> Result<()> {
        self.updated_at = Utc::now();
        let uncommitted_fields = self.uncommitted_fields.get_mut();
        match op {
            IndexWriteOperation::CreateField2 {
                field_id,
                field_path,
                is_array: _,
                field_type,
            } => {
                let field_type = match field_type {
                    IndexWriteOperationFieldType::Bool => {
                        uncommitted_fields
                            .bool_fields
                            .insert(field_id, UncommittedBoolField::empty(field_path.clone()));
                        FieldType::Bool
                    }
                    IndexWriteOperationFieldType::Number => {
                        uncommitted_fields
                            .number_fields
                            .insert(field_id, UncommittedNumberField::empty(field_path.clone()));
                        FieldType::Number
                    }
                    IndexWriteOperationFieldType::StringFilter => {
                        uncommitted_fields.string_filter_fields.insert(
                            field_id,
                            UncommittedStringFilterField::empty(field_path.clone()),
                        );
                        FieldType::StringFilter
                    }
                    IndexWriteOperationFieldType::String => {
                        uncommitted_fields
                            .string_fields
                            .insert(field_id, UncommittedStringField::empty(field_path.clone()));
                        FieldType::String
                    }
                    IndexWriteOperationFieldType::Embedding(model) => {
                        uncommitted_fields.vector_fields.insert(
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
                            if let Some(field) = uncommitted_fields.bool_fields.get_mut(&field_id) {
                                field.insert(doc_id, bool_value);
                            } else {
                                error!("Cannot find field {:?} in uncommitted fields", field_id);
                            }
                        }
                        IndexedValue::FilterNumber(field_id, number) => {
                            if let Some(field) = uncommitted_fields.number_fields.get_mut(&field_id)
                            {
                                field.insert(doc_id, number.0);
                            } else {
                                error!("Cannot find field {:?} in uncommitted fields", field_id);
                            }
                        }
                        IndexedValue::FilterString(field_id, string_value) => {
                            if let Some(field) =
                                uncommitted_fields.string_filter_fields.get_mut(&field_id)
                            {
                                field.insert(doc_id, string_value);
                            } else {
                                error!("Cannot find field {:?} in uncommitted fields", field_id);
                            }
                        }
                        IndexedValue::ScoreString(field_id, len, values) => {
                            if let Some(field) = uncommitted_fields.string_fields.get_mut(&field_id)
                            {
                                field.insert(doc_id, len, values);
                            } else {
                                error!("Cannot find field {:?} in uncommitted fields", field_id);
                            }
                        }
                    }
                }
            }
            IndexWriteOperation::IndexEmbedding { data } => {
                for (field_id, data) in data {
                    if let Some(field) = uncommitted_fields.vector_fields.get_mut(&field_id) {
                        for (doc_id, vectors) in data {
                            if let Err(e) = field.insert(doc_id, vectors) {
                                error!(error = ?e, "Cannot insert vector. Skip it.");
                            }
                        }
                    } else {
                        error!("Cannot find field {:?} in uncommitted fields", field_id);
                    }
                }
            }
            IndexWriteOperation::DeleteDocuments { doc_ids } => {
                let len = doc_ids.len() as u64;
                self.uncommitted_deleted_documents.extend(doc_ids);
                self.document_count = self.document_count.saturating_sub(len);
            }
        };

        Ok(())
    }

    pub fn has_field(&self, field_name: &str) -> bool {
        self.path_to_index_id_map.get(field_name).is_some()
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
                        threshold: None,
                        exact: false,
                    }),
                    "hybrid" => SearchMode::Hybrid(HybridMode {
                        term: mode_result.term.clone(),
                        similarity: Similarity(0.8),
                        threshold: None,
                        exact: false,
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
                let properties = self
                    .calculate_string_properties(&search_params.properties)
                    .await?;
                results.extend(
                    self.search_full_text(
                        &search_mode.term,
                        search_mode.threshold,
                        search_mode.exact,
                        properties,
                        boost,
                        filtered_doc_ids.as_ref(),
                        uncommitted_deleted_documents,
                    )
                    .await?,
                )
            }
            SearchMode::Vector(search_mode) => {
                let vector_properties = self.calculate_vector_properties().await?;
                results.extend(
                    self.search_vector(
                        &search_mode.term,
                        vector_properties,
                        search_mode.similarity,
                        filtered_doc_ids.as_ref(),
                        search_params.limit,
                        uncommitted_deleted_documents,
                    )
                    .await?,
                )
            }
            SearchMode::Hybrid(search_mode) => {
                let vector_properties = self.calculate_vector_properties().await?;
                let string_properties = self
                    .calculate_string_properties(&search_params.properties)
                    .await?;

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
                        search_mode.threshold,
                        search_mode.exact,
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

    pub async fn stats(&self, is_temp: bool) -> Result<IndexStats> {
        let mut fields_stats = Vec::new();

        let uncommitted_fields = self.uncommitted_fields.read().await;
        let committed_fields = self.committed_fields.read().await;

        fields_stats.extend(uncommitted_fields.bool_fields.iter().map(|(k, v)| {
            let path = v.field_path().join(".");
            IndexFieldStats {
                field_id: *k,
                field_path: path,
                stats: IndexFieldStatsType::UncommittedBoolean(v.stats()),
            }
        }));
        fields_stats.extend(uncommitted_fields.number_fields.iter().map(|(k, v)| {
            let path = v.field_path().join(".");
            IndexFieldStats {
                field_id: *k,
                field_path: path,
                stats: IndexFieldStatsType::UncommittedNumber(v.stats()),
            }
        }));
        fields_stats.extend(
            uncommitted_fields
                .string_filter_fields
                .iter()
                .map(|(k, v)| {
                    let path = v.field_path().join(".");
                    IndexFieldStats {
                        field_id: *k,
                        field_path: path,
                        stats: IndexFieldStatsType::UncommittedStringFilter(v.stats()),
                    }
                }),
        );
        fields_stats.extend(uncommitted_fields.string_fields.iter().map(|(k, v)| {
            let path = v.field_path().join(".");
            IndexFieldStats {
                field_id: *k,
                field_path: path,
                stats: IndexFieldStatsType::UncommittedString(v.stats()),
            }
        }));
        fields_stats.extend(uncommitted_fields.vector_fields.iter().map(|(k, v)| {
            let path = v.field_path().join(".");
            IndexFieldStats {
                field_id: *k,
                field_path: path,
                stats: IndexFieldStatsType::UncommittedVector(v.stats()),
            }
        }));

        fields_stats.extend(committed_fields.bool_fields.iter().filter_map(|(k, v)| {
            let path = v.field_path().join(".");
            Some(IndexFieldStats {
                field_id: *k,
                field_path: path,
                stats: IndexFieldStatsType::CommittedBoolean(v.stats().ok()?),
            })
        }));
        fields_stats.extend(committed_fields.number_fields.iter().filter_map(|(k, v)| {
            let path = v.field_path().join(".");
            Some(IndexFieldStats {
                field_id: *k,
                field_path: path,
                stats: IndexFieldStatsType::CommittedNumber(v.stats().ok()?),
            })
        }));
        fields_stats.extend(
            committed_fields
                .string_filter_fields
                .iter()
                .filter_map(|(k, v)| {
                    let path = v.field_path().join(".");
                    Some(IndexFieldStats {
                        field_id: *k,
                        field_path: path,
                        stats: IndexFieldStatsType::CommittedStringFilter(v.stats()),
                    })
                }),
        );
        fields_stats.extend(committed_fields.string_fields.iter().filter_map(|(k, v)| {
            let path = v.field_path().join(".");
            Some(IndexFieldStats {
                field_id: *k,
                field_path: path,
                stats: IndexFieldStatsType::CommittedString(v.stats().ok()?),
            })
        }));
        fields_stats.extend(committed_fields.vector_fields.iter().filter_map(|(k, v)| {
            let path = v.field_path().join(".");
            Some(IndexFieldStats {
                field_id: *k,
                field_path: path,
                stats: IndexFieldStatsType::CommittedVector(v.stats().ok()?),
            })
        }));

        fields_stats.sort_by_key(|e| e.field_id.0);

        Ok(IndexStats {
            id: self.id,
            default_locale: self.locale,
            document_count: self.document_count as usize,
            is_temp,
            fields_stats,
            created_at: self.created_at,
            updated_at: self.updated_at,
        })
    }

    pub async fn calculate_facets(
        &self,
        token_scores: &HashMap<DocumentId, f32>,
        facets: &HashMap<String, FacetDefinition>,
        res_facets: &mut HashMap<String, FacetResult>,
    ) -> Result<()> {
        if facets.is_empty() {
            return Ok(());
        }

        info!("Computing facets on {:?}", facets.keys());

        let uncommitted_fields = self.uncommitted_fields.read().await;
        let committed_fields = self.committed_fields.read().await;

        for (field_name, facet) in facets {
            let Some((field_id, field_type)) = self.path_to_index_id_map.get(field_name) else {
                bail!("Unknown field name {}", field_name);
            };
            let facet_definition =
                res_facets
                    .entry(field_name.clone())
                    .or_insert_with(|| FacetResult {
                        count: 0,
                        values: HashMap::new(),
                    });

            // This calculation is not efficient
            // we have the doc_ids that matches:
            // - filters
            // - search
            // We should use them to calculate the facets
            // Instead here we are building an hashset and
            // iter again on it to filter the doc_ids.
            // We could create a dedicated method in the indexes that
            // accepts the matching doc_ids + facet definition and returns the count
            // TODO: do it
            match (facet, field_type) {
                (FacetDefinition::Number(facet), FieldType::Number) => {
                    let uncommitted = uncommitted_fields.number_fields.get(&field_id);
                    let committed = committed_fields.number_fields.get(&field_id);

                    for range in &facet.ranges {
                        let filter = NumberFilter::Between((range.from, range.to));

                        let uncommitted_output = uncommitted.map(|f| f.filter(&filter));
                        let committed_output = committed.map(|f| f.filter(&filter));

                        let facet = match (committed_output, uncommitted_output) {
                            (Some(Ok(committed_output)), Some(uncommitted_output)) => {
                                committed_output
                                    .chain(uncommitted_output)
                                    .filter(|doc_id| token_scores.contains_key(doc_id))
                                    .collect()
                            }
                            (Some(Ok(committed_output)), None) => committed_output
                                .filter(|doc_id| token_scores.contains_key(doc_id))
                                .collect(),
                            (Some(Err(_)), Some(uncommitted_output))
                            | (None, Some(uncommitted_output)) => uncommitted_output
                                .filter(|doc_id| token_scores.contains_key(doc_id))
                                .collect(),
                            (Some(Err(_)), None) | (None, None) => HashSet::new(),
                        };

                        let c = facet_definition
                            .values
                            .entry(format!("{}-{}", range.from, range.to))
                            .or_default();
                        *c += facet.len();
                    }
                }
                (FacetDefinition::Bool(facets), FieldType::Bool) => {
                    let uncommitted = uncommitted_fields.bool_fields.get(&field_id);
                    let committed = committed_fields.bool_fields.get(&field_id);

                    if facets.r#true {
                        let committed_output = committed.map(|f| f.filter(true));
                        let uncommitted_output = uncommitted.map(|f| f.filter(true));
                        let true_facet = match (committed_output, uncommitted_output) {
                            (Some(Ok(committed_output)), Some(uncommitted_output)) => {
                                committed_output
                                    .chain(uncommitted_output)
                                    .filter(|doc_id| token_scores.contains_key(doc_id))
                                    .collect()
                            }
                            (Some(Ok(committed_output)), None) => committed_output
                                .filter(|doc_id| token_scores.contains_key(doc_id))
                                .collect(),
                            (Some(Err(_)), Some(uncommitted_output))
                            | (None, Some(uncommitted_output)) => uncommitted_output
                                .filter(|doc_id| token_scores.contains_key(doc_id))
                                .collect(),
                            (Some(Err(_)), None) | (None, None) => HashSet::new(),
                        };

                        let c = facet_definition
                            .values
                            .entry("true".to_string())
                            .or_default();
                        *c += true_facet.len();
                    }
                    if facets.r#false {
                        let committed_output = committed.map(|f| f.filter(false));
                        let uncommitted_output = uncommitted.map(|f| f.filter(false));
                        let false_facet = match (committed_output, uncommitted_output) {
                            (Some(Ok(committed_output)), Some(uncommitted_output)) => {
                                committed_output
                                    .chain(uncommitted_output)
                                    .filter(|doc_id| token_scores.contains_key(doc_id))
                                    .collect()
                            }
                            (Some(Ok(committed_output)), None) => committed_output
                                .filter(|doc_id| token_scores.contains_key(doc_id))
                                .collect(),
                            (Some(Err(_)), Some(uncommitted_output))
                            | (None, Some(uncommitted_output)) => uncommitted_output
                                .filter(|doc_id| token_scores.contains_key(doc_id))
                                .collect(),
                            (Some(Err(_)), None) | (None, None) => HashSet::new(),
                        };

                        let c = facet_definition
                            .values
                            .entry("false".to_string())
                            .or_default();
                        *c += false_facet.len();
                    }
                }
                (FacetDefinition::String(_), FieldType::StringFilter) => {
                    let uncommitted = uncommitted_fields.string_filter_fields.get(&field_id);
                    let committed = committed_fields.string_filter_fields.get(&field_id);

                    let mut all_values = HashSet::new();
                    if let Some(values) = committed.map(|f| f.get_string_values()) {
                        all_values.extend(values);
                    }
                    if let Some(values) = uncommitted.map(|f| f.get_string_values()) {
                        all_values.extend(values);
                    }

                    for filter in all_values {
                        let committed_output = committed.map(|f| f.filter(filter));
                        let uncommitted_output = uncommitted.map(|f| f.filter(filter));

                        let mut facets = HashSet::new();
                        if let Some(committed_output) = committed_output {
                            facets.extend(
                                committed_output
                                    .filter(|doc_id| token_scores.contains_key(doc_id))
                                    .collect::<HashSet<_>>(),
                            );
                        }
                        if let Some(uncommitted_output) = uncommitted_output {
                            facets.extend(
                                uncommitted_output
                                    .filter(|doc_id| token_scores.contains_key(doc_id))
                                    .collect::<HashSet<_>>(),
                            );
                        }
                        let c = facet_definition
                            .values
                            .entry(filter.to_string())
                            .or_default();
                        *c += facets.len();
                    }
                }
                (_, t) => {
                    bail!(
                        "Cannot calculate facet on field {:?}: wrong type. Expected {:?} filter",
                        field_name,
                        t
                    );
                }
            }

            facet_definition.count += facet_definition.values.len();
        }

        Ok(())
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

        let uncommitted_fields = self.uncommitted_fields.read().await;
        let committed_fields = self.committed_fields.read().await;

        let mut filtered: HashSet<DocumentId> = Default::default();
        for (k, filter) in where_filter {
            let (field_id, field_type) = match self.path_to_index_id_map.get(k) {
                None => {
                    // If the user specified a field that is not in the index,
                    // we should return an empty set.
                    return Ok(Some(HashSet::new()));
                }
                Some((field_id, field_type)) => (field_id, field_type),
            };
            match (field_type, filter) {
                (FieldType::Bool, Filter::Bool(filter_bool)) => {
                    let uncommitted_field = uncommitted_fields
                        .bool_fields
                        .get(&field_id)
                        .ok_or_else(|| {
                            anyhow::anyhow!("Cannot filter by \"{}\": unknown field", &k)
                        })?;

                    filtered.extend(uncommitted_field.filter(*filter_bool));

                    let committed_field = committed_fields.bool_fields.get(&field_id);
                    if let Some(committed_field) = committed_field {
                        let a = committed_field.filter(*filter_bool).with_context(|| {
                            format!("Cannot filter by \"{}\": unknown field", &k)
                        })?;
                        filtered.extend(a);
                    }
                }
                (FieldType::Number, Filter::Number(filter_number)) => {
                    let uncommitted_field = uncommitted_fields
                        .number_fields
                        .get(&field_id)
                        .ok_or_else(|| {
                            anyhow::anyhow!("Cannot filter by \"{}\": unknown field", &k)
                        })?;

                    filtered.extend(uncommitted_field.filter(filter_number));

                    let committed_field = committed_fields.number_fields.get(&field_id);
                    if let Some(committed_field) = committed_field {
                        let a = committed_field.filter(filter_number).with_context(|| {
                            format!("Cannot filter by \"{}\": unknown field", &k)
                        })?;
                        filtered.extend(a);
                    }
                }
                (FieldType::StringFilter, Filter::String(filter_string)) => {
                    let uncommitted_field = uncommitted_fields
                        .string_filter_fields
                        .get(&field_id)
                        .ok_or_else(|| {
                            anyhow::anyhow!("Cannot filter by \"{}\": unknown field", &k)
                        })?;

                    filtered.extend(uncommitted_field.filter(filter_string));

                    let committed_field = committed_fields.string_filter_fields.get(&field_id);
                    if let Some(committed_field) = committed_field {
                        let a = committed_field.filter(filter_string);
                        filtered.extend(a);
                    }
                }
                _ => {
                    // If the user specified a field that is not in the index,
                    // we should return an empty set.
                    return Ok(Some(HashSet::new()));
                }
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

    async fn calculate_vector_properties(&self) -> Result<Vec<FieldId>> {
        let uncommitted_fields = self.uncommitted_fields.read().await;
        let committed_fields = self.committed_fields.read().await;

        let properties: HashSet<_> = uncommitted_fields
            .vector_fields
            .keys()
            .chain(committed_fields.vector_fields.keys())
            .copied()
            .collect();

        Ok(properties.into_iter().collect())
    }

    async fn calculate_string_properties(&self, properties: &Properties) -> Result<Vec<FieldId>> {
        let properties: HashSet<_> = match properties {
            Properties::Specified(properties) => {
                let mut field_ids = HashSet::new();
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
                    field_ids.insert(field_id);
                }
                field_ids
            }
            Properties::None | Properties::Star => {
                let uncommitted_fields = self.uncommitted_fields.read().await;
                let committed_fields = self.committed_fields.read().await;

                uncommitted_fields
                    .string_fields
                    .keys()
                    .chain(committed_fields.string_fields.keys())
                    .copied()
                    .collect()
            }
        };

        Ok(properties.into_iter().collect())
    }

    async fn search_full_text(
        &self,
        term: &str,
        threshold: Option<Threshold>,
        exact: bool,
        properties: Vec<FieldId>,
        boost: HashMap<FieldId, f32>,
        filtered_doc_ids: Option<&HashSet<DocumentId>>,
        uncommitted_deleted_documents: &HashSet<DocumentId>,
    ) -> Result<HashMap<DocumentId, f32>> {
        let uncommitted_fields = self.uncommitted_fields.read().await;
        let committed_fields = self.committed_fields.read().await;

        let tokens = self.text_parser.tokenize_and_stem(term);
        let tokens: Vec<_> = if exact {
            tokens.into_iter().map(|e| e.0).collect()
        } else {
            tokens
                .into_iter()
                .flat_map(|e| std::iter::once(e.0).chain(e.1.into_iter()))
                .collect()
        };

        let mut scorer: BM25Scorer<DocumentId> = match threshold {
            Some(Threshold(threshold)) => {
                let perc = tokens.len() as f32 * threshold;
                let threshold = perc.floor() as u32;
                BM25Scorer::with_threshold(threshold)
            }
            None => BM25Scorer::plain(),
        };

        for field_id in properties {
            let Some(field) = uncommitted_fields.string_fields.get(&field_id) else {
                bail!("Cannot search on field {:?}: unknown field", field_id);
            };
            let committed = committed_fields.string_fields.get(&field_id);

            let global_info = if let Some(committed) = committed {
                committed.global_info() + field.global_info()
            } else {
                field.global_info()
            };

            let boost = boost.get(&field_id).copied().unwrap_or(1.0);

            scorer.reset_term();
            field
                .search(
                    &tokens,
                    exact,
                    boost,
                    &mut scorer,
                    filtered_doc_ids,
                    &global_info,
                    uncommitted_deleted_documents,
                )
                .context("Cannot perform search")?;
            if let Some(committed) = committed {
                scorer.reset_term();
                committed
                    .search(
                        &tokens,
                        exact,
                        boost,
                        &mut scorer,
                        filtered_doc_ids,
                        &global_info,
                        uncommitted_deleted_documents,
                    )
                    .context("Cannot perform search")?;
            }
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

        let uncommitted_fields = self.uncommitted_fields.read().await;
        let committed_fields = self.committed_fields.read().await;

        for field_id in properties {
            let Some(uncommitted) = uncommitted_fields.vector_fields.get(&field_id) else {
                bail!("Cannot search on field {:?}: unknown field", field_id);
            };
            let committed = committed_fields.vector_fields.get(&field_id);

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
    pub document_count: usize,
    pub is_temp: bool,
    pub fields_stats: Vec<IndexFieldStats>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
enum FieldType {
    Bool,
    Number,
    StringFilter,
    String,
    Vector,
}

#[derive(Debug, Serialize, Deserialize)]
struct DumpV1 {
    id: IndexId,
    document_count: u64,
    locale: Locale,
    aliases: Vec<IndexId>,
    bool_field_ids: Vec<(FieldId, BoolFieldInfo)>,
    number_field_ids: Vec<(FieldId, NumberFieldInfo)>,
    string_filter_field_ids: Vec<(FieldId, StringFilterFieldInfo)>,
    string_field_ids: Vec<(FieldId, StringFieldInfo)>,
    vector_field_ids: Vec<(FieldId, VectorFieldInfo)>,
    path_to_index_id_map: Vec<(Box<[String]>, (FieldId, FieldType))>,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
}
#[derive(Debug, Serialize, Deserialize)]
enum Dump {
    V1(DumpV1),
}
