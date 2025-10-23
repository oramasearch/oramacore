use chrono::{DateTime, Utc};
use committed_field::{
    BoolFieldInfo, CommittedBoolField, CommittedNumberField, CommittedStringField,
    CommittedStringFilterField, CommittedVectorField, NumberFieldInfo, StringFieldInfo,
    StringFilterFieldInfo, VectorFieldInfo,
};
use futures::join;
use group::Groupable;
use merge::{
    merge_bool_field, merge_number_field, merge_string_field, merge_string_filter_field,
    merge_vector_field,
};
use oramacore_lib::filters::{FilterResult, PlainFilterResult};
use path_to_index_id_map::PathToIndexId;
use search_context::FullTextSearchContext;
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, trace, warn};
use uncommitted_field::*;

use crate::{
    ai::{llms, OramaModel},
    collection_manager::{
        bm25::BM25Scorer,
        sides::{
            read::{
                context::ReadSideContext,
                index::{
                    committed_field::{
                        CommittedDateField, CommittedGeoPointField, DateFieldInfo,
                        GeoPointFieldInfo,
                    },
                    merge::{merge_date_field, merge_geopoint_field},
                },
                ReadError,
            },
            write::index::EnumStrategy,
            Offset,
        },
    },
    lock::{OramaAsyncLock, OramaAsyncLockReadGuard},
    metrics::{
        commit::{FIELD_INDEX_COMMIT_CALCULATION_TIME, INDEX_COMMIT_CALCULATION_TIME},
        search::{MATCHING_COUNT_CALCULTATION_COUNT, MATCHING_PERC_CALCULATION_COUNT},
        CollectionLabels, FieldIndexCollectionCommitLabels, IndexCollectionCommitLabels,
    },
    types::{
        CollectionId, DocumentId, FacetDefinition, FacetResult, Filter, FulltextMode, HybridMode,
        IndexId, Limit, Number, NumberFilter, Properties, SearchMode, SearchModeResult,
        SearchParams, Similarity, SortOrder, Threshold, TypeParsingStrategies, VectorMode,
        WhereFilter,
    },
};
use oramacore_lib::fs::{create_if_not_exists, BufferedFile};
use oramacore_lib::nlp::{locales::Locale, TextParser};

use super::collection::{IndexFieldStats, IndexFieldStatsType};
use super::OffloadFieldConfig;
mod committed_field;
mod group;
mod merge;
mod path_to_index_id_map;
mod search_context;
mod uncommitted_field;

use crate::{
    collection_manager::sides::{
        write::index::IndexedValue, IndexWriteOperation, IndexWriteOperationFieldType,
    },
    types::FieldId,
};
use anyhow::{anyhow, bail, Context, Result};
pub use committed_field::{
    CommittedBoolFieldStats, CommittedDateFieldStats, CommittedGeoPointFieldStats,
    CommittedNumberFieldStats, CommittedStringFieldStats, CommittedStringFilterFieldStats,
    CommittedVectorFieldStats,
};
use debug_panic::debug_panic;
pub use group::GroupValue;
use oramacore_lib::pin_rules::PinRulesReader;
use std::iter::Peekable;
use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};
pub use uncommitted_field::{
    UncommittedBoolFieldStats, UncommittedDateFieldStats, UncommittedGeoPointFieldStats,
    UncommittedNumberFieldStats, UncommittedStringFieldStats, UncommittedStringFilterFieldStats,
    UncommittedVectorFieldStats,
};

#[derive(Debug, Clone, Copy)]
pub enum DeletionReason {
    UserWanted,
    CollectionReindexed { temp_index_id: IndexId },
    IndexResynced { temp_index_id: IndexId },
    ExpiredTempIndex,
}

#[derive(Default)]
struct UncommittedFields {
    bool_fields: HashMap<FieldId, UncommittedBoolField>,
    number_fields: HashMap<FieldId, UncommittedNumberField>,
    string_filter_fields: HashMap<FieldId, UncommittedStringFilterField>,
    date_fields: HashMap<FieldId, UncommittedDateFilterField>,
    geopoint_fields: HashMap<FieldId, UncommittedGeoPointFilterField>,
    string_fields: HashMap<FieldId, UncommittedStringField>,
    vector_fields: HashMap<FieldId, UncommittedVectorField>,
}

#[derive(Default)]
struct CommittedFields {
    bool_fields: HashMap<FieldId, CommittedBoolField>,
    number_fields: HashMap<FieldId, CommittedNumberField>,
    string_filter_fields: HashMap<FieldId, CommittedStringFilterField>,
    date_fields: HashMap<FieldId, CommittedDateField>,
    geopoint_fields: HashMap<FieldId, CommittedGeoPointField>,
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

    context: ReadSideContext,
    offload_config: OffloadFieldConfig,

    document_count: u64,
    uncommitted_deleted_documents: HashSet<DocumentId>,

    uncommitted_fields: OramaAsyncLock<UncommittedFields>,
    committed_fields: OramaAsyncLock<CommittedFields>,

    path_to_index_id_map: PathToIndexId,

    is_new: AtomicBool,

    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,

    pin_rules_reader: OramaAsyncLock<PinRulesReader<DocumentId>>,
    enum_strategy: EnumStrategy,
}

impl Index {
    pub fn new(
        id: IndexId,
        text_parser: Arc<TextParser>,
        context: ReadSideContext,
        offload_config: OffloadFieldConfig,
        enum_strategy: EnumStrategy,
    ) -> Self {
        Self {
            id,
            locale: text_parser.locale(),
            text_parser,
            aliases: vec![],
            deleted: None,
            promoted_to_runtime_index: AtomicBool::new(false),

            context,
            offload_config,

            document_count: 0,
            uncommitted_deleted_documents: HashSet::new(),

            committed_fields: OramaAsyncLock::new("committed_fields", Default::default()),
            uncommitted_fields: OramaAsyncLock::new("uncommitted_fields", Default::default()),

            path_to_index_id_map: PathToIndexId::empty(),
            is_new: AtomicBool::new(true),

            created_at: Utc::now(),
            updated_at: Utc::now(),

            pin_rules_reader: OramaAsyncLock::new("pin_rules_reader", PinRulesReader::empty()),

            enum_strategy,
        }
    }

    pub fn try_load(
        index_id: IndexId,
        data_dir: PathBuf,
        context: ReadSideContext,
        offload_config: OffloadFieldConfig,
    ) -> Result<Self> {
        debug!("Reading index info");
        let dump: Dump = BufferedFile::open(data_dir.join("index.json"))
            .context("Cannot open index.json")?
            .read_json_data()
            .context("Cannot read index.json")?;
        let Dump::V1(dump) = dump;
        debug!("Index info read");

        debug_assert_eq!(
            dump.id, index_id,
            "Index id mismatch: expected {:?}, got {:?}",
            index_id, dump.id
        );
        debug!("DONE");

        let mut filter_fields: HashMap<Box<[String]>, (FieldId, FieldType)> = Default::default();
        let mut score_fields: HashMap<Box<[String]>, (FieldId, FieldType)> = Default::default();

        let mut uncommitted_fields = UncommittedFields::default();
        let mut committed_fields = CommittedFields::default();
        debug!("Loading bool fields");
        for (field_id, info) in dump.bool_field_ids {
            filter_fields.insert(info.field_path.clone(), (field_id, FieldType::Bool));

            uncommitted_fields.bool_fields.insert(
                field_id,
                UncommittedBoolField::empty(info.field_path.clone()),
            );
            debug!("CommittedBoolField::try_load for field_id {:?}", field_id);
            let field = CommittedBoolField::try_load(info).context("Cannot load bool field")?;
            debug!("DONE");
            committed_fields.bool_fields.insert(field_id, field);
        }
        debug!("Bool fields loaded");

        debug!("Loading number_field_ids");
        for (field_id, info) in dump.number_field_ids {
            filter_fields.insert(info.field_path.clone(), (field_id, FieldType::Number));

            uncommitted_fields.number_fields.insert(
                field_id,
                UncommittedNumberField::empty(info.field_path.clone()),
            );
            let field = CommittedNumberField::try_load(info).context("Cannot load number field")?;
            committed_fields.number_fields.insert(field_id, field);
        }
        debug!("Number fields loaded");

        debug!("Loading date_field_ids");
        for (field_id, info) in dump.date_field_ids {
            filter_fields.insert(info.field_path.clone(), (field_id, FieldType::Date));

            uncommitted_fields.date_fields.insert(
                field_id,
                UncommittedDateFilterField::empty(info.field_path.clone()),
            );
            let field = CommittedDateField::try_load(info).context("Cannot load date field")?;
            committed_fields.date_fields.insert(field_id, field);
        }

        debug!("Loading geopoint_field_ids");
        for (field_id, info) in dump.geopoint_field_ids {
            filter_fields.insert(info.field_path.clone(), (field_id, FieldType::GeoPoint));

            uncommitted_fields.geopoint_fields.insert(
                field_id,
                UncommittedGeoPointFilterField::empty(info.field_path.clone()),
            );
            let field =
                CommittedGeoPointField::try_load(info).context("Cannot load geopoint field")?;
            committed_fields.geopoint_fields.insert(field_id, field);
        }

        debug!("Loading string_filter_field_ids");
        for (field_id, info) in dump.string_filter_field_ids {
            filter_fields.insert(info.field_path.clone(), (field_id, FieldType::StringFilter));

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

        debug!("Loading string_field_ids");
        for (field_id, info) in dump.string_field_ids {
            score_fields.insert(info.field_path.clone(), (field_id, FieldType::String));

            uncommitted_fields.string_fields.insert(
                field_id,
                UncommittedStringField::empty(info.field_path.clone()),
            );
            let field = CommittedStringField::try_load(info, offload_config)
                .context("Cannot load string field")?;
            committed_fields.string_fields.insert(field_id, field);
        }

        debug!("Loading vector_field_ids");
        for (field_id, info) in dump.vector_field_ids {
            score_fields.insert(info.field_path.clone(), (field_id, FieldType::Vector));

            uncommitted_fields.vector_fields.insert(
                field_id,
                UncommittedVectorField::empty(info.field_path.clone(), info.model.0),
            );
            let field = CommittedVectorField::try_load(info, offload_config)
                .context("Cannot load vector field")?;
            committed_fields.vector_fields.insert(field_id, field);
        }
        debug!("Vector fields loaded");

        Ok(Self {
            id: dump.id,
            locale: dump.locale,
            text_parser: context.nlp_service.get(dump.locale),
            deleted: None,
            promoted_to_runtime_index: AtomicBool::new(false),
            aliases: dump.aliases,

            context,
            offload_config,

            document_count: dump.document_count,
            uncommitted_deleted_documents: HashSet::new(),

            committed_fields: OramaAsyncLock::new("committed_fields", committed_fields),
            uncommitted_fields: OramaAsyncLock::new("uncommitted_fields", uncommitted_fields),

            path_to_index_id_map: PathToIndexId::new(filter_fields, score_fields),
            is_new: AtomicBool::new(false),

            created_at: dump.created_at,
            updated_at: dump.updated_at,

            pin_rules_reader: OramaAsyncLock::new(
                "pin_rules_reader",
                PinRulesReader::try_new(data_dir.join("pin_rules"))?,
            ),
            enum_strategy: dump.enum_strategy,
        })
    }

    pub async fn get_all_document_ids(&self) -> Result<Vec<DocumentId>> {
        let properties = Properties::default();
        let properties = self.calculate_string_properties(&properties).await?;
        let output = self
            .search_full_text(
                "",
                None,
                false,
                None,
                properties,
                Default::default(),
                None,
                &Default::default(),
            )
            .await
            .expect("Failed to get all documents");
        let document_ids: Vec<_> = output.keys().copied().collect();

        Ok(document_ids)
    }

    pub async fn commit(
        &self,
        data_dir: PathBuf,
        offset: Offset,
        collection_id: CollectionId,
    ) -> Result<()> {
        debug!("Starting to commit index {:?}", self.id);

        let data_dir_with_offset = data_dir.join(format!("offset-{}", offset.0));

        let uncommitted_fields = self.uncommitted_fields.read("commit").await;

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
                .date_fields
                .iter()
                .any(|(_, field)| !field.is_empty())
            || uncommitted_fields
                .geopoint_fields
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

            self.try_unload_fields().await;

            return Ok(());
        }

        create_if_not_exists(data_dir_with_offset.clone())
            .context("Cannot create data directory")?;

        debug!("Committing index {:?}", self.id);

        let total_m = INDEX_COMMIT_CALCULATION_TIME.create(IndexCollectionCommitLabels {
            collection: collection_id.as_str().to_string(),
            index: self.id.as_str().to_string(),
            side: "read",
        });

        let committed_fields = self.committed_fields.read("commit").await;

        enum MergeResult<T> {
            Changed(T),
            Unchanged,
        }

        let m = FIELD_INDEX_COMMIT_CALCULATION_TIME.create(FieldIndexCollectionCommitLabels {
            collection: collection_id.as_str().to_string(),
            index: self.id.as_str().to_string(),
            side: "read",
            field_type: "bool",
        });
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
        drop(m);

        let m = FIELD_INDEX_COMMIT_CALCULATION_TIME.create(FieldIndexCollectionCommitLabels {
            collection: collection_id.as_str().to_string(),
            index: self.id.as_str().to_string(),
            side: "read",
            field_type: "number",
        });
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
        drop(m);

        let m = FIELD_INDEX_COMMIT_CALCULATION_TIME.create(FieldIndexCollectionCommitLabels {
            collection: collection_id.as_str().to_string(),
            index: self.id.as_str().to_string(),
            side: "read",
            field_type: "date",
        });
        let mut merged_dates = HashMap::new();
        let date_indexes: HashSet<_> = uncommitted_fields
            .date_fields
            .keys()
            .chain(committed_fields.date_fields.keys())
            .copied()
            .collect();
        for field_id in date_indexes {
            let data_dir = data_dir_with_offset.join(field_id.0.to_string());
            let uncommitted = uncommitted_fields.date_fields.get(&field_id);
            let committed = committed_fields.date_fields.get(&field_id);
            if let Some(merged) = merge_date_field(
                uncommitted,
                committed,
                data_dir,
                &self.uncommitted_deleted_documents,
                is_promoted,
            )
            .context("Cannot merge date field")?
            {
                merged_dates.insert(field_id, MergeResult::Changed(merged));
            } else {
                merged_dates.insert(field_id, MergeResult::Unchanged);
            }
        }
        drop(m);

        let m = FIELD_INDEX_COMMIT_CALCULATION_TIME.create(FieldIndexCollectionCommitLabels {
            collection: collection_id.as_str().to_string(),
            index: self.id.as_str().to_string(),
            side: "read",
            field_type: "geopoint",
        });
        let mut merged_geopoints = HashMap::new();
        let geopoint_indexes: HashSet<_> = uncommitted_fields
            .geopoint_fields
            .keys()
            .chain(committed_fields.geopoint_fields.keys())
            .copied()
            .collect();
        for field_id in geopoint_indexes {
            let data_dir = data_dir_with_offset.join(field_id.0.to_string());
            let uncommitted = uncommitted_fields.geopoint_fields.get(&field_id);
            let committed = committed_fields.geopoint_fields.get(&field_id);
            let output = merge_geopoint_field(
                uncommitted,
                committed,
                data_dir,
                &self.uncommitted_deleted_documents,
                is_promoted,
            )
            .context("Cannot merge geopoint field")?;

            if let Some(merged) = output {
                merged_geopoints.insert(field_id, MergeResult::Changed(merged));
            } else {
                merged_geopoints.insert(field_id, MergeResult::Unchanged);
            }
        }
        drop(m);

        let m = FIELD_INDEX_COMMIT_CALCULATION_TIME.create(FieldIndexCollectionCommitLabels {
            collection: collection_id.as_str().to_string(),
            index: self.id.as_str().to_string(),
            side: "read",
            field_type: "string_filter",
        });
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
        drop(m);

        let m = FIELD_INDEX_COMMIT_CALCULATION_TIME.create(FieldIndexCollectionCommitLabels {
            collection: collection_id.as_str().to_string(),
            index: self.id.as_str().to_string(),
            side: "read",
            field_type: "string",
        });
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
                &self.offload_config,
            )
            .context("Cannot merge string field")?
            {
                merged_strings.insert(field_id, MergeResult::Changed(merged));
            } else {
                merged_strings.insert(field_id, MergeResult::Unchanged);
            }
        }
        drop(m);

        let m = FIELD_INDEX_COMMIT_CALCULATION_TIME.create(FieldIndexCollectionCommitLabels {
            collection: collection_id.as_str().to_string(),
            index: self.id.as_str().to_string(),
            side: "read",
            field_type: "vector",
        });
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
                &self.offload_config,
            )
            .context("Cannot merge vector field")?
            {
                merged_vectors.insert(field_id, MergeResult::Changed(merged));
            } else {
                merged_vectors.insert(field_id, MergeResult::Unchanged);
            }
        }
        drop(m);

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
        let mut uncommitted_fields = self.uncommitted_fields.write("commit").await;
        let mut committed_fields = self.committed_fields.write("commit").await;

        for (field_id, merged) in merged_bools {
            match merged {
                MergeResult::Changed(merged) => {
                    #[cfg(debug_assertions)]
                    {
                        let data_dir = merged.get_field_info().data_dir;
                        let offset_str = data_dir.components().rev().nth(1).unwrap();
                        assert_eq!(
                            offset_str.as_os_str().to_str().unwrap(),
                            &format!("offset-{}", offset.0),
                            "Vector field data dir offset mismatch after commit"
                        );
                        assert!(std::fs::exists(data_dir).unwrap());
                    }
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
                    #[cfg(debug_assertions)]
                    {
                        let data_dir = merged.get_field_info().data_dir;
                        let offset_str = data_dir.components().rev().nth(1).unwrap();
                        assert_eq!(
                            offset_str.as_os_str().to_str().unwrap(),
                            &format!("offset-{}", offset.0),
                            "Vector field data dir offset mismatch after commit"
                        );
                        assert!(std::fs::exists(data_dir).unwrap());
                    }
                    committed_fields.number_fields.insert(field_id, merged);
                }
                MergeResult::Unchanged => {}
            }
            if let Some(uncommitted) = uncommitted_fields.number_fields.get_mut(&field_id) {
                uncommitted.clear();
            }
        }
        for (field_id, merged) in merged_dates {
            match merged {
                MergeResult::Changed(merged) => {
                    #[cfg(debug_assertions)]
                    {
                        let data_dir = merged.get_field_info().data_dir;
                        let offset_str = data_dir.components().rev().nth(1).unwrap();
                        assert_eq!(
                            offset_str.as_os_str().to_str().unwrap(),
                            &format!("offset-{}", offset.0),
                            "Vector field data dir offset mismatch after commit"
                        );
                        assert!(std::fs::exists(data_dir).unwrap());
                    }
                    committed_fields.date_fields.insert(field_id, merged);
                }
                MergeResult::Unchanged => {}
            }
            if let Some(uncommitted) = uncommitted_fields.date_fields.get_mut(&field_id) {
                uncommitted.clear();
            }
        }
        for (field_id, merged) in merged_geopoints {
            match merged {
                MergeResult::Changed(merged) => {
                    #[cfg(debug_assertions)]
                    {
                        let data_dir = merged.get_field_info().data_dir;
                        let offset_str = data_dir.components().rev().nth(1).unwrap();
                        assert_eq!(
                            offset_str.as_os_str().to_str().unwrap(),
                            &format!("offset-{}", offset.0),
                            "Vector field data dir offset mismatch after commit"
                        );
                        assert!(std::fs::exists(data_dir).unwrap());
                    }
                    committed_fields.geopoint_fields.insert(field_id, merged);
                }
                MergeResult::Unchanged => {}
            }
            if let Some(uncommitted) = uncommitted_fields.geopoint_fields.get_mut(&field_id) {
                uncommitted.clear();
            }
        }
        for (field_id, merged) in merged_string_filters {
            match merged {
                MergeResult::Changed(merged) => {
                    #[cfg(debug_assertions)]
                    {
                        let data_dir = merged.get_field_info().data_dir;
                        let offset_str = data_dir.components().rev().nth(1).unwrap();
                        assert_eq!(
                            offset_str.as_os_str().to_str().unwrap(),
                            &format!("offset-{}", offset.0),
                            "Vector field data dir offset mismatch after commit"
                        );
                        assert!(std::fs::exists(data_dir).unwrap());
                    }
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
                    #[cfg(debug_assertions)]
                    {
                        let data_dir = merged.get_field_info().data_dir;
                        let offset_str = data_dir.components().rev().nth(1).unwrap();
                        assert_eq!(
                            offset_str.as_os_str().to_str().unwrap(),
                            &format!("offset-{}", offset.0),
                            "Vector field data dir offset mismatch after commit"
                        );
                        assert!(std::fs::exists(data_dir).unwrap());
                    }
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
                    #[cfg(debug_assertions)]
                    {
                        let data_dir = merged.get_field_info().data_dir;
                        let offset_str = data_dir.components().rev().nth(1).unwrap();
                        assert_eq!(
                            offset_str.as_os_str().to_str().unwrap(),
                            &format!("offset-{}", offset.0),
                            "Vector field data dir offset mismatch after commit"
                        );
                        assert!(std::fs::exists(data_dir).unwrap());
                    }
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
            date_field_ids: committed_fields
                .date_fields
                .iter()
                .map(|(k, v)| (*k, v.get_field_info()))
                .collect(),
            geopoint_field_ids: committed_fields
                .geopoint_fields
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
            // Not used anymore. We calculate it on the fly
            path_to_index_id_map: Vec::new(),
            created_at: self.created_at,
            updated_at: self.updated_at,
            enum_strategy: self.enum_strategy,
        });

        drop(uncommitted_fields);
        drop(committed_fields);

        let mut pin_rules_reader = self.pin_rules_reader.write("commit").await;
        pin_rules_reader
            .commit(data_dir.join("pin_rules"))
            .context("Cannot commit pin rules")?;
        drop(pin_rules_reader);

        self.try_unload_fields().await;

        BufferedFile::create_or_overwrite(data_dir.join("index.json"))
            .context("Cannot create index.json")?
            .write_json_data(&dump)
            .context("Cannot write index.json")?;

        // Force flags after commit
        self.promoted_to_runtime_index
            .store(false, Ordering::Relaxed);
        self.is_new.store(false, Ordering::Relaxed);

        drop(total_m);

        debug!("Index committed: {:?}", self.id);

        Ok(())
    }

    pub async fn clean_up(&self, index_data_dir: PathBuf) -> Result<()> {
        info!("Clean up");
        let committed_fields = self.committed_fields.read("clean_up").await;

        let field_data_dirs: HashSet<_> = committed_fields
            .bool_fields
            .values()
            .map(|f| f.get_field_info().data_dir)
            .chain(
                committed_fields
                    .number_fields
                    .values()
                    .map(|f| f.get_field_info().data_dir),
            )
            .chain(
                committed_fields
                    .string_filter_fields
                    .values()
                    .map(|f| f.get_field_info().data_dir),
            )
            .chain(
                committed_fields
                    .date_fields
                    .values()
                    .map(|f| f.get_field_info().data_dir),
            )
            .chain(
                committed_fields
                    .geopoint_fields
                    .values()
                    .map(|f| f.get_field_info().data_dir),
            )
            .chain(
                committed_fields
                    .string_fields
                    .values()
                    .map(|f| f.get_field_info().data_dir),
            )
            .chain(
                committed_fields
                    .vector_fields
                    .values()
                    .map(|f| f.get_field_info().data_dir),
            )
            .map(|f| f.parent().unwrap().to_path_buf())
            .collect();

        let subfolders = match std::fs::read_dir(index_data_dir) {
            Ok(a) => a,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // No data dir, nothing to clean
                return Ok(());
            }
            Err(e) => {
                return Err(anyhow::anyhow!(
                    "Cannot read index data dir for cleanup: {:?}",
                    e
                ));
            }
        };
        let subfolders: Result<Vec<_>, _> = subfolders.collect();

        let subfolders = subfolders.context("Cannot read entry in index folder")?;

        for entry in subfolders {
            let a = entry.file_type().context("Cannot get file type")?;
            if !a.is_dir() {
                continue;
            }

            if field_data_dirs.contains(&entry.path()) {
                continue;
            }

            if !entry.file_name().to_str().unwrap().starts_with("offset-") {
                continue;
            }

            info!("Removing unused index data folder: {:?}", entry.path());
            std::fs::remove_dir_all(entry.path()).unwrap();
        }

        Ok(())
    }

    async fn try_unload_fields(&self) {
        let lock = self.committed_fields.read("try_unload_fields").await;
        for string_field in lock.string_fields.values() {
            string_field.unload_if_not_used();
        }
        for vector_field in lock.vector_fields.values() {
            vector_field.unload_if_not_used();
        }
        drop(lock);
    }

    #[inline]
    pub fn id(&self) -> IndexId {
        self.id
    }

    #[inline]
    pub fn aliases(&self) -> &[IndexId] {
        self.aliases.as_slice()
    }

    #[inline]
    pub fn created_at(&self) -> DateTime<Utc> {
        self.created_at
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
        debug_assert!(self.deleted.is_none(), "Index is already deleted");
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
                match field_type {
                    IndexWriteOperationFieldType::Bool => {
                        self.path_to_index_id_map.insert_filter_field(
                            field_path.clone(),
                            field_id,
                            FieldType::Bool,
                        );
                        uncommitted_fields
                            .bool_fields
                            .insert(field_id, UncommittedBoolField::empty(field_path));
                    }
                    IndexWriteOperationFieldType::Number => {
                        self.path_to_index_id_map.insert_filter_field(
                            field_path.clone(),
                            field_id,
                            FieldType::Number,
                        );
                        uncommitted_fields
                            .number_fields
                            .insert(field_id, UncommittedNumberField::empty(field_path));
                    }
                    IndexWriteOperationFieldType::StringFilter => {
                        self.path_to_index_id_map.insert_filter_field(
                            field_path.clone(),
                            field_id,
                            FieldType::StringFilter,
                        );
                        uncommitted_fields
                            .string_filter_fields
                            .insert(field_id, UncommittedStringFilterField::empty(field_path));
                    }
                    IndexWriteOperationFieldType::Date => {
                        self.path_to_index_id_map.insert_filter_field(
                            field_path.clone(),
                            field_id,
                            FieldType::Date,
                        );
                        uncommitted_fields
                            .date_fields
                            .insert(field_id, UncommittedDateFilterField::empty(field_path));
                    }
                    IndexWriteOperationFieldType::GeoPoint => {
                        self.path_to_index_id_map.insert_filter_field(
                            field_path.clone(),
                            field_id,
                            FieldType::GeoPoint,
                        );
                        uncommitted_fields
                            .geopoint_fields
                            .insert(field_id, UncommittedGeoPointFilterField::empty(field_path));
                    }
                    IndexWriteOperationFieldType::String => {
                        self.path_to_index_id_map.insert_score_field(
                            field_path.clone(),
                            field_id,
                            FieldType::String,
                        );
                        uncommitted_fields
                            .string_fields
                            .insert(field_id, UncommittedStringField::empty(field_path));
                    }
                    IndexWriteOperationFieldType::Embedding(model) => {
                        self.path_to_index_id_map.insert_score_field(
                            field_path.clone(),
                            field_id,
                            FieldType::Vector,
                        );
                        uncommitted_fields
                            .vector_fields
                            .insert(field_id, UncommittedVectorField::empty(field_path, model.0));
                    }
                };
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
                        IndexedValue::FilterDate(field_id, timestamp) => {
                            if let Some(field) = uncommitted_fields.date_fields.get_mut(&field_id) {
                                field.insert(doc_id, timestamp);
                            } else {
                                error!("Cannot find field {:?} in uncommitted fields", field_id);
                            }
                        }
                        IndexedValue::FilterGeoPoint(field_id, geopoint) => {
                            if let Some(field) =
                                uncommitted_fields.geopoint_fields.get_mut(&field_id)
                            {
                                field.insert(doc_id, geopoint);
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
            IndexWriteOperation::PinRule(op) => {
                let pin_rules_lock = self.pin_rules_reader.get_mut();
                pin_rules_lock
                    .update(op)
                    .context("Cannot apply pin rule operation")?;
            }
        };

        Ok(())
    }

    pub fn has_field(&self, field_name: &str) -> bool {
        self.path_to_index_id_map.get(field_name).is_some()
    }

    pub fn has_filter_field(&self, field_name: &str) -> bool {
        self.path_to_index_id_map
            .get_filter_field(field_name)
            .is_some()
    }

    pub async fn get_sort_iterator<'index>(
        &'index self,
        field_name: &str,
        order: SortOrder,
    ) -> Result<SortedField<'index>, ReadError> {
        let Some((field_id, field_type)) = self.path_to_index_id_map.get_filter_field(field_name)
        else {
            return Err(ReadError::SortFieldNotFound(field_name.to_string()));
        };

        let uncommitted_fields = self.uncommitted_fields.read("get_sort_iterator").await;
        let committed_fields = self.committed_fields.read("get_sort_iterator").await;

        match &field_type {
            FieldType::Number => {
                if !uncommitted_fields.number_fields.contains_key(&field_id) {
                    return Err(ReadError::Generic(anyhow::anyhow!(
                        "Field {} is not a number field",
                        field_name
                    )));
                };

                Ok(SortedField {
                    uncommitted_fields,
                    committed_fields,
                    field_id,
                    field_type,
                    order,
                })
            }
            FieldType::Bool => {
                if !uncommitted_fields.bool_fields.contains_key(&field_id) {
                    return Err(ReadError::Generic(anyhow::anyhow!(
                        "Field {} is not a bool field",
                        field_name
                    )));
                };

                Ok(SortedField {
                    uncommitted_fields,
                    committed_fields,
                    field_id,
                    field_type,
                    order,
                })
            }

            FieldType::Date => {
                if !uncommitted_fields.date_fields.contains_key(&field_id) {
                    return Err(ReadError::Generic(anyhow::anyhow!(
                        "Field {} is not a date field",
                        field_name
                    )));
                };

                Ok(SortedField {
                    uncommitted_fields,
                    committed_fields,
                    field_id,
                    field_type,
                    order,
                })
            }
            _ => Err(ReadError::InvalidSortField(
                field_name.to_string(),
                format!("{field_type:?}"),
            )),
        }
    }

    // Since we only have one embedding model for all indexes in a collection,
    // we can get the first index model and return it early.
    pub async fn get_model(&self) -> Option<OramaModel> {
        let uncommitted_fields = self.uncommitted_fields.read("get_model").await;
        uncommitted_fields
            .vector_fields
            .values()
            .next()
            .map(|f| f.get_model())
    }

    pub async fn calculate_token_score(
        &self,
        search_params: &SearchParams,
        results: &mut HashMap<DocumentId, f32>,
    ) -> Result<()> {
        let uncommitted_deleted_documents = &self.uncommitted_deleted_documents;

        let filtered_doc_ids = self
            .calculate_filtered_doc_ids(&search_params.where_filter, uncommitted_deleted_documents)
            .await
            .with_context(|| format!("Cannot calculate filtered doc in index {:?}", self.id))?;

        // Manage the "auto" search mode. OramaCore will automatically determine
        // wether to use full text search, vector search or hybrid search.
        let search_mode: SearchMode = match &search_params.mode {
            SearchMode::Auto(mode_result) => {
                let final_mode: String = self
                    .context
                    .llm_service
                    .run_known_prompt(
                        llms::KnownPrompts::Autoquery,
                        vec![],
                        vec![("query".to_string(), mode_result.term.clone())],
                        None,
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
                        tolerance: None,
                    }),
                    "hybrid" => SearchMode::Hybrid(HybridMode {
                        term: mode_result.term.clone(),
                        similarity: Similarity(0.8),
                        threshold: None,
                        exact: false,
                        tolerance: None,
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
                        search_mode.tolerance,
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
                        search_mode.tolerance,
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

    pub async fn stats(&self, is_temp: bool, with_keys: bool) -> Result<IndexStats> {
        let mut fields_stats = Vec::new();

        let uncommitted_fields = self.uncommitted_fields.read("stats").await;
        let committed_fields = self.committed_fields.read("stats").await;

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
        fields_stats.extend(uncommitted_fields.date_fields.iter().map(|(k, v)| {
            let path = v.field_path().join(".");
            IndexFieldStats {
                field_id: *k,
                field_path: path,
                stats: IndexFieldStatsType::UncommittedDate(v.stats()),
            }
        }));
        fields_stats.extend(uncommitted_fields.geopoint_fields.iter().map(|(k, v)| {
            let path = v.field_path().join(".");
            IndexFieldStats {
                field_id: *k,
                field_path: path,
                stats: IndexFieldStatsType::UncommittedGeoPoint(v.stats()),
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
                        stats: IndexFieldStatsType::UncommittedStringFilter(v.stats(with_keys)),
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
        fields_stats.extend(committed_fields.date_fields.iter().filter_map(|(k, v)| {
            let path = v.field_path().join(".");
            Some(IndexFieldStats {
                field_id: *k,
                field_path: path,
                stats: IndexFieldStatsType::CommittedDate(v.stats().ok()?),
            })
        }));
        fields_stats.extend(
            committed_fields
                .geopoint_fields
                .iter()
                .filter_map(|(k, v)| {
                    let path = v.field_path().join(".");
                    Some(IndexFieldStats {
                        field_id: *k,
                        field_path: path,
                        stats: IndexFieldStatsType::CommittedGeoPoint(v.stats().ok()?),
                    })
                }),
        );
        fields_stats.extend(committed_fields.string_filter_fields.iter().map(|(k, v)| {
            let path = v.field_path().join(".");
            IndexFieldStats {
                field_id: *k,
                field_path: path,
                stats: IndexFieldStatsType::CommittedStringFilter(v.stats(with_keys)),
            }
        }));
        fields_stats.extend(committed_fields.string_fields.iter().map(|(k, v)| {
            let path = v.field_path().join(".");
            IndexFieldStats {
                field_id: *k,
                field_path: path,
                stats: IndexFieldStatsType::CommittedString(v.stats()),
            }
        }));
        fields_stats.extend(committed_fields.vector_fields.iter().map(|(k, v)| {
            let path = v.field_path().join(".");
            IndexFieldStats {
                field_id: *k,
                field_path: path,
                stats: IndexFieldStatsType::CommittedVector(v.stats()),
            }
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
            type_parsing_strategies: TypeParsingStrategies {
                enum_strategy: self.enum_strategy,
            },
        })
    }

    pub fn document_count(&self) -> u64 {
        self.document_count
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

        let uncommitted_fields = self.uncommitted_fields.read("facets").await;
        let committed_fields = self.committed_fields.read("facets").await;

        for (field_name, facet) in facets {
            let Some((field_id, field_type)) =
                self.path_to_index_id_map.get_filter_field(field_name)
            else {
                warn!("Unknown field name '{}'", field_name);
                continue;
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
                        "Cannot calculate facet on field {:?}: wrong type. Expected {:?} facet",
                        field_name,
                        t
                    );
                }
            }

            facet_definition.count += facet_definition.values.len();
        }

        Ok(())
    }

    fn generate_group_combinations(
        field_ids: &[FieldId],
        all_variants: &HashMap<FieldId, HashMap<GroupValue, HashSet<DocumentId>>>,
        current_combination: &mut Vec<(FieldId, GroupValue)>,
        result: &mut Vec<Vec<(FieldId, GroupValue)>>,
    ) {
        if current_combination.len() == field_ids.len() {
            result.push(current_combination.clone());
            return;
        }

        let field_id = field_ids[current_combination.len()];
        if let Some(variants) = all_variants.get(&field_id) {
            for group_value in variants.keys() {
                current_combination.push((field_id, group_value.clone()));
                Self::generate_group_combinations(
                    field_ids,
                    all_variants,
                    current_combination,
                    result,
                );
                current_combination.pop();
            }
        }
    }

    pub async fn calculate_groups(
        &self,
        properties_on_group: &[String],
        results: &mut HashMap<Vec<GroupValue>, HashSet<DocumentId>>,
    ) -> Result<()> {
        let Some(properties) = properties_on_group
            .iter()
            .map(|field_name| self.path_to_index_id_map.get_filter_field(field_name))
            .collect::<Option<Vec<_>>>()
        else {
            // Index has to contain all the property.
            // Otherwise, its document cannot be inside the buckets
            return Ok(());
        };

        let committed = self.committed_fields.read("groups").await;
        let uncommitted = self.uncommitted_fields.read("groups").await;

        let mut all_variants = HashMap::<FieldId, HashMap<GroupValue, HashSet<DocumentId>>>::new();
        for (field_id, field_type) in &properties {
            let uncommitted_groupable: Box<&dyn Groupable> = match field_type {
                FieldType::Bool => {
                    let f = uncommitted
                        .bool_fields
                        .get(field_id)
                        .ok_or(anyhow!("Cannot find bool field {:?}", field_id))?;
                    Box::new(f)
                }
                FieldType::Number => {
                    let f = uncommitted
                        .number_fields
                        .get(field_id)
                        .ok_or(anyhow!("Cannot find number field {:?}", field_id))?;
                    Box::new(f)
                }
                FieldType::StringFilter => {
                    let f = uncommitted
                        .string_filter_fields
                        .get(field_id)
                        .ok_or(anyhow!("Cannot find string filter field {:?}", field_id))?;
                    Box::new(f)
                }
                FieldType::GeoPoint => {
                    bail!("Cannot calculate group on a GeoPoint field");
                }
                _ => {
                    debug_panic!("Invalid field type {:?} for group", field_type);
                    bail!("Cannot calculate group on {:?} field", field_type);
                }
            };

            let variants: Vec<_> = uncommitted_groupable.get_values().collect();
            if variants.len() > 500 {
                // Should we group by a field with more than 500 variant???
                // TODO: think about it. For now, no.
                bail!("Cannot calculate groups on a field with more than 500 variant");
            }

            for variant in &variants {
                let docs: HashSet<_> = uncommitted_groupable.get_doc_ids(variant)?.collect();
                let doc_ids_per_variant = all_variants.entry(*field_id).or_default();
                let g = doc_ids_per_variant.entry(variant.clone()).or_default();
                g.extend(docs);
            }

            let committed_groupable: Option<Box<&dyn Groupable>> = match field_type {
                FieldType::Bool => committed
                    .bool_fields
                    .get(field_id)
                    .map(|f| Box::<&dyn Groupable>::new(f)),
                FieldType::Number => committed
                    .number_fields
                    .get(field_id)
                    .map(|f| Box::<&dyn Groupable>::new(f)),
                FieldType::StringFilter => committed
                    .string_filter_fields
                    .get(field_id)
                    .map(|f| Box::<&dyn Groupable>::new(f)),
                FieldType::GeoPoint => {
                    bail!("Cannot calculate group on a GeoPoint field");
                }
                _ => {
                    debug_panic!("Invalid field type {:?} for group", field_type);
                    bail!("Cannot calculate group on {:?} field", field_type);
                }
            };

            if let Some(committed_groupable) = committed_groupable {
                let variants: Vec<_> = committed_groupable.get_values().collect();
                if variants.len() > 500 {
                    // Should we group by a field with more than 500 variant???
                    // TODO: think about it. For now, no.
                    bail!("Cannot calculate groups on a field with more than 500 variant");
                }

                for variant in &variants {
                    let docs: HashSet<_> = committed_groupable.get_doc_ids(variant)?.collect();
                    let doc_ids_per_variant = all_variants.entry(*field_id).or_default();
                    let g = doc_ids_per_variant.entry(variant.clone()).or_default();
                    g.extend(docs);
                }
            }
        }

        drop(committed);
        drop(uncommitted);

        let field_ids: Vec<_> = properties.iter().map(|(field_id, _)| *field_id).collect();
        let mut groups = Vec::new();
        Self::generate_group_combinations(&field_ids, &all_variants, &mut Vec::new(), &mut groups);

        // Filter groups by documents that are in token_scores and create final result
        for group_combination in groups {
            // Find intersection of all document sets for this combination
            let mut intersection: Option<HashSet<DocumentId>> = None;
            let mut group_values = Vec::new();

            for (field_id, group_value) in group_combination {
                let Some(docs) = all_variants
                    .get(&field_id)
                    .and_then(|map| map.get(&group_value))
                else {
                    debug_panic!("Cannot calculate group values on field {:?}", field_id);
                    continue;
                };

                intersection = if let Some(intersection) = intersection {
                    Some(intersection.intersection(docs).cloned().collect())
                } else {
                    Some(docs.clone())
                };

                group_values.push(group_value);
            }

            let doc_ids_for_key = results.entry(group_values).or_default();
            if let Some(intersection) = intersection {
                doc_ids_for_key.extend(intersection);
            }
        }

        Ok(())
    }

    async fn calculate_filtered_doc_ids(
        &self,
        where_filter: &WhereFilter,
        uncommitted_deleted_documents: &HashSet<DocumentId>,
    ) -> Result<Option<FilterResult<DocumentId>>> {
        if where_filter.is_empty() {
            return Ok(None);
        }

        trace!("Calculating filtered doc ids");

        let uncommitted_fields = self.uncommitted_fields.read("filters").await;
        let committed_fields = self.committed_fields.read("filters").await;

        fn calculate_filter(
            document_count_estimate: u64,
            path_to_index_id_map: &PathToIndexId,
            where_filter: &WhereFilter,
            uncommitted_fields: &UncommittedFields,
            committed_fields: &CommittedFields,
        ) -> Result<FilterResult<DocumentId>> {
            let mut results = Vec::new();

            for (k, filter) in &where_filter.filter_on_fields {
                let (field_id, field_type) = match path_to_index_id_map.get_filter_field(k) {
                    None => {
                        // If the user specified a field that is not in the index,
                        // we should return an empty set.
                        return Ok(FilterResult::Filter(PlainFilterResult::new(
                            document_count_estimate,
                        )));
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

                        // We could check if uncommitted_field is empty
                        // If so, we could skip the filter and the `and` operation
                        // TODO: improve this

                        let filtered = PlainFilterResult::from_iter(
                            document_count_estimate,
                            uncommitted_field.filter(*filter_bool),
                        );
                        let mut filtered = FilterResult::Filter(filtered);

                        let committed_field = committed_fields.bool_fields.get(&field_id);
                        if let Some(committed_field) = committed_field {
                            let committed_docs =
                                committed_field.filter(*filter_bool).with_context(|| {
                                    format!("Cannot filter by \"{}\": unknown field", &k)
                                })?;

                            filtered = FilterResult::or(
                                filtered,
                                FilterResult::Filter(PlainFilterResult::from_iter(
                                    document_count_estimate,
                                    committed_docs,
                                )),
                            );
                        }

                        results.push(filtered);
                    }
                    (FieldType::Date, Filter::Date(filter_date)) => {
                        let uncommitted_field = uncommitted_fields
                            .date_fields
                            .get(&field_id)
                            .ok_or_else(|| {
                                anyhow::anyhow!("Cannot filter by \"{}\": unknown field", &k)
                            })?;

                        let filtered = PlainFilterResult::from_iter(
                            document_count_estimate,
                            uncommitted_field.filter(filter_date),
                        );
                        let mut filtered = FilterResult::Filter(filtered);

                        let committed_field = committed_fields.date_fields.get(&field_id);
                        if let Some(committed_field) = committed_field {
                            let committed_docs =
                                committed_field.filter(filter_date).with_context(|| {
                                    format!("Cannot filter by \"{}\": unknown field", &k)
                                })?;

                            filtered = FilterResult::or(
                                filtered,
                                FilterResult::Filter(PlainFilterResult::from_iter(
                                    document_count_estimate,
                                    committed_docs,
                                )),
                            );
                        }

                        results.push(filtered);
                    }
                    (FieldType::Number, Filter::Number(filter_number)) => {
                        let uncommitted_field = uncommitted_fields
                            .number_fields
                            .get(&field_id)
                            .ok_or_else(|| {
                                anyhow::anyhow!("Cannot filter by \"{}\": unknown field", &k)
                            })?;

                        let filtered = PlainFilterResult::from_iter(
                            document_count_estimate,
                            uncommitted_field.filter(filter_number),
                        );
                        let mut filtered = FilterResult::Filter(filtered);

                        let committed_field = committed_fields.number_fields.get(&field_id);
                        if let Some(committed_field) = committed_field {
                            let committed_docs =
                                committed_field.filter(filter_number).with_context(|| {
                                    format!("Cannot filter by \"{}\": unknown field", &k)
                                })?;

                            filtered = FilterResult::or(
                                filtered,
                                FilterResult::Filter(PlainFilterResult::from_iter(
                                    document_count_estimate,
                                    committed_docs,
                                )),
                            );
                        }

                        results.push(filtered);
                    }
                    (FieldType::StringFilter, Filter::String(filter_string)) => {
                        let uncommitted_field = uncommitted_fields
                            .string_filter_fields
                            .get(&field_id)
                            .ok_or_else(|| {
                                anyhow::anyhow!("Cannot filter by \"{}\": unknown field", &k)
                            })?;

                        let filtered = PlainFilterResult::from_iter(
                            document_count_estimate,
                            uncommitted_field.filter(filter_string),
                        );
                        let mut filtered = FilterResult::Filter(filtered);

                        let committed_field = committed_fields.string_filter_fields.get(&field_id);
                        if let Some(committed_field) = committed_field {
                            let committed_docs = committed_field.filter(filter_string);

                            filtered = FilterResult::or(
                                filtered,
                                FilterResult::Filter(PlainFilterResult::from_iter(
                                    document_count_estimate,
                                    committed_docs,
                                )),
                            );
                        }

                        results.push(filtered);
                    }
                    (FieldType::GeoPoint, Filter::GeoPoint(geopoint_filter)) => {
                        let uncommitted_field = uncommitted_fields
                            .geopoint_fields
                            .get(&field_id)
                            .ok_or_else(|| {
                            anyhow::anyhow!("Cannot filter by \"{}\": unknown field", &k)
                        })?;

                        let filtered = PlainFilterResult::from_iter(
                            document_count_estimate,
                            uncommitted_field.filter(geopoint_filter),
                        );
                        let mut filtered = FilterResult::Filter(filtered);

                        let committed_field = committed_fields.geopoint_fields.get(&field_id);
                        if let Some(committed_field) = committed_field {
                            let committed_docs = committed_field.filter(geopoint_filter);

                            filtered = FilterResult::or(
                                filtered,
                                FilterResult::Filter(PlainFilterResult::from_iter(
                                    document_count_estimate,
                                    committed_docs,
                                )),
                            );
                        }

                        results.push(filtered);
                    }
                    _ => {
                        // If the user specified a field that is not in the index,
                        // we should return an empty set.
                        return Ok(FilterResult::Filter(PlainFilterResult::new(
                            document_count_estimate,
                        )));
                    }
                }
            }

            if let Some(filter) = where_filter.and.as_ref() {
                for f in filter {
                    let result = calculate_filter(
                        document_count_estimate,
                        path_to_index_id_map,
                        f,
                        uncommitted_fields,
                        committed_fields,
                    )?;
                    results.push(result);
                }
            }

            if let Some(filter) = where_filter.or.as_ref() {
                let mut or = vec![];
                for f in filter {
                    let result = calculate_filter(
                        document_count_estimate,
                        path_to_index_id_map,
                        f,
                        uncommitted_fields,
                        committed_fields,
                    )?;
                    or.push(result);
                }

                if or.is_empty() {
                    return Ok(FilterResult::Filter(PlainFilterResult::new(
                        document_count_estimate,
                    )));
                }

                let mut result = or.pop().expect("we should have at least one result");
                for f in or {
                    result = FilterResult::or(result, f);
                }

                results.push(result);
            }

            if let Some(filter) = where_filter.not.as_ref() {
                let result = calculate_filter(
                    document_count_estimate,
                    path_to_index_id_map,
                    filter,
                    uncommitted_fields,
                    committed_fields,
                )?;
                results.push(FilterResult::Not(Box::new(result)));
            }

            // Is this correct????
            if results.is_empty() {
                return Ok(FilterResult::Filter(PlainFilterResult::new(
                    document_count_estimate,
                )));
            }

            let mut result = results.pop().expect("we should have at least one result");
            for f in results {
                result = FilterResult::and(result, f);
            }

            Ok(result)
        }

        let mut output = calculate_filter(
            self.document_count,
            &self.path_to_index_id_map,
            where_filter,
            &uncommitted_fields,
            &committed_fields,
        )?;

        if !uncommitted_deleted_documents.is_empty() {
            output = FilterResult::and(
                output,
                FilterResult::Not(Box::new(FilterResult::Filter(
                    PlainFilterResult::from_iter(
                        self.document_count,
                        uncommitted_deleted_documents.iter().copied(),
                    ),
                ))),
            );
        }

        Ok(Some(output))
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
        let uncommitted_fields = self.uncommitted_fields.read("vector_props").await;
        let committed_fields = self.committed_fields.read("vector_props").await;

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
                    let Some((field_id, field_type)) = self.path_to_index_id_map.get(field) else {
                        continue;
                        // bail!("Cannot filter by \"{}\": unknown field", &field);
                    };
                    if !matches!(field_type, FieldType::String) {
                        continue;
                        // bail!("Cannot filter by \"{}\": wrong type", &field);
                    }
                    field_ids.insert(field_id);
                }
                field_ids
            }
            Properties::None | Properties::Star => {
                let uncommitted_fields = self.uncommitted_fields.read("string_props").await;
                let committed_fields = self.committed_fields.read("string_props").await;

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
        tolerance: Option<u8>,
        properties: Vec<FieldId>,
        boost: HashMap<FieldId, f32>,
        filtered_doc_ids: Option<&FilterResult<DocumentId>>,
        uncommitted_deleted_documents: &HashSet<DocumentId>,
    ) -> Result<HashMap<DocumentId, f32>> {
        let uncommitted_fields = self.uncommitted_fields.read("search_fulltext").await;
        let committed_fields = self.committed_fields.read("search_fulltext").await;

        let tokens = self.text_parser.tokenize_and_stem(term);
        let mut tokens: Vec<_> = if exact {
            tokens.into_iter().map(|e| e.0).collect()
        } else {
            tokens
                .into_iter()
                .flat_map(|e| std::iter::once(e.0).chain(e.1.into_iter()))
                .collect()
        };

        if tokens.is_empty() {
            // Ensure we have at least one token otherwise the result will be empty.
            tokens.push(String::from(""));
        }

        let mut scorer: BM25Scorer<DocumentId> = match threshold {
            Some(Threshold(threshold)) => {
                let perc = tokens.len() as f32 * threshold;
                let threshold = perc.floor() as u32;
                BM25Scorer::with_threshold(threshold)
            }
            None => BM25Scorer::plain(),
        };

        // Use the collection's total document count for canonical BM25F
        let total_documents = self.document_count as f32;

        let mut field_global_info: HashMap<FieldId, _> = HashMap::new();
        let mut corpus_dfs: HashMap<String, usize> = HashMap::new();

        // Single pass to compute both global info and corpus document frequencies
        for token in &tokens {
            let mut total_corpus_docs = HashSet::new();

            for field_id in &properties {
                let Some(field) = uncommitted_fields.string_fields.get(field_id) else {
                    continue;
                };
                let committed = committed_fields.string_fields.get(field_id);

                // Compute global info only once per field
                if !field_global_info.contains_key(field_id) {
                    let global_info = if let Some(committed) = committed {
                        committed.global_info() + field.global_info()
                    } else {
                        field.global_info()
                    };
                    field_global_info.insert(*field_id, global_info);
                }

                // Collect unique document IDs for this term across all fields for corpus DF calculation
                let mut corpus_scorer = BM25Scorer::plain();
                let single_token_slice = std::slice::from_ref(token);
                let mut corpus_context = FullTextSearchContext {
                    tokens: single_token_slice,
                    exact_match: exact,
                    boost: 1.0,
                    field_id: *field_id,
                    filtered_doc_ids: None,
                    global_info: field_global_info[field_id].clone(),
                    uncommitted_deleted_documents,
                    total_term_count: 1,
                };

                field
                    .search(&mut corpus_context, &mut corpus_scorer)
                    .context("Cannot perform corpus search")?;
                if let Some(committed_field) = committed {
                    committed_field
                        .search(&mut corpus_context, &mut corpus_scorer, tolerance)
                        .context("Cannot perform corpus search")?;
                }

                // Add documents found in this field to the corpus count
                total_corpus_docs.extend(corpus_scorer.get_scores().keys().cloned());
            }

            corpus_dfs.insert(token.clone(), total_corpus_docs.len().max(1));
        }

        // Now perform the actual scoring with pre-computed corpus frequencies
        for (term_index, token) in tokens.iter().enumerate() {
            scorer.reset_term();
            let corpus_df = corpus_dfs[token];

            // Then, add field contributions for this term with filtering applied
            for field_id in &properties {
                let Some(field) = uncommitted_fields.string_fields.get(field_id) else {
                    continue;
                };
                let committed = committed_fields.string_fields.get(field_id);

                let global_info = field_global_info[field_id].clone();

                let boost = boost.get(field_id).copied().unwrap_or(1.0);

                // Reuse the same token slice to avoid allocation
                let single_token_slice = std::slice::from_ref(token);
                let mut context = FullTextSearchContext {
                    tokens: single_token_slice,
                    exact_match: exact,
                    boost,
                    field_id: *field_id,
                    filtered_doc_ids,
                    global_info,
                    uncommitted_deleted_documents,
                    total_term_count: term_index as u64 + 1,
                };

                // Search this field for this specific term (with filters applied)
                field
                    .search(&mut context, &mut scorer)
                    .context("Cannot perform search")?;

                if let Some(committed_field) = committed {
                    committed_field
                        .search(&mut context, &mut scorer, tolerance)
                        .context("Cannot perform search")?;
                }
            }

            match &scorer {
                BM25Scorer::Plain(_) => {
                    scorer.finalize_term_plain(
                        corpus_df,
                        total_documents,
                        1.2, // k parameter
                        1.0, // phrase boost
                    );
                }
                BM25Scorer::WithThreshold(_) => {
                    scorer.finalize_term(
                        corpus_df,
                        total_documents,
                        1.2,             // k parameter
                        1.0,             // phrase boost
                        1 << term_index, // token_indexes: bit mask indicating which token this is
                    );
                }
            }

            // Move to next term
            scorer.next_term();
        }

        Ok(scorer.get_scores())
    }

    async fn search_vector(
        &self,
        term: &str,
        properties: Vec<FieldId>,
        similarity: Similarity,
        filtered_doc_ids: Option<&FilterResult<DocumentId>>,
        limit: Limit,
        uncommitted_deleted_documents: &HashSet<DocumentId>,
    ) -> Result<HashMap<DocumentId, f32>> {
        let mut output: HashMap<DocumentId, f32> = HashMap::new();

        let uncommitted_fields = self.uncommitted_fields.read("vector_search").await;
        let committed_fields = self.committed_fields.read("vector_search").await;

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
                .context
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

    pub async fn get_pin_rule_ids(&self) -> Vec<String> {
        let pin_rules_reader = self.pin_rules_reader.read("get_pin_rule_ids").await;
        pin_rules_reader.get_rule_ids()
    }

    pub async fn get_read_lock_on_pin_rules(
        &self,
    ) -> OramaAsyncLockReadGuard<'_, PinRulesReader<DocumentId>> {
        self.pin_rules_reader
            .read("get_read_lock_on_pin_rules")
            .await
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
    pub type_parsing_strategies: TypeParsingStrategies,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
enum FieldType {
    Bool,
    Number,
    StringFilter,
    Date,
    GeoPoint,
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
    #[serde(default)]
    date_field_ids: Vec<(FieldId, DateFieldInfo)>,
    #[serde(default)]
    geopoint_field_ids: Vec<(FieldId, GeoPointFieldInfo)>,
    string_filter_field_ids: Vec<(FieldId, StringFilterFieldInfo)>,
    string_field_ids: Vec<(FieldId, StringFieldInfo)>,
    vector_field_ids: Vec<(FieldId, VectorFieldInfo)>,
    path_to_index_id_map: Vec<(Box<[String]>, (FieldId, FieldType))>,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
    #[serde(default)]
    enum_strategy: EnumStrategy,
}

#[derive(Debug, Serialize, Deserialize)]
enum Dump {
    V1(DumpV1),
}

impl oramacore_lib::filters::DocId for DocumentId {
    #[inline]
    fn as_u64(&self) -> u64 {
        self.0
    }
}

pub struct SortedField<'index> {
    uncommitted_fields: OramaAsyncLockReadGuard<'index, UncommittedFields>,
    committed_fields: OramaAsyncLockReadGuard<'index, CommittedFields>,
    field_id: FieldId,
    field_type: FieldType,
    order: SortOrder,
}

impl<'index> SortedField<'index> {
    pub fn iter<'s>(&'s self) -> Box<dyn Iterator<Item = (Number, HashSet<DocumentId>)> + 's>
    where
        'index: 's,
    {
        match self.field_type {
            FieldType::Bool => {
                let uncommitted = self
                    .uncommitted_fields
                    .bool_fields
                    .get(&self.field_id)
                    .unwrap();

                fn bool_to_number(i: bool) -> Number {
                    if i {
                        Number::I32(1)
                    } else {
                        Number::I32(0)
                    }
                }

                let (true_h, false_h) = uncommitted.clone_inner();
                let iter1: Box<dyn Iterator<Item = (Number, HashSet<DocumentId>)> + '_> =
                    match self.order {
                        SortOrder::Ascending => Box::new(
                            std::iter::once((bool_to_number(false), false_h))
                                .chain(std::iter::once((bool_to_number(true), true_h))),
                        ),
                        SortOrder::Descending => Box::new(Box::new(
                            std::iter::once((bool_to_number(true), true_h))
                                .chain(std::iter::once((bool_to_number(false), false_h))),
                        )),
                    };

                let r: Box<dyn Iterator<Item = (Number, HashSet<DocumentId>)> + '_> =
                    if let Some(field) = self.committed_fields.bool_fields.get(&self.field_id) {
                        let (true_h, false_h) = field.clone_inner().unwrap();
                        let iter2: Box<dyn Iterator<Item = (Number, HashSet<DocumentId>)> + '_> =
                            match self.order {
                                SortOrder::Ascending => Box::new(
                                    std::iter::once((bool_to_number(false), false_h))
                                        .chain(std::iter::once((bool_to_number(true), true_h))),
                                ),
                                SortOrder::Descending => Box::new(
                                    std::iter::once((bool_to_number(true), true_h))
                                        .chain(std::iter::once((bool_to_number(false), false_h))),
                                ),
                            };
                        let iter = SortIterator::new(iter1, iter2, self.order);

                        Box::new(iter)
                    } else {
                        iter1
                    };

                r
            }
            FieldType::Number => {
                let field = self
                    .uncommitted_fields
                    .number_fields
                    .get(&self.field_id)
                    .unwrap();

                let iter1 = field.iter();
                let iter1: Box<dyn Iterator<Item = (Number, HashSet<DocumentId>)> + '_> =
                    match self.order {
                        SortOrder::Ascending => Box::new(iter1),
                        SortOrder::Descending => Box::new(iter1.rev()),
                    };

                let r: Box<dyn Iterator<Item = (Number, HashSet<DocumentId>)> + '_> =
                    if let Some(field) = self.committed_fields.number_fields.get(&self.field_id) {
                        let iter2 = field.iter();
                        let iter2: Box<dyn Iterator<Item = (Number, HashSet<DocumentId>)> + '_> =
                            match self.order {
                                SortOrder::Ascending => {
                                    Box::new(iter2.map(|(number, doc_ids)| (number.0, doc_ids)))
                                }
                                SortOrder::Descending => Box::new(
                                    iter2.rev().map(|(number, doc_ids)| (number.0, doc_ids)),
                                ),
                            };
                        let iter = SortIterator::new(iter1, iter2, self.order);

                        Box::new(iter)
                    } else {
                        iter1
                    };

                r
            }
            FieldType::Date => {
                let field = self
                    .uncommitted_fields
                    .date_fields
                    .get(&self.field_id)
                    .unwrap();

                fn i64_to_number(i: i64) -> Number {
                    if let Ok(i) = i32::try_from(i) {
                        Number::I32(i)
                    } else if i > 0 {
                        Number::I32(i32::MAX)
                    } else {
                        Number::I32(i32::MIN)
                    }
                }

                let iter1 = field
                    .iter()
                    .map(|(timestamp, doc_ids)| (i64_to_number(timestamp), doc_ids));
                let iter1: Box<dyn Iterator<Item = (Number, HashSet<DocumentId>)> + '_> =
                    match self.order {
                        SortOrder::Ascending => Box::new(iter1),
                        SortOrder::Descending => Box::new(iter1.rev()),
                    };

                let r = if let Some(field) = self.committed_fields.date_fields.get(&self.field_id) {
                    let iter2 = field
                        .iter()
                        .map(|(number, doc_ids)| (i64_to_number(number), doc_ids));
                    let iter2: Box<dyn Iterator<Item = (Number, HashSet<DocumentId>)> + '_> =
                        match self.order {
                            SortOrder::Ascending => Box::new(iter2),
                            SortOrder::Descending => Box::new(iter2.rev()),
                        };
                    let iter = SortIterator::new(iter1, iter2, self.order);

                    Box::new(iter)
                } else {
                    iter1
                };

                r
            }
            _ => unreachable!("We checked `field_type` before so, this branch cannot be reached"),
        }
    }
}

struct SortIterator<'s1, 's2, T: Ord + Clone> {
    iter1: Peekable<Box<dyn Iterator<Item = (T, HashSet<DocumentId>)> + 's1>>,
    iter2: Peekable<Box<dyn Iterator<Item = (T, HashSet<DocumentId>)> + 's2>>,
    order: SortOrder,
}

impl<'s1, 's2, T: Ord + Clone> SortIterator<'s1, 's2, T> {
    fn new(
        iter1: Box<dyn Iterator<Item = (T, HashSet<DocumentId>)> + 's1>,
        iter2: Box<dyn Iterator<Item = (T, HashSet<DocumentId>)> + 's2>,
        order: SortOrder,
    ) -> Self {
        Self {
            iter1: iter1.peekable(),
            iter2: iter2.peekable(),
            order,
        }
    }
}

impl<T: Ord + Clone> Iterator for SortIterator<'_, '_, T> {
    type Item = (T, HashSet<DocumentId>);

    fn next(&mut self) -> Option<Self::Item> {
        let el1 = self.iter1.peek();
        let el2 = self.iter2.peek();

        match (el1, el2) {
            (None, None) => None,
            (Some((_, _)), None) => self.iter1.next(),
            (None, Some((_, _))) => self.iter2.next(),
            (Some((k1, _)), Some((k2, _))) => match self.order {
                SortOrder::Ascending => {
                    if k1 < k2 {
                        self.iter1.next()
                    } else {
                        self.iter2.next()
                    }
                }
                SortOrder::Descending => {
                    if k1 > k2 {
                        self.iter1.next()
                    } else {
                        self.iter2.next()
                    }
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DocumentId, SortOrder};

    #[test]
    fn test_sort_iterator_ascending() {
        let data1 = vec![(1, 10), (3, 30), (5, 50)];
        let data2 = vec![(2, 20), (4, 40), (6, 60)];
        let iter = SortIterator::new(make_iter(data1), make_iter(data2), SortOrder::Ascending);
        let result: Vec<u64> = iter.flat_map(|(_, doc_ids)| doc_ids).map(|d| d.0).collect();
        assert_eq!(result, vec![10, 20, 30, 40, 50, 60]);
    }

    #[test]
    fn test_sort_iterator_descending() {
        let data1 = vec![(6, 60), (4, 40), (2, 20)];
        let data2 = vec![(5, 50), (3, 30), (1, 10)];
        let iter = SortIterator::new(make_iter(data1), make_iter(data2), SortOrder::Descending);
        let result: Vec<u64> = iter.flat_map(|(_, doc_ids)| doc_ids).map(|d| d.0).collect();

        assert_eq!(result, vec![60, 50, 40, 30, 20, 10]);
    }

    #[test]
    fn test_sort_iterator_mixed_lengths() {
        let data1 = vec![(1, 10), (3, 30)];
        let data2 = vec![(2, 20), (4, 40), (5, 50)];
        let iter = SortIterator::new(make_iter(data1), make_iter(data2), SortOrder::Ascending);
        let result: Vec<u64> = iter.flat_map(|(_, doc_ids)| doc_ids).map(|d| d.0).collect();
        assert_eq!(result, vec![10, 20, 30, 40, 50]);
    }

    #[test]
    fn test_sort_iterator_with_duplicates() {
        // Both iterators contain the same key and value
        let data1 = vec![(1, 10), (2, 20), (3, 30)];
        let data2 = vec![(2, 20), (3, 30), (4, 40)];
        let iter = SortIterator::new(make_iter(data1), make_iter(data2), SortOrder::Ascending);
        let result: Vec<u64> = iter.flat_map(|(_, doc_ids)| doc_ids).map(|d| d.0).collect();

        assert_eq!(result, vec![10, 20, 20, 30, 30, 40]);
    }

    fn make_iter(data: Vec<(u64, u64)>) -> Box<dyn Iterator<Item = (u64, HashSet<DocumentId>)>> {
        Box::new(
            data.into_iter()
                .map(|(k, v)| (k, HashSet::from([DocumentId(v)]))),
        )
    }
}
