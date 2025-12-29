use chrono::{DateTime, Utc};
use committed_field::{
    BoolFieldInfo, CommittedBoolField, CommittedNumberField, CommittedStringFilterField,
    NumberFieldInfo, StringFilterFieldInfo, VectorFieldInfo,
};
use path_to_index_id_map::PathToIndexId;
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, trace, warn};
use uncommitted_field::*;

use crate::{
    collection_manager::sides::{
        Offset, read::{
            context::ReadSideContext,
            index::{
                committed_field::{
                    CommittedDateField, CommittedGeoPointField, CommittedStringField,
                    CommittedVectorField, DateFieldInfo, GeoPointFieldInfo, StringFieldInfo,
                },
                merge::{
                    CommittedField, CommittedFieldMetadata, Field, UncommittedField, merge_field
                },
            },
        }, write::index::EnumStrategy
    },
    lock::{OramaAsyncLock, OramaAsyncLockReadGuard},
    metrics::{
        FieldIndexCollectionCommitLabels, IndexCollectionCommitLabels, commit::{FIELD_INDEX_COMMIT_CALCULATION_TIME, INDEX_COMMIT_CALCULATION_TIME}
    },
    python::embeddings::Model,
    types::{
        CollectionId, DocumentId, FulltextMode, IndexId, Limit,
        Properties, SearchMode, SortOrder, TypeParsingStrategies,
    },
};
use oramacore_lib::fs::{create_if_not_exists, BufferedFile};
use oramacore_lib::nlp::{locales::Locale, TextParser};

use super::collection::{IndexFieldStats, IndexFieldStatsType};
use super::OffloadFieldConfig;
mod committed_field;
pub mod facet;
pub mod filter;
pub mod group;
mod merge;
mod path_to_index_id_map;
mod sort;
pub mod token_score;
mod uncommitted_field;

use crate::{
    collection_manager::sides::{
        write::index::IndexedValue, IndexWriteOperation, IndexWriteOperationFieldType,
    },
    types::FieldId,
};
use anyhow::{Context, Result};
pub use committed_field::{
    CommittedBoolFieldStats, CommittedDateFieldStats, CommittedGeoPointFieldStats,
    CommittedNumberFieldStats, CommittedStringFieldStats, CommittedStringFilterFieldStats,
    CommittedVectorFieldStats,
};
pub use group::GroupValue;
pub use sort::IndexSortContext;
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
pub struct UncommittedFields {
    pub bool_fields: HashMap<FieldId, UncommittedBoolField>,
    pub number_fields: HashMap<FieldId, UncommittedNumberField>,
    pub string_filter_fields: HashMap<FieldId, UncommittedStringFilterField>,
    pub date_fields: HashMap<FieldId, UncommittedDateFilterField>,
    pub geopoint_fields: HashMap<FieldId, UncommittedGeoPointFilterField>,
    pub string_fields: HashMap<FieldId, UncommittedStringField>,
    pub vector_fields: HashMap<FieldId, UncommittedVectorField>,
}

#[derive(Default)]
pub struct CommittedFields {
    pub bool_fields: HashMap<FieldId, CommittedBoolField>,
    pub number_fields: HashMap<FieldId, CommittedNumberField>,
    pub string_filter_fields: HashMap<FieldId, CommittedStringFilterField>,
    pub date_fields: HashMap<FieldId, CommittedDateField>,
    pub geopoint_fields: HashMap<FieldId, CommittedGeoPointField>,
    pub string_fields: HashMap<FieldId, CommittedStringField>,
    pub vector_fields: HashMap<FieldId, CommittedVectorField>,
}

pub struct IndexSearchStore<'index> {
    pub document_count: u64,
    pub committed_fields: OramaAsyncLockReadGuard<'index, CommittedFields>,
    pub uncommitted_fields: OramaAsyncLockReadGuard<'index, UncommittedFields>,
    pub path_to_field_id_map: &'index PathToIndexId,
    pub uncommitted_deleted_documents: &'index HashSet<DocumentId>,
    pub text_parser: &'index TextParser,

    // I don't like this, but for now it's the easiest way
    pub read_side_context: &'index ReadSideContext,
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
            let field = CommittedBoolField::try_load(info, offload_config)
                .context("Cannot load bool field")?;
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
            let field = CommittedNumberField::try_load(info, offload_config)
                .context("Cannot load number field")?;
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
            let field = CommittedDateField::try_load(info, offload_config)
                .context("Cannot load date field")?;
            committed_fields.date_fields.insert(field_id, field);
        }

        debug!("Loading geopoint_field_ids");
        for (field_id, info) in dump.geopoint_field_ids {
            filter_fields.insert(info.field_path.clone(), (field_id, FieldType::GeoPoint));

            uncommitted_fields.geopoint_fields.insert(
                field_id,
                UncommittedGeoPointFilterField::empty(info.field_path.clone()),
            );
            let field = CommittedGeoPointField::try_load(info, offload_config)
                .context("Cannot load geopoint field")?;
            committed_fields.geopoint_fields.insert(field_id, field);
        }

        debug!("Loading string_filter_field_ids");
        for (field_id, info) in dump.string_filter_field_ids {
            filter_fields.insert(info.field_path.clone(), (field_id, FieldType::StringFilter));

            uncommitted_fields.string_filter_fields.insert(
                field_id,
                UncommittedStringFilterField::empty(info.field_path.clone()),
            );
            let field = CommittedStringFilterField::try_load(info, offload_config)
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
                UncommittedVectorField::empty(info.field_path.clone(), info.model),
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

            enum_strategy: dump.enum_strategy,
        })
    }

    pub async fn get_search_store<'index>(&'index self) -> IndexSearchStore<'index> {
        let (committed_fields, uncommitted_fields) = tokio::join!(
            self.committed_fields.read("get_all_document_ids"),
            self.uncommitted_fields.read("get_all_document_ids"),
        );

        IndexSearchStore {
            document_count: self.document_count,
            committed_fields,
            uncommitted_fields,
            path_to_field_id_map: &self.path_to_index_id_map,
            uncommitted_deleted_documents: &self.uncommitted_deleted_documents,
            text_parser: &self.text_parser,
            read_side_context: &self.context,
        }
    }

    pub async fn get_all_document_ids(&self) -> Result<Vec<DocumentId>> {
        let (committed_fields, uncommitted_fields) = tokio::join!(
            self.committed_fields.read("get_all_document_ids"),
            self.uncommitted_fields.read("get_all_document_ids"),
        );

        let context = token_score::TokenScoreContext::new(
            self.id,
            self.document_count,
            &**uncommitted_fields,
            &**committed_fields,
            &self.text_parser,
            &self.context,
            &self.path_to_index_id_map,
        );

        let mode = SearchMode::Default(FulltextMode {
            term: String::new(),
            threshold: None,
            exact: false,
            tolerance: None,
        });
        let properties = Properties::default();
        let boost = HashMap::new();

        let params = token_score::TokenScoreParams {
            mode: &mode,
            properties: &properties,
            boost: &boost,
            limit_hint: Limit(usize::MAX),
            filtered_doc_ids: None,
        };

        let mut output = HashMap::new();
        context
            .execute(&params, &mut output)
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

        let merged_bools = merge_type(
            &data_dir_with_offset,
            &uncommitted_fields.bool_fields,
            &committed_fields.bool_fields,
            &self.uncommitted_deleted_documents,
            is_promoted,
            &self.offload_config,
            FieldIndexCollectionCommitLabels {
                collection: collection_id.as_str().to_string(),
                index: self.id.as_str().to_string(),
                side: "read",
                field_type: "bool",
            },
        )
        .context("Cannot merge bool fields")?;

        let merged_numbers = merge_type(
            &data_dir_with_offset,
            &uncommitted_fields.number_fields,
            &committed_fields.number_fields,
            &self.uncommitted_deleted_documents,
            is_promoted,
            &self.offload_config,
            FieldIndexCollectionCommitLabels {
                collection: collection_id.as_str().to_string(),
                index: self.id.as_str().to_string(),
                side: "read",
                field_type: "number",
            },
        )
        .context("Cannot merge number fields")?;

        let merged_dates = merge_type(
            &data_dir_with_offset,
            &uncommitted_fields.date_fields,
            &committed_fields.date_fields,
            &self.uncommitted_deleted_documents,
            is_promoted,
            &self.offload_config,
            FieldIndexCollectionCommitLabels {
                collection: collection_id.as_str().to_string(),
                index: self.id.as_str().to_string(),
                side: "read",
                field_type: "date",
            },
        )
        .context("Cannot merge date fields")?;

        let merged_geopoints = merge_type(
            &data_dir_with_offset,
            &uncommitted_fields.geopoint_fields,
            &committed_fields.geopoint_fields,
            &self.uncommitted_deleted_documents,
            is_promoted,
            &self.offload_config,
            FieldIndexCollectionCommitLabels {
                collection: collection_id.as_str().to_string(),
                index: self.id.as_str().to_string(),
                side: "read",
                field_type: "geopoint",
            },
        )
        .context("Cannot merge geopoint fields")?;

        let merged_string_filters = merge_type(
            &data_dir_with_offset,
            &uncommitted_fields.string_filter_fields,
            &committed_fields.string_filter_fields,
            &self.uncommitted_deleted_documents,
            is_promoted,
            &self.offload_config,
            FieldIndexCollectionCommitLabels {
                collection: collection_id.as_str().to_string(),
                index: self.id.as_str().to_string(),
                side: "read",
                field_type: "string_filter",
            },
        )
        .context("Cannot merge string filter fields")?;

        let merged_strings = merge_type(
            &data_dir_with_offset,
            &uncommitted_fields.string_fields,
            &committed_fields.string_fields,
            &self.uncommitted_deleted_documents,
            is_promoted,
            &self.offload_config,
            FieldIndexCollectionCommitLabels {
                collection: collection_id.as_str().to_string(),
                index: self.id.as_str().to_string(),
                side: "read",
                field_type: "string",
            },
        )
        .context("Cannot merge string fields")?;

        let merged_vectors = merge_type(
            &data_dir_with_offset,
            &uncommitted_fields.vector_fields,
            &committed_fields.vector_fields,
            &self.uncommitted_deleted_documents,
            is_promoted,
            &self.offload_config,
            FieldIndexCollectionCommitLabels {
                collection: collection_id.as_str().to_string(),
                index: self.id.as_str().to_string(),
                side: "read",
                field_type: "vector",
            },
        )
        .context("Cannot merge vector fields")?;

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

        overwrite_committed(
            self.id,
            offset,
            merged_bools,
            &mut committed_fields.bool_fields,
            &mut uncommitted_fields.bool_fields,
        );
        overwrite_committed(
            self.id,
            offset,
            merged_numbers,
            &mut committed_fields.number_fields,
            &mut uncommitted_fields.number_fields,
        );
        overwrite_committed(
            self.id,
            offset,
            merged_dates,
            &mut committed_fields.date_fields,
            &mut uncommitted_fields.date_fields,
        );
        overwrite_committed(
            self.id,
            offset,
            merged_geopoints,
            &mut committed_fields.geopoint_fields,
            &mut uncommitted_fields.geopoint_fields,
        );
        overwrite_committed(
            self.id,
            offset,
            merged_string_filters,
            &mut committed_fields.string_filter_fields,
            &mut uncommitted_fields.string_filter_fields,
        );
        overwrite_committed(
            self.id,
            offset,
            merged_strings,
            &mut committed_fields.string_fields,
            &mut uncommitted_fields.string_fields,
        );
        overwrite_committed(
            self.id,
            offset,
            merged_vectors,
            &mut committed_fields.vector_fields,
            &mut uncommitted_fields.vector_fields,
        );

        let dump = Dump::V1(DumpV1 {
            id: self.id,
            document_count: self.document_count,
            locale: self.locale,
            aliases: self.aliases.clone(),
            bool_field_ids: committed_fields
                .bool_fields
                .iter()
                .map(|(k, v)| (*k, v.metadata()))
                .collect(),
            number_field_ids: committed_fields
                .number_fields
                .iter()
                .map(|(k, v)| (*k, v.metadata()))
                .collect(),
            date_field_ids: committed_fields
                .date_fields
                .iter()
                .map(|(k, v)| (*k, v.metadata()))
                .collect(),
            geopoint_field_ids: committed_fields
                .geopoint_fields
                .iter()
                .map(|(k, v)| (*k, v.metadata()))
                .collect(),
            string_filter_field_ids: committed_fields
                .string_filter_fields
                .iter()
                .map(|(k, v)| (*k, v.metadata()))
                .collect(),
            string_field_ids: committed_fields
                .string_fields
                .iter()
                .map(|(k, v)| (*k, v.metadata()))
                .collect(),
            vector_field_ids: committed_fields
                .vector_fields
                .iter()
                .map(|(k, v)| (*k, v.metadata()))
                .collect(),
            // Not used anymore. We calculate it on the fly
            path_to_index_id_map: Vec::new(),
            created_at: self.created_at,
            updated_at: self.updated_at,
            enum_strategy: self.enum_strategy,
        });

        drop(uncommitted_fields);
        drop(committed_fields);

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
        let committed_fields = self.committed_fields.read("clean_up").await;

        fn add<FM: CommittedFieldMetadata, CF: CommittedField<FieldMetadata = FM>>(
            set: &mut HashSet<PathBuf>,
            fields: &HashMap<FieldId, CF>,
        ) {
            set.extend(
                fields
                    .values()
                    .map(CF::metadata)
                    .map(|m| m.data_dir().clone()),
            );
        }

        let mut field_data_dirs: HashSet<PathBuf> = HashSet::new();
        add(&mut field_data_dirs, &committed_fields.bool_fields);
        add(&mut field_data_dirs, &committed_fields.number_fields);
        add(&mut field_data_dirs, &committed_fields.string_filter_fields);
        add(&mut field_data_dirs, &committed_fields.date_fields);
        add(&mut field_data_dirs, &committed_fields.geopoint_fields);
        add(&mut field_data_dirs, &committed_fields.string_fields);
        add(&mut field_data_dirs, &committed_fields.vector_fields);

        trace!("Field data dirs to keep: {:?}", field_data_dirs);

        let subfolders = match std::fs::read_dir(index_data_dir) {
            Ok(a) => a,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // No data dir, nothing to clean
                return Ok(());
            }
            Err(e) => {
                return Err(anyhow::anyhow!(
                    "Cannot read index data dir for cleanup: {e:?}"
                ));
            }
        };
        let subfolders = subfolders
            .collect::<Result<Vec<_>, _>>()
            .context("Cannot read entry in index folder")?;

        for entry in subfolders {
            let a = entry.file_type().context("Cannot get file type")?;
            if !a.is_dir() {
                continue;
            }

            let subfolder = entry.path();

            trace!("Checking subfolder: {:?}", subfolder);

            // Avoid use `eq` or `contains`. To be safier, use `starts_with`
            // So `field_data_dirs` could point to a folder inside `subfolder`
            // and we will not delete it.
            let is_used = field_data_dirs
                .iter()
                .any(|field_data_dir| field_data_dir.starts_with(&subfolder));
            if is_used {
                continue;
            }

            // Delete only offset folders
            if !entry.file_name().to_str().unwrap().starts_with("offset-") {
                continue;
            }

            info!("Removing unused index data folder: {:?}", subfolder);
            std::fs::remove_dir_all(subfolder).context("Cannot remove path")?;
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
                            .insert(field_id, UncommittedVectorField::empty(field_path, model));
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
            IndexWriteOperation::PinRule(_) => {
                warn!("Ignore this rule");
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

    pub async fn get_sort_context<'index>(
        &'index self,
        field_name: &str,
        order: SortOrder,
    ) -> IndexSortContext<'index> {
        let uncommitted_fields = self.uncommitted_fields.read("get_sort_context").await;
        let committed_fields = self.committed_fields.read("get_sort_context").await;

        IndexSortContext::new(
            &self.path_to_index_id_map,
            uncommitted_fields,
            committed_fields,
            field_name,
            order,
        )
    }

    // Since we only have one embedding model for all indexes in a collection,
    // we can get the first index model and return it early.
    pub async fn get_model(&self) -> Option<Model> {
        let uncommitted_fields = self.uncommitted_fields.read("get_model").await;
        uncommitted_fields
            .vector_fields
            .values()
            .next()
            .map(|f| f.get_model())
    }

    pub async fn stats(&self, is_temp: bool) -> Result<IndexStats> {
        let mut fields_stats: Vec<IndexFieldStats> = Vec::new();

        let uncommitted_fields = self.uncommitted_fields.read("stats").await;
        let committed_fields = self.committed_fields.read("stats").await;

        fn extrapolate_stats<S: Into<IndexFieldStatsType>, F: Field<FieldStats = S>>(
            fields: &HashMap<FieldId, F>,
        ) -> impl Iterator<Item = IndexFieldStats> + '_ {
            fields.iter().map(|(k, v)| (k, v).into())
        }

        // Uncommitted
        fields_stats.extend(extrapolate_stats(&uncommitted_fields.bool_fields));
        fields_stats.extend(extrapolate_stats(&uncommitted_fields.number_fields));
        fields_stats.extend(extrapolate_stats(&uncommitted_fields.date_fields));
        fields_stats.extend(extrapolate_stats(&uncommitted_fields.geopoint_fields));
        fields_stats.extend(extrapolate_stats(&uncommitted_fields.string_filter_fields));
        fields_stats.extend(extrapolate_stats(&uncommitted_fields.string_fields));
        fields_stats.extend(extrapolate_stats(&uncommitted_fields.vector_fields));

        // Committed
        fields_stats.extend(extrapolate_stats(&committed_fields.bool_fields));
        fields_stats.extend(extrapolate_stats(&committed_fields.number_fields));
        fields_stats.extend(extrapolate_stats(&committed_fields.date_fields));
        fields_stats.extend(extrapolate_stats(&committed_fields.geopoint_fields));
        fields_stats.extend(extrapolate_stats(&committed_fields.string_filter_fields));
        fields_stats.extend(extrapolate_stats(&committed_fields.string_fields));
        fields_stats.extend(extrapolate_stats(&committed_fields.vector_fields));

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
pub(crate) enum FieldType {
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

impl<S: Into<IndexFieldStatsType>, F: Field<FieldStats = S>> From<(&FieldId, &F)>
    for IndexFieldStats
{
    fn from((k, v): (&FieldId, &F)) -> Self {
        let stats = v.stats();
        let field_path = v.field_path().join(".");
        IndexFieldStats {
            field_id: *k,
            field_path,
            stats: stats.into(),
        }
    }
}

enum MergeResult<T> {
    Changed(T),
    Unchanged,
}

fn merge_type<
    UF: UncommittedField,
    CFM: CommittedFieldMetadata,
    CF: CommittedField<Uncommitted = UF, FieldMetadata = CFM>,
>(
    data_dir_with_offset: &PathBuf,
    uncommitted_fields: &HashMap<FieldId, UF>,
    committed_fields: &HashMap<FieldId, CF>,
    uncommitted_deleted_documents: &HashSet<DocumentId>,
    is_promoted: bool,
    offload_config: &OffloadFieldConfig,
    telemetry_labels: FieldIndexCollectionCommitLabels,
) -> Result<HashMap<FieldId, MergeResult<CF>>> {
    let m = FIELD_INDEX_COMMIT_CALCULATION_TIME.create(telemetry_labels);

    let mut merged_fields = HashMap::new();
    let field_ids: HashSet<_> = uncommitted_fields
        .keys()
        .chain(committed_fields.keys())
        .copied()
        .collect();
    for field_id in field_ids {
        let data_dir = data_dir_with_offset.join(field_id.0.to_string());
        let uncommitted = uncommitted_fields.get(&field_id);
        let committed = committed_fields.get(&field_id);
        if let Some(merged_field) = merge_field(
            uncommitted,
            committed,
            data_dir,
            uncommitted_deleted_documents,
            is_promoted,
            offload_config,
        )
        .context("Cannot merge field")?
        {
            merged_fields.insert(field_id, MergeResult::Changed(merged_field));
        } else {
            merged_fields.insert(field_id, MergeResult::Unchanged);
        }
    }

    drop(m);

    Ok(merged_fields)
}

fn overwrite_committed<
    UF: UncommittedField,
    CFM: CommittedFieldMetadata,
    CF: CommittedField<Uncommitted = UF, FieldMetadata = CFM>,
>(
    index_id: IndexId,
    offset: Offset,
    merged_bools_result: HashMap<FieldId, MergeResult<CF>>,
    committed_fields: &mut HashMap<FieldId, CF>,
    uncommitted_fields: &mut HashMap<FieldId, UF>,
) {
    for (field_id, merged) in merged_bools_result {
        match merged {
            MergeResult::Changed(merged) => {
                #[cfg(debug_assertions)]
                {
                    let metadata = merged.metadata();
                    let data_dir = metadata.data_dir();
                    let mut components = data_dir.components().rev();
                    let offset_str = components.nth(1).unwrap();
                    assert_eq!(
                        offset_str.as_os_str().to_str().unwrap(),
                        &format!("offset-{}", offset.0),
                        "Field data dir offset mismatch after commit"
                    );
                    // Check if promoting to runtime index was done correctly
                    let index_id_on_fs = components.next().unwrap();
                    assert_eq!(
                        index_id_on_fs.as_os_str().to_str().unwrap(),
                        index_id.as_str(),
                        "Field index id mismatch after commit"
                    );
                    assert!(std::fs::exists(data_dir).unwrap());
                }
                committed_fields.insert(field_id, merged);
            }
            MergeResult::Unchanged => {}
        }
        if let Some(uncommitted) = uncommitted_fields.get_mut(&field_id) {
            uncommitted.clear();
        }
    }
}
