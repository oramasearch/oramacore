use chrono::{DateTime, Utc};
use committed_field::{
    BoolFieldInfo,
    NumberFieldInfo, StringFilterFieldInfo, VectorFieldInfo,
};
use path_to_index_id_map::PathToIndexId;
use serde::{Deserialize, Serialize};
use tracing::{debug, error, info, trace, warn};

use crate::{
    collection_manager::sides::{
        read::{
            context::ReadSideContext,
            index::{
                committed_field::{
                    DateFieldInfo, GeoPointFieldInfo, StringFieldInfo,
                },
            },
        },
        write::index::EnumStrategy,
        Offset,
    },
    lock::{OramaAsyncLock, OramaAsyncLockReadGuard},
    metrics::{
        commit::INDEX_COMMIT_CALCULATION_TIME,
        IndexCollectionCommitLabels,
    },
    python::embeddings::Model,
    types::{
        CollectionId, DocumentId, FulltextMode, IndexId, Limit, Properties, ScoreMode,
        TypeParsingStrategies,
    },
};
use oramacore_lib::fs::{create_if_not_exists, BufferedFile};
use oramacore_lib::nlp::{locales::Locale, TextParser};

use self::bool_field::BoolFieldStorage;
use self::date_field::DateFieldStorage;
use self::embedding_field::EmbeddingFieldStorage;
use self::geopoint_field::GeoPointFieldStorage;
use self::number_field::NumberFieldStorage;
use self::string_field::StringFieldStorage;
use self::string_filter_field::StringFilterFieldStorage;
use super::collection::{IndexFieldStats, IndexFieldStatsType};
use super::OffloadFieldConfig;
pub mod bool_field;
mod committed_field;
pub mod date_field;
pub mod embedding_field;
pub mod facet;
pub mod filter;
pub mod geopoint_field;
pub mod group;
mod merge;
pub mod number_field;
mod path_to_index_id_map;
mod sort;
pub mod string_field;
pub mod string_filter_field;
pub mod token_score;
mod uncommitted_field;

use crate::{
    collection_manager::sides::{
        write::index::IndexedValue, IndexWriteOperation, IndexWriteOperationFieldType,
    },
    types::FieldId,
};
use anyhow::{Context, Result};
pub use group::GroupValue;
pub use sort::{DocBatch, DocBatchIntoIter, DocBatchIter, IndexSortContext, SortedDocIdsIter};
use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
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
    // No more string_fields here -- they are managed by StringFieldStorage on Index directly.
}

#[derive(Default)]
pub struct CommittedFields {
    // No more string_fields here -- they are managed by StringFieldStorage on Index directly.
}

pub struct IndexSearchStore<'index> {
    pub document_count: u64,
    pub committed_fields: OramaAsyncLockReadGuard<'index, CommittedFields>,
    pub uncommitted_fields: OramaAsyncLockReadGuard<'index, UncommittedFields>,
    pub bool_fields: &'index HashMap<FieldId, BoolFieldStorage>,
    pub number_fields: &'index HashMap<FieldId, NumberFieldStorage>,
    pub date_fields: &'index HashMap<FieldId, DateFieldStorage>,
    pub geopoint_fields: &'index HashMap<FieldId, GeoPointFieldStorage>,
    pub string_filter_fields: &'index HashMap<FieldId, StringFilterFieldStorage>,
    pub string_fields: &'index HashMap<FieldId, StringFieldStorage>,
    pub embedding_fields: &'index HashMap<FieldId, EmbeddingFieldStorage>,
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

    // Base data directory for this index. Used to create stable paths
    // for BoolFieldStorage instances created during update().
    // Always known from construction time.
    data_dir: PathBuf,

    document_count: u64,
    uncommitted_deleted_documents: HashSet<DocumentId>,

    // Bool fields are managed directly by BoolFieldStorage (backed by oramacore_fields)
    // instead of being split across uncommitted/committed field maps.
    bool_fields: HashMap<FieldId, BoolFieldStorage>,

    // GeoPoint fields are managed directly by GeoPointFieldStorage (backed by oramacore_fields)
    // instead of being split across uncommitted/committed field maps.
    geopoint_fields: HashMap<FieldId, GeoPointFieldStorage>,

    // Date fields are managed directly by DateFieldStorage (backed by oramacore_fields I64Storage)
    // instead of being split across uncommitted/committed field maps.
    date_fields: HashMap<FieldId, DateFieldStorage>,

    // Number fields are managed directly by NumberFieldStorage (backed by dual oramacore_fields I64/F64 storages)
    // instead of being split across uncommitted/committed field maps.
    number_fields: HashMap<FieldId, NumberFieldStorage>,

    // StringFilter fields are managed directly by StringFilterFieldStorage (backed by oramacore_fields)
    // instead of being split across uncommitted/committed field maps.
    string_filter_fields: HashMap<FieldId, StringFilterFieldStorage>,

    // Embedding fields are managed directly by EmbeddingFieldStorage (backed by oramacore_fields)
    // instead of being split across uncommitted/committed field maps.
    embedding_fields: HashMap<FieldId, EmbeddingFieldStorage>,

    // String (fulltext/BM25) fields are managed directly by StringFieldStorage
    // (backed by oramacore_fields StringStorage) instead of being split across
    // uncommitted/committed field maps.
    string_fields: HashMap<FieldId, StringFieldStorage>,

    uncommitted_fields: OramaAsyncLock<UncommittedFields>,
    committed_fields: OramaAsyncLock<CommittedFields>,

    path_to_index_id_map: PathToIndexId,

    is_new: AtomicBool,

    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,

    enum_strategy: EnumStrategy,

    /// Custom multipliers (OMC - Orama Custom Multiplier) for documents.
    /// These multipliers are applied to document scores during search.
    /// The first HashMap is uncommitted OMC values, the second is committed.
    omc: OramaAsyncLock<(HashMap<DocumentId, f32>, HashMap<DocumentId, f32>)>,
}

impl Index {
    pub fn new(
        id: IndexId,
        text_parser: Arc<TextParser>,
        context: ReadSideContext,
        offload_config: OffloadFieldConfig,
        enum_strategy: EnumStrategy,
        data_dir: PathBuf,
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

            data_dir,

            document_count: 0,
            uncommitted_deleted_documents: HashSet::new(),

            bool_fields: HashMap::new(),
            number_fields: HashMap::new(),
            date_fields: HashMap::new(),
            geopoint_fields: HashMap::new(),
            string_filter_fields: HashMap::new(),
            embedding_fields: HashMap::new(),
            string_fields: HashMap::new(),

            committed_fields: OramaAsyncLock::new("committed_fields", Default::default()),
            uncommitted_fields: OramaAsyncLock::new("uncommitted_fields", Default::default()),

            path_to_index_id_map: PathToIndexId::empty(),
            is_new: AtomicBool::new(true),

            created_at: Utc::now(),
            updated_at: Utc::now(),

            enum_strategy,

            omc: OramaAsyncLock::new("omc", (HashMap::new(), HashMap::new())),
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

        let uncommitted_fields = UncommittedFields::default();
        let committed_fields = CommittedFields::default();
        let mut bool_fields = HashMap::with_capacity(dump.bool_field_ids.len());
        debug!("Loading bool fields");
        for (field_id, info) in dump.bool_field_ids {
            filter_fields.insert(info.field_path.clone(), (field_id, FieldType::Bool));

            debug!("BoolFieldStorage::try_load for field_id {:?}", field_id);
            let field = BoolFieldStorage::try_load(info)
                .context("Cannot load bool field")?;
            debug!("DONE");
            bool_fields.insert(field_id, field);
        }
        debug!("Bool fields loaded");

        debug!("Loading number_field_ids");
        let mut number_fields = HashMap::with_capacity(dump.number_field_ids.len());
        for (field_id, info) in dump.number_field_ids {
            filter_fields.insert(info.field_path.clone(), (field_id, FieldType::Number));

            debug!("NumberFieldStorage::try_load for field_id {:?}", field_id);
            let field = NumberFieldStorage::try_load(info)
                .context("Cannot load number field")?;
            debug!("DONE");
            number_fields.insert(field_id, field);
        }
        debug!("Number fields loaded");

        debug!("Loading date_field_ids");
        let mut date_fields = HashMap::with_capacity(dump.date_field_ids.len());
        for (field_id, info) in dump.date_field_ids {
            filter_fields.insert(info.field_path.clone(), (field_id, FieldType::Date));

            debug!("DateFieldStorage::try_load for field_id {:?}", field_id);
            let field = DateFieldStorage::try_load(info)
                .context("Cannot load date field")?;
            debug!("DONE");
            date_fields.insert(field_id, field);
        }

        debug!("Loading geopoint_field_ids");
        let mut geopoint_fields = HashMap::with_capacity(dump.geopoint_field_ids.len());
        for (field_id, info) in dump.geopoint_field_ids {
            filter_fields.insert(info.field_path.clone(), (field_id, FieldType::GeoPoint));

            debug!("GeoPointFieldStorage::try_load for field_id {:?}", field_id);
            let field = GeoPointFieldStorage::try_load(info)
                .context("Cannot load geopoint field")?;
            debug!("DONE");
            geopoint_fields.insert(field_id, field);
        }

        debug!("Loading string_filter_field_ids");
        let mut string_filter_fields = HashMap::with_capacity(dump.string_filter_field_ids.len());
        for (field_id, info) in dump.string_filter_field_ids {
            filter_fields.insert(info.field_path.clone(), (field_id, FieldType::StringFilter));

            let field = StringFilterFieldStorage::try_load(info)
                .context("Cannot load string filter field")?;
            string_filter_fields.insert(field_id, field);
        }

        debug!("Loading string_field_ids");
        let mut string_fields = HashMap::with_capacity(dump.string_field_ids.len());
        for (field_id, info) in dump.string_field_ids {
            score_fields.insert(info.field_path.clone(), (field_id, FieldType::String));

            debug!("StringFieldStorage::try_load for field_id {:?}", field_id);
            let field = StringFieldStorage::try_load(info)
                .context("Cannot load string field")?;
            debug!("DONE");
            string_fields.insert(field_id, field);
        }

        debug!("Loading vector_field_ids");
        let mut embedding_fields = HashMap::with_capacity(dump.vector_field_ids.len());
        for (field_id, info) in dump.vector_field_ids {
            score_fields.insert(info.field_path.clone(), (field_id, FieldType::Vector));

            debug!("EmbeddingFieldStorage::try_load for field_id {:?}", field_id);
            let field = EmbeddingFieldStorage::try_load(info)
                .context("Cannot load embedding field")?;
            debug!("DONE");
            embedding_fields.insert(field_id, field);
        }
        debug!("Vector fields loaded");

        // Load OMC from dedicated versioned file, with backward compatibility
        // for existing indexes that stored OMC inside index.json (dump.omc).
        let committed_omc: HashMap<DocumentId, f32> = BufferedFile::open(data_dir.join("omc.bin"))
            .and_then(|f| f.read_bincode_data::<OmcDump>())
            .map(|dump| {
                let OmcDump::V1(v1) = dump;
                v1.data
            })
            .unwrap_or_else(|_| {
                // Fallback 1: try reading old ocm.bin for backward compatibility
                BufferedFile::open(data_dir.join("ocm.bin"))
                    .and_then(|f| f.read_bincode_data::<OmcDump>())
                    .map(|dump| {
                        let OmcDump::V1(v1) = dump;
                        v1.data
                    })
                    .unwrap_or_else(|_| {
                        // Fallback 2: dump.ocm for backward compatibility with existing indexes
                        dump.ocm.unwrap_or_default()
                    })
            });

        Ok(Self {
            id: dump.id,
            locale: dump.locale,
            text_parser: context.nlp_service.get(dump.locale),
            deleted: None,
            promoted_to_runtime_index: AtomicBool::new(false),
            aliases: dump.aliases,

            context,
            offload_config,

            data_dir,

            document_count: dump.document_count,
            uncommitted_deleted_documents: HashSet::new(),

            bool_fields,
            number_fields,
            date_fields,
            geopoint_fields,
            string_filter_fields,
            embedding_fields,
            string_fields,

            committed_fields: OramaAsyncLock::new("committed_fields", committed_fields),
            uncommitted_fields: OramaAsyncLock::new("uncommitted_fields", uncommitted_fields),

            path_to_index_id_map: PathToIndexId::new(filter_fields, score_fields),
            is_new: AtomicBool::new(false),

            created_at: dump.created_at,
            updated_at: dump.updated_at,

            enum_strategy: dump.enum_strategy,

            // Use OMC loaded from dedicated file or from dump (backward compatibility)
            omc: OramaAsyncLock::new("omc", (HashMap::new(), committed_omc)),
        })
    }

    pub fn get_text_parser(&self) -> &TextParser {
        self.text_parser.as_ref()
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
            bool_fields: &self.bool_fields,
            number_fields: &self.number_fields,
            date_fields: &self.date_fields,
            geopoint_fields: &self.geopoint_fields,
            string_filter_fields: &self.string_filter_fields,
            string_fields: &self.string_fields,
            embedding_fields: &self.embedding_fields,
            path_to_field_id_map: &self.path_to_index_id_map,
            uncommitted_deleted_documents: &self.uncommitted_deleted_documents,
            text_parser: &self.text_parser,
            read_side_context: &self.context,
        }
    }

    pub async fn get_all_document_ids(&self) -> Result<Vec<DocumentId>> {
        let context = token_score::TokenScoreContext::new(
            self.id,
            self.document_count,
            &self.embedding_fields,
            &self.string_fields,
            &self.text_parser,
            &self.context,
            &self.path_to_index_id_map,
        );

        let mode = ScoreMode::Default(FulltextMode {
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
        let something_to_commit = self
            .bool_fields
            .values()
            .any(|field| field.has_pending_ops())
            || self
                .geopoint_fields
                .values()
                .any(|field| field.has_pending_ops())
            || self
                .string_filter_fields
                .values()
                .any(|field| field.has_pending_ops())
            || self
                .date_fields
                .values()
                .any(|field| field.has_pending_ops())
            || self
                .number_fields
                .values()
                .any(|field| field.has_pending_ops())
            || self
                .embedding_fields
                .values()
                .any(|field| field.has_pending_ops())
            || self
                .string_fields
                .values()
                .any(|field| field.has_pending_ops())
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

        // Compact bool fields directly (they manage their own persistence)
        for (_, bool_field) in &self.bool_fields {
            if bool_field.has_pending_ops() {
                bool_field.compact(offset.0 as u64)
                    .context("Cannot compact bool field")?;
            }
        }

        // Compact geopoint fields directly (they manage their own persistence)
        for (_, geopoint_field) in &self.geopoint_fields {
            if geopoint_field.has_pending_ops() {
                geopoint_field.compact(offset.0 as u64)
                    .context("Cannot compact geopoint field")?;
            }
        }

        // Compact string_filter fields directly (they manage their own persistence)
        for (_, string_filter_field) in &self.string_filter_fields {
            if string_filter_field.has_pending_ops() {
                string_filter_field.compact(offset.0 as u64)
                    .context("Cannot compact string_filter field")?;
            }
        }

        // Compact date fields directly (they manage their own persistence via I64Storage)
        for (_, date_field) in &self.date_fields {
            if date_field.has_pending_ops() {
                date_field.compact(offset.0 as u64)
                    .context("Cannot compact date field")?;
            }
        }

        // Compact number fields directly (they manage their own persistence via dual I64/F64 storages)
        for (_, number_field) in &self.number_fields {
            if number_field.has_pending_ops() {
                number_field.compact(offset.0 as u64)
                    .context("Cannot compact number field")?;
            }
        }

        // Compact embedding fields directly (they manage their own persistence via EmbeddingStorage)
        for (_, embedding_field) in &self.embedding_fields {
            if embedding_field.has_pending_ops() {
                embedding_field.compact(offset.0 as u64)
                    .context("Cannot compact embedding field")?;
            }
        }

        // Compact string fields directly (they manage their own persistence via StringStorage)
        for (_, string_field) in &self.string_fields {
            if string_field.has_pending_ops() {
                string_field.compact(offset.0 as u64)
                    .context("Cannot compact string field")?;
            }
        }

        // Drop read locks acquired earlier (no more merge cycle needed for string fields)
        drop(uncommitted_fields);
        drop(committed_fields);

        // Merge and commit OMC values
        // Get write lock to merge uncommitted_omc into committed_omc
        let mut omc_lock = self.omc.write("commit").await;

        // Collect uncommitted OMC values first to avoid borrow issues
        let uncommitted_omc_values: Vec<_> = omc_lock.0.drain().collect();

        // Merge uncommitted OMC into committed OMC
        for (doc_id, multiplier) in uncommitted_omc_values {
            omc_lock.1.insert(doc_id, multiplier);
        }

        // Remove deleted documents from committed OMC
        for doc_id in &self.uncommitted_deleted_documents {
            omc_lock.1.remove(doc_id);
        }

        // Save OMC to dedicated versioned file (only if not empty)
        if !omc_lock.1.is_empty() {
            let omc_dump = OmcDump::V1(OmcDumpV1 {
                data: omc_lock.1.clone(),
            });
            BufferedFile::create_or_overwrite(data_dir.join("omc.bin"))
                .context("Cannot create omc.bin")?
                .write_bincode_data(&omc_dump)
                .context("Cannot write omc.bin")?;
        }
        drop(omc_lock);

        let dump = Dump::V1(DumpV1 {
            id: self.id,
            document_count: self.document_count,
            locale: self.locale,
            aliases: self.aliases.clone(),
            bool_field_ids: self
                .bool_fields
                .iter()
                .map(|(k, v)| (*k, v.metadata()))
                .collect(),
            number_field_ids: self
                .number_fields
                .iter()
                .map(|(k, v)| (*k, v.metadata()))
                .collect(),
            date_field_ids: self
                .date_fields
                .iter()
                .map(|(k, v)| (*k, v.metadata()))
                .collect(),
            geopoint_field_ids: self
                .geopoint_fields
                .iter()
                .map(|(k, v)| (*k, v.metadata()))
                .collect(),
            string_filter_field_ids: self
                .string_filter_fields
                .iter()
                .map(|(k, v)| (*k, v.metadata()))
                .collect(),
            string_field_ids: self
                .string_fields
                .iter()
                .map(|(k, v)| (*k, v.metadata()))
                .collect(),
            vector_field_ids: self
                .embedding_fields
                .iter()
                .map(|(k, v)| (*k, v.metadata()))
                .collect(),
            // Not used anymore. We calculate it on the fly
            path_to_index_id_map: Vec::new(),
            created_at: self.created_at,
            updated_at: self.updated_at,
            enum_strategy: self.enum_strategy,
            // OMC is now stored in dedicated omc.bin file; set to None for new data
            ocm: None,
        });

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
        let mut field_data_dirs: HashSet<PathBuf> = HashSet::new();
        // Bool fields are managed separately by BoolFieldStorage
        field_data_dirs.extend(
            self.bool_fields
                .values()
                .map(|f| f.base_path().to_path_buf()),
        );
        // Number fields are managed separately by NumberFieldStorage
        field_data_dirs.extend(
            self.number_fields
                .values()
                .map(|f| f.base_path().to_path_buf()),
        );
        // StringFilter fields are managed separately by StringFilterFieldStorage
        field_data_dirs.extend(
            self.string_filter_fields
                .values()
                .map(|f| f.base_path().to_path_buf()),
        );
        // Date fields are managed separately by DateFieldStorage
        field_data_dirs.extend(
            self.date_fields
                .values()
                .map(|f| f.base_path().to_path_buf()),
        );
        // GeoPoint fields are managed separately by GeoPointFieldStorage
        field_data_dirs.extend(
            self.geopoint_fields
                .values()
                .map(|f| f.base_path().to_path_buf()),
        );
        // Embedding fields are managed separately by EmbeddingFieldStorage
        field_data_dirs.extend(
            self.embedding_fields
                .values()
                .map(|f| f.base_path().to_path_buf()),
        );
        // String fields are managed separately by StringFieldStorage
        field_data_dirs.extend(
            self.string_fields
                .values()
                .map(|f| f.base_path().to_path_buf()),
        );

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

        // Clean up old bool field versions (they manage their own persistence)
        for bool_field in self.bool_fields.values() {
            bool_field.cleanup();
        }

        // Clean up old geopoint field versions (they manage their own persistence)
        for geopoint_field in self.geopoint_fields.values() {
            geopoint_field.cleanup();
        }

        // Clean up old string_filter field versions (they manage their own persistence)
        for string_filter_field in self.string_filter_fields.values() {
            string_filter_field.cleanup();
        }

        // Clean up old date field versions (they manage their own persistence via I64Storage)
        for date_field in self.date_fields.values() {
            date_field.cleanup();
        }

        // Clean up old number field versions (they manage their own persistence via dual I64/F64 storages)
        for number_field in self.number_fields.values() {
            number_field.cleanup();
        }

        // Clean up old embedding field versions (they manage their own persistence via EmbeddingStorage)
        for embedding_field in self.embedding_fields.values() {
            embedding_field.cleanup();
        }

        // Clean up old string field versions (they manage their own persistence via StringStorage)
        for string_field in self.string_fields.values() {
            string_field.cleanup();
        }

        Ok(())
    }

    async fn try_unload_fields(&self) {
        // StringStorage uses mmap-based segments, no manual unloading needed.
        // EmbeddingStorage uses mmap-based segments, no manual unloading needed.
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

    pub fn promote_to_runtime_index(
        &mut self,
        runtime_index_id: IndexId,
        new_data_dir: PathBuf,
    ) -> Result<()> {
        let previous_id = self.id;
        self.id = runtime_index_id;
        self.aliases.push(previous_id);
        self.promoted_to_runtime_index
            .store(true, Ordering::Relaxed);
        // We need to update the created_at and updated_at fields
        self.updated_at = Utc::now();

        // Relocate bool fields from temp_indexes/ path to the permanent indexes/ path.
        // Other fields handle this in commit() via merge_type + copy_items,
        // but bool fields manage their own persistence (mmap'd sorted arrays)
        // and need explicit relocation here (where we have &mut self).
        let old_data_dir = std::mem::replace(&mut self.data_dir, new_data_dir.clone());
        for (field_id, old_storage) in self.bool_fields.iter_mut() {
            // Flush any pending in-memory ops to disk before copying the directory,
            // otherwise uncommitted inserts/deletes would be lost during promotion.
            if old_storage.has_pending_ops() {
                let version = old_storage.current_version_number() + 1;
                old_storage
                    .compact(version)
                    .context("Cannot compact bool field before promotion")?;
                old_storage.cleanup();
            }

            let old_path = old_storage.base_path().to_path_buf();
            let new_path = new_data_dir.join("bool").join(field_id.0.to_string());

            if old_path == new_path {
                continue;
            }

            // Copy old directory to new location (same strategy as merge.rs)
            if old_path.exists() {
                let options = fs_extra::dir::CopyOptions::new().overwrite(true);
                if let Some(parent) = new_path.parent() {
                    oramacore_lib::fs::create_if_not_exists(parent.to_path_buf())
                        .context("Cannot create bool field parent directory")?;
                }
                fs_extra::copy_items(&[&old_path], new_path.parent().expect("bool path must have parent"), &options)
                    .context("Cannot copy bool field directory during promotion")?;
            }

            // Recreate BoolFieldStorage at the new path
            let field_path = old_storage.field_path().into();
            let new_storage = BoolFieldStorage::new(field_path, new_path)
                .context("Cannot create BoolFieldStorage at new path after promotion")?;
            *old_storage = new_storage;
        }

        // Relocate geopoint fields from temp_indexes/ path to the permanent indexes/ path.
        for (field_id, old_storage) in self.geopoint_fields.iter_mut() {
            if old_storage.has_pending_ops() {
                let version = old_storage.current_version_id() + 1;
                old_storage
                    .compact(version)
                    .context("Cannot compact geopoint field before promotion")?;
                old_storage.cleanup();
            }

            let old_path = old_storage.base_path().to_path_buf();
            let new_path = new_data_dir.join("geopoint").join(field_id.0.to_string());

            if old_path == new_path {
                continue;
            }

            if old_path.exists() {
                let options = fs_extra::dir::CopyOptions::new().overwrite(true);
                if let Some(parent) = new_path.parent() {
                    oramacore_lib::fs::create_if_not_exists(parent.to_path_buf())
                        .context("Cannot create geopoint field parent directory")?;
                }
                fs_extra::copy_items(&[&old_path], new_path.parent().expect("geopoint path must have parent"), &options)
                    .context("Cannot copy geopoint field directory during promotion")?;
            }

            let field_path = old_storage.field_path().into();
            let new_storage = GeoPointFieldStorage::new(field_path, new_path)
                .context("Cannot create GeoPointFieldStorage at new path after promotion")?;
            *old_storage = new_storage;
        }

        // Relocate string_filter fields from temp_indexes/ path to the permanent indexes/ path.
        for (field_id, old_storage) in self.string_filter_fields.iter_mut() {
            if old_storage.has_pending_ops() {
                let version = old_storage.current_version_number() + 1;
                old_storage
                    .compact(version)
                    .context("Cannot compact string_filter field before promotion")?;
                old_storage.cleanup();
            }

            let old_path = old_storage.base_path().to_path_buf();
            let new_path = new_data_dir.join("string_filter").join(field_id.0.to_string());

            if old_path == new_path {
                continue;
            }

            if old_path.exists() {
                let options = fs_extra::dir::CopyOptions::new().overwrite(true);
                if let Some(parent) = new_path.parent() {
                    oramacore_lib::fs::create_if_not_exists(parent.to_path_buf())
                        .context("Cannot create string_filter field parent directory")?;
                }
                fs_extra::copy_items(&[&old_path], new_path.parent().expect("string_filter path must have parent"), &options)
                    .context("Cannot copy string_filter field directory during promotion")?;
            }

            let field_path = old_storage.field_path().into();
            let new_storage = StringFilterFieldStorage::new(field_path, new_path)
                .context("Cannot create StringFilterFieldStorage at new path after promotion")?;
            *old_storage = new_storage;
        }

        // Relocate date fields from temp_indexes/ path to the permanent indexes/ path.
        for (field_id, old_storage) in self.date_fields.iter_mut() {
            if old_storage.has_pending_ops() {
                let version = old_storage.current_offset() + 1;
                old_storage
                    .compact(version)
                    .context("Cannot compact date field before promotion")?;
                old_storage.cleanup();
            }

            let old_path = old_storage.base_path().to_path_buf();
            let new_path = new_data_dir.join("date").join(field_id.0.to_string());

            if old_path == new_path {
                continue;
            }

            if old_path.exists() {
                let options = fs_extra::dir::CopyOptions::new().overwrite(true);
                if let Some(parent) = new_path.parent() {
                    oramacore_lib::fs::create_if_not_exists(parent.to_path_buf())
                        .context("Cannot create date field parent directory")?;
                }
                fs_extra::copy_items(&[&old_path], new_path.parent().expect("date path must have parent"), &options)
                    .context("Cannot copy date field directory during promotion")?;
            }

            let field_path = old_storage.field_path().into();
            let new_storage = DateFieldStorage::new(field_path, new_path)
                .context("Cannot create DateFieldStorage at new path after promotion")?;
            *old_storage = new_storage;
        }

        // Relocate number fields from temp_indexes/ path to the permanent indexes/ path.
        for (field_id, old_storage) in self.number_fields.iter_mut() {
            if old_storage.has_pending_ops() {
                let version = old_storage.current_offset() + 1;
                old_storage
                    .compact(version)
                    .context("Cannot compact number field before promotion")?;
                old_storage.cleanup();
            }

            let old_path = old_storage.base_path().to_path_buf();
            let new_path = new_data_dir.join("number").join(field_id.0.to_string());

            if old_path == new_path {
                continue;
            }

            if old_path.exists() {
                let options = fs_extra::dir::CopyOptions::new().overwrite(true);
                if let Some(parent) = new_path.parent() {
                    oramacore_lib::fs::create_if_not_exists(parent.to_path_buf())
                        .context("Cannot create number field parent directory")?;
                }
                fs_extra::copy_items(&[&old_path], new_path.parent().expect("number path must have parent"), &options)
                    .context("Cannot copy number field directory during promotion")?;
            }

            let field_path = old_storage.field_path().into();
            let new_storage = NumberFieldStorage::new(field_path, new_path)
                .context("Cannot create NumberFieldStorage at new path after promotion")?;
            *old_storage = new_storage;
        }

        // Relocate embedding fields from temp_indexes/ path to the permanent indexes/ path.
        for (field_id, old_storage) in self.embedding_fields.iter_mut() {
            if old_storage.has_pending_ops() {
                let version = old_storage.current_version_number() + 1;
                old_storage
                    .compact(version)
                    .context("Cannot compact embedding field before promotion")?;
                old_storage.cleanup();
            }

            let old_path = old_storage.base_path().to_path_buf();
            let new_path = new_data_dir.join("embedding").join(field_id.0.to_string());

            if old_path == new_path {
                continue;
            }

            if old_path.exists() {
                let options = fs_extra::dir::CopyOptions::new().overwrite(true);
                if let Some(parent) = new_path.parent() {
                    oramacore_lib::fs::create_if_not_exists(parent.to_path_buf())
                        .context("Cannot create embedding field parent directory")?;
                }
                fs_extra::copy_items(&[&old_path], new_path.parent().expect("embedding path must have parent"), &options)
                    .context("Cannot copy embedding field directory during promotion")?;
            }

            let field_path = old_storage.field_path().into();
            let model = old_storage.model();
            let new_storage = EmbeddingFieldStorage::new(field_path, new_path, model)
                .context("Cannot create EmbeddingFieldStorage at new path after promotion")?;
            *old_storage = new_storage;
        }

        // Clean up old bool directory if it exists and is different
        let old_bool_dir = old_data_dir.join("bool");
        if old_bool_dir.exists() {
            if let Err(e) = std::fs::remove_dir_all(&old_bool_dir) {
                warn!("Failed to remove old bool directory {:?}: {}", old_bool_dir, e);
            }
        }

        // Clean up old geopoint directory if it exists and is different
        let old_geopoint_dir = old_data_dir.join("geopoint");
        if old_geopoint_dir.exists() {
            if let Err(e) = std::fs::remove_dir_all(&old_geopoint_dir) {
                warn!("Failed to remove old geopoint directory {:?}: {}", old_geopoint_dir, e);
            }
        }

        // Clean up old string_filter directory if it exists and is different
        let old_string_filter_dir = old_data_dir.join("string_filter");
        if old_string_filter_dir.exists() {
            if let Err(e) = std::fs::remove_dir_all(&old_string_filter_dir) {
                warn!("Failed to remove old string_filter directory {:?}: {}", old_string_filter_dir, e);
            }
        }

        // Clean up old date directory if it exists and is different
        let old_date_dir = old_data_dir.join("date");
        if old_date_dir.exists() {
            if let Err(e) = std::fs::remove_dir_all(&old_date_dir) {
                warn!("Failed to remove old date directory {:?}: {}", old_date_dir, e);
            }
        }

        // Clean up old number directory if it exists and is different
        let old_number_dir = old_data_dir.join("number");
        if old_number_dir.exists() {
            if let Err(e) = std::fs::remove_dir_all(&old_number_dir) {
                warn!("Failed to remove old number directory {:?}: {}", old_number_dir, e);
            }
        }

        // Relocate string fields from temp_indexes/ path to the permanent indexes/ path.
        for (field_id, old_storage) in self.string_fields.iter_mut() {
            if old_storage.has_pending_ops() {
                let version = old_storage.current_version_number() + 1;
                old_storage
                    .compact(version)
                    .context("Cannot compact string field before promotion")?;
                old_storage.cleanup();
            }

            let old_path = old_storage.base_path().to_path_buf();
            let new_path = new_data_dir.join("string").join(field_id.0.to_string());

            if old_path == new_path {
                continue;
            }

            if old_path.exists() {
                let options = fs_extra::dir::CopyOptions::new().overwrite(true);
                if let Some(parent) = new_path.parent() {
                    oramacore_lib::fs::create_if_not_exists(parent.to_path_buf())
                        .context("Cannot create string field parent directory")?;
                }
                fs_extra::copy_items(&[&old_path], new_path.parent().expect("string path must have parent"), &options)
                    .context("Cannot copy string field directory during promotion")?;
            }

            let field_path = old_storage.field_path().into();
            let new_storage = StringFieldStorage::new(field_path, new_path)
                .context("Cannot create StringFieldStorage at new path after promotion")?;
            *old_storage = new_storage;
        }

        // Clean up old embedding directory if it exists and is different
        let old_embedding_dir = old_data_dir.join("embedding");
        if old_embedding_dir.exists() {
            if let Err(e) = std::fs::remove_dir_all(&old_embedding_dir) {
                warn!("Failed to remove old embedding directory {:?}: {}", old_embedding_dir, e);
            }
        }

        // Clean up old string directory if it exists and is different
        let old_string_dir = old_data_dir.join("string");
        if old_string_dir.exists() {
            if let Err(e) = std::fs::remove_dir_all(&old_string_dir) {
                warn!("Failed to remove old string directory {:?}: {}", old_string_dir, e);
            }
        }

        Ok(())
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
                        // Create BoolFieldStorage with a stable path under the index data dir.
                        // data_dir is always known from construction time.
                        let bool_base_path = self.data_dir.join("bool").join(field_id.0.to_string());
                        let storage = BoolFieldStorage::new(field_path, bool_base_path)
                            .context("Failed to create BoolFieldStorage")?;
                        self.bool_fields.insert(field_id, storage);
                    }
                    IndexWriteOperationFieldType::Number => {
                        self.path_to_index_id_map.insert_filter_field(
                            field_path.clone(),
                            field_id,
                            FieldType::Number,
                        );
                        // Create NumberFieldStorage with a stable path under the index data dir.
                        let number_base_path = self.data_dir.join("number").join(field_id.0.to_string());
                        let storage = NumberFieldStorage::new(field_path, number_base_path)
                            .context("Failed to create NumberFieldStorage")?;
                        self.number_fields.insert(field_id, storage);
                    }
                    IndexWriteOperationFieldType::StringFilter => {
                        self.path_to_index_id_map.insert_filter_field(
                            field_path.clone(),
                            field_id,
                            FieldType::StringFilter,
                        );
                        // Create StringFilterFieldStorage with a stable path under the index data dir.
                        let string_filter_base_path = self.data_dir.join("string_filter").join(field_id.0.to_string());
                        let storage = StringFilterFieldStorage::new(field_path, string_filter_base_path)
                            .context("Failed to create StringFilterFieldStorage")?;
                        self.string_filter_fields.insert(field_id, storage);
                    }
                    IndexWriteOperationFieldType::Date => {
                        self.path_to_index_id_map.insert_filter_field(
                            field_path.clone(),
                            field_id,
                            FieldType::Date,
                        );
                        // Create DateFieldStorage with a stable path under the index data dir.
                        let date_base_path = self.data_dir.join("date").join(field_id.0.to_string());
                        let storage = DateFieldStorage::new(field_path, date_base_path)
                            .context("Failed to create DateFieldStorage")?;
                        self.date_fields.insert(field_id, storage);
                    }
                    IndexWriteOperationFieldType::GeoPoint => {
                        self.path_to_index_id_map.insert_filter_field(
                            field_path.clone(),
                            field_id,
                            FieldType::GeoPoint,
                        );
                        let geopoint_path = self.data_dir.join("geopoint").join(field_id.0.to_string());
                        let storage = GeoPointFieldStorage::new(field_path, geopoint_path)
                            .context("Failed to create GeoPointFieldStorage")?;
                        self.geopoint_fields.insert(field_id, storage);
                    }
                    IndexWriteOperationFieldType::String => {
                        self.path_to_index_id_map.insert_score_field(
                            field_path.clone(),
                            field_id,
                            FieldType::String,
                        );
                        let string_base_path = self.data_dir.join("string").join(field_id.0.to_string());
                        let storage = StringFieldStorage::new(field_path, string_base_path)
                            .context("Failed to create StringFieldStorage")?;
                        self.string_fields.insert(field_id, storage);
                    }
                    IndexWriteOperationFieldType::Embedding(model) => {
                        self.path_to_index_id_map.insert_score_field(
                            field_path.clone(),
                            field_id,
                            FieldType::Vector,
                        );
                        let embedding_path = self.data_dir.join("embedding").join(field_id.0.to_string());
                        let storage = EmbeddingFieldStorage::new(field_path, embedding_path, model)
                            .context("Failed to create EmbeddingFieldStorage")?;
                        self.embedding_fields.insert(field_id, storage);
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
                            if let Some(field) = self.bool_fields.get(&field_id) {
                                field.insert(doc_id, bool_value);
                            } else {
                                error!("Cannot find bool field {:?}", field_id);
                            }
                        }
                        IndexedValue::FilterNumber(field_id, number) => {
                            if let Some(field) = self.number_fields.get(&field_id) {
                                if let Err(e) = field.insert(doc_id, number.0) {
                                    error!("Failed to insert number value: {e:?}");
                                }
                            } else {
                                error!("Cannot find number field {:?}", field_id);
                            }
                        }
                        IndexedValue::FilterNumber2(field_id, ref number_indexed_value) => {
                            if let Some(field) = self.number_fields.get(&field_id) {
                                if let Err(e) = field.insert_indexed(doc_id, number_indexed_value) {
                                    error!("Failed to insert indexed number value: {e:?}");
                                }
                            } else {
                                error!("Cannot find number field {:?}", field_id);
                            }
                        }
                        IndexedValue::FilterString(field_id, string_value) => {
                            if let Some(field) = self.string_filter_fields.get(&field_id) {
                                field.insert(doc_id, string_value);
                            } else {
                                error!("Cannot find string_filter field {:?}", field_id);
                            }
                        }
                        IndexedValue::FilterString2(field_id, ref string_filter_indexed_value) => {
                            if let Some(field) = self.string_filter_fields.get(&field_id) {
                                field.insert_indexed(doc_id, string_filter_indexed_value);
                            } else {
                                error!("Cannot find string_filter field {:?}", field_id);
                            }
                        }
                        IndexedValue::ScoreString(field_id, len, values) => {
                            if let Some(field) = self.string_fields.get(&field_id) {
                                field.insert_legacy(doc_id, len, values);
                            } else {
                                error!("Cannot find string field {:?}", field_id);
                            }
                        }
                        IndexedValue::ScoreString2(field_id, indexed_value) => {
                            if let Some(field) = self.string_fields.get(&field_id) {
                                field.insert(doc_id, indexed_value);
                            } else {
                                error!("Cannot find string field {:?}", field_id);
                            }
                        }
                        IndexedValue::FilterDate(field_id, timestamp) => {
                            if let Some(field) = self.date_fields.get(&field_id) {
                                if let Err(e) = field.insert(doc_id, timestamp) {
                                    error!("Cannot insert date for field {:?}: {}", field_id, e);
                                }
                            } else {
                                error!("Cannot find date field {:?}", field_id);
                            }
                        }
                        IndexedValue::FilterDate2(field_id, ref date_indexed_value) => {
                            if let Some(field) = self.date_fields.get(&field_id) {
                                if let Err(e) = field.insert_indexed(doc_id, date_indexed_value) {
                                    error!("Cannot insert indexed date for field {:?}: {}", field_id, e);
                                }
                            } else {
                                error!("Cannot find date field {:?}", field_id);
                            }
                        }
                        IndexedValue::FilterGeoPoint(field_id, geopoint) => {
                            if let Some(field) = self.geopoint_fields.get(&field_id) {
                                if let Err(e) = field.insert(doc_id, geopoint) {
                                    error!("Cannot insert geopoint for field {:?}: {}", field_id, e);
                                }
                            } else {
                                error!("Cannot find geopoint field {:?}", field_id);
                            }
                        }
                        IndexedValue::FilterBool2(field_id, ref bool_indexed_value) => {
                            if let Some(field) = self.bool_fields.get(&field_id) {
                                field.insert_indexed(doc_id, bool_indexed_value);
                            } else {
                                error!("Cannot find bool field {:?}", field_id);
                            }
                        }
                        IndexedValue::FilterGeoPoint2(field_id, ref geopoint_indexed_value) => {
                            if let Some(field) = self.geopoint_fields.get(&field_id) {
                                field.insert_indexed(doc_id, geopoint_indexed_value);
                            } else {
                                error!("Cannot find geopoint field {:?}", field_id);
                            }
                        }
                    }
                }
            }
            IndexWriteOperation::Index2 {
                doc_id,
                indexed_values,
                omc,
            } => {
                // Handle Index2: same as Index but also stores the OMC value if present
                self.document_count += 1;

                // Store OMC value if provided
                if let Some(multiplier) = omc {
                    let (ref mut uncommitted_omc, _) = *self.omc.get_mut();
                    uncommitted_omc.insert(doc_id, multiplier);
                }

                for indexed_value in indexed_values {
                    match indexed_value {
                        IndexedValue::FilterBool(field_id, bool_value) => {
                            if let Some(field) = self.bool_fields.get(&field_id) {
                                field.insert(doc_id, bool_value);
                            } else {
                                error!("Cannot find bool field {:?}", field_id);
                            }
                        }
                        IndexedValue::FilterNumber(field_id, number) => {
                            if let Some(field) = self.number_fields.get(&field_id) {
                                if let Err(e) = field.insert(doc_id, number.0) {
                                    error!("Failed to insert number value: {e:?}");
                                }
                            } else {
                                error!("Cannot find number field {:?}", field_id);
                            }
                        }
                        IndexedValue::FilterNumber2(field_id, ref number_indexed_value) => {
                            if let Some(field) = self.number_fields.get(&field_id) {
                                if let Err(e) = field.insert_indexed(doc_id, number_indexed_value) {
                                    error!("Failed to insert indexed number value: {e:?}");
                                }
                            } else {
                                error!("Cannot find number field {:?}", field_id);
                            }
                        }
                        IndexedValue::FilterString(field_id, string_value) => {
                            if let Some(field) = self.string_filter_fields.get(&field_id) {
                                field.insert(doc_id, string_value);
                            } else {
                                error!("Cannot find string_filter field {:?}", field_id);
                            }
                        }
                        IndexedValue::FilterString2(field_id, ref string_filter_indexed_value) => {
                            if let Some(field) = self.string_filter_fields.get(&field_id) {
                                field.insert_indexed(doc_id, string_filter_indexed_value);
                            } else {
                                error!("Cannot find string_filter field {:?}", field_id);
                            }
                        }
                        IndexedValue::ScoreString(field_id, len, values) => {
                            if let Some(field) = self.string_fields.get(&field_id) {
                                field.insert_legacy(doc_id, len, values);
                            } else {
                                error!("Cannot find string field {:?}", field_id);
                            }
                        }
                        IndexedValue::ScoreString2(field_id, indexed_value) => {
                            if let Some(field) = self.string_fields.get(&field_id) {
                                field.insert(doc_id, indexed_value);
                            } else {
                                error!("Cannot find string field {:?}", field_id);
                            }
                        }
                        IndexedValue::FilterDate(field_id, timestamp) => {
                            if let Some(field) = self.date_fields.get(&field_id) {
                                if let Err(e) = field.insert(doc_id, timestamp) {
                                    error!("Cannot insert date for field {:?}: {}", field_id, e);
                                }
                            } else {
                                error!("Cannot find date field {:?}", field_id);
                            }
                        }
                        IndexedValue::FilterDate2(field_id, ref date_indexed_value) => {
                            if let Some(field) = self.date_fields.get(&field_id) {
                                if let Err(e) = field.insert_indexed(doc_id, date_indexed_value) {
                                    error!("Cannot insert indexed date for field {:?}: {}", field_id, e);
                                }
                            } else {
                                error!("Cannot find date field {:?}", field_id);
                            }
                        }
                        IndexedValue::FilterGeoPoint(field_id, geopoint) => {
                            if let Some(field) = self.geopoint_fields.get(&field_id) {
                                if let Err(e) = field.insert(doc_id, geopoint) {
                                    error!("Cannot insert geopoint for field {:?}: {}", field_id, e);
                                }
                            } else {
                                error!("Cannot find geopoint field {:?}", field_id);
                            }
                        }
                        IndexedValue::FilterBool2(field_id, ref bool_indexed_value) => {
                            if let Some(field) = self.bool_fields.get(&field_id) {
                                field.insert_indexed(doc_id, bool_indexed_value);
                            } else {
                                error!("Cannot find bool field {:?}", field_id);
                            }
                        }
                        IndexedValue::FilterGeoPoint2(field_id, ref geopoint_indexed_value) => {
                            if let Some(field) = self.geopoint_fields.get(&field_id) {
                                field.insert_indexed(doc_id, geopoint_indexed_value);
                            } else {
                                error!("Cannot find geopoint field {:?}", field_id);
                            }
                        }
                    }
                }
            }
            IndexWriteOperation::IndexEmbedding { data } => {
                for (field_id, data) in data {
                    if let Some(field) = self.embedding_fields.get(&field_id) {
                        for (doc_id, vectors) in data {
                            field.insert(doc_id, vectors);
                        }
                    } else {
                        error!("Cannot find embedding field {:?}", field_id);
                    }
                }
            }
            IndexWriteOperation::DeleteDocuments { doc_ids } => {
                let len = doc_ids.len() as u64;

                // Also remove OMC values for deleted documents
                let (ref mut uncommitted_omc, ref mut committed_omc) = *self.omc.get_mut();
                for doc_id in &doc_ids {
                    uncommitted_omc.remove(doc_id);
                    committed_omc.remove(doc_id);
                }

                // Delete from bool fields directly (they manage their own deletion)
                for doc_id in &doc_ids {
                    for (_, bool_field) in &self.bool_fields {
                        bool_field.delete(*doc_id);
                    }
                }

                // Delete from geopoint fields directly (they manage their own deletion)
                for doc_id in &doc_ids {
                    for (_, geopoint_field) in &self.geopoint_fields {
                        geopoint_field.delete(*doc_id);
                    }
                }

                // Delete from string_filter fields directly (they manage their own deletion)
                for doc_id in &doc_ids {
                    for (_, string_filter_field) in &self.string_filter_fields {
                        string_filter_field.delete(*doc_id);
                    }
                }

                // Delete from date fields directly (they manage their own deletion via I64Storage)
                for doc_id in &doc_ids {
                    for (_, date_field) in &self.date_fields {
                        date_field.delete(*doc_id);
                    }
                }

                // Delete from number fields directly (they manage their own deletion via dual I64/F64 storages)
                for doc_id in &doc_ids {
                    for (_, number_field) in &self.number_fields {
                        number_field.delete(*doc_id);
                    }
                }

                // Delete from embedding fields directly (they manage their own deletion via EmbeddingStorage)
                for doc_id in &doc_ids {
                    for (_, embedding_field) in &self.embedding_fields {
                        embedding_field.delete(*doc_id);
                    }
                }

                // Delete from string fields directly (they manage their own deletion via StringStorage)
                for doc_id in &doc_ids {
                    for (_, string_field) in &self.string_fields {
                        string_field.delete(*doc_id);
                    }
                }

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

    /// Get all OMC values at once for batch operations.
    /// Returns a reference to both uncommitted and committed OMC maps.
    pub async fn get_all_omc(
        &self,
    ) -> crate::lock::OramaAsyncLockReadGuard<
        '_,
        (HashMap<DocumentId, f32>, HashMap<DocumentId, f32>),
    > {
        self.omc.read("get_all_omc").await
    }

    // Since we only have one embedding model for all indexes in a collection,
    // we can get the first index model and return it early.
    pub fn get_model(&self) -> Option<Model> {
        self.embedding_fields
            .values()
            .next()
            .map(|f| f.model())
    }

    pub async fn stats(&self, is_temp: bool) -> Result<IndexStats> {
        let mut fields_stats: Vec<IndexFieldStats> = Vec::new();

        // Bool fields (managed by BoolFieldStorage, not split across uncommitted/committed)
        for (field_id, bool_field) in &self.bool_fields {
            let stats = bool_field.stats();
            fields_stats.push(IndexFieldStats {
                field_id: *field_id,
                field_path: bool_field.field_path().join("."),
                stats: IndexFieldStatsType::BoolFieldStorage(stats),
            });
        }

        // GeoPoint fields (managed by GeoPointFieldStorage, not split across uncommitted/committed)
        for (field_id, geopoint_field) in &self.geopoint_fields {
            let stats = geopoint_field.stats();
            fields_stats.push(IndexFieldStats {
                field_id: *field_id,
                field_path: geopoint_field.field_path().join("."),
                stats: IndexFieldStatsType::GeoPointFieldStorage(stats),
            });
        }

        // StringFilter fields (managed by StringFilterFieldStorage, not split across uncommitted/committed)
        for (field_id, string_filter_field) in &self.string_filter_fields {
            let stats = string_filter_field.stats();
            fields_stats.push(IndexFieldStats {
                field_id: *field_id,
                field_path: string_filter_field.field_path().join("."),
                stats: IndexFieldStatsType::StringFilterFieldStorage(stats),
            });
        }

        // Date fields (managed by DateFieldStorage, not split across uncommitted/committed)
        for (field_id, date_field) in &self.date_fields {
            let stats = date_field.stats();
            fields_stats.push(IndexFieldStats {
                field_id: *field_id,
                field_path: date_field.field_path().join("."),
                stats: IndexFieldStatsType::DateFieldStorage(stats),
            });
        }

        // Number fields (managed by NumberFieldStorage, not split across uncommitted/committed)
        for (field_id, number_field) in &self.number_fields {
            let stats = number_field.stats();
            fields_stats.push(IndexFieldStats {
                field_id: *field_id,
                field_path: number_field.field_path().join("."),
                stats: IndexFieldStatsType::NumberFieldStorage(stats),
            });
        }

        // String fields (managed by StringFieldStorage, not split across uncommitted/committed)
        for (field_id, string_field) in &self.string_fields {
            fields_stats.push(IndexFieldStats {
                field_id: *field_id,
                field_path: string_field.field_path().join("."),
                stats: IndexFieldStatsType::StringFieldStorage(string_field.stats()),
            });
        }

        // Embedding fields (managed by EmbeddingFieldStorage, not split across uncommitted/committed)
        for (field_id, embedding_field) in &self.embedding_fields {
            let stats = embedding_field.stats();
            fields_stats.push(IndexFieldStats {
                field_id: *field_id,
                field_path: embedding_field.field_path().join("."),
                stats: IndexFieldStatsType::EmbeddingFieldStorage(stats),
            });
        }

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
pub enum FieldType {
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
    /// OMC (Orama Custom Multiplier) values for documents.
    /// DEPRECATED: OMC is now stored in a dedicated `omc.bin` file.
    /// This field is kept for backward compatibility when reading existing data.
    #[serde(default)]
    ocm: Option<HashMap<DocumentId, f32>>,
}

/// Versioned dump format for OMC (Orama Custom Multiplier) data.
/// Stored separately in `omc.bin` for better separation of concerns.
#[derive(Debug, Serialize, Deserialize)]
struct OmcDumpV1 {
    data: HashMap<DocumentId, f32>,
}

/// Versioned envelope for OMC dump data, allowing future format migrations.
#[derive(Debug, Serialize, Deserialize)]
enum OmcDump {
    V1(OmcDumpV1),
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

