use anyhow::{anyhow, bail, Context, Result};
use std::{
    collections::HashMap,
    io::ErrorKind,
    ops::{Deref, DerefMut},
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use axum::extract::State;
use chrono::{DateTime, Utc};
use debug_panic::debug_panic;
use orama_js_pool::OutputChannel;
use oramacore_lib::{
    hook_storage::{HookReader, HookType},
    pin_rules::PinRulesReader,
};
use serde::{Deserialize, Serialize};
use tokio::time::Instant;
use tracing::{debug, error, info, warn};

use crate::{
    ai::advanced_autoquery::{AdvancedAutoQuery, AdvancedAutoQuerySteps, QueryMappedSearchResult},
    collection_manager::sides::{
        read::{
            analytics::SearchAnalyticEventOrigin, context::ReadSideContext,
            CommittedDateFieldStats, CommittedGeoPointFieldStats, CommittedStringFieldStats,
            ReadError, UncommittedDateFieldStats, UncommittedGeoPointFieldStats,
        },
        write::index::EnumStrategy,
        CollectionWriteOperation, Offset, ReplaceIndexReason,
    },
    lock::{OramaAsyncLock, OramaAsyncLockReadGuard, OramaAsyncLockWriteGuard, OramaSyncLock},
    types::{
        ApiKey, CollectionId, CollectionStatsRequest, DocumentId, FieldId, IndexId,
        InteractionMessage, Number, OramaDate, Role,
    },
};

use super::{
    index::{Index, IndexStats},
    CollectionCommitConfig, CommittedBoolFieldStats, CommittedNumberFieldStats,
    CommittedStringFilterFieldStats, CommittedVectorFieldStats, DeletionReason, OffloadFieldConfig,
    ReadSide, UncommittedBoolFieldStats, UncommittedNumberFieldStats, UncommittedStringFieldStats,
    UncommittedStringFilterFieldStats, UncommittedVectorFieldStats,
};

use crate::types::NLPSearchRequest;
use oramacore_lib::fs::*;
use oramacore_lib::nlp::locales::Locale;

#[derive(Serialize)]
pub struct FilterableFieldBool {
    pub field_path: String,
    pub field_type: String,
    pub count_true: usize,
    pub count_false: usize,
    pub count: usize,
}

#[derive(Serialize)]
pub struct FilterableFieldGeoPoint {
    pub field_path: String,
    pub field_type: String,
    pub count: usize,
}

#[derive(Serialize)]
pub struct FilterableFieldDate {
    pub field_path: String,
    pub field_type: String,
    pub min: Option<OramaDate>,
    pub max: Option<OramaDate>,
}

#[derive(Serialize)]
pub struct FilterableFieldNumber {
    pub field_path: String,
    pub field_type: String,
    pub min: f64,
    pub max: f64,
}

#[derive(Serialize)]
pub struct FilterableFieldString {
    pub field_path: String,
    pub field_type: String,
    pub count: usize,
}

#[derive(Serialize)]
#[serde(untagged)]
pub enum FilterableField {
    Bool(FilterableFieldBool),
    GeoPoint(FilterableFieldGeoPoint),
    Date(FilterableFieldDate),
    Number(FilterableFieldNumber),
    String(FilterableFieldString),
}

#[derive(Serialize)]
pub struct FilterableFieldsStats {
    pub index_id: IndexId,
    pub fields: Vec<FilterableField>,
}

pub struct CollectionReader {
    data_dir: PathBuf,
    id: CollectionId,
    description: Option<String>,
    mcp_description: OramaAsyncLock<Option<String>>,
    default_locale: Locale,
    deleted: bool,

    read_api_key: ApiKey,
    write_api_key: Option<ApiKey>,
    context: ReadSideContext,
    offload_config: OffloadFieldConfig,
    commit_config: CollectionCommitConfig,

    indexes: OramaAsyncLock<Vec<Index>>,

    temp_indexes: OramaAsyncLock<Vec<Index>>,

    hook: OramaAsyncLock<HookReader>,

    created_at: DateTime<Utc>,
    updated_at: OramaAsyncLock<DateTime<Utc>>,

    // Per-collection offset tracking - single unified offset
    offset: OramaAsyncLock<Offset>,

    document_count_estimation: AtomicU64,

    // Counter for operations applied since last commit
    pending_operations: Arc<AtomicU64>,

    /// Timestamp of the last successful commit
    /// For new collections, initialized to (now - threshold) to allow immediate commit
    /// For loaded collections, initialized to now to provide grace period
    /// Updated after each successful commit to track elapsed time
    last_commit_time: OramaSyncLock<Instant>,

    pin_rules_reader: OramaAsyncLock<PinRulesReader<DocumentId>>,
}

impl CollectionReader {
    pub fn empty(
        data_dir: PathBuf,
        id: CollectionId,
        description: Option<String>,
        mcp_description: Option<String>,
        default_locale: Locale,
        read_api_key: ApiKey,
        write_api_key: Option<ApiKey>,
        context: ReadSideContext,
        offload_config: OffloadFieldConfig,
        commit_config: CollectionCommitConfig,
    ) -> Result<Self> {
        Ok(Self {
            id,
            description,
            mcp_description: OramaAsyncLock::new("collection_mcp_description", mcp_description),
            default_locale,
            deleted: false,

            read_api_key,
            write_api_key,

            context,
            offload_config,
            commit_config,

            indexes: OramaAsyncLock::new("collection_indexes", Default::default()),
            temp_indexes: OramaAsyncLock::new("collection_temp_indexes", Default::default()),

            hook: OramaAsyncLock::new(
                "collection_hook",
                HookReader::try_new(data_dir.join("hooks"))?,
            ),

            created_at: Utc::now(),
            updated_at: OramaAsyncLock::new("collection_updated_at", Utc::now()),

            // Initialize per-collection offset to 0
            offset: OramaAsyncLock::new("collection_offset", Offset(0)),

            document_count_estimation: AtomicU64::new(0),

            // Initialize pending operations counter to 0
            pending_operations: Arc::new(AtomicU64::new(0)),

            // Initialize to past time to allow immediate commit when operations arrive
            last_commit_time: OramaSyncLock::new(
                "collection_last_commit_time",
                Instant::now() - commit_config.time_threshold,
            ),

            pin_rules_reader: OramaAsyncLock::new("pin_rules_reader", PinRulesReader::empty()),

            data_dir,
        })
    }

    pub fn try_load(
        context: ReadSideContext,
        data_dir: PathBuf,
        offload_config: OffloadFieldConfig,
        commit_config: CollectionCommitConfig,
        global_offset: Offset,
    ) -> Result<Self> {
        debug!("Loading collection info");
        let dump: Dump = BufferedFile::open(data_dir.join("collection.json"))
            .context("Cannot open collection.json")?
            .read_json_data()
            .context("Cannot read collection.json")?;

        // Handle V1 to V2 migration
        let dump = match dump {
            Dump::V1(v1) => {
                info!("Migrating collection {:?} from V1 to V2", v1.id);
                let dump = Dump::V2(DumpV2 {
                    id: v1.id,
                    description: v1.description,
                    mcp_description: v1.mcp_description,
                    default_locale: v1.default_locale,
                    read_api_key: v1.read_api_key,
                    write_api_key: v1.write_api_key,
                    index_ids: v1.index_ids,
                    temp_index_ids: v1.temp_index_ids,
                    created_at: v1.created_at,
                    updated_at: v1.updated_at,
                    offset: global_offset,
                });

                BufferedFile::create_or_overwrite(data_dir.join("collection.json"))
                    .context("Cannot open collection.json")?
                    .write_json_data(&dump)
                    .context("Cannot read collection.json")?;

                match dump {
                    Dump::V2(v2) => v2,
                    _ => unreachable!("Just wrote V2 dump"),
                }
            }
            Dump::V2(mut v2) => {
                // This collection offset could be:
                // - higher than global_offset: this line doesn't change anything
                // - equal to global_offset: this line doesn't change anything
                // - lower than global_offset: this will not change anything
                //   because the reader dequeues operations from global offset
                v2.offset = v2.offset.max(global_offset);

                // Use persisted offset value
                v2
            }
        };
        let offset = dump.offset;
        debug!("Collection info loaded");

        let mut indexes: Vec<Index> = Vec::with_capacity(dump.index_ids.len());
        for index_id in dump.index_ids {
            debug!("Loading index {:?}", index_id);
            let index = Index::try_load(
                index_id,
                data_dir.join("indexes").join(index_id.as_str()),
                context.clone(),
                offload_config,
            )?;
            indexes.push(index);
            debug!("Index {:?} loaded", index_id);
        }

        let mut temp_indexes: Vec<Index> = Vec::with_capacity(dump.temp_index_ids.len());
        for index_id in dump.temp_index_ids {
            debug!("Loading temp index {:?}", index_id);
            let index = Index::try_load(
                index_id,
                data_dir.join("temp_indexes").join(index_id.as_str()),
                context.clone(),
                offload_config,
            )?;
            temp_indexes.push(index);
            debug!("Temp index {:?} loaded", index_id);
        }

        let document_count_estimation = indexes.iter().map(|i| i.document_count()).sum::<u64>();

        let s = Self {
            id: dump.id,
            description: dump.description,
            mcp_description: OramaAsyncLock::new(
                "collection_mcp_description",
                dump.mcp_description,
            ),
            default_locale: dump.default_locale,
            deleted: false,

            read_api_key: dump.read_api_key,
            write_api_key: dump.write_api_key,

            context,
            offload_config,
            commit_config,

            indexes: OramaAsyncLock::new("collection_indexes", indexes),
            temp_indexes: OramaAsyncLock::new("collection_temp_indexes", temp_indexes),

            hook: OramaAsyncLock::new(
                "collection_hook",
                HookReader::try_new(data_dir.join("hooks"))?,
            ),

            created_at: dump.created_at,
            updated_at: OramaAsyncLock::new("collection_updated_at", dump.updated_at),

            // Initialize per-collection offset from loaded data
            offset: OramaAsyncLock::new("collection_offset", offset),

            document_count_estimation: AtomicU64::new(document_count_estimation),

            // Always start at 0 after load (all operations before restart are committed)
            pending_operations: Arc::new(AtomicU64::new(0)),

            // Set to now after restart to give 5-minute grace period before first commit
            last_commit_time: OramaSyncLock::new("collection_last_commit_time", Instant::now()),

            pin_rules_reader: OramaAsyncLock::new(
                "pin_rules_reader",
                PinRulesReader::try_new(data_dir.join("pin_rules"))?,
            ),

            data_dir,
        };

        Ok(s)
    }

    pub async fn commit(&self, force: bool) -> Result<Offset> {
        if !self.should_commit(force) {
            // Return current offset without committing

            let dump: Dump = BufferedFile::open(self.data_dir.join("collection.json"))
                .context("Cannot create collection.json")?
                .read_json_data()
                .context("Cannot write collection.json")?;

            let offset = match dump {
                Dump::V1(_) => **self.offset.read("commit").await,
                Dump::V2(v2) => v2.offset,
            };

            return Ok(offset);
        }

        // During the commit we have:
        // 1. indexes till alive
        // 2. indexes deleted by the user
        // 3. temp indexes promoted to runtime indexes due to resync
        // 4. temp indexes promoted to runtime indexes due to reindexing
        //
        // So:
        // 1. We have to keep it and commit it
        // 2. We have to remove it from the in memory indexes and remove it from the disk
        // 3-4. We have to commit it and remove the temp indexes from the disk
        //
        // NB: For deleted index, the documents are removed
        //     because the writer sends an appropriate command for that.
        //     Hence, we don't need to do anything here.

        // Read current offset - this is what we actually processed
        let offset = **self.offset.read("commit").await;

        let indexes_lock = self.indexes.read("commit").await;
        let mut index_ids = Vec::with_capacity(indexes_lock.len());

        let mut index_id_to_remove = vec![];
        let mut temp_index_id_to_remove = vec![];

        let mut count = 0;
        let indexes_dir = self.data_dir.join("indexes");
        for (i, index) in indexes_lock.iter().enumerate() {
            match index.get_deletion_reason() {
                Some(DeletionReason::UserWanted) => {
                    index_id_to_remove.push((i, index.id()));
                    continue;
                }
                Some(DeletionReason::IndexResynced { temp_index_id })
                | Some(DeletionReason::CollectionReindexed { temp_index_id }) => {
                    // This is the old version of the index.
                    // In the vec, there'll be the new version of this index
                    // So, we should remove the old temp index.
                    temp_index_id_to_remove.push(temp_index_id);
                    continue;
                }
                Some(DeletionReason::ExpiredTempIndex) => {
                    // This shouldn't happen on regular indexes, only on temp indexes
                    index_id_to_remove.push((i, index.id()));
                    continue;
                }
                None => {}
            }
            count += index.document_count();
            index_ids.push(index.id());
            let dir = indexes_dir.join(index.id().as_str());
            index.commit(dir, offset, self.id).await?;
        }
        drop(indexes_lock);

        let temp_indexes_lock = self.temp_indexes.read("commit").await;
        let mut temp_index_ids = Vec::with_capacity(temp_indexes_lock.len());
        let temp_indexes_dir = self.data_dir.join("temp_indexes");
        for index in temp_indexes_lock.iter() {
            if index.is_deleted() {
                // Handle deleted temp indexes - collect them for removal
                match index.get_deletion_reason() {
                    Some(DeletionReason::ExpiredTempIndex) => {
                        // Add expired temp index to removal list
                        temp_index_id_to_remove.push(index.id());
                    }
                    Some(reason) => {
                        warn!(
                            "Unexpected deletion reason for temp index {}: {:?}",
                            index.id(),
                            reason
                        );
                    }
                    None => {
                        warn!(
                            "Temp index {} marked as deleted but no deletion reason",
                            index.id()
                        );
                    }
                }
                continue;
            }
            temp_index_ids.push(index.id());
            let dir = temp_indexes_dir.join(index.id().as_str());
            index.commit(dir, offset, self.id).await?;
        }
        drop(temp_indexes_lock);

        let mut hook_lock = self.hook.write("commit").await;
        hook_lock.commit()?;
        drop(hook_lock);

        let mut pin_rules_reader = self.pin_rules_reader.write("commit").await;
        pin_rules_reader
            .commit(self.data_dir.join("pin_rules"))
            .context("Cannot commit pin rules")?;
        drop(pin_rules_reader);

        // Save DumpV2 with single offset
        let dump = Dump::V2(DumpV2 {
            id: self.id,
            description: self.description.clone(),
            mcp_description: self.mcp_description.read("commit").await.clone(),
            default_locale: self.default_locale,
            read_api_key: self.read_api_key,
            write_api_key: self.write_api_key,
            index_ids,
            temp_index_ids,
            created_at: self.created_at,
            updated_at: **self.updated_at.read("commit").await,
            // Per-collection offset tracking - single unified offset
            offset,
        });

        BufferedFile::create_or_overwrite(self.data_dir.join("collection.json"))
            .context("Cannot create collection.json")?
            .write_json_data(&dump)
            .context("Cannot write collection.json")?;

        // Remove deleted indexes from in memory
        let mut indexes_lock = self.indexes.write("commit").await;
        let (kept, deleted): (Vec<_>, Vec<_>) = std::mem::take(&mut **indexes_lock)
            .into_iter()
            .enumerate()
            .partition(|(i, _)| !index_id_to_remove.iter().any(|(ind, _)| ind == i));
        **indexes_lock = kept.into_iter().map(|(_, index)| index).collect();
        drop(indexes_lock);

        // Remove deleted indexes from disk: here it is safe because we already save the collection dump to fs
        for (_, index) in deleted {
            let dir = indexes_dir.join(index.id().as_str());
            if let Err(err) = std::fs::remove_dir_all(dir) {
                // This happen when the index is never committed
                if err.kind() != ErrorKind::NotFound {
                    error!(error = ?err, "Cannot remove index dir {:?}", index.id());
                }
            }
        }

        // Remove deleted temp indexes from in memory
        let temp_indexes_to_remove: std::collections::HashSet<_> =
            temp_index_id_to_remove.iter().collect();
        if !temp_indexes_to_remove.is_empty() {
            let mut temp_indexes_lock = self.temp_indexes.write("commit_retain").await;
            temp_indexes_lock.retain(|index| !temp_indexes_to_remove.contains(&index.id()));
            drop(temp_indexes_lock);
        }

        // Remove temp deleted indexes from disk because they are promoted to runtime indexes or expired.
        // Here it is safe because we already save the collection dump to fs
        for index_id in temp_index_id_to_remove {
            let dir = temp_indexes_dir.join(index_id.as_str());
            if let Err(err) = std::fs::remove_dir_all(dir) {
                // This happen when the index is never committed
                if err.kind() != ErrorKind::NotFound {
                    error!(error = ?err, "Cannot remove index dir {:?}", index_id);
                }
            }
        }

        self.document_count_estimation
            .store(count, std::sync::atomic::Ordering::Relaxed);

        // Reset pending operations counter after successful commit
        self.pending_operations.store(0, Ordering::Relaxed);

        // Update last commit timestamp
        let mut last_commit_lock = self.last_commit_time.write("commit_complete").unwrap();
        **last_commit_lock = Instant::now();
        drop(last_commit_lock);

        Ok(offset)
    }

    pub async fn clean_up(&self) -> Result<()> {
        // Skip deleted index is important because when we replace an index,
        // with a temp one, the old index is marked as deleted.
        // In this situation, the old index and the new index share the same data dir.
        // Anyway the old index has an old offset, so during the clean up,
        // it will remove data that are still used by the new index.

        let indexes_lock = self.indexes.read("clean_up").await;

        let indexes_dir = self.data_dir.join("indexes");
        for index in indexes_lock.iter() {
            if !index.is_deleted() {
                let dir = indexes_dir.join(index.id().as_str());
                index
                    .clean_up(dir)
                    .await
                    .with_context(|| format!("index id {:?}", index.id()))?;
            }
        }
        drop(indexes_lock);

        let temp_indexes_lock = self.temp_indexes.read("clean_up").await;
        let temp_indexes_dir = self.data_dir.join("temp_indexes");
        for index in temp_indexes_lock.iter() {
            if !index.is_deleted() {
                let dir = temp_indexes_dir.join(index.id().as_str());
                index
                    .clean_up(dir)
                    .await
                    .with_context(|| format!("temp index id {:?}", index.id()))?;
            }
        }
        drop(temp_indexes_lock);

        Ok(())
    }

    #[inline]
    pub fn id(&self) -> CollectionId {
        self.id
    }

    pub fn is_deleted(&self) -> bool {
        self.deleted
    }

    pub fn mark_as_deleted(&mut self) {
        *self.updated_at.get_mut() = Utc::now();
        self.deleted = true;
    }

    pub fn get_document_count_estimation(&self) -> u64 {
        self.document_count_estimation
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    #[inline]
    #[allow(clippy::result_large_err)]
    pub fn check_read_api_key(
        &self,
        api_key: ApiKey,
        master_api_key: Option<ApiKey>,
    ) -> Result<(), ReadError> {
        if api_key == self.read_api_key {
            return Ok(());
        }
        if let Some(write_api_key) = self.write_api_key {
            if write_api_key == api_key {
                return Ok(());
            }
        }
        if let Some(master_api_key) = master_api_key {
            if master_api_key == api_key {
                return Ok(());
            }
        }

        Err(ReadError::Generic(anyhow!("Invalid read api key")))
    }

    #[inline]
    pub fn get_id(&self) -> CollectionId {
        self.id
    }

    pub fn get_hook_storage(&self) -> &OramaAsyncLock<HookReader> {
        &self.hook
    }

    pub async fn update_mcp_description(&self, mcp_description: Option<String>) -> Result<()> {
        let mut mcp_description_lock = self.mcp_description.write("update").await;
        **mcp_description_lock = mcp_description;
        drop(mcp_description_lock);

        Ok(())
    }

    pub async fn nlp_search(
        &self,
        read_side: State<Arc<ReadSide>>,
        read_api_key: ApiKey,
        collection_id: CollectionId,
        search_params: &NLPSearchRequest,
        collection_stats: CollectionStats,
        log_sender: Option<Arc<tokio::sync::broadcast::Sender<(OutputChannel, String)>>>,
    ) -> Result<Vec<QueryMappedSearchResult>> {
        let llm_service = self.context.llm_service.clone();
        let llm_config = search_params.llm_config.clone();
        let query = search_params.query.clone();

        let advanced_autoquery = AdvancedAutoQuery::new(collection_stats, llm_service, llm_config);
        let conversation = vec![InteractionMessage {
            role: Role::User,
            content: query,
        }];

        let search_results = advanced_autoquery
            .run(
                read_side.clone(),
                read_api_key,
                collection_id,
                conversation,
                log_sender,
                SearchAnalyticEventOrigin::NLP,
                search_params.user_id.clone(),
            )
            .await?;

        Ok(search_results)
    }

    pub async fn nlp_search_stream(
        &self,
        read_side: State<Arc<ReadSide>>,
        read_api_key: ApiKey,
        collection_id: CollectionId,
        search_params: &NLPSearchRequest,
        collection_stats: CollectionStats,
        log_sender: Option<Arc<tokio::sync::broadcast::Sender<(OutputChannel, String)>>>,
    ) -> Result<impl tokio_stream::Stream<Item = Result<AdvancedAutoQuerySteps>>, ReadError> {
        let llm_service = self.context.llm_service.clone();
        let llm_config = search_params.llm_config.clone();
        let query = search_params.query.clone();

        let advanced_autoquery = AdvancedAutoQuery::new(collection_stats, llm_service, llm_config);
        let conversation = vec![InteractionMessage {
            role: Role::User,
            content: query,
        }];

        Ok(advanced_autoquery
            .run_stream(
                read_side.clone(),
                read_api_key,
                collection_id,
                conversation,
                log_sender,
                SearchAnalyticEventOrigin::NLP,
                search_params.user_id.clone(),
            )
            .await)
    }

    pub async fn get_all_index_ids(&self) -> Result<Vec<IndexId>> {
        let lock = self.indexes.read("get_all_index_ids").await;
        let all_indexes = lock
            .iter()
            .filter(|index| !index.is_deleted())
            .map(|i| i.id())
            .collect::<Vec<_>>();
        Ok(all_indexes)
    }

    pub async fn get_indexes_lock<'collection, 'search>(
        &'collection self,
        index_ids: &'search [IndexId],
    ) -> Result<ReadIndexesLockGuard<'collection, 'search>> {
        let lock = self.indexes.read("get_indexes_lock").await;

        let unknown_index = index_ids.iter().find(|id| {
            !lock
                .iter()
                .any(|index| !index.is_deleted() && index.id() == **id)
        });
        if let Some(unknown_index) = unknown_index {
            bail!("Unknown index: {unknown_index:?}");
        }

        Ok(ReadIndexesLockGuard::new(lock, index_ids))
    }

    pub async fn get_index_ids(
        &self,
        indexes_from_user: Option<&Vec<IndexId>>,
    ) -> Result<Vec<IndexId>> {
        let all_indexes = self.get_all_index_ids().await?;
        // No indexes from user -> search on all indexes
        let Some(indexes_from_user) = indexes_from_user else {
            return Ok(all_indexes);
        };
        // Empty indexes from user -> search on all indexes
        // It doens't make sense to search on empty indexes list
        if indexes_from_user.is_empty() {
            return Ok(all_indexes);
        }

        let unknown_indexes = indexes_from_user
            .iter()
            .filter(|index| !all_indexes.contains(index))
            .collect::<Vec<_>>();
        if !unknown_indexes.is_empty() {
            bail!("Unknown indexes: {unknown_indexes:?}. Available indexes: {all_indexes:?}")
        }

        let res = all_indexes
            .into_iter()
            .filter(|index| indexes_from_user.contains(index))
            .collect();
        Ok(res)
    }

    pub async fn get_index(&self, index_id: IndexId) -> Option<IndexReadLock<'_>> {
        let indexes_lock = self.indexes.read("get_index").await;
        IndexReadLock::try_new(indexes_lock, index_id)
    }

    pub async fn update(
        &self,
        offset: Offset,
        collection_operation: CollectionWriteOperation,
    ) -> Result<()> {
        // Check if operation already applied to THIS collection
        let current = self.offset.read("update").await;
        if offset <= **current && !current.is_zero() {
            warn!(
                collection_id=?self.id,
                offset=?offset,
                "Already applied to this collection"
            );
            return Ok(());
        }
        drop(current);

        let mut updated_at_lock = self.updated_at.write("update").await;
        **updated_at_lock = Utc::now();
        drop(updated_at_lock);

        match collection_operation {
            CollectionWriteOperation::CreateIndex2 { index_id, locale } => {
                let mut indexes_lock = self.indexes.write("create_index").await;
                let index = Index::new(
                    index_id,
                    self.context.nlp_service.get(locale),
                    self.context.clone(),
                    self.offload_config,
                    EnumStrategy::default(),
                );
                let contains = get_index_in_vector(&indexes_lock, index_id).is_some();
                if contains {
                    warn!("Index {} already exists", index_id);
                    debug_panic!("Index {} already exists", index_id);
                } else {
                    indexes_lock.push(index);
                }
                drop(indexes_lock);
            }
            CollectionWriteOperation::CreateIndex3 {
                index_id,
                locale,
                enum_strategy,
            } => {
                let mut indexes_lock = self.indexes.write("create_index").await;
                let index = Index::new(
                    index_id,
                    self.context.nlp_service.get(locale),
                    self.context.clone(),
                    self.offload_config,
                    enum_strategy,
                );
                let contains = get_index_in_vector(&indexes_lock, index_id).is_some();
                if contains {
                    warn!("Index {} already exists", index_id);
                    debug_panic!("Index {} already exists", index_id);
                } else {
                    indexes_lock.push(index);
                }
                drop(indexes_lock);
            }
            CollectionWriteOperation::CreateTemporaryIndex2 { index_id, locale } => {
                let mut temp_indexes_lock = self.temp_indexes.write("create_temp_index").await;
                let index = Index::new(
                    index_id,
                    self.context.nlp_service.get(locale),
                    self.context.clone(),
                    self.offload_config,
                    EnumStrategy::default(),
                );
                let contains = get_index_in_vector(&temp_indexes_lock, index_id).is_some();
                if contains {
                    warn!("Temp index {} already exists", index_id);
                    debug_panic!("Temp index {} already exists", index_id);
                } else {
                    temp_indexes_lock.push(index);
                }
                drop(temp_indexes_lock);
            }
            CollectionWriteOperation::CreateTemporaryIndex3 {
                index_id,
                locale,
                enum_strategy,
            } => {
                let mut temp_indexes_lock = self.temp_indexes.write("create_temp_index").await;
                let index = Index::new(
                    index_id,
                    self.context.nlp_service.get(locale),
                    self.context.clone(),
                    self.offload_config,
                    enum_strategy,
                );
                let contains = get_index_in_vector(&temp_indexes_lock, index_id).is_some();
                if contains {
                    warn!("Temp index {} already exists", index_id);
                    debug_panic!("Temp index {} already exists", index_id);
                } else {
                    temp_indexes_lock.push(index);
                }
                drop(temp_indexes_lock);
            }
            CollectionWriteOperation::DeleteIndex2 { index_id } => {
                let index = self.get_index_mut(index_id).await;
                if let Some(mut index) = index {
                    index.mark_as_deleted(DeletionReason::UserWanted);
                } else {
                    warn!("Cannot mark index {} as deleted. Ignored.", index_id);
                }
            }
            CollectionWriteOperation::DeleteTempIndex { temp_index_id } => {
                let temp_index = self.get_temp_index_mut(temp_index_id).await;
                if let Some(mut temp_index) = temp_index {
                    temp_index.mark_as_deleted(DeletionReason::ExpiredTempIndex);
                } else {
                    warn!(
                        "Cannot mark temp index {} as deleted. Ignored.",
                        temp_index_id
                    );
                }
            }
            CollectionWriteOperation::IndexWriteOperation(index_id, index_op) => {
                let temp_index = self.get_temp_index_mut(index_id).await;
                if let Some(mut temp_index) = temp_index {
                    temp_index
                        .update(index_op)
                        .with_context(|| format!("Cannot update index {index_id:?}"))?;
                } else {
                    let index = self.get_index_mut(index_id).await;
                    let Some(mut index) = index else {
                        bail!("Index {index_id} not found")
                    };
                    index
                        .update(index_op)
                        .with_context(|| format!("Cannot update index {index_id:?}"))?;
                }
            }
            CollectionWriteOperation::ReplaceIndex {
                runtime_index_id,
                temp_index_id,
                reference,
                reason,
            } => {
                let mut temp_index_lock = self.temp_indexes.write("replace_temp_index").await;
                let mut runtime_index_lock = self.indexes.write("replace_runtime_index").await;

                let runtime_i = get_index_in_vector(&runtime_index_lock, runtime_index_id);
                let temp_i = get_index_in_vector(&temp_index_lock, temp_index_id);

                let Some(runtime_i) = runtime_i else {
                    bail!("Runtime index {runtime_index_id} not found");
                };
                let Some(temp_i) = temp_i else {
                    bail!("Temp index {temp_index_id} not found");
                };

                let old_index = runtime_index_lock
                    .get_mut(runtime_i)
                    // This should not happen, since we already checked that the index exists
                    .unwrap();
                match reason {
                    ReplaceIndexReason::CollectionReindexed => {
                        old_index
                            .mark_as_deleted(DeletionReason::CollectionReindexed { temp_index_id });
                    }
                    ReplaceIndexReason::IndexResynced => {
                        old_index.mark_as_deleted(DeletionReason::IndexResynced { temp_index_id });
                    }
                };

                let mut new_index = temp_index_lock
                    // This should not happen, since we already checked that the index exists
                    .remove(temp_i);

                // Replace the temp index id with the new one
                new_index.promote_to_runtime_index(runtime_index_id);
                runtime_index_lock.push(new_index);

                drop(temp_index_lock);
                drop(runtime_index_lock);

                if let Some(notifier) = self.context.notifier.as_ref() {
                    match notifier
                        .notify_collection_substitution(
                            self.id,
                            runtime_index_id,
                            temp_index_id,
                            reference,
                        )
                        .await
                    {
                        Ok(_) => {
                            info!("Collection {} notified", self.id);
                        }
                        Err(e) => {
                            error!("Error notifying collection {}: {:?}", self.id, e);
                        }
                    }
                }
            }
            CollectionWriteOperation::Hook(op) => {
                let mut lock = self.hook.write("update_hook").await;
                lock.update(op)?;
                drop(lock);
            }
            CollectionWriteOperation::UpdateMcpDescription { mcp_description } => {
                self.update_mcp_description(mcp_description).await?;
            }
            CollectionWriteOperation::PinRule(op) => {
                println!("Applying pin rule operation: {op:?}");
                let mut pin_rules_lock = self.pin_rules_reader.write("update_pin_rule").await;
                pin_rules_lock
                    .update(op)
                    .context("Cannot apply pin rule operation")?;
                drop(pin_rules_lock);
            }
        }

        // Update offset after successful operation
        let mut current = self.offset.write("update").await;
        **current = offset;
        drop(current);

        // Commit collection if threshold reached
        // NB: this commits only this collection
        let pending = self.pending_operations.fetch_add(1, Ordering::SeqCst) + 1;
        if pending >= self.commit_config.operation_threshold {
            info!(
                collection_id=?self.id,
                offset=?offset,
                pending=pending,
                "Collection threshold reached, committing"
            );
            // Force commit since we've reached the per-collection threshold
            self.commit(true).await?;
        }

        Ok(())
    }

    pub async fn stats(&self, _req: CollectionStatsRequest) -> Result<CollectionStats, ReadError> {
        let indexes_lock = self.indexes.read("stats").await;
        let mut indexes_stats = Vec::with_capacity(indexes_lock.len());
        let mut embedding_model: Option<String> = None;

        for i in indexes_lock.iter() {
            if i.is_deleted() {
                continue;
            }

            // This should only happen on the first iteration
            if embedding_model.is_none() {
                if let Some(model) = i.get_model().await {
                    let serializable_model = model.to_string();
                    embedding_model = Some(serializable_model.to_string());
                }
            }

            indexes_stats.push(i.stats(false).await?);
        }
        drop(indexes_lock);

        let temp_indexes_lock = self.temp_indexes.read("stats").await;
        for i in temp_indexes_lock.iter() {
            if i.is_deleted() {
                continue;
            }
            indexes_stats.push(i.stats(true).await?);
        }

        let lock = self.hook.read("stats").await;
        let hooks = lock.list()?;
        let hooks = hooks
            .into_iter()
            .filter_map(|(t, c)| c.map(|_| t))
            .collect();
        drop(lock);

        Ok(CollectionStats {
            id: self.get_id(),
            document_count: indexes_stats
                .iter()
                .map(|i| i.document_count)
                .sum::<usize>(),
            description: self.description.clone(),
            mcp_description: self.mcp_description.read("stats").await.clone(),
            default_locale: self.default_locale,
            embedding_model,
            indexes_stats,
            hooks,
            created_at: self.created_at,
            updated_at: **self.updated_at.read("stats").await,
        })
    }

    pub async fn get_filterable_fields(
        &self,
        with_keys: bool,
    ) -> Result<Vec<FilterableFieldsStats>, ReadError> {
        let mut stats = self.stats(CollectionStatsRequest { with_keys }).await?;

        stats.indexes_stats = stats
            .indexes_stats
            .into_iter()
            .map(|mut index_stats| {
                index_stats.fields_stats.retain(|stat| {
                    !matches!(
                        &stat.stats,
                        IndexFieldStatsType::CommittedString(_)
                            | IndexFieldStatsType::UncommittedString(_)
                            | IndexFieldStatsType::CommittedVector(_)
                            | IndexFieldStatsType::UncommittedVector(_)
                    )
                });
                index_stats
            })
            .collect();

        let mut result: Vec<FilterableFieldsStats> = Vec::new();

        for stat in stats.indexes_stats.iter() {
            let mut final_stats: HashMap<FieldId, FilterableField> = HashMap::new();

            for field in stat.fields_stats.iter() {
                match &field.stats {
                    IndexFieldStatsType::CommittedBoolean(CommittedBoolFieldStats {
                        false_count,
                        true_count,
                    }) => {
                        final_stats.insert(
                            field.field_id,
                            FilterableField::Bool(FilterableFieldBool {
                                field_path: field.field_path.clone(),
                                field_type: "boolean".to_string(),
                                count_true: *true_count,
                                count_false: *false_count,
                                count: *true_count + *false_count,
                            }),
                        );
                    }
                    IndexFieldStatsType::UncommittedBoolean(UncommittedBoolFieldStats {
                        false_count,
                        true_count,
                    }) => {
                        if *false_count > 0 || *true_count > 0 {
                            if let Some(FilterableField::Bool(bool_stats)) =
                                final_stats.get(&field.field_id)
                            {
                                final_stats.insert(
                                    field.field_id,
                                    FilterableField::Bool(FilterableFieldBool {
                                        field_path: field.field_path.clone(),
                                        field_type: "boolean".to_string(),
                                        count_true: bool_stats.count_true + *true_count,
                                        count_false: bool_stats.count_false + *false_count,
                                        count: bool_stats.count + *true_count + *false_count,
                                    }),
                                );
                            }
                        }
                    }
                    IndexFieldStatsType::CommittedGeoPoint(CommittedGeoPointFieldStats {
                        count,
                    }) => {
                        final_stats.insert(
                            field.field_id,
                            FilterableField::GeoPoint(FilterableFieldGeoPoint {
                                field_path: field.field_path.clone(),
                                field_type: "geopoint".to_string(),
                                count: *count,
                            }),
                        );
                    }
                    IndexFieldStatsType::UncommittedGeoPoint(UncommittedGeoPointFieldStats {
                        count,
                    }) => {
                        if *count > 0 {
                            if let Some(FilterableField::GeoPoint(geo_stats)) =
                                final_stats.get(&field.field_id)
                            {
                                final_stats.insert(
                                    field.field_id,
                                    FilterableField::GeoPoint(FilterableFieldGeoPoint {
                                        field_path: field.field_path.clone(),
                                        field_type: "geopoint".to_string(),
                                        count: geo_stats.count + *count,
                                    }),
                                );
                            }
                        }
                    }
                    IndexFieldStatsType::CommittedDate(CommittedDateFieldStats { min, max }) => {
                        final_stats.insert(
                            field.field_id,
                            FilterableField::Date(FilterableFieldDate {
                                field_path: field.field_path.clone(),
                                field_type: "date".to_string(),
                                min: min.clone(),
                                max: max.clone(),
                            }),
                        );
                    }
                    IndexFieldStatsType::UncommittedDate(UncommittedDateFieldStats {
                        min,
                        max,
                        ..
                    }) => {
                        if let Some(FilterableField::Date(date_stats)) =
                            final_stats.get(&field.field_id)
                        {
                            let new_min = match (date_stats.min.clone(), min) {
                                (Some(existing_min), Some(new_min)) => {
                                    Some(OramaDate::try_from_i64(existing_min.as_i64().min(new_min.as_i64())).expect("Unable to conver i64 back to date format. This is a bug. Please report at https://github.com/oramasearch/oramacore"))
                                }
                                (Some(existing_min), None) => Some(existing_min),
                                (None, Some(new_min)) => Some(new_min.clone()),
                                (None, None) => None,
                            };
                            let new_max = match (date_stats.max.clone(), max) {
                                (Some(existing_max), Some(new_max)) => {
                                    Some(OramaDate::try_from_i64(existing_max.as_i64().max(new_max.as_i64())).expect("Unable to conver i64 back to date format. This is a bug. Please report at https://github.com/oramasearch/oramacore"))
                                }
                                (Some(existing_max), None) => Some(existing_max),
                                (None, Some(new_max)) => Some(new_max.clone()),
                                (None, None) => None,
                            };
                            final_stats.insert(
                                field.field_id,
                                FilterableField::Date(FilterableFieldDate {
                                    field_path: field.field_path.clone(),
                                    field_type: "date".to_string(),
                                    min: new_min,
                                    max: new_max,
                                }),
                            );
                        }
                    }
                    IndexFieldStatsType::CommittedNumber(CommittedNumberFieldStats {
                        min,
                        max,
                    }) => {
                        final_stats.insert(
                            field.field_id,
                            FilterableField::Number(FilterableFieldNumber {
                                field_path: field.field_path.clone(),
                                field_type: "number".to_string(),
                                min: match min {
                                    Number::I32(i) => *i as f64,
                                    Number::F32(f) => *f as f64,
                                },
                                max: match max {
                                    Number::I32(i) => *i as f64,
                                    Number::F32(f) => *f as f64,
                                },
                            }),
                        );
                    }
                    IndexFieldStatsType::UncommittedNumber(UncommittedNumberFieldStats {
                        min,
                        max,
                        ..
                    }) => {
                        if let Some(FilterableField::Number(number_stats)) =
                            final_stats.get(&field.field_id)
                        {
                            let min_as_f64 = match min {
                                Number::I32(i) => *i as f64,
                                Number::F32(f) => *f as f64,
                            };
                            let max_as_f64 = match max {
                                Number::I32(i) => *i as f64,
                                Number::F32(f) => *f as f64,
                            };

                            let new_min = min_as_f64.min(number_stats.min);
                            let new_max = max_as_f64.max(number_stats.max);

                            final_stats.insert(
                                field.field_id,
                                FilterableField::Number(FilterableFieldNumber {
                                    field_path: field.field_path.clone(),
                                    field_type: "number".to_string(),
                                    min: new_min,
                                    max: new_max,
                                }),
                            );
                        }
                    }
                    IndexFieldStatsType::CommittedStringFilter(
                        CommittedStringFilterFieldStats { key_count, .. },
                    ) => {
                        final_stats.insert(
                            field.field_id,
                            FilterableField::String(FilterableFieldString {
                                field_path: field.field_path.clone(),
                                field_type: "enum".to_string(),
                                count: *key_count,
                            }),
                        );
                    }
                    IndexFieldStatsType::UncommittedStringFilter(
                        UncommittedStringFilterFieldStats { key_count, .. },
                    ) => {
                        if *key_count > 0 {
                            if let Some(FilterableField::String(string_stats)) =
                                final_stats.get(&field.field_id)
                            {
                                final_stats.insert(
                                    field.field_id,
                                    FilterableField::String(FilterableFieldString {
                                        field_path: field.field_path.clone(),
                                        field_type: "enum".to_string(),
                                        count: string_stats.count + *key_count,
                                    }),
                                );
                            }
                        }
                    }
                    _ => {}
                }
            }

            result.push(FilterableFieldsStats {
                index_id: stat.id,
                fields: final_stats.into_values().collect(),
            });
        }

        Ok(result)
    }

    async fn get_index_mut(&self, index_id: IndexId) -> Option<IndexWriteLock<'_>> {
        let indexes_lock = self.indexes.write("get_index_mut").await;
        IndexWriteLock::try_new(indexes_lock, index_id)
    }

    async fn get_temp_index_mut(&self, index_id: IndexId) -> Option<IndexWriteLock<'_>> {
        let indexes_lock = self.temp_indexes.write("get_temp_index_mut").await;
        IndexWriteLock::try_new(indexes_lock, index_id)
    }

    pub async fn get_pin_rules_reader(
        &self,
        reason: &'static str,
    ) -> OramaAsyncLockReadGuard<'_, PinRulesReader<DocumentId>> {
        self.pin_rules_reader.read(reason).await
    }

    fn should_commit(&self, force: bool) -> bool {
        // We commit only if:
        // 1. Forced (from per-collection threshold or shutdown)
        // 2. Collection operation threshold reached (configurable via collection_commit.operation_threshold)
        // 3. Time threshold reached AND we have pending operations
        // 4. No pending operations (to update last commit time)

        if force {
            return true;
        }

        let pending = self.pending_operations.load(Ordering::SeqCst);
        // Fast path: operation threshold reached
        if pending >= self.commit_config.operation_threshold {
            info!(
                collection_id=?self.id,
                pending=pending,
                "Collection operation threshold reached, committing"
            );
            true
        }
        // Check time threshold if we have pending operations
        else {
            let last_commit_lock = self.last_commit_time.read("commit_check").unwrap();
            let elapsed = last_commit_lock.elapsed();
            let should_commit_by_time = elapsed >= self.commit_config.time_threshold;
            drop(last_commit_lock);

            if pending > 0 {
                should_commit_by_time
            } else {
                // In this case we commit even if no pending opertations
                // So, we will update the last commit time even if no operations
                true
            }
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Committed {
    pub epoch: u64,
}

use derive_more::From;
#[derive(Serialize, Debug, From)]
#[serde(tag = "type")]
pub enum IndexFieldStatsType {
    #[serde(rename = "uncommitted_bool")]
    UncommittedBoolean(UncommittedBoolFieldStats),
    #[serde(rename = "committed_bool")]
    CommittedBoolean(CommittedBoolFieldStats),

    #[serde(rename = "uncommitted_number")]
    UncommittedNumber(UncommittedNumberFieldStats),
    #[serde(rename = "committed_number")]
    CommittedNumber(CommittedNumberFieldStats),

    #[serde(rename = "uncommitted_date")]
    UncommittedDate(UncommittedDateFieldStats),
    #[serde(rename = "committed_date")]
    CommittedDate(CommittedDateFieldStats),

    #[serde(rename = "uncommitted_geopoint")]
    UncommittedGeoPoint(UncommittedGeoPointFieldStats),
    #[serde(rename = "committed_geopoint")]
    CommittedGeoPoint(CommittedGeoPointFieldStats),

    #[serde(rename = "uncommitted_string_filter")]
    UncommittedStringFilter(UncommittedStringFilterFieldStats),
    #[serde(rename = "committed_string_filter")]
    CommittedStringFilter(CommittedStringFilterFieldStats),

    #[serde(rename = "uncommitted_string")]
    UncommittedString(UncommittedStringFieldStats),
    #[serde(rename = "committed_string")]
    CommittedString(CommittedStringFieldStats),

    #[serde(rename = "uncommitted_vector")]
    UncommittedVector(UncommittedVectorFieldStats),
    #[serde(rename = "committed_vector")]
    CommittedVector(CommittedVectorFieldStats),
}

#[derive(Serialize, Debug)]
pub struct IndexFieldStats {
    pub field_id: FieldId,
    pub field_path: String,
    #[serde(flatten)]
    pub stats: IndexFieldStatsType,
}

#[derive(Serialize, Debug)]
pub struct CollectionStats {
    pub id: CollectionId,
    pub document_count: usize,
    pub description: Option<String>,
    pub mcp_description: Option<String>,
    pub default_locale: Locale,
    pub embedding_model: Option<String>,
    pub indexes_stats: Vec<IndexStats>,
    pub hooks: Vec<HookType>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

pub struct IndexReadLock<'guard> {
    lock: OramaAsyncLockReadGuard<'guard, Vec<Index>>,
    index: usize,
}

impl<'guard> IndexReadLock<'guard> {
    pub fn try_new(
        indexes_lock: OramaAsyncLockReadGuard<'guard, Vec<Index>>,
        id: IndexId,
    ) -> Option<Self> {
        get_index_in_vector(&indexes_lock, id).map(|index| Self {
            lock: indexes_lock,
            index,
        })
    }
}

impl Deref for IndexReadLock<'_> {
    type Target = Index;

    fn deref(&self) -> &Self::Target {
        // Safety: the index map contains the id because we checked it before
        // no one can remove the collection from the map because we hold a read lock
        &self.lock[self.index]
    }
}

pub struct IndexWriteLock<'guard> {
    lock: OramaAsyncLockWriteGuard<'guard, Vec<Index>>,
    index: usize,
}

impl<'guard> IndexWriteLock<'guard> {
    pub fn try_new(
        indexes_lock: OramaAsyncLockWriteGuard<'guard, Vec<Index>>,
        id: IndexId,
    ) -> Option<Self> {
        get_index_in_vector(&indexes_lock, id).map(|index| Self {
            lock: indexes_lock,
            index,
        })
    }
}
impl Deref for IndexWriteLock<'_> {
    type Target = Index;

    fn deref(&self) -> &Self::Target {
        // Safety: the index map contains the id because we checked it before
        // no one can remove the collection from the map because we hold a read lock
        &self.lock[self.index]
    }
}
impl DerefMut for IndexWriteLock<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // Safety: the index map contains the id because we checked it before
        // no one can remove the collection from the map because we hold a read lock
        &mut self.lock[self.index]
    }
}

fn get_index_in_vector(vec: &[Index], wanted: IndexId) -> Option<usize> {
    // fast path
    for (i, index) in vec.iter().enumerate() {
        if index.is_deleted() {
            continue;
        }
        if index.id() == wanted {
            return Some(i);
        }
    }

    // check in alias too
    for (i, index) in vec.iter().enumerate() {
        if index.is_deleted() {
            continue;
        }
        if index.aliases().contains(&wanted) {
            return Some(i);
        }
    }

    None
}

#[derive(Debug, Serialize, Deserialize)]
struct DumpV1 {
    id: CollectionId,
    description: Option<String>,
    mcp_description: Option<String>,
    default_locale: Locale,
    read_api_key: ApiKey,
    write_api_key: Option<ApiKey>,
    index_ids: Vec<IndexId>,
    temp_index_ids: Vec<IndexId>,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
struct DumpV2 {
    id: CollectionId,
    description: Option<String>,
    mcp_description: Option<String>,
    default_locale: Locale,
    read_api_key: ApiKey,
    write_api_key: Option<ApiKey>,
    index_ids: Vec<IndexId>,
    temp_index_ids: Vec<IndexId>,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
    // Per-collection offset tracking - single unified offset
    offset: Offset,
}

#[derive(Debug, Serialize, Deserialize)]
enum Dump {
    V1(DumpV1),
    V2(DumpV2),
}

pub struct ReadIndexesLockGuard<'collection, 'search> {
    lock: OramaAsyncLockReadGuard<'collection, Vec<Index>>,
    index_ids: &'search [IndexId],
}
impl<'collection, 'search> ReadIndexesLockGuard<'collection, 'search> {
    pub fn new(
        lock: OramaAsyncLockReadGuard<'collection, Vec<Index>>,
        index_ids: &'search [IndexId],
    ) -> Self {
        Self { lock, index_ids }
    }

    pub fn len(&self) -> usize {
        self.index_ids.len()
    }

    pub fn get_index(&self, index_id: IndexId) -> Option<&Index> {
        if !self.index_ids.contains(&index_id) {
            debug_panic!(
                "Index id {} not in the requested index ids {:?}",
                index_id,
                self.index_ids
            );
            warn!(
                "Index id {} not in the requested index ids {:?}",
                index_id, self.index_ids
            );
            return None;
        }

        self.lock
            .iter()
            .find(|index| !index.is_deleted() && index.id() == index_id)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Index> {
        self.lock
            .iter()
            .filter(|index| !index.is_deleted() && self.index_ids.contains(&index.id()))
    }
}
