use std::{
    collections::HashMap,
    io::ErrorKind,
    ops::{Deref, DerefMut},
    path::PathBuf,
    sync::{atomic::AtomicU64, Arc},
};

use anyhow::{anyhow, bail, Context, Result};

use chrono::{DateTime, Utc};
use debug_panic::debug_panic;
use serde::{Deserialize, Serialize};
use tokio::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use tracing::{error, info, warn};

use crate::{
    ai::{llms::LLMService, AIService},
    collection_manager::sides::{CollectionWriteOperation, Offset, ReplaceIndexReason},
    file_utils::BufferedFile,
    nlp::{locales::Locale, NLPService},
    types::{ApiKey, CollectionId, DocumentId, FacetResult, FieldId, IndexId, SearchParams},
};

use super::{
    index::{Index, IndexStats},
    notify::Notifier,
    CommittedBoolFieldStats, CommittedNumberFieldStats, CommittedStringFieldStats,
    CommittedStringFilterFieldStats, CommittedVectorFieldStats, DeletionReason,
    UncommittedBoolFieldStats, UncommittedNumberFieldStats, UncommittedStringFieldStats,
    UncommittedStringFilterFieldStats, UncommittedVectorFieldStats,
};

pub struct CollectionReader {
    id: CollectionId,
    description: Option<String>,
    default_locale: Locale,
    deleted: bool,

    read_api_key: ApiKey,
    ai_service: Arc<AIService>,
    nlp_service: Arc<NLPService>,
    llm_service: Arc<LLMService>,

    indexes: RwLock<Vec<Index>>,

    temp_indexes: RwLock<Vec<Index>>,

    notifier: Option<Arc<Notifier>>,

    created_at: DateTime<Utc>,
    updated_at: RwLock<DateTime<Utc>>,

    document_count_estimation: AtomicU64,
}

impl CollectionReader {
    pub fn empty(
        id: CollectionId,
        description: Option<String>,
        default_locale: Locale,
        read_api_key: ApiKey,
        ai_service: Arc<AIService>,
        nlp_service: Arc<NLPService>,
        llm_service: Arc<LLMService>,
        notifier: Option<Arc<Notifier>>,
    ) -> Self {
        Self {
            id,
            description,
            default_locale,
            deleted: false,

            read_api_key,
            ai_service,
            nlp_service,
            llm_service,

            indexes: Default::default(),
            temp_indexes: Default::default(),
            notifier,

            created_at: Utc::now(),
            updated_at: RwLock::new(Utc::now()),

            document_count_estimation: AtomicU64::new(0),
        }
    }

    pub fn try_load(
        ai_service: Arc<AIService>,
        nlp_service: Arc<NLPService>,
        llm_service: Arc<LLMService>,
        notifier: Option<Arc<Notifier>>,
        data_dir: PathBuf,
    ) -> Result<Self> {
        let dump: Dump = BufferedFile::open(data_dir.join("collection.json"))
            .context("Cannot open collection.json")?
            .read_json_data()
            .context("Cannot read collection.json")?;
        let Dump::V1(dump) = dump;

        let mut indexes: Vec<Index> = Vec::with_capacity(dump.index_ids.len());
        for index_id in dump.index_ids {
            let index = Index::try_load(
                index_id,
                data_dir.join("indexes").join(index_id.as_str()),
                nlp_service.clone(),
                llm_service.clone(),
                ai_service.clone(),
            )?;
            indexes.push(index);
        }

        let mut temp_indexes: Vec<Index> = Vec::with_capacity(dump.temp_index_ids.len());
        for index_id in dump.temp_index_ids {
            let index = Index::try_load(
                index_id,
                data_dir.join("temp_indexes").join(index_id.as_str()),
                nlp_service.clone(),
                llm_service.clone(),
                ai_service.clone(),
            )?;
            temp_indexes.push(index);
        }

        let document_count_estimation = indexes.iter().map(|i| i.document_count()).sum::<u64>();

        let s = Self {
            id: dump.id,
            description: dump.description,
            default_locale: dump.default_locale,
            deleted: false,

            read_api_key: dump.read_api_key,
            ai_service,
            nlp_service,
            llm_service,

            indexes: RwLock::new(indexes),
            temp_indexes: RwLock::new(temp_indexes),
            notifier,

            created_at: dump.created_at,
            updated_at: RwLock::new(dump.updated_at),

            document_count_estimation: AtomicU64::new(document_count_estimation),
        };

        Ok(s)
    }

    pub async fn commit(&self, data_dir: PathBuf, offset: Offset) -> Result<()> {
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
        // NB: the document from the document_storage are removed
        //     because the writer sends an appropriate command for that

        let indexes_lock = self.indexes.read().await;
        let mut index_ids = Vec::with_capacity(indexes_lock.len());

        let mut index_id_to_remove = vec![];
        let mut temp_index_id_to_remove = vec![];

        let mut count = 0;
        let indexes_dir = data_dir.join("indexes");
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
                None => {}
            }
            count += index.document_count();
            index_ids.push(index.id());
            let dir = indexes_dir.join(index.id().as_str());
            index.commit(dir, offset).await?;
        }
        drop(indexes_lock);

        let temp_indexes_lock = self.temp_indexes.read().await;
        let mut temp_index_ids = Vec::with_capacity(temp_indexes_lock.len());
        let temp_indexes_dir = data_dir.join("temp_indexes");
        for index in temp_indexes_lock.iter() {
            if index.is_deleted() {
                debug_panic!("It is not possible to delete a temp index (yet)");
                continue;
            }
            temp_index_ids.push(index.id());
            let dir = temp_indexes_dir.join(index.id().as_str());
            index.commit(dir, offset).await?;
        }
        drop(temp_indexes_lock);

        let dump = Dump::V1(DumpV1 {
            id: self.id,
            description: self.description.clone(),
            default_locale: self.default_locale,
            read_api_key: self.read_api_key,
            index_ids,
            temp_index_ids,
            created_at: self.created_at,
            updated_at: *self.updated_at.read().await,
        });

        BufferedFile::create_or_overwrite(data_dir.join("collection.json"))
            .context("Cannot create collection.json")?
            .write_json_data(&dump)
            .context("Cannot write collection.json")?;

        // Remove deleted indexes from in memory
        let mut indexes_lock = self.indexes.write().await;
        let (kept, deleted): (Vec<_>, Vec<_>) = std::mem::take(&mut *indexes_lock)
            .into_iter()
            .enumerate()
            .partition(|(i, _)| !index_id_to_remove.iter().any(|(ind, _)| ind == i));
        *indexes_lock = kept.into_iter().map(|(_, index)| index).collect();
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

        // Remove temp deleted indexes from disk because they are promoted to runtime indexes.
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

    #[inline]
    pub fn check_read_api_key(&self, api_key: ApiKey) -> Result<()> {
        if api_key != self.read_api_key {
            return Err(anyhow!("Invalid read api key"));
        }
        Ok(())
    }

    #[inline]
    pub fn get_id(&self) -> CollectionId {
        self.id
    }

    pub async fn search(
        &self,
        search_params: &SearchParams,
    ) -> Result<HashMap<DocumentId, f32>, anyhow::Error> {
        let document_count_estimation = self
            .document_count_estimation
            .load(std::sync::atomic::Ordering::Relaxed);

        // Let's suppose the number of matching document is 1/3 of the total number of documents
        // This is a very rough estimation, but it is better than nothing
        // This allows us to avoid reallocation of the result map
        let mut result: HashMap<DocumentId, f32> =
            HashMap::with_capacity((document_count_estimation / 3) as usize);
        let indexes_lock = self.indexes.read().await;

        let indexes_to_search_on =
            calculate_index_to_search_on(&indexes_lock, search_params.indexes.as_ref())?;

        for index in indexes_lock.iter() {
            if index.is_deleted() {
                continue;
            }
            if !indexes_to_search_on.contains(&index.id()) {
                continue;
            }
            index.search(search_params, &mut result).await?;
        }

        if result.is_empty() && !search_params.where_filter.is_empty() {
            // We want to return an error if there's some filter on unknown field
            // It is checked in this way because:
            // - we don't want to affect the performance of the search
            // - we want to return a meaningful error message
            let fields_in_filter = search_params.where_filter.get_all_keys();

            // We don't handle the case when the field type is different in different indexes
            // We should dedicate a message error for that
            // TODO: do it

            for field_in_filter in fields_in_filter {
                if indexes_lock
                    .iter()
                    .all(|index| !index.has_field(&field_in_filter))
                {
                    return Err(anyhow!(
                        "Cannot filter by \"{}\": unknown field",
                        field_in_filter
                    ));
                }
            }
        }

        Ok(result)
    }

    pub async fn calculate_facets(
        &self,
        token_scores: &HashMap<DocumentId, f32>,
        search_params: &SearchParams,
    ) -> Result<HashMap<String, FacetResult>> {
        let mut result: HashMap<String, FacetResult> = HashMap::new();
        let indexes_lock = self.indexes.read().await;

        let indexes_to_search_on =
            calculate_index_to_search_on(&indexes_lock, search_params.indexes.as_ref())?;

        for index in indexes_lock.iter() {
            if index.is_deleted() {
                continue;
            }
            if !indexes_to_search_on.contains(&index.id()) {
                continue;
            }

            index
                .calculate_facets(token_scores, &search_params.facets, &mut result)
                .await?;
        }

        Ok(result)
    }

    pub async fn get_index(&self, index_id: IndexId) -> Option<IndexReadLock<'_>> {
        let indexes_lock = self.indexes.read().await;
        IndexReadLock::try_new(indexes_lock, index_id)
    }

    pub async fn update(&self, collection_operation: CollectionWriteOperation) -> Result<()> {
        let mut updated_at_lock = self.updated_at.write().await;
        *updated_at_lock = Utc::now();
        drop(updated_at_lock);

        match collection_operation {
            CollectionWriteOperation::CreateIndex2 { index_id, locale } => {
                let mut indexes_lock = self.indexes.write().await;
                let index = Index::new(
                    index_id,
                    self.nlp_service.get(locale),
                    self.llm_service.clone(),
                    self.ai_service.clone(),
                );
                let contains = get_index_in_vector(&indexes_lock, index_id).is_some();
                if contains {
                    warn!("Index {} already exists", index_id);
                    debug_panic!("Index {} already exists", index_id);
                }
                indexes_lock.push(index);
                drop(indexes_lock);
            }
            CollectionWriteOperation::CreateTemporaryIndex2 { index_id, locale } => {
                let mut temp_indexes_lock = self.temp_indexes.write().await;
                let index = Index::new(
                    index_id,
                    self.nlp_service.get(locale),
                    self.llm_service.clone(),
                    self.ai_service.clone(),
                );
                let contains = get_index_in_vector(&temp_indexes_lock, index_id).is_some();
                if contains {
                    warn!("Temp index {} already exists", index_id);
                    debug_panic!("Temp index {} already exists", index_id);
                }
                temp_indexes_lock.push(index);
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
            CollectionWriteOperation::IndexWriteOperation(index_id, index_op) => {
                let temp_index = self.get_temp_index_mut(index_id).await;
                if let Some(mut temp_index) = temp_index {
                    temp_index
                        .update(index_op)
                        .with_context(|| format!("Cannot update index {:?}", index_id))?;
                } else {
                    let index = self.get_index_mut(index_id).await;
                    let Some(mut index) = index else {
                        bail!("Index {} not found", index_id)
                    };
                    index
                        .update(index_op)
                        .with_context(|| format!("Cannot update index {:?}", index_id))?;
                }
            }
            CollectionWriteOperation::ReplaceIndex {
                runtime_index_id,
                temp_index_id,
                reference,
                reason,
            } => {
                let mut temp_index_lock = self.temp_indexes.write().await;
                let mut runtime_index_lock = self.indexes.write().await;

                let runtime_i = get_index_in_vector(&runtime_index_lock, runtime_index_id);
                let temp_i = get_index_in_vector(&temp_index_lock, temp_index_id);

                let Some(runtime_i) = runtime_i else {
                    bail!("Runtime index {} not found", runtime_index_id);
                };
                let Some(temp_i) = temp_i else {
                    bail!("Temp index {} not found", temp_index_id);
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

                if let Some(notifier) = self.notifier.as_ref() {
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
        }

        Ok(())
    }

    pub async fn stats(&self) -> Result<CollectionStats> {
        let indexes_lock = self.indexes.read().await;
        let mut indexes_stats = Vec::with_capacity(indexes_lock.len());
        for i in indexes_lock.iter() {
            if i.is_deleted() {
                continue;
            }
            indexes_stats.push(i.stats(false).await?);
        }
        drop(indexes_lock);

        let temp_indexes_lock = self.temp_indexes.read().await;
        for i in temp_indexes_lock.iter() {
            if i.is_deleted() {
                continue;
            }
            indexes_stats.push(i.stats(true).await?);
        }

        Ok(CollectionStats {
            id: self.get_id(),
            document_count: indexes_stats
                .iter()
                .map(|i| i.document_count)
                .sum::<usize>(),
            description: self.description.clone(),
            default_locale: self.default_locale,
            indexes_stats,
            created_at: self.created_at,
            updated_at: *self.updated_at.read().await,
        })
    }

    async fn get_index_mut(&self, index_id: IndexId) -> Option<IndexWriteLock<'_>> {
        let indexes_lock = self.indexes.write().await;
        IndexWriteLock::try_new(indexes_lock, index_id)
    }

    async fn get_temp_index_mut(&self, index_id: IndexId) -> Option<IndexWriteLock<'_>> {
        let indexes_lock = self.temp_indexes.write().await;
        IndexWriteLock::try_new(indexes_lock, index_id)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Committed {
    pub epoch: u64,
}

#[derive(Serialize, Debug)]
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
    pub default_locale: Locale,
    pub indexes_stats: Vec<IndexStats>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

pub struct IndexReadLock<'guard> {
    lock: RwLockReadGuard<'guard, Vec<Index>>,
    index: usize,
}

impl<'guard> IndexReadLock<'guard> {
    pub fn try_new(indexes_lock: RwLockReadGuard<'guard, Vec<Index>>, id: IndexId) -> Option<Self> {
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
    lock: RwLockWriteGuard<'guard, Vec<Index>>,
    index: usize,
}

impl<'guard> IndexWriteLock<'guard> {
    pub fn try_new(
        indexes_lock: RwLockWriteGuard<'guard, Vec<Index>>,
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

fn get_index_in_vector(vec: &Vec<Index>, wanted: IndexId) -> Option<usize> {
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

fn calculate_index_to_search_on(
    indexes: &[Index],
    indexes_from_user: Option<&Vec<IndexId>>,
) -> Result<Vec<IndexId>> {
    let all_indexes = indexes.iter().map(|i| i.id()).collect::<Vec<_>>();
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
        bail!(
            "Unknown indexes: {:?}. Available indexes: {:?}",
            unknown_indexes,
            all_indexes
        )
    }

    let res = all_indexes
        .into_iter()
        .filter(|index| indexes_from_user.contains(index))
        .collect();
    Ok(res)
}

#[derive(Debug, Serialize, Deserialize)]
struct DumpV1 {
    id: CollectionId,
    description: Option<String>,
    default_locale: Locale,
    read_api_key: ApiKey,
    index_ids: Vec<IndexId>,
    temp_index_ids: Vec<IndexId>,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
}
#[derive(Debug, Serialize, Deserialize)]
enum Dump {
    V1(DumpV1),
}
