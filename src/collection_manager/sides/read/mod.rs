mod collection;
mod collections;
mod document_storage;

use duration_str::deserialize_duration;
use std::time::Duration;
use std::{collections::HashMap, path::PathBuf, sync::Arc};
use tokio::time::{Instant, MissedTickBehavior};

use anyhow::{Context, Result};
use collections::CollectionsReader;
use document_storage::{DocumentStorage, DocumentStorageConfig};
use ordered_float::NotNan;
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, RwLock};
use tracing::{error, info, trace, warn};

use crate::file_utils::BufferedFile;
use crate::{
    ai::AIService,
    capped_heap::CappedHeap,
    collection_manager::dto::{ApiKey, SearchParams, SearchResult, SearchResultHit, TokenScore},
    metrics::{
        CollectionAddedLabels, CollectionOperationLabels, COLLECTION_ADDED_COUNTER,
        COLLECTION_OPERATION_COUNTER,
    },
    nlp::NLPService,
    types::{CollectionId, DocumentId},
};

use super::{
    CollectionWriteOperation, InputSideChannelType, Offset, OperationReceiver,
    OperationReceiverCreator, WriteOperation,
};

#[derive(Deserialize, Clone)]
pub struct ReadSideConfig {
    pub input: InputSideChannelType,
    pub config: IndexesConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct IndexesConfig {
    pub data_dir: PathBuf,
    #[serde(default = "default_insert_batch_commit_size")]
    pub insert_batch_commit_size: u64,
    #[serde(deserialize_with = "deserialize_duration")]
    pub commit_interval: Duration,
}

pub struct ReadSide {
    collections: CollectionsReader,
    document_storage: DocumentStorage,
    operation_counter: RwLock<u64>,
    insert_batch_commit_size: u64,
    data_dir: PathBuf,
    live_offset: RwLock<Offset>,
    // This offset will update everytime a change is made to the read side.
    commit_insert_mutex: Mutex<Offset>,
}

impl ReadSide {
    pub async fn try_load(
        operation_receiver_creator: OperationReceiverCreator,
        ai_service: Arc<AIService>,
        nlp_service: Arc<NLPService>,
        config: ReadSideConfig,
    ) -> Result<Arc<Self>> {
        let mut document_storage = DocumentStorage::try_new(DocumentStorageConfig {
            data_dir: config.config.data_dir.join("docs"),
        })
        .context("Cannot create document storage")?;

        let insert_batch_commit_size = config.config.insert_batch_commit_size;
        let commit_interval = config.config.commit_interval;
        let data_dir = config.config.data_dir.clone();

        let collections_reader =
            CollectionsReader::try_load(ai_service, nlp_service, config.config)
                .await
                .context("Cannot load collections")?;
        document_storage
            .load()
            .context("Cannot load document storage")?;

        let read_info: Result<ReadInfo> = BufferedFile::open(data_dir.join("read.info"))
            .and_then(|f| f.read_json_data())
            .context("Cannot read offset file");
        let last_offset = match read_info {
            Ok(ReadInfo::V1(info)) => info.offset,
            Err(_) => {
                warn!("Cannot read 'read.info' file. Starting from 0");
                Offset(0)
            }
        };
        info!(offset=?last_offset, "Starting read side");

        let read_side = ReadSide {
            collections: collections_reader,
            document_storage,
            operation_counter: Default::default(),
            insert_batch_commit_size,
            data_dir,
            live_offset: RwLock::new(last_offset),
            commit_insert_mutex: Mutex::new(last_offset),
        };

        let operation_receiver = operation_receiver_creator.create(last_offset).await?;

        let read_side = Arc::new(read_side);

        start_commit_loop(read_side.clone(), commit_interval);
        start_receive_operations(read_side.clone(), operation_receiver);

        Ok(read_side)
    }

    pub async fn commit(&self) -> Result<()> {
        // We stop insertion operations while we are committing
        let commit_insert_mutex_lock = self.commit_insert_mutex.lock().await;

        self.collections.commit().await?;
        self.document_storage
            .commit()
            .await
            .context("Cannot commit document storage")?;

        let offset: Offset = *(self.live_offset.read().await);
        BufferedFile::create_or_overwrite(self.data_dir.join("read.info"))
            .context("Cannot create read.info file")?
            .write_json_data(&ReadInfo::V1(ReadInfoV1 { offset }))
            .context("Cannot write read.info file")?;

        drop(commit_insert_mutex_lock);

        Ok(())
    }

    pub async fn search(
        &self,
        read_api_key: ApiKey,
        collection_id: CollectionId,
        mut search_params: SearchParams,
    ) -> Result<SearchResult> {
        let facets = std::mem::take(&mut search_params.facets);
        let limit = search_params.limit;

        let collection = self
            .collections
            .get_collection(collection_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("Collection not found"))?;
        collection.check_read_api_key(read_api_key)?;

        let token_scores = collection.search(search_params).await?;

        let facets = collection.calculate_facets(&token_scores, facets).await?;

        let count = token_scores.len();

        let top_results: Vec<TokenScore> = top_n(token_scores, limit.0);

        trace!("Top results: {:?}", top_results);
        let docs = self
            .document_storage
            .get_documents_by_ids(top_results.iter().map(|m| m.document_id).collect())
            .await?;

        trace!("Calculates hits");
        let hits: Vec<_> = top_results
            .into_iter()
            .zip(docs)
            .map(|(token_score, document)| {
                let id = document
                    .as_ref()
                    .and_then(|d| d.id.clone())
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

    pub async fn update(&self, op: (Offset, WriteOperation)) -> Result<()> {
        trace!(offset=?op.0, "Updating read side");

        // We stop commit operations while we are updating
        let commit_insert_mutex_lock = self.commit_insert_mutex.lock().await;

        let (offset, op) = op;

        let mut live_offset = self.live_offset.write().await;
        *live_offset = offset;
        drop(live_offset);

        // Already applied. We can skip this operation.
        if offset <= *commit_insert_mutex_lock && !commit_insert_mutex_lock.is_zero() {
            warn!(offset=?offset, "Operation already applied. Skipping");
            return Ok(());
        }

        match op {
            WriteOperation::CreateCollection { id, read_api_key } => {
                COLLECTION_ADDED_COUNTER
                    .create(CollectionAddedLabels {
                        collection: id.0.clone(),
                    })
                    .increment_by_one();
                self.collections
                    .create_collection(offset, id, read_api_key)
                    .await?;
            }
            WriteOperation::Collection(collection_id, collection_operation) => {
                COLLECTION_OPERATION_COUNTER
                    .create(CollectionOperationLabels {
                        collection: collection_id.0.clone(),
                    })
                    .increment_by_one();

                let collection = self
                    .collections
                    .get_collection(collection_id)
                    .await
                    .ok_or_else(|| anyhow::anyhow!("Collection not found"))?;

                if let CollectionWriteOperation::DeleteDocuments { doc_ids } = &collection_operation
                {
                    for doc_id in doc_ids {
                        self.document_storage.delete_document(doc_id).await?;
                    }
                }

                if let CollectionWriteOperation::InsertDocument { doc_id, doc } =
                    collection_operation
                {
                    trace!(?doc_id, "Inserting document");
                    collection.increment_document_count();
                    self.document_storage.add_document(doc_id, doc.0).await?;
                    trace!(?doc_id, "Document inserted");
                } else {
                    collection.update(offset, collection_operation).await?;
                }
            }
        }

        let mut lock = self.operation_counter.write().await;
        *lock += 1;
        let should_commit = if *lock >= self.insert_batch_commit_size {
            *lock = 0;
            true
        } else {
            false
        };
        drop(lock);

        drop(commit_insert_mutex_lock);

        if should_commit {
            info!(insert_batch_commit_size=?self.insert_batch_commit_size, "insert_batch_commit_size reached, committing");
            self.commit().await?;
        } else {
            trace!(insert_batch_commit_size=?self.insert_batch_commit_size, "insert_batch_commit_size not reached, not committing");
        }

        trace!("Updating done");

        Ok(())
    }

    // This is wrong. We should not expose the ai service to the read side.
    // TODO: Remove this method.
    pub fn get_ai_service(&self) -> Arc<AIService> {
        self.collections.get_ai_service()
    }

    pub async fn count_document_in_collection(&self, collection_id: CollectionId) -> Option<u64> {
        let collection = self.collections.get_collection(collection_id).await?;
        Some(collection.count_documents())
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
        .map(|(value, key)| TokenScore {
            document_id: key,
            score: value.into_inner(),
        })
        .collect();

    result
}

fn default_insert_batch_commit_size() -> u64 {
    300
}

fn start_commit_loop(read_side: Arc<ReadSide>, insert_batch_commit_size: Duration) {
    tokio::task::spawn(async move {
        let start = Instant::now() + insert_batch_commit_size;
        let mut interval = tokio::time::interval_at(start, insert_batch_commit_size);

        // If for some reason we miss a tick, we skip it.
        // In fact, the commit is blocked only by `update` method.
        // If the collection is under heavy load,
        // the commit will be run due to the `insert_batch_commit_size` config.
        interval.set_missed_tick_behavior(MissedTickBehavior::Skip);

        loop {
            interval.tick().await;
            info!(
                "{:?} time reached. Committing read side",
                insert_batch_commit_size.clone()
            );
            if let Err(e) = read_side.commit().await {
                tracing::error!(?e, "Cannot commit read side");
            }
        }
    });
}

fn start_receive_operations(read_side: Arc<ReadSide>, mut operation_receiver: OperationReceiver) {
    tokio::task::spawn(async move {
        info!("Starting operation receiver");
        while let Some(op) = operation_receiver.recv().await {
            trace!(?op, "Received operation");
            if let Err(e) = read_side.update(op).await {
                tracing::error!(?e, "Cannot update read side");
            }
        }
        error!("Operation receiver is closed");
    });
}

#[derive(Deserialize, Serialize)]
struct ReadInfoV1 {
    offset: Offset,
}

#[derive(Deserialize, Serialize)]
enum ReadInfo {
    V1(ReadInfoV1),
}

#[cfg(test)]
mod tests {
    use crate::collection_manager::sides::read::{
        collection::CollectionReader, collections::CollectionsReader,
    };

    #[test]
    fn test_side_read_sync_send() {
        fn assert_sync_send<T: Sync + Send>() {}
        assert_sync_send::<CollectionsReader>();
        assert_sync_send::<CollectionReader>();
    }
}
