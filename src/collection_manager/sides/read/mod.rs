mod collection;
mod collections;
mod document_storage;

use std::{collections::HashMap, path::PathBuf, sync::Arc};

use anyhow::{Context, Result};
use collections::CollectionsReader;
use document_storage::{DocumentStorage, DocumentStorageConfig};
use ordered_float::NotNan;
use serde::Deserialize;
use tokio::sync::RwLock;
use tracing::{info, trace};

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

use super::{CollectionWriteOperation, Offset, WriteOperation};

#[derive(Debug, Deserialize, Clone)]
pub struct IndexesConfig {
    pub data_dir: PathBuf,
    #[serde(default = "default_insert_batch_commit_size")]
    pub insert_batch_commit_size: u64,
}

pub struct ReadSide {
    collections: CollectionsReader,
    document_storage: DocumentStorage,
    operation_counter: RwLock<u64>,
    insert_batch_commit_size: u64,
}

impl ReadSide {
    pub fn try_new(
        ai_service: Arc<AIService>,
        nlp_service: Arc<NLPService>,
        indexes_config: IndexesConfig,
    ) -> Result<Self> {
        let document_storage = DocumentStorage::try_new(DocumentStorageConfig {
            data_dir: indexes_config.data_dir.join("docs"),
        })
        .context("Cannot create document storage")?;

        let insert_batch_commit_size = indexes_config.insert_batch_commit_size;

        Ok(Self {
            collections: CollectionsReader::try_new(ai_service, nlp_service, indexes_config)?,
            document_storage,
            operation_counter: Default::default(),
            insert_batch_commit_size,
        })
    }

    pub async fn load(&mut self) -> Result<()> {
        self.collections.load().await?;

        self.document_storage
            .load()
            .context("Cannot load document storage")?;

        Ok(())
    }

    pub async fn commit(&self) -> Result<()> {
        self.collections.commit().await?;

        self.document_storage
            .commit()
            .await
            .context("Cannot commit document storage")?;

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

        let (offset, op) = op;
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
                    self.document_storage.add_document(doc_id, doc).await?;
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
