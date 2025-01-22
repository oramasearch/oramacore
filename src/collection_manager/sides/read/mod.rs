mod collection;
mod collections;
mod document_storage;

use std::{collections::HashMap, sync::Arc};

use anyhow::{Context, Result};
use collections::CollectionsReader;
pub use collections::IndexesConfig;
use document_storage::{DocumentStorage, DocumentStorageConfig};
use ordered_float::NotNan;

use crate::{
    ai::AIService,
    capped_heap::CappedHeap,
    collection_manager::dto::{SearchParams, SearchResult, SearchResultHit, TokenScore},
    metrics::{
        CollectionAddedLabels, CollectionOperationLabels, COLLECTION_ADDED_COUNTER,
        COLLECTION_OPERATION_COUNTER,
    },
    nlp::NLPService,
    types::{CollectionId, DocumentId},
};

use super::{CollectionWriteOperation, Offset, WriteOperation};

pub struct ReadSide {
    collections: CollectionsReader,
    document_storage: DocumentStorage,
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

        Ok(Self {
            collections: CollectionsReader::try_new(ai_service, nlp_service, indexes_config)?,
            document_storage,
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
            .context("Cannot commit document storage")?;

        Ok(())
    }

    pub async fn search(
        &self,
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
        let token_scores = collection.search(search_params).await?;

        let facets = collection.calculate_facets(&token_scores, facets)?;

        let count = token_scores.len();

        let top_results: Vec<TokenScore> = top_n(token_scores, limit.0);

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
        let (offset, op) = op;
        match op {
            WriteOperation::CreateCollection { id } => {
                COLLECTION_ADDED_COUNTER
                    .create(CollectionAddedLabels {
                        collection: id.0.clone(),
                    })
                    .increment_by_one();
                self.collections.create_collection(offset, id).await?;
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

                if let CollectionWriteOperation::InsertDocument { doc_id, doc } =
                    collection_operation
                {
                    collection.increment_document_count();
                    self.document_storage.add_document(doc_id, doc).await?;
                } else {
                    collection.update(offset, collection_operation).await?;
                }
            }
        }

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
