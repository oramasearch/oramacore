use std::{sync::Arc, time::Duration};

use anyhow::{bail, Result};
use fake::{Fake, Faker};
use futures::future::BoxFuture;
use futures::future::FutureExt;
use serde_json::json;
use tokio::time::sleep;

use crate::types::SearchParams;
use crate::types::SearchResult;
use crate::{
    build_orama,
    collection_manager::sides::{index::CreateIndexRequest, ReadSide, WriteSide},
    tests::utils::init_log,
    types::{ApiKey, CollectionId, CreateCollection, DocumentList, IndexId, InsertDocumentsResult},
    OramacoreConfig,
};

use super::utils::{create_grpc_server, create_oramacore_config};

async fn wait_for<'i, 'b, I, R>(i: &'i I, f: impl Fn(&I) -> BoxFuture<'b, Result<R>>) -> Result<R>
where
    'b: 'i,
{
    // 10msec * 1000 attempts = 10 sec
    const MAX_ATTEMPTS: usize = 1_000;
    let mut attempts = 0;
    loop {
        attempts += 1;
        match f(i).await {
            Ok(r) => break Ok(r),
            Err(e) => {
                if attempts > MAX_ATTEMPTS {
                    break Err(e);
                }
                sleep(Duration::from_millis(10)).await
            }
        }
    }
}

struct TestContext {
    reader: Arc<ReadSide>,
    writer: Arc<WriteSide>,
    pub master_api_key: ApiKey,
}
impl TestContext {
    async fn new() -> Self {
        let mut config: OramacoreConfig = create_oramacore_config();
        config.writer_side.master_api_key = Self::generate_api_key();
        Self::new_with_config(config).await
    }

    async fn new_with_config(mut config: OramacoreConfig) -> Self {
        if config.ai_server.port == 0 {
            let address = create_grpc_server().await.unwrap();
            config.ai_server.host = address.ip().to_string();
            config.ai_server.port = address.port();
        }

        let master_api_key = config.writer_side.master_api_key;
        let (writer, reader) = build_orama(config).await.unwrap();
        let writer = writer.unwrap();
        let reader = reader.unwrap();

        TestContext {
            reader,
            writer,
            master_api_key,
        }
    }

    async fn create_collection(&self) -> Result<TestCollectionClient> {
        let id = Self::generate_collection_id();
        let write_api_key = Self::generate_api_key();
        let read_api_key = Self::generate_api_key();

        self.writer
            .create_collection(
                self.master_api_key,
                CreateCollection {
                    id,
                    description: None,
                    read_api_key,
                    write_api_key,
                    language: None,
                    embeddings: None,
                },
            )
            .await?;

        wait_for(self, |s| {
            let reader = s.reader.clone();
            async move { reader.collection_stats(read_api_key, id).await }.boxed()
        })
        .await?;

        Ok(TestCollectionClient {
            collection_id: id,
            write_api_key,
            read_api_key,
            reader: self.reader.clone(),
            writer: self.writer.clone(),
        })
    }

    fn generate_collection_id() -> CollectionId {
        let id: String = Faker.fake();
        CollectionId::try_new(id).unwrap()
    }
    fn generate_api_key() -> ApiKey {
        let id: String = Faker.fake();
        ApiKey::try_new(id).unwrap()
    }
}

struct TestCollectionClient {
    pub collection_id: CollectionId,
    pub write_api_key: ApiKey,
    pub read_api_key: ApiKey,
    reader: Arc<ReadSide>,
    writer: Arc<WriteSide>,
}
impl TestCollectionClient {
    pub async fn create_index(&self) -> Result<TestIndexClient> {
        let index_id = Self::generate_index_id();
        self.writer
            .create_index(
                self.write_api_key,
                self.collection_id,
                CreateIndexRequest {
                    id: index_id,
                    embedding_field_definition: vec![],
                },
            )
            .await?;

        wait_for(self, |s| {
            let reader = s.reader.clone();
            let read_api_key = s.read_api_key;
            let collection_id = s.collection_id;
            async move {
                let stats = reader.collection_stats(read_api_key, collection_id).await?;

                stats
                    .indexes_stats
                    .iter()
                    .find(|index| index.id == index_id)
                    .ok_or_else(|| anyhow::anyhow!("Index not found"))?;

                Ok(())
            }
            .boxed()
        })
        .await?;

        Ok(TestIndexClient {
            collection_id: self.collection_id,
            index_id,
            write_api_key: self.write_api_key,
            read_api_key: self.read_api_key,
            reader: self.reader.clone(),
            writer: self.writer.clone(),
        })
    }

    pub async fn search(&self, search_params: SearchParams) -> Result<SearchResult> {
        self.reader
            .search(self.read_api_key, self.collection_id, search_params)
            .await
    }

    fn generate_index_id() -> IndexId {
        let id: String = Faker.fake();
        IndexId::try_new(id).unwrap()
    }
}

struct TestIndexClient {
    pub collection_id: CollectionId,
    pub index_id: IndexId,
    pub write_api_key: ApiKey,
    pub read_api_key: ApiKey,
    reader: Arc<ReadSide>,
    writer: Arc<WriteSide>,
}
impl TestIndexClient {
    pub async fn insert_documents(&self, documents: DocumentList) -> Result<InsertDocumentsResult> {
        let stats = self
            .reader
            .collection_stats(self.read_api_key, self.collection_id)
            .await?;
        let index_stats = stats
            .indexes_stats
            .iter()
            .find(|index| index.id == self.index_id)
            .ok_or_else(|| anyhow::anyhow!("Index not found"))?;
        let document_count = index_stats.document_count;
        let expected_document_count = documents.len() as u64 + document_count;

        let result = self
            .writer
            .insert_documents(
                self.write_api_key,
                self.collection_id,
                self.index_id,
                documents,
            )
            .await?;

        wait_for(self, |s| {
            let reader = s.reader.clone();
            let read_api_key = s.read_api_key;
            let collection_id = s.collection_id;
            async move {
                let stats = reader.collection_stats(read_api_key, collection_id).await?;
                let index_stats = stats
                    .indexes_stats
                    .iter()
                    .find(|index| index.id == self.index_id)
                    .ok_or_else(|| anyhow::anyhow!("Index not found"))?;
                if index_stats.document_count < expected_document_count {
                    bail!(
                        "Document count mismatch: expected {}, got {}",
                        expected_document_count,
                        index_stats.document_count
                    );
                }

                Ok(())
            }
            .boxed()
        })
        .await?;

        Ok(result)
    }

    pub async fn delete_documents(&self, ids: Vec<String>) -> Result<()> {
        let stats = self
            .reader
            .collection_stats(self.read_api_key, self.collection_id)
            .await?;
        let index_stats = stats
            .indexes_stats
            .iter()
            .find(|index| index.id == self.index_id)
            .ok_or_else(|| anyhow::anyhow!("Index not found"))?;
        let document_count = index_stats.document_count;
        let expected_document_count = document_count - ids.len() as u64;

        self.writer
            .delete_documents(self.write_api_key, self.collection_id, self.index_id, ids)
            .await?;

        wait_for(self, |s| {
            let reader = s.reader.clone();
            let read_api_key = s.read_api_key;
            let collection_id = s.collection_id;
            async move {
                let stats = reader
                    .collection_stats(read_api_key, collection_id)
                    .await?;
                let index_stats = stats
                    .indexes_stats
                    .iter()
                    .find(|index| index.id == self.index_id)
                    .ok_or_else(|| anyhow::anyhow!("Index not found"))?;

                if index_stats.document_count > expected_document_count {
                    bail!(
                        "Document count mismatch: expected {}, got {}",
                        expected_document_count,
                        index_stats.document_count
                    );
                }
                    
                Ok(())
            }
            .boxed()
        })
        .await?;

        Ok(())

    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_fulltext_search() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();

    sleep(Duration::from_secs(1)).await;

    let index_client = collection_client.create_index().await.unwrap();

    index_client.insert_documents(json!([
        {
            "id": "1",
            "text": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        },
        {
            "id": "2",
            "text": "Curabitur sem tortor, interdum in rutrum in, dignissim vestibulum metus.",
        }
    ]).try_into().unwrap()).await.unwrap();

    let output = collection_client
        .search(
            json!({
                "term": "Lorem ipsum",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();


    assert_eq!(output.count, 1);
    assert_eq!(output.hits.len(), 1);
    assert_eq!(output.hits[0].id, format!("{}:{}", index_client.index_id, "1"));
    assert!(output.hits[0].score > 0.);

    drop(test_context);
}


#[tokio::test(flavor = "multi_thread")]
async fn test_delete_search() {
    init_log();

    let test_context = TestContext::new().await;

    let collection_client = test_context.create_collection().await.unwrap();

    sleep(Duration::from_secs(1)).await;

    let index_client = collection_client.create_index().await.unwrap();

    index_client.insert_documents(json!([
        {
            "id": "1",
            "text": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        },
        {
            "id": "2",
            "text": "Curabitur sem tortor, interdum in rutrum in, dignissim vestibulum metus.",
        }
    ]).try_into().unwrap()).await.unwrap();
    index_client.delete_documents(vec!["1".to_string()]).await.unwrap();

    let output = collection_client
        .search(
            json!({
                "term": "Lorem ipsum",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.count, 0);
    assert_eq!(output.hits.len(), 0);

    let output = collection_client
        .search(
            json!({
                "term": "Curabitur sem tortor",
            })
            .try_into()
            .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(output.count, 1);
    assert_eq!(output.hits.len(), 1);

    drop(test_context);
}
