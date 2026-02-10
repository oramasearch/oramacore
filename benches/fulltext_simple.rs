use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use fake::{Fake, Faker};
use serde_json::json;
use std::time::Duration;
use tokio::runtime::Runtime;

// Import the necessary modules directly since tests module is not available in bench context
use std::{path::PathBuf, sync::Arc};

use anyhow::{bail, Result};
use futures::{future::BoxFuture, FutureExt};
use tokio::time::sleep;

use anyhow::Context;
use duration_string::DurationString;
use oramacore::{
    ai::{AIServiceConfig, AIServiceLLMConfig},
    build_orama,
    collection_manager::sides::{
        read::{
            CollectionCommitConfig, IndexesConfig, OffloadFieldConfig, ReadSide, ReadSideConfig,
            SearchRequest,
        },
        write::{CollectionsWriterConfig, TempIndexCleanupConfig, WriteSide, WriteSideConfig},
        InputSideChannelType, OutputSideChannelType,
    },
    types::{
        ApiKey, CollectionId, CollectionStatsRequest, CreateCollection, CreateIndexRequest,
        DocumentList, IndexId, InsertDocumentsResult, ReadApiKey, SearchParams, SearchResult,
        WriteApiKey,
    },
    web_server::HttpConfig,
    OramacoreConfig,
};

/// Generate a new temporary path for test data
pub fn generate_new_path() -> PathBuf {
    let tmp_dir = tempfile::tempdir().expect("Cannot create temp dir");
    let dir = tmp_dir.path().to_path_buf();
    std::fs::create_dir_all(dir.clone()).expect("Cannot create dir");
    dir
}

/// Create Oramacore configuration for benchmarks
pub fn create_oramacore_config() -> OramacoreConfig {
    OramacoreConfig {
        log: Default::default(),
        http: HttpConfig {
            host: "127.0.0.1".parse().unwrap(),
            port: 2222,
            allow_cors: false,
            with_prometheus: false,
        },
        ai_server: AIServiceConfig {
            embeddings: None,
            llm: AIServiceLLMConfig {
                local: true,
                host: "localhost".to_string(),
                port: Some(8000),
                model: "Qwen/Qwen2.5-3b-Instruct".to_string(),
                api_key: String::new(),
            },
            remote_llms: None,
            models_cache_dir: "/tmp/fastembed_cache".to_string(),
            total_threads: 12,
        },
        writer_side: WriteSideConfig {
            master_api_key: ApiKey::try_new("my-master-api-key").unwrap(),
            output: OutputSideChannelType::InMemory { capacity: 100 },
            config: CollectionsWriterConfig {
                data_dir: generate_new_path(),
                embedding_queue_limit: 50,
                default_embedding_model: oramacore::python::embeddings::Model::BGESmall,
                insert_batch_commit_size: 10_000,
                javascript_queue_limit: 10_000,
                commit_interval: Duration::from_secs(3_000),
                temp_index_cleanup: TempIndexCleanupConfig {
                    cleanup_interval: Duration::from_secs(3600),
                    max_age: Duration::from_secs(43200),
                },
            },
            hooks: Default::default(),
            jwt: None,
        },
        reader_side: ReadSideConfig {
            master_api_key: None,
            input: InputSideChannelType::InMemory { capacity: 100 },
            config: IndexesConfig {
                data_dir: generate_new_path(),
                insert_batch_commit_size: 10_000,
                commit_interval: Duration::from_secs(3_000),
                notifier: None,
                offload_field: OffloadFieldConfig {
                    unload_window: DurationString::from_string("30m".to_string()).unwrap(),
                    slot_count_exp: 8,
                    slot_size_exp: 4,
                },
                force_commit: 4,
                collection_commit: CollectionCommitConfig::default(),
            },
            analytics: None,
            hooks: Default::default(),
            jwt: None,
        },
    }
}

/// Wait for async operations with timeout
pub async fn wait_for<'i, 'b, I, R>(
    i: &'i I,
    f: impl Fn(&'i I) -> BoxFuture<'b, Result<R>>,
) -> Result<R>
where
    'b: 'i,
{
    const MAX_ATTEMPTS: usize = 2_000;
    let mut attempts = 0;
    loop {
        attempts += 1;
        match f(i).await {
            Ok(r) => break Ok(r),
            Err(e) => {
                if attempts > MAX_ATTEMPTS {
                    break Err(e);
                }
                sleep(Duration::from_millis(20)).await
            }
        }
    }
}

/// Test context for benchmarks
pub struct TestContext {
    pub config: OramacoreConfig,
    pub reader: Arc<ReadSide>,
    pub writer: Arc<WriteSide>,
    pub master_api_key: ApiKey,
}

impl TestContext {
    pub async fn new() -> Self {
        let mut config: OramacoreConfig = create_oramacore_config();
        config.writer_side.master_api_key = Self::generate_api_key();
        Self::new_with_config(config).await
    }

    pub async fn new_with_config(config: OramacoreConfig) -> Self {
        let master_api_key = config.writer_side.master_api_key;
        let (writer, reader) = build_orama(config.clone()).await.unwrap();
        let writer = writer.unwrap();
        let reader = reader.unwrap();

        TestContext {
            config,
            reader,
            writer,
            master_api_key,
        }
    }

    pub async fn create_collection(&self) -> Result<TestCollectionClient> {
        let id = Self::generate_collection_id();
        let write_api_key = Self::generate_api_key();
        let read_api_key = Self::generate_api_key();

        self.writer
            .create_collection(
                self.master_api_key,
                CreateCollection {
                    id,
                    description: None,
                    mcp_description: None,
                    read_api_key,
                    write_api_key,
                    language: None,
                    embeddings_model: Some(oramacore::python::embeddings::Model::BGESmall),
                },
            )
            .await?;

        let read_api_key = ReadApiKey::ApiKey(read_api_key);
        let read_api_key_for_wait = read_api_key.clone();
        wait_for(self, |s| {
            let reader = s.reader.clone();
            let read_api_key = read_api_key_for_wait.clone();
            async move {
                reader
                    .collection_stats(
                        &read_api_key,
                        id,
                        CollectionStatsRequest { with_keys: false },
                    )
                    .await
                    .context("")
            }
            .boxed()
        })
        .await?;

        self.get_test_collection_client(id, WriteApiKey::from_api_key(write_api_key), read_api_key)
    }

    pub fn get_test_collection_client(
        &self,
        collection_id: CollectionId,
        write_api_key: WriteApiKey,
        read_api_key: ReadApiKey,
    ) -> Result<TestCollectionClient> {
        Ok(TestCollectionClient {
            collection_id,
            write_api_key,
            read_api_key,
            reader: self.reader.clone(),
            writer: self.writer.clone(),
        })
    }

    pub async fn commit_all(&self) -> Result<()> {
        self.writer.commit().await?;
        self.reader.commit(true).await?;
        Ok(())
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

/// Test collection client for benchmarks
pub struct TestCollectionClient {
    pub collection_id: CollectionId,
    pub write_api_key: WriteApiKey,
    pub read_api_key: ReadApiKey,
    pub reader: Arc<ReadSide>,
    pub writer: Arc<WriteSide>,
}

impl TestCollectionClient {
    pub async fn create_index(&self) -> Result<TestIndexClient> {
        let index_id = Self::generate_index_id("index");
        self.writer
            .create_index(
                self.write_api_key,
                self.collection_id,
                CreateIndexRequest {
                    index_id,
                    embedding: None,
                    type_strategy: Default::default(),
                },
            )
            .await?;

        wait_for(self, |s| {
            let reader = s.reader.clone();
            let read_api_key = s.read_api_key.clone();
            let collection_id = s.collection_id;
            async move {
                let stats = reader
                    .collection_stats(
                        &read_api_key,
                        collection_id,
                        CollectionStatsRequest { with_keys: false },
                    )
                    .await?;

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
            read_api_key: self.read_api_key.clone(),
            reader: self.reader.clone(),
            writer: self.writer.clone(),
        })
    }

    pub async fn search(&self, search_params: SearchParams) -> Result<SearchResult> {
        self.reader
            .search(
                &self.read_api_key,
                self.collection_id,
                SearchRequest {
                    search_params,
                    analytics_metadata: None,
                    interaction_id: None,
                    search_analytics_event_origin: None,
                },
            )
            .await
            .context("")
    }

    fn generate_index_id(prefix: &str) -> IndexId {
        let id: String = format!("{}_{}", prefix, Faker.fake::<String>());
        IndexId::try_new(id).unwrap()
    }
}

/// Test index client for benchmarks
pub struct TestIndexClient {
    pub collection_id: CollectionId,
    pub index_id: IndexId,
    pub write_api_key: WriteApiKey,
    pub read_api_key: ReadApiKey,
    pub reader: Arc<ReadSide>,
    pub writer: Arc<WriteSide>,
}

impl TestIndexClient {
    pub async fn insert_documents(&self, documents: DocumentList) -> Result<InsertDocumentsResult> {
        let stats = self
            .reader
            .collection_stats(
                &self.read_api_key,
                self.collection_id,
                CollectionStatsRequest { with_keys: false },
            )
            .await?;
        let index_stats = stats
            .indexes_stats
            .iter()
            .find(|index| index.id == self.index_id)
            .ok_or_else(|| anyhow::anyhow!("Index not found"))?;
        let document_count = index_stats.document_count;
        let expected_document_count = documents.len() + document_count;

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
            let read_api_key = s.read_api_key.clone();
            let collection_id = s.collection_id;
            async move {
                let stats = reader
                    .collection_stats(
                        &read_api_key,
                        collection_id,
                        CollectionStatsRequest { with_keys: false },
                    )
                    .await?;
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
}

/// Benchmark scales - focused on smaller scales for faster execution
const SCALES: &[usize] = &[1_000, 5_000];

/// Generate test documents with simple, predictable content for benchmarking
fn generate_test_documents(count: usize) -> Vec<serde_json::Value> {
    (0..count)
        .map(|i| {
            json!({
                "id": i.to_string(),
                "text": format!("document content technology software development number {}", i),
                "category": match i % 3 {
                    0 => "technology",
                    1 => "software",
                    _ => "development"
                },
            })
        })
        .collect()
}

/// Setup function to prepare test data
async fn setup_test_data(doc_count: usize) -> (TestContext, TestCollectionClient) {
    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // Insert documents in batches
    let documents = generate_test_documents(doc_count);
    let batch_size = 500;

    for chunk in documents.chunks(batch_size) {
        index_client
            .insert_documents(json!(chunk).try_into().unwrap())
            .await
            .unwrap();
    }

    (test_context, collection_client)
}

/// Benchmark basic fulltext search on uncommitted indexes
fn bench_fulltext_search_basic(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    for &scale in SCALES {
        let mut group = c.benchmark_group(format!("fulltext_basic_{scale}"));
        group.throughput(Throughput::Elements(scale as u64));
        group.sample_size(10);
        group.measurement_time(Duration::from_secs(3));

        // Setup data for this scale
        let (test_context, collection_client) = rt.block_on(async { setup_test_data(scale).await });

        // Simple single word search
        group.bench_function("single_word", |b| {
            b.to_async(&rt).iter(|| async {
                let search_params = black_box(json!({"term": "technology"}).try_into().unwrap());
                let result = collection_client.search(search_params).await.unwrap();
                black_box(result);
            });
        });

        // Multi-word search
        group.bench_function("multi_word", |b| {
            b.to_async(&rt).iter(|| async {
                let search_params =
                    black_box(json!({"term": "technology software"}).try_into().unwrap());
                let result = collection_client.search(search_params).await.unwrap();
                black_box(result);
            });
        });

        // Search with limit
        group.bench_function("with_limit", |b| {
            b.to_async(&rt).iter(|| async {
                let search_params = black_box(
                    json!({"term": "development", "limit": 10})
                        .try_into()
                        .unwrap(),
                );
                let result = collection_client.search(search_params).await.unwrap();
                black_box(result);
            });
        });

        group.finish();
        drop(test_context);
    }
}

/// Benchmark fulltext search on committed indexes
fn bench_fulltext_search_committed(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    for &scale in SCALES {
        let mut group = c.benchmark_group(format!("fulltext_committed_{scale}"));
        group.throughput(Throughput::Elements(scale as u64));
        group.sample_size(10);
        group.measurement_time(Duration::from_secs(3));

        // Setup data and commit for this scale
        let (test_context, collection_client) = rt.block_on(async {
            let (test_context, collection_client) = setup_test_data(scale).await;
            test_context.commit_all().await.unwrap();
            (test_context, collection_client)
        });

        // Simple single word search
        group.bench_function("single_word", |b| {
            b.to_async(&rt).iter(|| async {
                let search_params = black_box(json!({"term": "technology"}).try_into().unwrap());
                let result = collection_client.search(search_params).await.unwrap();
                black_box(result);
            });
        });

        // Multi-word search
        group.bench_function("multi_word", |b| {
            b.to_async(&rt).iter(|| async {
                let search_params =
                    black_box(json!({"term": "technology software"}).try_into().unwrap());
                let result = collection_client.search(search_params).await.unwrap();
                black_box(result);
            });
        });

        // Search with limit
        group.bench_function("with_limit", |b| {
            b.to_async(&rt).iter(|| async {
                let search_params = black_box(
                    json!({"term": "development", "limit": 10})
                        .try_into()
                        .unwrap(),
                );
                let result = collection_client.search(search_params).await.unwrap();
                black_box(result);
            });
        });

        group.finish();
        drop(test_context);
    }
}

criterion_group!(
    fulltext_simple_benches,
    bench_fulltext_search_basic,
    bench_fulltext_search_committed
);
criterion_main!(fulltext_simple_benches);
