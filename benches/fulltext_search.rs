use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use fake::{Fake, Faker};
use serde_json::json;
use std::time::Duration;
use tempfile;
use tokio::runtime::Runtime;

// Import the necessary modules directly since tests module is not available in bench context
use std::{path::PathBuf, sync::Arc};

use anyhow::{bail, Result};
use futures::{future::BoxFuture, FutureExt};
use tokio::time::sleep;

use anyhow::Context;
use duration_string::DurationString;
use http::uri::Scheme;
use oramacore::{
    ai::{AIServiceConfig, AIServiceLLMConfig, OramaModel},
    build_orama,
    collection_manager::sides::{
        read::{
            AnalyticSearchEventInvocationType, IndexesConfig, OffloadFieldConfig, ReadSide,
            ReadSideConfig,
        },
        write::{
            CollectionsWriterConfig, OramaModelSerializable, TempIndexCleanupConfig, WriteSide,
            WriteSideConfig,
        },
        InputSideChannelType, OutputSideChannelType,
    },
    types::{
        ApiKey, CollectionId, CollectionStatsRequest, CreateCollection, CreateIndexRequest,
        DocumentList, IndexId, InsertDocumentsResult, SearchParams, SearchResult, WriteApiKey,
    },
    web_server::HttpConfig,
    OramacoreConfig,
};

/// Initialize logging for benchmarks
pub fn init_log() {
    if let Ok(a) = std::env::var("LOG") {
        if a == "info" {
            std::env::set_var("RUST_LOG", "oramacore=info,warn");
        } else {
            std::env::set_var("RUST_LOG", "oramacore=trace,warn");
        }
    }
    let _ = tracing_subscriber::fmt::try_init();
}

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
            host: "0.0.0.0".parse().unwrap(),
            port: 0,
            api_key: None,
            max_connections: 1,
            scheme: Scheme::HTTP,
            embeddings: None,
            llm: AIServiceLLMConfig {
                local: true,
                host: "localhost".to_string(),
                port: Some(8000),
                model: "Qwen/Qwen2.5-3b-Instruct".to_string(),
                api_key: String::new(),
            },
            remote_llms: None,
        },
        writer_side: WriteSideConfig {
            master_api_key: ApiKey::try_new("my-master-api-key").unwrap(),
            output: OutputSideChannelType::InMemory { capacity: 100 },
            config: CollectionsWriterConfig {
                data_dir: generate_new_path(),
                embedding_queue_limit: 50,
                default_embedding_model: OramaModelSerializable(OramaModel::BgeSmall),
                insert_batch_commit_size: 10_000,
                javascript_queue_limit: 10_000,
                commit_interval: Duration::from_secs(3_000),
                temp_index_cleanup: TempIndexCleanupConfig {
                    cleanup_interval: Duration::from_secs(3600),
                    max_age: Duration::from_secs(43200),
                },
            },
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
            },
            analytics: None,
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
                    read_api_key,
                    write_api_key,
                    language: None,
                    embeddings_model: Some(OramaModelSerializable(OramaModel::BgeSmall)),
                },
            )
            .await?;

        wait_for(self, |s| {
            let reader = s.reader.clone();
            async move {
                reader
                    .collection_stats(
                        read_api_key,
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
        read_api_key: ApiKey,
    ) -> Result<TestCollectionClient> {
        Ok(TestCollectionClient {
            collection_id,
            write_api_key,
            read_api_key,
            master_api_key: self.master_api_key,
            reader: self.reader.clone(),
            writer: self.writer.clone(),
        })
    }

    pub async fn commit_all(&self) -> Result<()> {
        self.writer.commit().await?;
        self.reader.commit().await?;
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
    pub read_api_key: ApiKey,
    master_api_key: ApiKey,
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
                },
            )
            .await?;

        wait_for(self, |s| {
            let reader = s.reader.clone();
            let read_api_key = s.read_api_key;
            let collection_id = s.collection_id;
            async move {
                let stats = reader
                    .collection_stats(
                        read_api_key,
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
            read_api_key: self.read_api_key,
            reader: self.reader.clone(),
            writer: self.writer.clone(),
        })
    }

    pub async fn search(&self, search_params: SearchParams) -> Result<SearchResult> {
        self.reader
            .search(
                self.read_api_key,
                self.collection_id,
                search_params,
                AnalyticSearchEventInvocationType::Direct,
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
    pub read_api_key: ApiKey,
    pub reader: Arc<ReadSide>,
    pub writer: Arc<WriteSide>,
}

impl TestIndexClient {
    pub async fn insert_documents(&self, documents: DocumentList) -> Result<InsertDocumentsResult> {
        let stats = self
            .reader
            .collection_stats(
                self.read_api_key,
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
            let read_api_key = s.read_api_key;
            let collection_id = s.collection_id;
            async move {
                let stats = reader
                    .collection_stats(
                        read_api_key,
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

/// Different scales of data to benchmark
const SCALES: &[usize] = &[1_000, 10_000];

/// Query types to benchmark
#[derive(Debug, Clone)]
enum QueryType {
    SingleWord,
    MultiWord,
    Phrase,
    ExactMatch,
    FuzzySearch,
    WithThreshold,
}

impl QueryType {
    fn name(&self) -> &'static str {
        match self {
            QueryType::SingleWord => "single_word",
            QueryType::MultiWord => "multi_word",
            QueryType::Phrase => "phrase",
            QueryType::ExactMatch => "exact_match",
            QueryType::FuzzySearch => "fuzzy_search",
            QueryType::WithThreshold => "with_threshold",
        }
    }
}

/// Generate realistic text content for documents
fn generate_document_content(id: usize) -> String {
    let base_words = vec![
        "technology",
        "innovation",
        "development",
        "software",
        "engineering",
        "artificial",
        "intelligence",
        "machine",
        "learning",
        "data",
        "science",
        "computer",
        "programming",
        "algorithm",
        "optimization",
        "performance",
        "scalability",
        "efficiency",
        "automation",
        "digital",
        "transformation",
        "cloud",
        "computing",
        "database",
        "search",
        "analytics",
        "business",
        "strategy",
        "marketing",
        "product",
        "design",
        "user",
        "experience",
        "interface",
        "mobile",
        "web",
        "application",
        "platform",
        "network",
        "security",
        "privacy",
        "compliance",
        "governance",
        "architecture",
        "framework",
        "library",
        "open",
        "source",
        "collaboration",
        "team",
    ];

    let sentence_count = (id % 5) + 1; // 1-5 sentences
    let mut content = String::new();

    for _ in 0..sentence_count {
        let word_count = (id % 10) + 5; // 5-14 words per sentence
        let mut sentence = Vec::new();

        for _ in 0..word_count {
            sentence.push(base_words[id % base_words.len()]);
        }

        content.push_str(&sentence.join(" "));
        content.push_str(". ");
    }

    // Add some variety with specific patterns for different document types
    match id % 4 {
        0 => format!("Advanced {}", content),
        1 => format!("{} This document discusses innovative approaches.", content),
        2 => format!("Research paper: {}", content),
        3 => format!("{} Implementation guide and best practices.", content),
        _ => content,
    }
}

/// Generate test documents at different scales
fn generate_test_documents(count: usize) -> Vec<serde_json::Value> {
    (0..count)
        .map(|i| {
            json!({
                "id": i.to_string(),
                "title": format!("Document {}", i),
                "content": generate_document_content(i),
                "category": match i % 5 {
                    0 => "technology",
                    1 => "business",
                    2 => "science",
                    3 => "engineering",
                    _ => "general"
                },
                "priority": i % 10,
                "tags": vec![
                    format!("tag_{}", i % 20),
                    format!("category_{}", i % 5)
                ]
            })
        })
        .collect()
}

/// Create search queries based on query type
fn create_search_query(query_type: &QueryType) -> serde_json::Value {
    match query_type {
        QueryType::SingleWord => json!({
            "term": "technology"
        }),
        QueryType::MultiWord => json!({
            "term": "artificial intelligence machine learning"
        }),
        QueryType::Phrase => json!({
            "term": "innovative approaches",
            "exact": false
        }),
        QueryType::ExactMatch => json!({
            "term": "Research paper",
            "exact": true
        }),
        QueryType::FuzzySearch => json!({
            "term": "technlogy", // intentional typo
            "tolerance": 1
        }),
        QueryType::WithThreshold => json!({
            "term": "software engineering development",
            "threshold": 0.8
        }),
    }
}

/// Setup function to prepare test data and context
async fn setup_test_data(doc_count: usize) -> (TestContext, TestCollectionClient) {
    let test_context = TestContext::new().await;
    let collection_client = test_context.create_collection().await.unwrap();
    let index_client = collection_client.create_index().await.unwrap();

    // Insert documents in batches to avoid overwhelming the system
    let documents = generate_test_documents(doc_count);
    let batch_size = 1000;

    for chunk in documents.chunks(batch_size) {
        index_client
            .insert_documents(json!(chunk).try_into().unwrap())
            .await
            .unwrap();
    }

    (test_context, collection_client)
}

/// Benchmark fulltext search performance on uncommitted indexes
fn bench_fulltext_uncommitted(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    for &scale in SCALES {
        let mut group = c.benchmark_group(format!("fulltext_uncommitted_{}", scale));
        group.throughput(Throughput::Elements(scale as u64));
        group.sample_size(10); // Reduced sample size for faster benchmarks
        group.measurement_time(Duration::from_secs(5));

        // Setup data once for this scale
        let (test_context, collection_client) = rt.block_on(async { setup_test_data(scale).await });

        let query_types = vec![
            QueryType::SingleWord,
            QueryType::MultiWord,
            QueryType::Phrase,
            QueryType::ExactMatch,
            QueryType::FuzzySearch,
            QueryType::WithThreshold,
        ];

        for query_type in query_types {
            let query = create_search_query(&query_type);

            group.bench_with_input(
                BenchmarkId::new("query", query_type.name()),
                &query,
                |b, query| {
                    b.to_async(&rt).iter(|| async {
                        let search_params = black_box(query.clone()).try_into().unwrap();
                        let result = collection_client.search(search_params).await.unwrap();
                        black_box(result);
                    });
                },
            );
        }

        group.finish();
        drop(test_context);
    }
}

/// Benchmark fulltext search performance on committed indexes
fn bench_fulltext_committed(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    for &scale in SCALES {
        let mut group = c.benchmark_group(format!("fulltext_committed_{}", scale));
        group.throughput(Throughput::Elements(scale as u64));
        group.sample_size(10);
        group.measurement_time(Duration::from_secs(5));

        // Setup data and commit for this scale
        let (test_context, collection_client) = rt.block_on(async {
            let (test_context, collection_client) = setup_test_data(scale).await;

            // Commit the data to test committed index performance
            test_context.commit_all().await.unwrap();

            (test_context, collection_client)
        });

        let query_types = vec![
            QueryType::SingleWord,
            QueryType::MultiWord,
            QueryType::Phrase,
            QueryType::ExactMatch,
            QueryType::FuzzySearch,
            QueryType::WithThreshold,
        ];

        for query_type in query_types {
            let query = create_search_query(&query_type);

            group.bench_with_input(
                BenchmarkId::new("query", query_type.name()),
                &query,
                |b, query| {
                    b.to_async(&rt).iter(|| async {
                        let search_params = black_box(query.clone()).try_into().unwrap();
                        let result = collection_client.search(search_params).await.unwrap();
                        black_box(result);
                    });
                },
            );
        }

        group.finish();
        drop(test_context);
    }
}

/// Benchmark search result pagination performance
fn bench_fulltext_pagination(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let scale = 25_000; // Use a large dataset for pagination testing
    let mut group = c.benchmark_group("fulltext_pagination");
    group.throughput(Throughput::Elements(scale as u64));
    group.sample_size(10);

    let (test_context, collection_client) = rt.block_on(async {
        let (test_context, collection_client) = setup_test_data(scale).await;
        test_context.commit_all().await.unwrap();
        (test_context, collection_client)
    });

    // Test different pagination scenarios
    let pagination_tests = vec![
        ("first_page", 0, 10),
        ("mid_page", 100, 10),
        ("large_page", 0, 100),
        ("deep_page", 1000, 10),
    ];

    for (name, offset, limit) in pagination_tests {
        let query = json!({
            "term": "technology software",
            "offset": offset,
            "limit": limit
        });

        group.bench_with_input(BenchmarkId::new("pagination", name), &query, |b, query| {
            b.to_async(&rt).iter(|| async {
                let search_params = black_box(query.clone()).try_into().unwrap();
                let result = collection_client.search(search_params).await.unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
    drop(test_context);
}

/// Benchmark complex query scenarios
fn bench_fulltext_complex_queries(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let scale = 15_000;
    let mut group = c.benchmark_group("fulltext_complex_queries");
    group.throughput(Throughput::Elements(scale as u64));
    group.sample_size(10);

    let (test_context, collection_client) = rt.block_on(async {
        let (test_context, collection_client) = setup_test_data(scale).await;
        test_context.commit_all().await.unwrap();
        (test_context, collection_client)
    });

    // Complex query scenarios
    let complex_queries = vec![
        (
            "high_threshold",
            json!({
                "term": "artificial intelligence machine learning data science",
                "threshold": 0.9,
                "limit": 50
            }),
        ),
        (
            "low_threshold",
            json!({
                "term": "technology innovation development software",
                "threshold": 0.3,
                "limit": 100
            }),
        ),
        (
            "exact_with_tolerance",
            json!({
                "term": "technlogy innovtion", // intentional typos
                "exact": false,
                "tolerance": 2,
                "limit": 20
            }),
        ),
        (
            "long_query",
            json!({
                "term": "advanced technology innovation development software engineering artificial intelligence machine learning data science computer programming algorithm optimization performance scalability efficiency automation digital transformation cloud computing database search analytics business strategy",
                "threshold": 0.4,
                "limit": 30
            }),
        ),
        (
            "empty_term",
            json!({
                "term": "",
                "limit": 100
            }),
        ),
    ];

    for (name, query) in complex_queries {
        group.bench_with_input(BenchmarkId::new("complex", name), &query, |b, query| {
            b.to_async(&rt).iter(|| async {
                let search_params = black_box(query.clone()).try_into().unwrap();
                let result = collection_client.search(search_params).await.unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
    drop(test_context);
}

criterion_group!(
    fulltext_benches,
    bench_fulltext_uncommitted,
    bench_fulltext_committed,
    bench_fulltext_pagination,
    bench_fulltext_complex_queries
);
criterion_main!(fulltext_benches);
