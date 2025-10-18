use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use serde_json::json;
use std::time::Duration;
use tokio::runtime::Runtime;

// Import the necessary modules directly
use std::{path::PathBuf, sync::Arc};

use anyhow::Result;
use futures::{future::BoxFuture, FutureExt};
use tokio::time::sleep;

use anyhow::Context;
use duration_string::DurationString;
use http::uri::Scheme;
use oramacore::{
    ai::{AIServiceConfig, AIServiceLLMConfig},
    build_orama,
    collection_manager::sides::{
        read::{IndexesConfig, OffloadFieldConfig, ReadSide, ReadSideConfig, SearchRequest},
        write::{CollectionsWriterConfig, TempIndexCleanupConfig, WriteSide, WriteSideConfig},
        InputSideChannelType, OutputSideChannelType,
    },
    types::{
        ApiKey, CollectionId, CollectionStatsRequest, CreateCollection, CreateIndexRequest,
        DocumentList, IndexId, SearchParams, SearchResult, WriteApiKey,
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

/// Create minimal Oramacore configuration without AI server dependency
pub fn create_minimal_config() -> OramacoreConfig {
    OramacoreConfig {
        log: Default::default(),
        http: HttpConfig {
            host: "127.0.0.1".parse().unwrap(),
            port: 2223, // Different port to avoid conflicts
            allow_cors: false,
            with_prometheus: false,
        },
        ai_server: AIServiceConfig {
            embeddings: None,
            llm: AIServiceLLMConfig {
                local: false, // Disable local LLM
                host: "localhost".to_string(),
                port: None,
                model: "disabled".to_string(),
                api_key: String::new(),
            },
            remote_llms: None,
        },
        writer_side: WriteSideConfig {
            master_api_key: ApiKey::try_new("bench-master-key").unwrap(),
            output: OutputSideChannelType::InMemory { capacity: 1000 },
            config: CollectionsWriterConfig {
                data_dir: generate_new_path(),
                embedding_queue_limit: 10,
                default_embedding_model: oramacore::python::embeddings::Model::BGELarge,
                insert_batch_commit_size: 1_000, // Smaller for faster benchmarks
                javascript_queue_limit: 1_000,
                commit_interval: Duration::from_secs(300), // Shorter interval
                temp_index_cleanup: TempIndexCleanupConfig {
                    cleanup_interval: Duration::from_secs(3600),
                    max_age: Duration::from_secs(43200),
                },
            },
            jwt: None,
        },
        reader_side: ReadSideConfig {
            master_api_key: None,
            input: InputSideChannelType::InMemory { capacity: 1000 },
            config: IndexesConfig {
                data_dir: generate_new_path(),
                insert_batch_commit_size: 1_000,
                commit_interval: Duration::from_secs(300),
                notifier: None,
                offload_field: OffloadFieldConfig {
                    unload_window: DurationString::from_string("5m".to_string()).unwrap(),
                    slot_count_exp: 6, // Smaller for benchmarks
                    slot_size_exp: 3,
                },
            },
            analytics: None,
        },
    }
}

/// Wait for async operations with shorter timeout for benchmarks
pub async fn wait_for_quick<'i, 'b, I, R>(
    i: &'i I,
    f: impl Fn(&'i I) -> BoxFuture<'b, Result<R>>,
) -> Result<R>
where
    'b: 'i,
{
    const MAX_ATTEMPTS: usize = 500; // Shorter timeout
    let mut attempts = 0;
    loop {
        attempts += 1;
        match f(i).await {
            Ok(r) => break Ok(r),
            Err(e) => {
                if attempts > MAX_ATTEMPTS {
                    break Err(e);
                }
                sleep(Duration::from_millis(10)).await // Faster polling
            }
        }
    }
}

/// Minimal benchmark context
pub struct BenchContext {
    pub reader: Arc<ReadSide>,
    pub writer: Arc<WriteSide>,
    pub master_api_key: ApiKey,
    pub collection_id: CollectionId,
    pub read_api_key: ApiKey,
    pub write_api_key: WriteApiKey,
}

impl BenchContext {
    pub async fn new() -> Self {
        let config = create_minimal_config();
        let master_api_key = config.writer_side.master_api_key;
        let (writer, reader) = build_orama(config).await.expect("Failed to build orama");
        let writer = writer.expect("Writer should be enabled");
        let reader = reader.expect("Reader should be enabled");

        // Create collection
        let collection_id = CollectionId::try_new("bench_collection").unwrap();
        let write_api_key = ApiKey::try_new("bench_write_key").unwrap();
        let read_api_key = ApiKey::try_new("bench_read_key").unwrap();

        writer
            .create_collection(
                master_api_key,
                CreateCollection {
                    id: collection_id,
                    description: None,
                    mcp_description: None,
                    read_api_key,
                    write_api_key,
                    language: None,
                    embeddings_model: None, // No embeddings to avoid AI server dependency
                },
            )
            .await
            .expect("Failed to create collection");

        // Wait for collection to be ready
        let reader_clone = reader.clone();
        wait_for_quick(&reader_clone, |r| {
            async move {
                r.collection_stats(
                    read_api_key,
                    collection_id,
                    CollectionStatsRequest { with_keys: false },
                )
                .await
                .context("Collection not ready")
            }
            .boxed()
        })
        .await
        .expect("Collection setup timeout");

        BenchContext {
            reader,
            writer,
            master_api_key,
            collection_id,
            read_api_key,
            write_api_key: WriteApiKey::from_api_key(write_api_key),
        }
    }

    pub async fn create_index(&self) -> IndexId {
        let index_id = IndexId::try_new("bench_index").unwrap();

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
            .await
            .expect("Failed to create index");

        // Wait for index to be ready
        let reader_clone = self.reader.clone();
        let collection_id = self.collection_id;
        let read_api_key = self.read_api_key;
        wait_for_quick(&reader_clone, |r| {
            async move {
                let stats = r
                    .collection_stats(
                        read_api_key,
                        collection_id,
                        CollectionStatsRequest { with_keys: false },
                    )
                    .await?;

                stats
                    .indexes_stats
                    .iter()
                    .find(|idx| idx.id == index_id)
                    .ok_or_else(|| anyhow::anyhow!("Index not found"))?;

                Ok(())
            }
            .boxed()
        })
        .await
        .expect("Index setup timeout");

        index_id
    }

    pub async fn insert_documents(&self, index_id: IndexId, documents: DocumentList) -> Result<()> {
        self.writer
            .insert_documents(self.write_api_key, self.collection_id, index_id, documents)
            .await?;

        Ok(())
    }

    pub async fn search(&self, search_params: SearchParams) -> Result<SearchResult> {
        self.reader
            .search(
                self.read_api_key,
                self.collection_id,
                SearchRequest {
                    search_params,
                    analytics_metadata: None,
                    interaction_id: None,
                    search_analytics_event_origin: None,
                },
            )
            .await
            .map_err(|e| anyhow::anyhow!("Search failed: {}", e))
    }

    pub async fn commit_all(&self) -> Result<()> {
        self.writer.commit().await?;
        self.reader.commit().await?;
        Ok(())
    }
}

/// Generate simple test documents
fn generate_simple_docs(count: usize) -> Vec<serde_json::Value> {
    (0..count)
        .map(|i| {
            json!({
                "id": i.to_string(),
                "text": format!("technology software development document number {}", i),
            })
        })
        .collect()
}

/// Benchmark single word search performance
fn bench_single_word_search(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let scales = [1000, 5000];

    for scale in scales {
        let mut group = c.benchmark_group(format!("single_word_search_{scale}"));
        group.throughput(Throughput::Elements(scale as u64));
        group.sample_size(20);
        group.measurement_time(Duration::from_secs(5));

        let (bench_context, _) = rt.block_on(async {
            let bench_context = BenchContext::new().await;
            let index_id = bench_context.create_index().await;

            // Insert test documents
            let documents = generate_simple_docs(scale);
            for chunk in documents.chunks(500) {
                bench_context
                    .insert_documents(index_id, json!(chunk).try_into().unwrap())
                    .await
                    .unwrap();
            }

            (bench_context, index_id)
        });

        // Benchmark uncommitted search
        group.bench_function("uncommitted", |b| {
            b.to_async(&rt).iter(|| async {
                let search_params = black_box(json!({"term": "technology"}).try_into().unwrap());
                let result = bench_context.search(search_params).await.unwrap();
                black_box(result);
            });
        });

        // Commit and benchmark committed search
        rt.block_on(async {
            bench_context.commit_all().await.unwrap();
        });

        group.bench_function("committed", |b| {
            b.to_async(&rt).iter(|| async {
                let search_params = black_box(json!({"term": "technology"}).try_into().unwrap());
                let result = bench_context.search(search_params).await.unwrap();
                black_box(result);
            });
        });

        group.finish();
    }
}

/// Benchmark multi-word search performance
fn bench_multi_word_search(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let scale = 2000; // Smaller scale for multi-word complexity
    let mut group = c.benchmark_group("multi_word_search");
    group.throughput(Throughput::Elements(scale as u64));
    group.sample_size(15);
    group.measurement_time(Duration::from_secs(3));

    let (bench_context, _index_id) = rt.block_on(async {
        let bench_context = BenchContext::new().await;
        let index_id = bench_context.create_index().await;

        // Insert test documents with varied content
        let documents = (0..scale)
            .map(|i| {
                json!({
                    "id": i.to_string(),
                    "text": format!("technology software development programming number {}", i),
                })
            })
            .collect::<Vec<_>>();

        for chunk in documents.chunks(500) {
            bench_context
                .insert_documents(index_id, json!(chunk).try_into().unwrap())
                .await
                .unwrap();
        }

        bench_context.commit_all().await.unwrap();
        (bench_context, index_id)
    });

    // Test different query complexities
    let queries = [
        ("two_terms", json!({"term": "technology software"})),
        (
            "three_terms",
            json!({"term": "technology software development"}),
        ),
        (
            "with_limit",
            json!({"term": "technology development", "limit": 10}),
        ),
        (
            "with_threshold",
            json!({"term": "technology programming", "threshold": 0.8}),
        ),
    ];

    for (name, query) in queries {
        group.bench_function(name, |b| {
            b.to_async(&rt).iter(|| async {
                let search_params = black_box(query.clone().try_into().unwrap());
                let result = bench_context.search(search_params).await.unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

criterion_group!(
    minimal_benches,
    bench_single_word_search,
    bench_multi_word_search
);
criterion_main!(minimal_benches);
