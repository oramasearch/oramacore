use criterion::{black_box, criterion_group, criterion_main, Criterion};
use serde_json::json;
use std::fs;
use std::time::Duration;
use tokio::runtime::Runtime;

use std::{path::PathBuf, sync::Arc};

use anyhow::Result;
use futures::{future::BoxFuture, FutureExt};
use tempfile;
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
        DocumentList, IndexId, SearchParams, WriteApiKey,
    },
    web_server::HttpConfig,
    OramacoreConfig,
};

/// Generate temporary path
pub fn generate_new_path() -> PathBuf {
    let tmp_dir = tempfile::tempdir().expect("Cannot create temp dir");
    let dir = tmp_dir.path().to_path_buf();
    std::fs::create_dir_all(dir.clone()).expect("Cannot create dir");
    dir
}

/// Create minimal config
pub fn create_minimal_config() -> OramacoreConfig {
    OramacoreConfig {
        log: Default::default(),
        http: HttpConfig {
            host: "127.0.0.1".parse().unwrap(),
            port: 2224, // Unique port
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
                local: false,
                host: "localhost".to_string(),
                port: None,
                model: "disabled".to_string(),
                api_key: String::new(),
            },
            remote_llms: None,
        },
        writer_side: WriteSideConfig {
            master_api_key: ApiKey::try_new("bench-master-key").unwrap(),
            output: OutputSideChannelType::InMemory { capacity: 100 },
            config: CollectionsWriterConfig {
                data_dir: generate_new_path(),
                embedding_queue_limit: 10,
                default_embedding_model: OramaModelSerializable(OramaModel::BgeSmall),
                insert_batch_commit_size: 500,
                javascript_queue_limit: 100,
                commit_interval: Duration::from_secs(60),
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
                insert_batch_commit_size: 500,
                commit_interval: Duration::from_secs(60),
                notifier: None,
                offload_field: OffloadFieldConfig {
                    unload_window: DurationString::from_string("5m".to_string()).unwrap(),
                    slot_count_exp: 4,
                    slot_size_exp: 2,
                },
            },
            analytics: None,
        },
    }
}

/// Quick wait function for benchmarks
pub async fn wait_quick<'i, 'b, I, R>(
    i: &'i I,
    f: impl Fn(&'i I) -> BoxFuture<'b, Result<R>>,
) -> Result<R>
where
    'b: 'i,
{
    const MAX_ATTEMPTS: usize = 100; // Much shorter timeout
    let mut attempts = 0;
    loop {
        attempts += 1;
        match f(i).await {
            Ok(r) => break Ok(r),
            Err(e) => {
                if attempts > MAX_ATTEMPTS {
                    break Err(e);
                }
                sleep(Duration::from_millis(50)).await // Longer sleep, fewer attempts
            }
        }
    }
}

/// Fast fulltext search benchmark
fn bench_fulltext_fast(c: &mut Criterion) {
    // Use a smaller runtime to avoid hanging
    let rt = Runtime::new().unwrap();

    let doc_count = 1000; // Small scale for fast benchmarks

    let (reader, writer, collection_id, read_api_key, write_api_key, index_id) =
        rt.block_on(async {
            let config = create_minimal_config();
            let master_api_key = config.writer_side.master_api_key;

            let (writer, reader) = build_orama(config).await.expect("Failed to build orama");
            let writer = writer.expect("Writer should be enabled");
            let reader = reader.expect("Reader should be enabled");

            // Create collection
            let collection_id = CollectionId::try_new("fast_bench_collection").unwrap();
            let write_api_key = ApiKey::try_new("fast_bench_write_key").unwrap();
            let read_api_key = ApiKey::try_new("fast_bench_read_key").unwrap();

            writer
                .create_collection(
                    master_api_key,
                    CreateCollection {
                        id: collection_id,
                        description: None,
                        read_api_key,
                        write_api_key,
                        language: None,
                        embeddings_model: None,
                    },
                )
                .await
                .expect("Failed to create collection");

            // Wait for collection
            wait_quick(&reader, |r| {
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

            // Create index
            let index_id = IndexId::try_new("fast_bench_index").unwrap();

            writer
                .create_index(
                    WriteApiKey::from_api_key(write_api_key),
                    collection_id,
                    CreateIndexRequest {
                        index_id,
                        embedding: None,
                    },
                )
                .await
                .expect("Failed to create index");

            // Wait for index
            wait_quick(&reader, |r| {
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

            // Load games.json data
            let games_data =
                fs::read_to_string("benches/games.json").expect("Failed to read games.json");
            let games: Vec<serde_json::Value> =
                serde_json::from_str(&games_data).expect("Failed to parse games.json");
            let documents = games.into_iter().take(doc_count).collect::<Vec<_>>();

            writer
                .insert_documents(
                    WriteApiKey::from_api_key(write_api_key),
                    collection_id,
                    index_id,
                    json!(documents).try_into().unwrap(),
                )
                .await
                .expect("Failed to insert documents");

            (
                reader,
                writer,
                collection_id,
                read_api_key,
                WriteApiKey::from_api_key(write_api_key),
                index_id,
            )
        });

    // Game-based search benchmarks
    c.bench_function("fulltext_rpg_search", |b| {
        b.to_async(&rt).iter(|| async {
            let search_params = black_box(json!({"term": "RPG"}).try_into().unwrap());
            let result = reader
                .search(
                    read_api_key,
                    collection_id,
                    search_params,
                    AnalyticSearchEventInvocationType::Direct,
                )
                .await
                .expect("Search failed");
            black_box(result);
        });
    });

    // Multi-word game search benchmark
    c.bench_function("fulltext_adventure_indie", |b| {
        b.to_async(&rt).iter(|| async {
            let search_params = black_box(json!({"term": "adventure indie"}).try_into().unwrap());
            let result = reader
                .search(
                    read_api_key,
                    collection_id,
                    search_params,
                    AnalyticSearchEventInvocationType::Direct,
                )
                .await
                .expect("Search failed");
            black_box(result);
        });
    });

    // Commit the data and test committed search
    rt.block_on(async {
        writer.commit().await.expect("Writer commit failed");
        reader.commit().await.expect("Reader commit failed");
    });

    // Committed search benchmark
    c.bench_function("fulltext_committed_platform", |b| {
        b.to_async(&rt).iter(|| async {
            let search_params = black_box(json!({"term": "platform"}).try_into().unwrap());
            let result = reader
                .search(
                    read_api_key,
                    collection_id,
                    search_params,
                    AnalyticSearchEventInvocationType::Direct,
                )
                .await
                .expect("Search failed");
            black_box(result);
        });
    });
}

criterion_group!(fast_benches, bench_fulltext_fast);
criterion_main!(fast_benches);
