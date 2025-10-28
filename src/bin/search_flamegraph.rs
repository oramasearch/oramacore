use anyhow::{Context, Result};
use core::panic;
use duration_string::DurationString;
use serde_json::json;
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};
use tokio::time::sleep;
use tracing::{level_filters::LevelFilter, warn};

use oramacore::{
    ai::{AIServiceConfig, AIServiceLLMConfig},
    build_orama,
    collection_manager::sides::{
        read::{IndexesConfig, OffloadFieldConfig, ReadSideConfig, SearchRequest},
        write::{CollectionsWriterConfig, TempIndexCleanupConfig, WriteSideConfig},
        InputSideChannelType, OutputSideChannelType,
    },
    python::embeddings::Model,
    types::{
        ApiKey, CollectionId, CreateCollection, CreateIndexRequest, DeleteDocuments, DocumentList,
        IndexEmbeddingsCalculation, IndexId, LanguageDTO, SearchParams, WriteApiKey,
    },
    web_server::HttpConfig,
    LogConfig, OramacoreConfig,
};

// Configuration constants
const INITIAL_DOCUMENTS: usize = 10_000;
const DELETE_DOCUMENTS: usize = 1_000;
const SEARCH_ITERATIONS: usize = 1_000;
const BATCH_SIZE: usize = 100;

// Common search terms for fulltext search
const SEARCH_TERMS: &[&str] = &[
    "Lorem ipsum",
    "dolor sit amet",
    "consectetur",
    "adipiscing elit",
    "sed do eiusmod",
    "tempor incididunt",
    "ut labore",
    "dolore magna",
    "aliqua",
    "enim ad minim",
];

/// Generate realistic text content for documents
fn generate_document_content(id: usize) -> String {
    let base_texts = [
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
        "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.",
        "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
        "Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium.",
        "Totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt.",
        "Explicabo nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit.",
        "Sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt.",
    ];

    let base_text = base_texts[id % base_texts.len()];
    let additional_words = [
        "technology",
        "innovation",
        "development",
        "solution",
        "research",
        "analysis",
        "implementation",
        "optimization",
        "performance",
        "quality",
    ];
    let additional_word = additional_words[id % additional_words.len()];

    format!("{base_text} Additional content about {additional_word} for document ID: {id}")
}

/// Create OramaCore configuration for testing
fn create_test_config(build: bool) -> OramacoreConfig {
    let temp_dir = std::env::temp_dir().join("oramacore_flamegraph_test");
    if build {
        std::fs::remove_dir_all(&temp_dir).ok(); // Clean up previous runs
        std::fs::create_dir_all(&temp_dir).expect("Cannot create temp dir");
    }

    OramacoreConfig {
        log: LogConfig {
            file_path: None,
            levels: HashMap::from([("oramacore".to_string(), LevelFilter::INFO)]),
            ..Default::default()
        },
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
        },
        writer_side: WriteSideConfig {
            master_api_key: ApiKey::try_new("master-key").unwrap(),
            output: OutputSideChannelType::InMemory { capacity: 1000 },
            config: CollectionsWriterConfig {
                data_dir: temp_dir.join("write"),
                embedding_queue_limit: 50,
                default_embedding_model: Model::BGESmall,
                insert_batch_commit_size: 100,
                javascript_queue_limit: 1000,
                commit_interval: Duration::from_secs(60),
                temp_index_cleanup: TempIndexCleanupConfig {
                    cleanup_interval: Duration::from_secs(60),
                    max_age: Duration::from_secs(3600),
                },
            },
            jwt: None,
        },
        reader_side: ReadSideConfig {
            master_api_key: None,
            input: InputSideChannelType::InMemory { capacity: 1000 },
            config: IndexesConfig {
                data_dir: temp_dir.join("read"),
                insert_batch_commit_size: 100,
                commit_interval: Duration::from_secs(60),
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

#[tokio::main]
async fn main() -> Result<()> {
    let arguments: Vec<_> = std::env::args().collect();
    let (build, run) = match arguments.get(1).map(|s| s.as_str()) {
        None => (true, true),
        Some("build") => (true, false),
        Some("run") => (false, true),
        _ => {
            panic!("Unknown command. only 'build' and 'run' are allowed. Nothing means both");
        }
    };

    // Initialize logging
    tracing_subscriber::fmt::init();
    println!("Starting search flamegraph analysis");

    // Build OramaCore instance
    let config = create_test_config(build);
    let (write_side, read_side) = build_orama(config)
        .await
        .context("Failed to build OramaCore")?;
    let write_side = write_side.context("Write side not configured")?;
    let read_side = read_side.context("Read side not configured")?;

    println!("OramaCore instance created successfully");

    let collection_id =
        CollectionId::try_new("test-collection").context("Invalid collection ID")?;
    let read_api_key = ApiKey::try_new("read-key").context("Invalid read API key")?;
    let write_api_key =
        WriteApiKey::ApiKey(ApiKey::try_new("write-key").context("Invalid write API key")?);
    let master_api_key = ApiKey::try_new("master-key").context("Invalid master API key")?;

    if build {
        // Step 1: Create collection
        write_side
            .create_collection(
                master_api_key,
                CreateCollection {
                    id: collection_id,
                    description: Some("Search performance test collection".to_string()),
                    mcp_description: Some(
                        "Performance test collection for search benchmarking".to_string(),
                    ),
                    language: Some(LanguageDTO::English),
                    read_api_key,
                    write_api_key: match write_api_key {
                        WriteApiKey::ApiKey(ref key) => *key,
                        _ => return Err(anyhow::anyhow!("Expected API key for write access")),
                    },
                    embeddings_model: None,
                },
            )
            .await
            .context("Failed to create collection")?;

        println!("Collection created: {collection_id:?}");

        // Step 2: Create index
        let index_id = IndexId::try_new("test-index").context("Invalid index ID")?;
        write_side
            .create_index(
                write_api_key,
                collection_id,
                CreateIndexRequest {
                    index_id,
                    embedding: Some(IndexEmbeddingsCalculation::AllProperties),
                    type_strategy: Default::default(),
                },
            )
            .await
            .context("Failed to create index")?;

        println!("Index created: {index_id:?}");

        // Step 3: Insert documents in batches
        println!("Inserting {INITIAL_DOCUMENTS} documents in batches of {BATCH_SIZE}");
        let start_time = Instant::now();

        for batch_start in (0..INITIAL_DOCUMENTS).step_by(BATCH_SIZE) {
            let batch_end = (batch_start + BATCH_SIZE).min(INITIAL_DOCUMENTS);
            let mut documents = Vec::new();

            for i in batch_start..batch_end {
                documents.push(json!({
                    "id": format!("doc_{}", i),
                    "title": format!("Document Title {}", i),
                    "content": generate_document_content(i),
                    "category": format!("category_{}", i % 10),
                    "timestamp": chrono::Utc::now().timestamp() + i as i64,
                }));
            }

            let document_list: DocumentList = json!(documents)
                .try_into()
                .context("Failed to convert documents to DocumentList")?;

            write_side
                .insert_documents(write_api_key, collection_id, index_id, document_list)
                .await
                .context("Failed to insert documents")?;

            if batch_start % (BATCH_SIZE * 10) == 0 {
                println!("Inserted {} documents", batch_start + BATCH_SIZE);
            }
        }

        let insert_duration = start_time.elapsed();
        println!("Inserted {INITIAL_DOCUMENTS} documents in {insert_duration:?}");

        // Step 4: Commit data
        println!("Committing initial data");
        let commit_start = Instant::now();
        read_side
            .commit()
            .await
            .context("Failed to commit initial data")?;
        let commit_duration = commit_start.elapsed();
        println!("Initial commit completed in {commit_duration:?}");

        // Wait for commit to complete
        sleep(Duration::from_millis(100)).await;

        // Step 5: Delete some documents
        println!("Deleting {DELETE_DOCUMENTS} random documents");
        let delete_start = Instant::now();

        let mut document_ids_to_delete = Vec::new();
        for i in 0..DELETE_DOCUMENTS {
            document_ids_to_delete.push(format!("doc_{}", i * 2)); // Delete every other document from the first batch
        }

        let delete_documents: DeleteDocuments = document_ids_to_delete;

        write_side
            .delete_documents(write_api_key, collection_id, index_id, delete_documents)
            .await
            .context("Failed to delete documents")?;

        let delete_duration = delete_start.elapsed();
        println!("Deleted {DELETE_DOCUMENTS} documents in {delete_duration:?}");

        // Step 6: Commit deletions
        println!("Committing deletions");
        let commit2_start = Instant::now();
        read_side
            .commit()
            .await
            .context("Failed to commit deletions")?;
        let commit2_duration = commit2_start.elapsed();
        println!("Deletion commit completed in {commit2_duration:?}");
    }

    if run {
        // Wait for commit to complete
        sleep(Duration::from_millis(100)).await;

        // Step 7: Warm up with a few searches
        println!("Warming up with initial searches");
        for i in 0..10 {
            let search_term = SEARCH_TERMS[i % SEARCH_TERMS.len()];
            let search_params: SearchParams = json!({
                "term": search_term,
                "limit": 10,
                "offset": 0
            })
            .try_into()
            .context("Failed to create search params")?;

            read_side
                .search(
                    read_api_key,
                    collection_id,
                    SearchRequest {
                        search_params,
                        analytics_metadata: None,
                        interaction_id: None,
                        search_analytics_event_origin: None,
                    },
                )
                .await
                .unwrap();
        }

        println!("Warmup completed");

        // Step 8: Performance benchmark - This is where we want to measure performance
        println!("Starting fulltext search benchmark with {SEARCH_ITERATIONS} iterations");

        let benchmark_start = Instant::now();
        let mut total_search_time = Duration::ZERO;
        let mut successful_searches = 0;

        for i in 0..SEARCH_ITERATIONS {
            let search_term = SEARCH_TERMS[i % SEARCH_TERMS.len()];
            let search_params: SearchParams = json!({
                "term": search_term,
                "limit": 20,
                "offset": 0
            })
            .try_into()
            .context("Failed to create search params")?;

            let search_start = Instant::now();

            match read_side
                .search(
                    read_api_key,
                    collection_id,
                    SearchRequest {
                        search_params,
                        analytics_metadata: None,
                        interaction_id: None,
                        search_analytics_event_origin: None,
                    },
                )
                .await
            {
                Ok(results) => {
                    let search_duration = search_start.elapsed();
                    total_search_time += search_duration;
                    successful_searches += 1;

                    if i % 100 == 0 {
                        println!(
                            "Completed {} searches, found {} results for '{}' in {:?}",
                            i + 1,
                            results.count,
                            search_term,
                            search_duration
                        );
                    }
                }
                Err(e) => {
                    warn!("Search {} failed: {}", i, e);
                }
            }
        }

        let benchmark_duration = benchmark_start.elapsed();

        // Final statistics
        println!("=== PERFORMANCE ANALYSIS COMPLETE ===");
        println!("Total benchmark time: {benchmark_duration:?}");
        println!("Successful searches: {successful_searches}/{SEARCH_ITERATIONS}");

        if successful_searches > 0 {
            let avg_search_time = total_search_time / successful_searches as u32;
            let searches_per_second = successful_searches as f64 / benchmark_duration.as_secs_f64();

            println!("Average search time: {avg_search_time:?}");
            println!("Searches per second: {searches_per_second:.2}");
            println!("Total search processing time: {total_search_time:?}");
        }

        println!("Analysis complete. Use the flamegraph commands to profile this binary.");
    }

    Ok(())
}
