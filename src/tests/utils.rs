use std::{
    collections::HashMap,
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, Once},
    time::Duration,
};

use anyhow::{bail, Result};
use axum::{response::sse::Event, Json};
use duration_string::DurationString;
use fake::Fake;
use fake::Faker;
use futures::{future::BoxFuture, FutureExt};
use oramacore_lib::hook_storage::HookType;
use pyo3::Python;
use tokio::{
    sync::{mpsc, RwLock},
    time::sleep,
};
use tokio_stream::wrappers::ReceiverStream;
use tracing::warn;

use crate::ai::AIServiceEmbeddingsConfig;
use crate::ai::PythonLoggingLevel;
use crate::{
    ai::{AIServiceConfig, AIServiceLLMConfig},
    build_orama,
    collection_manager::sides::{
        read::{
            CollectionCommitConfig, CollectionStats, IndexesConfig, OffloadFieldConfig, ReadSide,
            ReadSideConfig, SearchRequest, ShelfWithDocuments,
        },
        write::{
            CollectionsWriterConfig, TempIndexCleanupConfig, WriteError, WriteSide, WriteSideConfig,
        },
        InputSideChannelType, OutputSideChannelType, ReplaceIndexReason,
    },
    python::embeddings::Model,
    types::{
        ApiKey, CollectionId, CollectionStatsRequest, CreateCollection, CreateIndexRequest,
        DescribeCollectionResponse, Document, DocumentList, IndexId, InsertDocumentsResult,
        LanguageDTO, ReadApiKey, ReplaceIndexRequest, SearchParams, SearchResult,
        TypeParsingStrategies, UpdateDocumentRequest, UpdateDocumentsResult, WriteApiKey,
    },
    web_server::HttpConfig,
    HooksConfig, OramacoreConfig,
};
use crate::{collection_manager::sides::read::ReadError, types::SearchResultHit};
use anyhow::Context;
use oramacore_lib::pin_rules::PinRule;
use oramacore_lib::shelves::{Shelf, ShelfId};

// Ensure Python is initialized only once across all tests.
// Python::initialize() must be called from a consistent thread and only once
// to avoid race conditions that can cause SIGSEGV.
static PYTHON_INIT: Once = Once::new();
// Ensure logging and environment variables are set only once to avoid
// race conditions with unsafe std::env::set_var in multi-threaded tests.
static LOG_INIT: Once = Once::new();

pub fn init_log() {
    PYTHON_INIT.call_once(|| {
        Python::initialize();
    });

    LOG_INIT.call_once(|| {
        if let Ok(a) = std::env::var("LOG") {
            if a == "info" {
                unsafe { std::env::set_var("RUST_LOG", "oramacore=info,oramacore_lib=info,warn") };
            } else {
                unsafe {
                    std::env::set_var("RUST_LOG", "oramacore=trace,oramacore_lib=trace,warn")
                };
            }
        }
        let _ = tracing_subscriber::fmt::try_init();
    });
}

pub fn generate_new_path() -> PathBuf {
    let tmp_dir = tempfile::tempdir().expect("Cannot create temp dir");
    let dir = tmp_dir.path().to_path_buf();
    std::fs::create_dir_all(dir.clone()).expect("Cannot create dir");
    dir
}

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
            // Use a process-unique cache directory to avoid race conditions
            // when multiple test processes access ONNX model files simultaneously.
            models_cache_dir: "/tmp/fastembed_cache".to_string(),
            total_threads: 4,
            embeddings: Some(AIServiceEmbeddingsConfig {
                automatic_embeddings_selector: None,
                execution_providers: vec!["CPUExecutionProvider".to_string()],
                default_model_group: "all".to_string(),
                total_threads: 4,
                dynamically_load_models: true,
                level: PythonLoggingLevel::Error,
            }),
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
            hooks: HooksConfig::default(),
            config: CollectionsWriterConfig {
                data_dir: generate_new_path(),
                embedding_queue_limit: 50,
                default_embedding_model: Model::BGESmall,
                // Lot of tests commit to test it.
                // So, we put an high value to avoid problems.
                insert_batch_commit_size: 10_000,
                javascript_queue_limit: 10_000,
                commit_interval: Duration::from_secs(3_000),
                temp_index_cleanup: TempIndexCleanupConfig {
                    cleanup_interval: Duration::from_secs(3600), // 1 hour
                    max_age: Duration::from_secs(43200),         // 12 hours
                },
            },
            jwt: None,
        },
        reader_side: ReadSideConfig {
            master_api_key: None,
            input: InputSideChannelType::InMemory { capacity: 100 },
            config: IndexesConfig {
                data_dir: generate_new_path(),
                // Lot of tests commit to test it.
                // So, we put an high value to avoid problems.
                insert_batch_commit_size: 10_000,
                commit_interval: Duration::from_secs(3_000),
                notifier: None,
                // Not offload during tests
                offload_field: OffloadFieldConfig {
                    unload_window: DurationString::from_string("30m".to_string()).unwrap(),
                    slot_count_exp: 8,
                    slot_size_exp: 4,
                },
                // Use default commit thresholds for tests
                collection_commit: CollectionCommitConfig {
                    operation_threshold: 300,
                    time_threshold: Duration::from_secs(5 * 60),
                },
                force_commit: u32::MAX,
            },
            hooks: HooksConfig::default(),
            analytics: None,
            jwt: None,
            secrets_manager: None,
        },
    }
}

pub async fn create_ai_server_mock(
    completitions_mock: Arc<RwLock<Vec<Vec<String>>>>,
    completitions_req_mock: Arc<RwLock<Vec<serde_json::Value>>>,
) -> Result<SocketAddr> {
    use axum::routing::post;

    let (sender, receiver) = tokio::sync::oneshot::channel();
    tokio::task::spawn(async {
        use axum::Router;
        use std::net::*;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);

        let app = Router::new().route(
            "/v1/chat/completions",
            post(|r: Json<serde_json::Value>| async move {
                use axum::response::Sse;
                use serde_json::json;

                let mut lock = completitions_req_mock.write().await;
                lock.push(r.0);
                drop(lock);

                let (http_sender, http_receiver) = mpsc::channel::<Result<Event, &str>>(10);
                tokio::spawn(async move {
                    let mut lock = completitions_mock.write().await;
                    let first = lock.remove(0);
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs();
                    let model = "gpt-3.5-turbo-0301";
                    let id = "chatcmpl-mock";
                    for (idx, s) in first.into_iter().enumerate() {
                        let chunk = json!({
                            "id": id,
                            "object": "chat.completion.chunk",
                            "created": now,
                            "model": model,
                            "choices": [
                                {
                                    "delta": { "content": s },
                                    "index": idx,
                                    "finish_reason": null
                                }
                            ]
                        });
                        let ev = Event::default().json_data(chunk).unwrap();
                        http_sender.send(Ok(ev)).await.unwrap();
                    }
                    let ev = Event::default().data("[DONE]");
                    http_sender.send(Ok(ev)).await.unwrap();
                });

                let rx_stream = ReceiverStream::new(http_receiver);
                Sse::new(rx_stream).keep_alive(
                    axum::response::sse::KeepAlive::new().interval(Duration::from_secs(15)),
                )
            }),
        );
        let listener = tokio::net::TcpListener::bind(addr).await.unwrap();

        let _ = sender.send(addr);

        axum::serve(listener, app).await.unwrap();
    });

    let addr = receiver.await.unwrap();

    Ok(addr)
}

pub async fn wait_for<'i, 'b, I, R>(
    i: &'i I,
    f: impl Fn(&'i I) -> BoxFuture<'b, Result<R>>,
) -> Result<R>
where
    'b: 'i,
{
    // 20 msec * 2_000 attempts = 40_000 msec = 40 sec
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

    /// Creates a new TestContext with JWT configuration for the reader side.
    /// This is useful for testing JWT validation flows end-to-end.
    pub async fn new_with_jwt_config(jwt_config: crate::auth::JwtConfig) -> Self {
        let mut config = create_oramacore_config();
        config.writer_side.master_api_key = Self::generate_api_key();
        config.reader_side.jwt = Some(jwt_config);
        Self::new_with_config(config).await
    }

    pub async fn reload(self) -> Self {
        self.reader.stop().await.unwrap();

        let config = self.config.clone();
        let (writer, reader) = build_orama(config.clone()).await.unwrap();
        let writer = writer.unwrap();
        let reader = reader.unwrap();

        TestContext {
            config,
            reader,
            writer,
            master_api_key: self.master_api_key,
        }
    }

    pub async fn get_writer_collections(&self) -> Vec<DescribeCollectionResponse> {
        self.writer
            .list_collections(self.master_api_key)
            .await
            .unwrap()
    }

    pub async fn create_collection(&self) -> Result<TestCollectionClient> {
        let id = Self::generate_collection_id();
        let write_api_key = Self::generate_api_key();
        let read_api_key_raw = Self::generate_api_key();
        let read_api_key = ReadApiKey::from_api_key(read_api_key_raw);

        self.writer
            .create_collection(
                self.master_api_key,
                CreateCollection {
                    id,
                    description: None,
                    mcp_description: None,
                    read_api_key: read_api_key_raw,
                    write_api_key,
                    language: None,
                    embeddings_model: Some(Model::BGESmall),
                },
            )
            .await?;

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
            master_api_key: self.master_api_key,
            reader: self.reader.as_ref(),
            writer: self.writer.as_ref(),
        })
    }

    pub async fn commit_all(&self) -> Result<()> {
        self.writer.commit().await?;
        // Force commit in tests to ensure all data is written to disk
        self.reader.commit(true).await?;
        Ok(())
    }

    pub fn generate_collection_id() -> CollectionId {
        let id: String = Faker.fake();
        CollectionId::try_new(id).unwrap()
    }
    pub fn generate_api_key() -> ApiKey {
        let id: String = Faker.fake();
        ApiKey::try_new(id).unwrap()
    }
}

impl Drop for TestContext {
    fn drop(&mut self) {
        // First, stop the reader while in tokio context
        let output = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                // Some tests may close the connection intentionally
                // so we ignore the error here
                self.reader.stop().await
            })
        });

        if let Err(err) = output {
            warn!("Error stopping reader: {}", err);
        }

        // Then acquire GIL to ensure Python objects are properly cleaned up
        // This must happen after the async cleanup to avoid deadlocks
        Python::attach(|_py| {
            // GIL is held, any Python object drops that happen now are safe
        });
    }
}

pub struct TestCollectionClient<'test> {
    pub collection_id: CollectionId,
    pub write_api_key: WriteApiKey,
    pub read_api_key: ReadApiKey,
    master_api_key: ApiKey,
    pub reader: &'test ReadSide,
    pub writer: &'test WriteSide,
}
impl TestCollectionClient<'_> {
    pub async fn create_index(&self) -> Result<TestIndexClient> {
        self.create_index_with_explicit_type_strategy(TypeParsingStrategies::default())
            .await
    }

    pub async fn create_index_with_explicit_type_strategy(
        &self,
        type_strategy: TypeParsingStrategies,
    ) -> Result<TestIndexClient> {
        let index_id = Self::generate_index_id("index");
        self.writer
            .create_index(
                self.write_api_key,
                self.collection_id,
                CreateIndexRequest {
                    index_id,
                    embedding: None,
                    type_strategy,
                },
            )
            .await?;

        wait_for(self, |s| {
            let reader = s.reader;
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

        self.get_test_index_client(index_id)
    }

    pub async fn create_temp_index(&self, copy_from: IndexId) -> Result<TestIndexClient> {
        let index_id = Self::generate_index_id("temp");
        self.writer
            .create_temp_index(
                self.write_api_key,
                self.collection_id,
                copy_from,
                CreateIndexRequest {
                    index_id,
                    embedding: None,
                    type_strategy: Default::default(),
                },
            )
            .await?;

        sleep(Duration::from_millis(50)).await;

        self.get_test_index_client(index_id)
    }

    pub fn get_test_index_client(&self, index_id: IndexId) -> Result<TestIndexClient> {
        Ok(TestIndexClient {
            collection_id: self.collection_id,
            index_id,
            write_api_key: self.write_api_key,
            read_api_key: self.read_api_key.clone(),
            reader: self.reader,
            writer: self.writer,
        })
    }

    pub async fn insert_hook(&self, hook_type: HookType, code: String) -> Result<()> {
        let collection = self
            .writer
            .get_collection(self.collection_id, self.write_api_key)
            .await?;

        collection.set_hook(hook_type, code).await?;

        Ok(())
    }

    pub async fn replace_index(
        &self,
        runtime_index_id: IndexId,
        temp_index_id: IndexId,
    ) -> Result<()> {
        let req = ReplaceIndexRequest {
            runtime_index_id,
            temp_index_id,
            reference: None,
        };
        self.writer
            .replace_index(
                self.write_api_key,
                self.collection_id,
                req,
                ReplaceIndexReason::IndexResynced,
            )
            .await?;

        wait_for(self, |s| {
            let reader = s.reader;
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
                let old_index = stats
                    .indexes_stats
                    .iter()
                    .find(|index| index.id == temp_index_id);

                if old_index.is_some() {
                    bail!("Old index still exists");
                }

                Ok(())
            }
            .boxed()
        })
        .await?;

        Ok(())
    }

    pub async fn delete(&self) -> Result<()> {
        self.writer
            .delete_collection(self.master_api_key, self.collection_id)
            .await?;

        wait_for(self, |s| {
            let reader = s.reader;
            let read_api_key = s.read_api_key.clone();
            let collection_id = s.collection_id;
            async move {
                if reader
                    .collection_stats(
                        &read_api_key,
                        collection_id,
                        CollectionStatsRequest { with_keys: false },
                    )
                    .await
                    .is_err()
                {
                    return Ok(());
                }
                bail!("Collection still exists: {collection_id}");
            }
            .boxed()
        })
        .await?;

        Ok(())
    }

    pub async fn reader_stats(&self) -> Result<CollectionStats> {
        self.reader
            .collection_stats(
                &self.read_api_key,
                self.collection_id,
                CollectionStatsRequest { with_keys: false },
            )
            .await
            .context("")
    }

    pub async fn search(&self, search_params: SearchParams) -> Result<SearchResult, ReadError> {
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
    }

    pub async fn rebuild_index(&self, language: LanguageDTO) -> Result<()> {
        self.writer
            .reindex(
                self.write_api_key,
                self.collection_id,
                language,
                Model::BGESmall,
                None,
            )
            .await?;

        Ok(())
    }

    /// Batch retrieves documents by their string IDs for testing purposes.
    pub async fn batch_get_documents(
        &self,
        doc_ids: Vec<String>,
    ) -> Result<HashMap<String, Document>> {
        self.reader
            .batch_get_documents(&self.read_api_key, self.collection_id, doc_ids)
            .await
            .map_err(|e| e.into())
    }

    pub async fn insert_shelf(&self, shelf: Shelf<String>) -> Result<()> {
        let collection = self
            .writer
            .get_collection(self.collection_id, self.write_api_key)
            .await?;

        collection
            .insert_shelf(shelf)
            .await
            .context("Failed to insert shelf")?;

        Ok(())
    }

    pub async fn delete_shelf(&self, shelf_id: String) -> Result<()> {
        let collection = self
            .writer
            .get_collection(self.collection_id, self.write_api_key)
            .await?;

        let shelf_id =
            ShelfId::try_new(shelf_id).map_err(|e| anyhow::anyhow!("Invalid shelf ID: {e}"))?;

        collection
            .delete_shelf(shelf_id)
            .await
            .context("Failed to delete shelf")?;

        Ok(())
    }

    pub async fn list_shelves(&self) -> Result<Vec<Shelf<String>>> {
        let collection = self
            .writer
            .get_collection(self.collection_id, self.write_api_key)
            .await?;

        let shelves = collection
            .list_shelves()
            .await
            .context("Failed to list shelves from writer")?;

        Ok(shelves)
    }

    pub async fn get_shelf_documents(&self, shelf_id: String) -> Result<ShelfWithDocuments> {
        let collection = self
            .reader
            .get_collection(self.collection_id, &self.read_api_key)
            .await?;

        let shelf_id_typed =
            ShelfId::try_new(shelf_id).map_err(|e| anyhow::anyhow!("Invalid shelf ID: {e}"))?;

        let shelf_with_documents = collection
            .get_shelf_documents(shelf_id_typed)
            .await
            .context("Failed to get shelf documents from reader")?;

        Ok(shelf_with_documents)
    }

    fn generate_index_id(prefix: &str) -> IndexId {
        let id: String = format!("{}_{}", prefix, Faker.fake::<String>());
        IndexId::try_new(id).unwrap()
    }
}

pub struct TestIndexClient<'test> {
    pub collection_id: CollectionId,
    pub index_id: IndexId,
    pub write_api_key: WriteApiKey,
    pub read_api_key: ReadApiKey,
    pub reader: &'test ReadSide,
    pub writer: &'test WriteSide,
}
impl TestIndexClient<'_> {
    pub async fn unchecked_insert_documents(
        &self,
        documents: DocumentList,
    ) -> Result<InsertDocumentsResult, WriteError> {
        self.writer
            .insert_documents(
                self.write_api_key,
                self.collection_id,
                self.index_id,
                documents,
            )
            .await
    }

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
            let reader = s.reader;
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

    pub async fn delete_documents(&self, ids: Vec<String>) -> Result<()> {
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
        let expected_document_count = document_count - ids.len();

        self.writer
            .delete_documents(self.write_api_key, self.collection_id, self.index_id, ids)
            .await?;

        wait_for(self, |s| {
            let reader = s.reader;
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

    pub async fn delete(&self) -> Result<()> {
        self.writer
            .delete_index(self.write_api_key, self.collection_id, self.index_id)
            .await?;

        wait_for(self, |s| {
            let reader = s.reader;
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
                if stats
                    .indexes_stats
                    .iter()
                    .find(|index| index.id == self.index_id)
                    .ok_or_else(|| anyhow::anyhow!("Index not found"))
                    .is_ok()
                {
                    bail!("Index still exists");
                }

                Ok(())
            }
            .boxed()
        })
        .await?;

        Ok(())
    }

    pub async fn update_documents(
        &self,
        req: UpdateDocumentRequest,
    ) -> Result<UpdateDocumentsResult, WriteError> {
        self.writer
            .update_documents(self.write_api_key, self.collection_id, self.index_id, req)
            .await
    }

    pub async fn uncheck_delete_documents(&self, ids: Vec<String>) -> Result<()> {
        self.writer
            .delete_documents(self.write_api_key, self.collection_id, self.index_id, ids)
            .await?;

        sleep(Duration::from_millis(100)).await;

        Ok(())
    }

    pub async fn insert_pin_rules(&self, rule: PinRule<String>) -> Result<()> {
        let collection = self
            .writer
            .get_collection(self.collection_id, self.write_api_key)
            .await?;

        let rule_id = rule.id.clone();

        collection
            .insert_merchandising_pin_rule(rule)
            .await
            .context("Failed to insert pin_rule")?;

        wait_for(self, |s| {
            let reader = s.reader;
            let read_api_key = s.read_api_key.clone();
            let collection_id = s.collection_id;
            let r = &rule_id;
            async move {
                let collection = reader.get_collection(collection_id, &read_api_key).await?;
                let reader = collection.get_pin_rules_reader("test").await;
                let ids = reader.get_rule_ids();

                if ids.contains(r) {
                    return Ok(());
                }

                bail!("Pin rule does not exist");
            }
            .boxed()
        })
        .await?;

        Ok(())
    }

    pub async fn delete_pin_rules(&self, rule_id: String) -> Result<()> {
        let collection = self
            .writer
            .get_collection(self.collection_id, self.write_api_key)
            .await?;

        collection
            .delete_merchandising_pin_rule(rule_id.clone())
            .await
            .context("Failed to delete pin_rule")?;

        wait_for(self, |s| {
            let reader = s.reader;
            let read_api_key = s.read_api_key.clone();
            let collection_id = s.collection_id;
            let r = &rule_id;
            async move {
                let collection = reader.get_collection(collection_id, &read_api_key).await?;
                let reader = collection.get_pin_rules_reader("test").await;
                let ids = reader.get_rule_ids();

                if !ids.contains(r) {
                    return Ok(());
                }

                bail!("Pin rule still exist");
            }
            .boxed()
        })
        .await?;

        Ok(())
    }
}

pub fn extrapolate_ids_from_result(result: &SearchResult) -> Vec<String> {
    extrapolate_ids_from_result_hits(&result.hits)
}

pub fn extrapolate_ids_from_result_hits(hits: &[SearchResultHit]) -> Vec<String> {
    hits.iter()
        .map(|h| {
            let id = h.id.clone();
            assert!(h.document.is_some(), "Document not found");
            id.split(":").nth(1).map(|id| id.to_string()).unwrap()
        })
        .collect()
}
