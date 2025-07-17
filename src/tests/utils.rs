use std::{
    net::{SocketAddr, TcpListener},
    path::PathBuf,
    sync::{Arc, OnceLock},
    time::Duration,
};

use anyhow::{bail, Result};
use axum::{response::sse::Event, Json};
use fake::Fake;
use fake::Faker;
use fastembed::{
    EmbeddingModel, InitOptions, InitOptionsUserDefined, Pooling, TextEmbedding, TokenizerFiles,
    UserDefinedEmbeddingModel,
};
use futures::{future::BoxFuture, FutureExt};
use grpc_def::Embedding;
use hook_storage::HookType;
use http::uri::Scheme;
use tokio::{
    sync::{mpsc, RwLock},
    time::sleep,
};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{transport::Server, Status};
use tracing::warn;

use crate::{
    ai::{AIServiceConfig, AIServiceLLMConfig, OramaModel},
    build_orama,
    collection_manager::sides::{
        read::{
            AnalyticSearchEventInvocationType, CollectionStats, IndexesConfig, ReadSide,
            ReadSideConfig,
        },
        triggers::Trigger,
        write::{
            CollectionsWriterConfig, OramaModelSerializable, WriteError, WriteSide, WriteSideConfig,
        },
        InputSideChannelType, OutputSideChannelType, ReplaceIndexReason,
    },
    types::{
        ApiKey, CollectionId, CollectionStatsRequest, CreateCollection, CreateIndexRequest,
        DescribeCollectionResponse, DocumentList, IndexId, InsertDocumentsResult, LanguageDTO,
        ReplaceIndexRequest, SearchParams, SearchResult, UpdateDocumentRequest,
        UpdateDocumentsResult, WriteApiKey,
    },
    web_server::HttpConfig,
    OramacoreConfig,
};
use anyhow::Context;

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

pub fn generate_new_path() -> PathBuf {
    let tmp_dir = tempfile::tempdir().expect("Cannot create temp dir");
    let dir = tmp_dir.path().to_path_buf();
    std::fs::create_dir_all(dir.clone()).expect("Cannot create dir");
    dir
}

/*
pub fn hooks_runtime_config() -> HooksRuntimeConfig {
    HooksRuntimeConfig {
        select_embeddings_properties: SelectEmbeddingsPropertiesHooksRuntimeConfig {
            check_interval: DurationString::from_str("1s").unwrap(),
            max_idle_time: DurationString::from_str("1s").unwrap(),
            instances_count_per_code: 1,
            queue_capacity: 1,
            max_execution_time: DurationString::from_str("1s").unwrap(),
            max_startup_time: DurationString::from_str("1s").unwrap(),
        },
    }
}
*/

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
                host: "localhost".to_string(),
                port: 8000,
                model: "Qwen/Qwen2.5-3b-Instruct".to_string(),
            },
            remote_llms: None,
        },
        writer_side: WriteSideConfig {
            master_api_key: ApiKey::try_new("my-master-api-key").unwrap(),
            output: OutputSideChannelType::InMemory { capacity: 100 },
            // hooks: hooks_runtime_config(),
            config: CollectionsWriterConfig {
                data_dir: generate_new_path(),
                embedding_queue_limit: 50,
                default_embedding_model: OramaModelSerializable(OramaModel::BgeSmall),
                // Lot of tests commit to test it.
                // So, we put an high value to avoid problems.
                insert_batch_commit_size: 10_000,
                javascript_queue_limit: 10_000,
                commit_interval: Duration::from_secs(3_000),
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
            },
            analytics: None,
        },
    }
}

static CELL: OnceLock<Result<(Arc<TextEmbedding>, Arc<TextEmbedding>)>> = OnceLock::new();
pub async fn create_grpc_server() -> Result<SocketAddr> {
    let model = EmbeddingModel::BGESmallENV15;

    let text_embedding = CELL.get_or_init(|| {
        let mut cwd = std::env::current_dir()
            .unwrap()
            // This join is needed for the blow loop.
            .join("foo");

        let cwd = loop {
            let Some(parent) = cwd.parent() else {
                break None;
            };
            if std::fs::exists(parent.join(".custom_models")).unwrap_or(false) {
                break Some(parent.to_path_buf());
            }
            cwd = parent.to_path_buf();
        };

        let Some(cwd) = cwd else {
            return Err(anyhow::anyhow!(
                "Cache not found. Run 'download-test-model.sh' to create the cache locally"
            ));
        };

        let cache_dir = cwd.join(".custom_models");
        std::fs::create_dir_all(&cache_dir).expect("Cannot create cache dir");

        let init_option = InitOptions::new(model.clone())
            .with_cache_dir(cache_dir.clone())
            .with_show_download_progress(false);

        let context_evaluator_model_dir =
            cache_dir.join("sentenceTransformersParaphraseMultilingualMiniLML12v2");
        std::fs::create_dir_all(&context_evaluator_model_dir).expect("Cannot create cache dir");

        let onnx_file = std::fs::read(context_evaluator_model_dir.join("model.onnx"))
            .expect("Cache not found. Run 'download-test-model.sh' to create the cache locally");
        let config_file = std::fs::read(context_evaluator_model_dir.join("config.json"))
            .expect("Cache not found. Run 'download-test-model.sh' to create the cache locally");
        let tokenizer_file = std::fs::read(context_evaluator_model_dir.join("tokenizer.json"))
            .expect("Cache not found. Run 'download-test-model.sh' to create the cache locally");
        let special_tokens_map_file = std::fs::read(
            context_evaluator_model_dir.join("special_tokens_map.json"),
        )
        .expect("Cache not found. Run 'download-test-model.sh' to create the cache locally");
        let tokenizer_config_file = std::fs::read(
            context_evaluator_model_dir.join("tokenizer_config.json"),
        )
        .expect("Cache not found. Run 'download-test-model.sh' to create the cache locally");
        let tokenizer_files = TokenizerFiles {
            config_file,
            special_tokens_map_file,
            tokenizer_file,
            tokenizer_config_file,
        };
        let user_defined_model =
            UserDefinedEmbeddingModel::new(onnx_file, tokenizer_files).with_pooling(Pooling::Mean);

        // Try creating a TextEmbedding instance from the user-defined model
        let context_evaluator: TextEmbedding = TextEmbedding::try_new_from_user_defined(
            user_defined_model,
            InitOptionsUserDefined::default(),
        )
        .unwrap();

        let text_embedding = TextEmbedding::try_new(init_option)
            .with_context(|| format!("Failed to initialize the Fastembed: {model}"))?;
        Ok((Arc::new(text_embedding), Arc::new(context_evaluator)))
    });
    let (text_embedding, context_evaluator) = text_embedding.as_ref().unwrap().clone();

    let server = GRPCServer {
        fastembed_model: text_embedding,
        context_evaluator,
    };

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);

    tokio::spawn(async move {
        Server::builder()
            .add_service(grpc_def::llm_service_server::LlmServiceServer::new(server))
            .serve(addr)
            .await
            .expect("Nooope");
    });

    // Waiting for the server to start
    loop {
        let c =
            grpc_def::llm_service_client::LlmServiceClient::connect(format!("http://{addr}")).await;
        if c.is_ok() {
            break;
        }
        sleep(Duration::from_millis(100)).await;
    }

    Ok(addr)
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
                    let mut idx = 0;
                    for s in first {
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
                        idx += 1;
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

pub mod grpc_def {
    tonic::include_proto!("orama_ai_service");
}

pub struct GRPCServer {
    fastembed_model: Arc<TextEmbedding>,
    context_evaluator: Arc<TextEmbedding>,
}

#[tonic::async_trait]
impl grpc_def::llm_service_server::LlmService for GRPCServer {
    async fn check_health(
        &self,
        _req: tonic::Request<grpc_def::HealthCheckRequest>,
    ) -> Result<tonic::Response<grpc_def::HealthCheckResponse>, Status> {
        Ok(tonic::Response::new(grpc_def::HealthCheckResponse {
            status: "ok".to_string(),
        }))
    }

    async fn get_embedding(
        &self,
        req: tonic::Request<grpc_def::EmbeddingRequest>,
    ) -> Result<tonic::Response<grpc_def::EmbeddingResponse>, Status> {
        let req = req.into_inner();
        // `0` means `BgeSmall`
        // `6` means `SentenceTransformersParaphraseMultilingualMiniLML12v2`
        let model = match req.model {
            0 => self.fastembed_model.clone(),
            6 => self.context_evaluator.clone(),
            _ => return Err(Status::invalid_argument("Invalid model")),
        };

        let embed = model
            .embed(req.input, None)
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(tonic::Response::new(grpc_def::EmbeddingResponse {
            embeddings_result: embed
                .into_iter()
                .map(|v| Embedding { embeddings: v })
                .collect(),
            dimensions: 384,
        }))
    }
}

pub async fn wait_for<'i, 'b, I, R>(
    i: &'i I,
    f: impl Fn(&I) -> BoxFuture<'b, Result<R>>,
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

    pub async fn new_with_config(mut config: OramacoreConfig) -> Self {
        if config.ai_server.port == 0 {
            let address = create_grpc_server().await.unwrap();
            config.ai_server.host = address.ip().to_string();
            config.ai_server.port = address.port();
        }

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

impl Drop for TestContext {
    fn drop(&mut self) {
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
    }
}

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
            read_api_key: self.read_api_key,
            reader: self.reader.clone(),
            writer: self.writer.clone(),
        })
    }

    pub async fn insert_hook(&self, hook_type: HookType, code: String) -> Result<()> {
        let hook_storage = self
            .writer
            .get_hooks_storage(self.write_api_key, self.collection_id)
            .await?;

        hook_storage.insert_hook(hook_type, code).await?;

        Ok(())
    }

    pub async fn insert_trigger(
        &self,
        trigger: Trigger,
        trigger_id: Option<String>,
    ) -> Result<Trigger> {
        let trigger_interface = self
            .writer
            .get_triggers_manager(self.write_api_key, self.collection_id)
            .await?;

        let trigger = trigger_interface
            .insert_trigger(trigger, trigger_id.clone())
            .await?;

        wait_for(self, |coll| {
            let reader = coll.reader.clone();
            let read_api_key = coll.read_api_key;
            let collection_id = coll.collection_id;
            let trigger_id = trigger_id.clone();
            async move {
                let trigger_interface = reader
                    .get_triggers_manager(read_api_key, collection_id)
                    .await?;

                let trigger = trigger_interface.get_trigger(trigger_id.unwrap()).await?;

                if trigger.is_none() {
                    bail!("Trigger not found");
                }

                Ok(())
            }
            .boxed()
        })
        .await?;

        Ok(trigger)
    }

    pub async fn get_trigger(&self, trigger_id: String) -> Result<Option<Trigger>> {
        let trigger_interface = self
            .reader
            .get_triggers_manager(self.read_api_key, self.collection_id)
            .await?;

        trigger_interface.get_trigger(trigger_id).await.context("")
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
            let reader = s.reader.clone();
            let read_api_key = s.read_api_key;
            let collection_id = s.collection_id;
            async move {
                if reader
                    .collection_stats(
                        read_api_key,
                        collection_id,
                        CollectionStatsRequest { with_keys: false },
                    )
                    .await
                    .is_err()
                {
                    return Ok(());
                }
                bail!("Collection still exists: {}", collection_id);
            }
            .boxed()
        })
        .await?;

        Ok(())
    }

    pub async fn reader_stats(&self) -> Result<CollectionStats> {
        self.reader
            .collection_stats(
                self.read_api_key,
                self.collection_id,
                CollectionStatsRequest { with_keys: false },
            )
            .await
            .context("")
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

    pub async fn rebuild_index(&self, language: LanguageDTO) -> Result<()> {
        self.writer
            .reindex(
                self.write_api_key,
                self.collection_id,
                language,
                OramaModel::BgeSmall,
                None,
            )
            .await?;

        Ok(())
    }

    fn generate_index_id(prefix: &str) -> IndexId {
        let id: String = format!("{}_{}", prefix, Faker.fake::<String>());
        IndexId::try_new(id).unwrap()
    }
}

pub struct TestIndexClient {
    pub collection_id: CollectionId,
    pub index_id: IndexId,
    pub write_api_key: WriteApiKey,
    pub read_api_key: ApiKey,
    pub reader: Arc<ReadSide>,
    pub writer: Arc<WriteSide>,
}
impl TestIndexClient {
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

    pub async fn delete_documents(&self, ids: Vec<String>) -> Result<()> {
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
        let expected_document_count = document_count - ids.len();

        self.writer
            .delete_documents(self.write_api_key, self.collection_id, self.index_id, ids)
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
}
