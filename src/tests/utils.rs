use std::{
    net::{SocketAddr, TcpListener},
    path::PathBuf,
    str::FromStr,
    sync::{Arc, OnceLock},
    time::Duration,
};

use anyhow::{bail, Result};
use duration_string::DurationString;
use fake::Fake;
use fake::Faker;
use fastembed::{
    EmbeddingModel, InitOptions, InitOptionsUserDefined, Pooling, TextEmbedding, TokenizerFiles,
    UserDefinedEmbeddingModel,
};
use futures::{future::BoxFuture, FutureExt};
use grpc_def::Embedding;
use http::uri::Scheme;
use tokio::time::sleep;
use tonic::{transport::Server, Status};
use tracing::info;

use crate::{
    ai::{AIServiceConfig, AIServiceLLMConfig, OramaModel},
    build_orama,
    collection_manager::sides::{
        hooks::{HooksRuntimeConfig, SelectEmbeddingsPropertiesHooksRuntimeConfig},
        index::CreateIndexRequest,
        CollectionStats, CollectionsWriterConfig, IndexesConfig, InputSideChannelType,
        OramaModelSerializable, OutputSideChannelType, ReadSide, ReadSideConfig, WriteSide,
        WriteSideConfig,
    },
    types::{
        ApiKey, CollectionId, CreateCollection, DocumentList, IndexId, InsertDocumentsResult,
        SearchParams, SearchResult,
    },
    web_server::HttpConfig,
    OramacoreConfig,
};
use anyhow::Context;

pub fn init_log() {
    if std::env::var("LOG").is_ok() {
        std::env::set_var("RUST_LOG", "oramacore=trace,warn");
    }
    let _ = tracing_subscriber::fmt::try_init();
}

pub fn generate_new_path() -> PathBuf {
    let tmp_dir = tempfile::tempdir().expect("Cannot create temp dir");
    info!("Temp dir: {:?}", tmp_dir.path());
    let dir = tmp_dir.path().to_path_buf();
    std::fs::create_dir_all(dir.clone()).expect("Cannot create dir");
    dir
}

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
            hooks: hooks_runtime_config(),
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
        },
        reader_side: ReadSideConfig {
            input: InputSideChannelType::InMemory { capacity: 100 },
            config: IndexesConfig {
                data_dir: generate_new_path(),
                // Lot of tests commit to test it.
                // So, we put an high value to avoid problems.
                insert_batch_commit_size: 10_000,
                commit_interval: Duration::from_secs(3_000),
                notifier: None,
            },
        },
    }
}

static CELL: OnceLock<Result<(Arc<TextEmbedding>, Arc<TextEmbedding>)>> = OnceLock::new();
pub async fn create_grpc_server() -> Result<SocketAddr> {
    let model = EmbeddingModel::BGESmallENV15;

    let text_embedding = CELL.get_or_init(|| {
        let cwd = std::env::current_dir().unwrap();
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
        let c = grpc_def::llm_service_client::LlmServiceClient::connect(format!("http://{}", addr))
            .await;
        if c.is_ok() {
            break;
        }
        sleep(Duration::from_millis(100)).await;
    }

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

pub struct TestContext {
    reader: Arc<ReadSide>,
    writer: Arc<WriteSide>,
    pub master_api_key: ApiKey,
}
impl TestContext {
    pub async fn new() -> Self {
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
            master_api_key: self.master_api_key,
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

pub struct TestCollectionClient {
    pub collection_id: CollectionId,
    pub write_api_key: ApiKey,
    pub read_api_key: ApiKey,
    master_api_key: ApiKey,
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

    pub async fn delete(&self) -> Result<()> {
        self.writer
            .delete_collection(self.master_api_key, self.collection_id)
            .await?;

        wait_for(self, |s| {
            let reader = s.reader.clone();
            let read_api_key = s.read_api_key;
            let collection_id = s.collection_id;
            async move {
                if reader.collection_stats(read_api_key, collection_id).await.is_err() {
                    return Ok(());
                }
                bail!(
                    "Collection still exists: {}",
                    collection_id
                );
            }
            .boxed()
        }).await?;

        Ok(())
    }

    pub async fn reader_stats(&self) -> Result<CollectionStats> {
        self.reader
            .collection_stats(self.read_api_key, self.collection_id)
            .await
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

pub struct TestIndexClient {
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
                let stats = reader.collection_stats(read_api_key, collection_id).await?;
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
                let stats = reader.collection_stats(read_api_key, collection_id).await?;
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

    pub async fn uncheck_delete_documents(&self, ids: Vec<String>) -> Result<()> {
        self.writer
            .delete_documents(self.write_api_key, self.collection_id, self.index_id, ids)
            .await?;

        sleep(Duration::from_millis(100)).await;

        Ok(())
    }
}
